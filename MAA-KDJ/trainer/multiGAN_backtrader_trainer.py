import torch
import torch.nn as nn
import copy

from utils.evaluate_visualization import *
from utils.backtrader import *
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F


import logging  # NEW
from utils.util import get_autocast_context
from torch.amp import GradScaler

scaler = GradScaler()


def train_multi_gan(args,
                    generators, discriminators,
                    dataloaders, test_dataloaders,
                    window_sizes, y_scaler,
                    train_xes, train_y,
                    val_xes, val_y, val_labels,
                    distill_epochs, cross_finetune_epochs,
                    num_epochs,
                    output_dir,
                    device,
                    backtrader,
                    init_GDweight=[
                        [1, 0, 0, 1.0],  # alphas_init
                        [0, 1, 0, 1.0],  # betas_init
                        [0., 0, 1, 1.0]  # gammas_init...
                    ],
                    final_GDweight=[
                        [0.333, 0.333, 0.333, 1.0],  # alphas_final
                        [0.333, 0.333, 0.333, 1.0],  # betas_final
                        [0.333, 0.333, 0.333, 1.0]  # gammas_final...,
                    ],
                    logger=None,
                    dynamic_weight=False):
    N = len(generators)

    assert N == len(discriminators)
    assert N == len(window_sizes)
    assert N >= 1

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5
    # 修改了return_loss_weight的值
    return_loss_weight = 0.5

    # 二元交叉熵【损失函数，可能会有问题
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizers_G = [torch.optim.AdamW(model.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    for model in generators]

    # 为每个优化器设置 ReduceLROnPlateau 调度器
    # 模式改为 'max' 以监控 CumReturn
    schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=16, min_lr=1e-7)
                  for optimizer in optimizers_G]

    optimizers_D = [torch.optim.Adam(model.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
                    for model in discriminators]

    best_epoch_mse = [-1 for _ in range(N)]
    best_epoch_acc = [-1 for _ in range(N)]

    # 定义生成历史记录的关键字
    keys = []
    g_keys = [f'G{i}' for i in range(1, N + 1)]
    d_keys = [f'D{i}' for i in range(1, N + 1)]
    MSE_g_keys = [f'MSE_G{i}' for i in range(1, N + 1)]
    val_loss_keys = [f'val_G{i}' for i in range(1, N + 1)]
    acc_keys = [f'acc_G{i}' for i in range(1, N + 1)]

    keys.extend(g_keys)
    keys.extend(d_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)
    keys.extend(acc_keys)

    d_g_keys = []
    for g_key in g_keys:
        for d_key in d_keys:
            d_g_keys.append(d_key + "_" + g_key)
    keys.extend(d_g_keys)

    # 创建包含每个值为np.zeros(num_epochs)的字典
    hists_dict = {key: np.zeros(num_epochs) for key in keys}

    best_mse = [float('inf') for _ in range(N)]
    best_acc = [0.0 for _ in range(N)]
    # 从 train_baseframe 移植的变量
    best_cum_return = [float("-inf") for _ in range(N)]  # 为每个生成器追踪最佳累积收益
    best_model_state = [None for _ in range(N)]  # 为每个生成器保存最佳模型

    patience_counter = 0
    patience = args.patience
    feature_num = train_xes[0].shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    for epoch in range(num_epochs):
        epo_start = time.time()

        if epoch < 10:
            weight_matrix = torch.tensor(init_GDweight).to(device)
        elif dynamic_weight:
            # —— 动态计算 G-D weight 矩阵 ——
            # 从上一轮的 validation loss 里拿到每个 G 的损失
            # val_loss_keys = ['val_G1', 'val_G2', ..., 'val_GN']
            losses = torch.stack([
                torch.tensor(hists_dict[val_loss_keys[i]][epoch - 1])
                for i in range(N)
            ]).to(device)  # shape: [N]

            # 性能 Perf_i = -loss_i，beta 控制“硬度”
            perf = torch.exp(-losses)  # shape: [N]
            probs = perf / perf.sum()  # shape: [N], softmax over generators

            # 构造训练 Generator 时用的 N×N 矩阵：每行都是同一分布
            weight_G = probs.unsqueeze(0).repeat(N, 1)  # shape: [N, N]
            weight_G = weight_G + torch.eye(N, device=device)

            # 构造训练 Discriminator 时的 N×(N+1) 矩阵：最后一列保持 1.0（给真数据）
            ones = torch.ones((N, 1), device=device)
            weight_matrix = torch.cat([weight_G, ones], dim=1)  # shape: [N, N+1]
        else:
            weight_matrix = torch.tensor(final_GDweight).to(device)

        keys = []
        keys.extend(g_keys)
        keys.extend(d_keys)
        keys.extend(MSE_g_keys)
        keys.extend(d_g_keys)

        loss_dict = {key: [] for key in keys}

        # use the gap the equalize the length of different generators
        gaps = [window_sizes[-1] - window_sizes[i] for i in range(N - 1)]

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloaders[-1]):
            # TODO: maybe try to random select a gap from the whole time windows
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)
            # print(x_last.shape, y_last.shape, label_last.shape)

            X = []
            Y = []
            LABELS = []

            for gap in gaps:
                X.append(x_last[:, gap:, :])
                Y.append(y_last[:, gap:, :])
                LABELS.append(label_last[:, gap:, :].long())
            X.append(x_last.to(device))
            Y.append(y_last.to(device))
            LABELS.append(label_last.to(device).long())

            for i in range(N):
                generators[i].eval()
                discriminators[i].train()

            loss_D, lossD_G = discriminate_fake(args, X, Y, LABELS,
                                                generators, discriminators,
                                                window_sizes, target_num,
                                                criterion, weight_matrix,
                                                device, mode="train_D",
                                                return_loss_weight=return_loss_weight)  # 传递 return_loss_weight

            # 3. 存入 loss_dict
            for i in range(N):
                loss_dict[d_keys[i]].append(loss_D[i].item())

            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    key = f'D{i}_G{j}'
                    loss_dict[key].append(lossD_G[i - 1, j - 1].item())

            # 根据批次的奇偶性交叉训练两个GAN
            # if batch_idx% 2 == 0:
            for optimizer_D in optimizers_D:
                optimizer_D.zero_grad()

            # TODO: to see whether there is need to add together
            scaler.scale(loss_D.sum(dim=0)).backward()

            for i in range(N):
                # optimizers_D[i].step()
                scaler.step(optimizers_D[i])
                scaler.update()

                discriminators[i].eval()
                generators[i].train()

            '''训练生成器'''
            weight = weight_matrix[:, :-1].clone().detach()  # [N, N]

            loss_G, loss_mse_G, loss_cls_G, loss_return_G, loss_combined_G = discriminate_fake(args, X, Y, LABELS,
                                                                                               # 增加返回的损失项
                                                                                               generators,
                                                                                               discriminators,
                                                                                               window_sizes, target_num,
                                                                                               criterion, weight,
                                                                                               device,
                                                                                               mode="train_G",
                                                                                               return_loss_weight=return_loss_weight)  # 传递 return_loss_weight

            for i in range(N):
                # loss_dict[g_keys[i]].append(loss_G[i].item()) # 这个是 GAN Loss，不是最终 Combined Loss
                loss_dict[g_keys[i]].append(loss_combined_G[i].item())  # 使用 Combined Loss 进行记录
                loss_dict["MSE_" + g_keys[i]].append(loss_mse_G[i].item())

            for optimizer_G in optimizers_G:
                optimizer_G.zero_grad()

            scaler.scale(loss_combined_G).sum(dim=0).backward()  # 对 Combined Loss 进行反向传播

            for optimizer_G in optimizers_G:
                # optimizer_G.step()
                scaler.step(optimizer_G)
                scaler.update()

        for key in loss_dict.keys():
            hists_dict[key][epoch] = np.mean(loss_dict[key])

        # # 移除原来的 improved 列表，因为早停逻辑将集中在 CumReturn
        # improved = [False] * N # 保持，用于 MSE/Acc 调度器，但不再用于早停

        # 评估阶段
        current_cum_returns = [None for _ in range(N)]  # 初始化每个生成器的当前累积收益

        for i in range(N):
            hists_dict[val_loss_keys[i]][epoch], hists_dict[acc_keys[i]][epoch] = validate_with_label(generators[i],
                                                                                                      val_xes[i], val_y,
                                                                                                      val_labels[i])
            # 对于每个生成器，根据 MSE 更新调度器
            schedulers[i].step(hists_dict[val_loss_keys[i]][epoch])

            # if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
            #     best_mse[i] = hists_dict[val_loss_keys[i]][epoch]
            #     # best_model_state[i] = copy.deepcopy(generators[i].state_dict()) # 最佳模型保存逻辑挪到 CumReturn 部分
            #     best_epoch_mse[i] = epoch + 1
            #     # improved[i] = True # 不再用于早停
            if hists_dict[acc_keys[i]][epoch] > best_acc[i]:
                best_acc[i] = hists_dict[acc_keys[i]][epoch]
                best_epoch_acc[i] = epoch + 1


        # 动态生成打印字符串
        log_str_mse = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]:.8f}"
            for i, key in enumerate(val_loss_keys)
        )
        log_str_acc = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch] * 100:.2f} %"
            for i, key in enumerate(acc_keys)
        )

        logging.info("Epoch %d | Validation MSE: %s | Accuracy: %s", epoch + 1, log_str_mse, log_str_acc)  # NEW

        # 金融指标评估和早停逻辑 (移植自 train_baseframe)
        if backtrader:
            improved_cum_return = False
            for i in range(N):
                _, val_metrics = validate_financial_metric(generators[i], train_xes[i], train_y, val_xes[i], val_y,
                                                           y_scaler)
                current_cum_returns[i] = val_metrics['cumulative_return']  # 获取当前的 CumReturn

                log_message = (
                    f"Generator {i + 1} | "
                    f"Sharpe: {val_metrics['sharpe_ratio']:.4f} | "
                    f"MaxDrawdown: {val_metrics['max_drawdown']:.2f}% | "
                    f"AnnualReturn: {val_metrics['annualized_return']:.2f}% | "
                    f"WinRate: {val_metrics['win_rate']:.2f}% | "
                    f"CumReturn: {val_metrics['cumulative_return']:.2f}% | "
                    f"AvgPosition: {val_metrics['avg_position']:.2f}%"
                )
                logging.info(log_message)

                # 早停逻辑修改：基于 CumReturn
                if current_cum_returns[i] > best_cum_return[i]:  # CumReturn 越高越好
                    best_cum_return[i] = current_cum_returns[i]
                    best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                    improved_cum_return = True  # 只要有一个生成器改善，就重置耐心
                    print(f'Generator {i + 1}: New best CumReturn found: {best_cum_return[i]:.2f}%')
                # 注意：这里不对每个生成器单独维护 patience_counter，而是整个训练过程共享一个

            # 所有生成器中最差的 CumReturn 也未改善，则耐心计数器增加
            if not improved_cum_return:
                patience_counter += 1
                print(
                    f'Patience remaining: {patience - patience_counter}, Current best CumReturns: {[f"{cr:.2f}%" for cr in best_cum_return]}, Current CumReturns: {[f"{cr:.2f}%" for cr in current_cum_returns]}')
            else:
                patience_counter = 0

            # 调度器现在基于最佳生成器的 CumReturn，或者可以考虑所有生成器的平均 CumReturn
            # 这里选择对所有生成器的 CumReturn 进行调度，或者只取最好的一个来调度。
            # 鉴于 schedulers 是按生成器分别定义的，我们让它们各自基于 val_loss_keys 进行调度，
            # 而早停则基于 cum_return。
            # 如果要让调度器也基于 CumReturn，则需要将 schedulers 的 mode 改为 'max'，并且 step 的参数改为 current_cum_returns[i]
            # for i in range(N):
            #     if current_cum_returns[i] is not None:
            #         schedulers[i].step(current_cum_returns[i]) # 如果调度器模式是 'max'
        else:  # 如果 backtrader 不为 True，则回退到基于 best_mse 的早停
            # 检查是否有任何生成器的 MSE 改善
            mse_improved_this_epoch = False
            for i in range(N):
                if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
                    best_mse[i] = hists_dict[val_loss_keys[i]][epoch].item()
                    best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                    best_epoch_mse[i] = epoch + 1
                    mse_improved_this_epoch = True
                    print(f'Generator {i + 1}: New best MSE found: {best_mse[i]:.4f}')

            if not mse_improved_this_epoch:
                patience_counter += 1
                print(
                    f'Patience remaining: {patience - patience_counter}, Current best MSEs: {[f"{bm:.4f}" for bm in best_mse]}, Current MSEs: {[f"{hists_dict[val_loss_keys[j]][epoch].item():.4f}" for j in range(N)]}')
            else:
                patience_counter = 0

        print(f"Patience Counter:{patience_counter}/{patience}")
        if patience_counter >= patience:
            # 明确是基于什么触发的早停
            trigger_metric = "CumReturn" if backtrader else "Validation MSE"
            print(
                f"Early stopping triggered at epoch {epoch + 1} due to no improvement in {trigger_metric} for {patience} epochs.")
            break

        epo_end = time.time()
        print(f"Epoch time: {epo_end - epo_start:.4f}")

    # 加载最佳模型状态
    for i in range(N):
        if best_model_state[i]:
            generators[i].load_state_dict(best_model_state[i])
        else:
            logging.warning(
                f"No improvement found for Generator {i + 1} during training. Using the model from the last epoch.")

    data_G = [[[] for _ in range(4)] for _ in range(N)]
    data_D = [[[] for _ in range(4)] for _ in range(N)]

    for i in range(N):
        for j in range(N + 1):
            if j < N:
                data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch]
                data_D[i][j] = hists_dict[f"D{i + 1}_G{j + 1}"][:epoch]
            elif j == N:
                data_G[i][j] = hists_dict[g_keys[i]][:epoch]
                data_D[i][j] = hists_dict[d_keys[i]][:epoch]

    plot_generator_losses(data_G, output_dir)
    plot_discriminator_losses(data_D, output_dir)

    # overall G&D
    visualize_overall_loss([data_G[i][N] for i in range(N)], [data_D[i][N] for i in range(N)], output_dir)

    hist_MSE_G = [[] for _ in range(N)]
    hist_val_loss = [[] for _ in range(N)]
    for i in range(N):
        hist_MSE_G[i] = hists_dict[f"MSE_G{i + 1}"][:epoch]
        hist_val_loss[i] = hists_dict[f"val_G{i + 1}"][:epoch]

    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch, output_dir)

    mse_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])
    acc_info = ", ".join([f"G{i + 1}:{best_epoch_acc[i]}" for i in range(N)])
    acc_value_info = ", ".join([f"G{i + 1}: {best_acc[i] * 100:.2f}%" for i in range(N)])
    # 由于 best_mse 在 backtrader 模式下可能不更新，这里获取实际的最好值
    mse_value_info = ", ".join(
        [f"G{i + 1}: {hists_dict[val_loss_keys[i]][best_epoch_mse[i] - 1]:.6f}" if best_epoch_mse[i] != -1 else "N/A"
         for i in range(N)])

    print(f"[Best MSE Epochs]     {mse_info}")
    print(f"[Best MSE Values]     {mse_value_info}")
    print(f"[Best ACC Epochs]     {acc_info}")
    print(f"[Best Accuracy Values]{acc_value_info}")

    logging.info(f"[Best MSE Epochs]     {mse_info}")
    logging.info(f"[Best MSE Values]     {mse_value_info}")
    logging.info(f"[Best ACC Epochs]     {acc_info}")
    logging.info(f"[Best Accuracy Values]{acc_value_info}")

    # 最终的金融指标评估
    all_val_metrics = []
    if backtrader:
        for i in range(N):
            _, val_metrics = validate_financial_metric(generators[i], train_xes[i], train_y, val_xes[i], val_y,
                                                       y_scaler)
            all_val_metrics.append(val_metrics)

    return all_val_metrics, best_acc, best_model_state


def discriminate_fake(args, X, Y, LABELS,
                      generators, discriminators,
                      window_sizes, target_num,
                      criterion, weight_matrix,
                      device,
                      mode,
                      return_loss_weight):  # 增加 return_loss_weight 参数
    assert mode in ["train_D", "train_G"]

    N = len(generators)

    # discriminator output for real data
    with get_autocast_context(args.amp_dtype):
        # 自动混合精度上下文
        dis_real_outputs = [model(y, label) for (model, y, label) in zip(discriminators, Y, LABELS)]
        outputs = [generator(x) for (generator, x) in zip(generators, X)]  # cannot be omitted
        real_labels = [torch.ones_like(dis_real_output).to(device) for dis_real_output in dis_real_outputs]
        fake_data_G, fake_logits_G = zip(*outputs)
        # 假设 fake_logits_G 是一个 list，每个元素是 [batch_size, num_classes] 的 tensor
        fake_cls_G = [torch.argmax(logit, dim=1) for logit in fake_logits_G]  # shape: [batch_size]

        lossD_real = [criterion(dis_real_output, real_label) for (dis_real_output, real_label) in
                      zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        # G1生成的数据
        fake_data_temp_G = [fake_data.detach() for fake_data in fake_data_G]
        # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
        fake_data_temp_G = [torch.cat([label[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (label, window_size, fake_data) in zip(Y, window_sizes, fake_data_temp_G)]
        # G1生成的cls logits
        fake_cls_temp_G = [fake_logits.detach() for fake_logits in fake_cls_G]
        # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
        fake_cls_temp_G = [torch.cat([label[:, :window_size, :], fake_cls.reshape(-1, 1, target_num)], axis=1)
                           for (label, window_size, fake_cls) in zip(Y, window_sizes, fake_cls_temp_G)]
    elif mode == "train_G":
        # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_G)]
        fake_cls_temp_G = [torch.cat([label[:, :window_size, :], fake_cls.reshape(-1, 1, target_num)], axis=1)
                           for (label, window_size, fake_cls) in zip(LABELS, window_sizes, fake_cls_G)]

    # 判别器对伪造数据损失
    # 三个生成器的结果的数据对齐
    fake_data_GtoD = {}
    fake_cls_GtoD = {}
    for i in range(N):
        for j in range(N):
            if i < j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :window_sizes[j] - window_sizes[i], :], fake_data_temp_G[i]], axis=1)
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [LABELS[j][:, :window_sizes[j] - window_sizes[i], :], fake_cls_temp_G[i]], axis=1)
            elif i > j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
            elif i == j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_temp_G[i]

    fake_labels = [torch.zeros_like(real_label).to(device) for real_label in real_labels]

    with get_autocast_context(args.amp_dtype):
        # 自动混合精度上下文
        dis_fake_outputD = []
        for i in range(N):
            row = []
            for j in range(N):
                out = discriminators[i](fake_data_GtoD[f"G{j + 1}ToD{i + 1}"],
                                        fake_cls_GtoD[f"G{j + 1}ToD{i + 1}"].long())
                row.append(out)
            if mode == "train_D":
                row.append(lossD_real[i])
            dis_fake_outputD.append(row)  # dis_fake_outputD[i][j] = Di(Gj)

        if mode == "train_D":
            loss_matrix = torch.zeros(N, N + 1, device=device)  # device 取决于你的模型位置
            weight = weight_matrix.clone().detach()  # [N, N+1]
            for i in range(N):
                for j in range(N + 1):
                    if j < N:
                        loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], fake_labels[i])
                    elif j == N:
                        loss_matrix[i, j] = dis_fake_outputD[i][j]
            loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)  # [N, N] --> [N, ]
            return loss_DorG, loss_matrix
        elif mode == "train_G":
            # 判别器对伪造数据损失 (来自 train_baseframe 的 GAN Loss 部分)
            gan_losses = torch.zeros(N, N, device=device)
            weight = weight_matrix.clone().detach()  # [N, N]
            for i in range(N):
                for j in range(N):
                    # For Generator, we want Discriminator to classify fake data as real
                    gan_losses[i, j] = criterion(dis_fake_outputD[i][j], real_labels[i])
            gan_loss = torch.multiply(weight, gan_losses).sum(dim=1)  # [N,]

            # MSE Loss (来自 train_baseframe)
            mse_losses = [F.mse_loss(fake_data.squeeze(), y[:, -1, :].squeeze()) for (fake_data, y) in
                          zip(fake_data_G, Y)]
            mse_loss = torch.stack(mse_losses).to(device)

            # Classification Loss (来自 train_baseframe)
            cls_losses = [F.cross_entropy(fake_cls, l[:, -1, :].squeeze()) for (fake_cls, l) in
                          zip(fake_logits_G, LABELS)]
            cls_loss = torch.stack(cls_losses).to(device)

            # Return Loss (来自 train_baseframe)
            return_losses = []
            for i in range(N):
                predicted_probs = F.softmax(fake_logits_G[i], dim=1)

                if Y[i].shape[1] >= 2:  # 确保至少有两个时间步来计算价格变化
                    price_change = (Y[i][:, -1, :] - Y[i][:, -2, :]).squeeze()
                else:  # 如果序列长度不足，则 price_change 为 0
                    price_change = torch.zeros_like(Y[i][:, -1, :]).squeeze()

                action_coefficients = torch.tensor([-1.0, 0.0, 1.0], device=price_change.device).unsqueeze(0)
                price_change_expanded = price_change.unsqueeze(1)
                potential_returns_per_action = price_change_expanded * action_coefficients
                expected_returns = (predicted_probs * potential_returns_per_action).sum(dim=1)
                return_loss = -expected_returns.mean()
                return_losses.append(return_loss)
            return_loss = torch.stack(return_losses).to(device)


            total_loss = cls_loss + mse_loss
            # `combined_loss` 对应 `(1 - return_loss_weight) * total_loss + return_loss_weight * return_loss`
            combined_loss = (1 - return_loss_weight) * total_loss + return_loss_weight * return_loss

            # 最终返回生成器总损失（用于反向传播）以及各个子损失，方便记录
            return gan_loss, mse_loss, cls_loss, return_loss, combined_loss


if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    # 将当前文件所在目录的上级加入 sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))