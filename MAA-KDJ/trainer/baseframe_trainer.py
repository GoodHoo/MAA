import copy
import torch
import torch.nn as nn
from utils.evaluate_visualization import * # 假设这个模块包含了 validate_with_label 和 validate_financial_metric
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from utils.backtrader import * # 假设这个模块与回测相关
import logging


def train_baseframe(generator, dataloader, test_dataloader,
                    y_scaler, train_x, train_y, val_x, val_y,
                    train_y_labels, val_y_labels,
                    num_epochs,
                    device,
                    backtrader,
                    logger=None):
    g_learning_rate = 2e-5
    clip_norm = 1.0
    return_loss_weight = 1

    optimizers_G = torch.optim.AdamW(generator.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))

    # 调度器可以根据损失或性能指标进行调整
    # 如果主要关注 CumReturn，可以考虑将其作为调度器的监控指标
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizers_G, mode='min', factor=0.1, patience=16, min_lr=1e-7)

    best_epoch = -1

    keys = []
    g_keys = 'G1'
    MSE_g_keys = 'MSE_G1'
    val_loss_keys = 'val_G1'
    test_loss_keys = 'test_G1'

    keys.extend([g_keys])
    keys.extend([MSE_g_keys])
    keys.extend([val_loss_keys])
    keys.extend([test_loss_keys])

    # 修改 best_loss 为 best_cum_return，并初始化为负无穷，因为我们希望最大化 CumReturn
    best_cum_return = float("-inf")
    best_acc = -1 # 保持对准确率的追踪
    best_model_state = None

    patience_counter = 0
    patience = 10 # 保持初始耐心值

    print("Starting training...")
    for epoch in range(num_epochs):
        generator.train()

        epoch_loss_keys = ['cls_loss', 'mse_loss', 'return_loss', 'combined_loss']
        loss_dict = {key: [] for key in epoch_loss_keys}

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloader):
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)

            outputs = generator(x_last)
            fake_data_G, fake_data_cls = outputs

            cls_loss = F.cross_entropy(fake_data_cls, label_last[:, -1, :].long().squeeze())
            mse_loss = F.mse_loss(fake_data_G.squeeze(), y_last[:, -1, :].squeeze())

            total_loss = cls_loss + mse_loss

            predicted_probs = F.softmax(fake_data_cls, dim=1)
            price_change = (y_last[:, -1, :] - y_last[:, -2, :]).squeeze()
            action_coefficients = torch.tensor([-1.0, 0.0, 1.0], device=price_change.device).unsqueeze(0)
            price_change_expanded = price_change.unsqueeze(1)
            potential_returns_per_action = price_change_expanded * action_coefficients
            expected_returns = (predicted_probs * potential_returns_per_action).sum(dim=1)

            return_loss = -expected_returns.mean()

            combined_loss = (1 - return_loss_weight) * total_loss + return_loss_weight * return_loss

            optimizers_G.zero_grad()
            combined_loss.backward()

            #torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_norm, norm_type=2)

            optimizers_G.step()

            loss_dict['cls_loss'].append(cls_loss.item())
            loss_dict['mse_loss'].append(mse_loss.item())
            loss_dict['return_loss'].append(return_loss.item())
            loss_dict['combined_loss'].append(combined_loss.item())

        avg_combined_loss = sum(loss_dict['combined_loss']) / len(loss_dict['combined_loss'])
        # 调度器可以保持基于 avg_combined_loss，因为这是训练过程中直接优化的目标
        scheduler.step(avg_combined_loss)

        # Evaluation phase
        generator.eval()

        val_loss, val_acc = validate_with_label(generator, val_x, val_y, val_y_labels)

        current_cum_return = None # 初始化
        # Logging
        log_message = (
            f"Epoch {epoch + 1:3d} | "
            f"Val MSE: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f} | "
            f"Avg Combined Loss: {avg_combined_loss:.4f}"
        )
        if backtrader:
            train_metrics, val_metrics = validate_financial_metric(generator, train_x, train_y, val_x, val_y, y_scaler)
            current_cum_return = val_metrics['cumulative_return'] # 获取当前的 CumReturn
            log_message += (
                f" | Sharpe: {val_metrics['sharpe_ratio']:.4f} | MaxDrawdown: {val_metrics['max_drawdown']:.2f}% | "
                f"AnnualReturn: {val_metrics['annualized_return']:.2f}% | WinRate: {val_metrics['win_rate']:.2f}% | "
                f"CumReturn: {val_metrics['cumulative_return']:.2f}% | AvgPosition: {val_metrics['avg_position']:.2f}%"
            )
        logging.info(log_message)

        # 早停逻辑修改：基于 CumReturn
        # 只有在 backtrader 为 True 且 current_cum_return 可用时，才使用 CumReturn 进行早停
        if backtrader and current_cum_return is not None:
            if current_cum_return < best_cum_return: # CumReturn 越低越差
                patience_counter += 1
                print(
                    f'Patience remaining: {patience - patience_counter}, Current best CumReturn: {best_cum_return:.2f}%, Current CumReturn: {current_cum_return:.2f}%')
            else: # CumReturn 提高，重置 patience_counter
                patience_counter = 0
                best_model_state = copy.deepcopy(generator.state_dict())
                best_cum_return = current_cum_return
                print(f'New best CumReturn found: {best_cum_return:.2f}%')
        else: # 如果 backtrader 不为 True，或者无法获取 CumReturn，则回退到基于 avg_combined_loss 的早停
            # 注意：这里需要一个 best_loss 的初始值，以及一个独立的 patience_counter 用于此分支
            # 为简化，在此处沿用原有的 patience_counter 和 best_loss 逻辑
            # 但是为了严谨，你可能需要一个独立的变量来追踪 avg_combined_loss 的最佳值
            if avg_combined_loss > best_loss: # 假设 best_loss 在函数外部或初始化时定义
                patience_counter += 1
                print(
                    f'Patience remaining: {patience - patience_counter}, Current best combined loss: {best_loss:.4f}, Current combined loss: {avg_combined_loss:.4f}')
            else:
                patience_counter = 0
                best_model_state = copy.deepcopy(generator.state_dict())
                best_loss = avg_combined_loss # 更新 best_loss
                print(f'New best combined loss found: {best_loss:.4f}')


        if val_acc > best_acc:
            best_acc = val_acc

        if patience_counter > patience:
            # 明确是基于什么触发的早停
            trigger_metric = "CumReturn" if backtrader and current_cum_return is not None else "combined loss"
            print(
                f"Early stopping triggered at epoch {epoch + 1} due to no improvement in {trigger_metric} for {patience} epochs.")
            break

    if best_model_state:
        generator.load_state_dict(best_model_state)
    else:
        logging.warning("No improvement found during training. Using the model from the last epoch.")

    if backtrader:
        _, val_metrics = validate_financial_metric(generator, train_x, train_y, val_x, val_y, y_scaler)
    else:
        val_metrics = None

    return [val_metrics], [best_acc.item()], [best_model_state]


if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))