import numpy as np
import torch
from evaluate_visualization import *

# 年化收益
def calculate_annualized_return(returns, periods_per_year=252):
    cumulative_return = np.prod([1 + r for r in returns]) - 1
    years = len(returns) / periods_per_year
    if years == 0:
        return 0
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    return annualized_return * 100

# 夏普比率（默认无风险利率为0）
def calculate_sharpe_ratio(returns, periods_per_year=252):
    excess_returns = np.array(returns)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    if std_excess == 0:
        return 0
    sharpe = (mean_excess * np.sqrt(periods_per_year)) / std_excess
    return sharpe

# 最大回撤
def calculate_max_drawdown(returns):
    cumulative_returns = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdowns)
    return abs(max_drawdown) * 100


# 胜率（可选）
def calculate_win_rate(returns):
    returns = np.array(returns)
    profitable_trades = np.sum(returns > 0)
    total_trades = np.sum(returns != 0)
    return (profitable_trades / total_trades) * 100 if total_trades > 0 else 0



def calculate_returns_with_position(prices, pred_reg,pred_cls):
    """
    计算给定价格数据和预测动作的回报。

    Args:
    - prices (np.array or torch.Tensor): 形状为 (N,) 或 (N, 1) 的价格数据。
    - pred_labels (np.array or torch.Tensor): 形状为 (N, T, 1) 的预测动作标签。

    Returns:
    - returns (list of lists): 每个子列表对应一个batch的动作回报。
    """
    # 转换为 NumPy 并压缩维度
    # prices = prices.squeeze()  # shape: (64,)
    pred_labels = pred_cls.squeeze()  # shape: (64, 16)

    all_returns = []

    for batch_idx in range(pred_labels.shape[1]):
        returns = []
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i - 1]) / prices[i - 1]
            action = pred_labels[i, batch_idx]

            if action == 1:  # Buy
                returns.append(price_change)
            elif action == 2:  # Sell
                returns.append(-price_change)
            else:  # Hold
                returns.append(0)
        all_returns.append(returns)

    return all_returns  # shape: (16, 63)



def calculate_returns(prices, pred_labels):
    """
    计算给定价格数据和预测动作的回报。

    Args:
    - prices (np.array or torch.Tensor): 形状为 (N,) 或 (N, 1) 的价格数据。
    - pred_labels (np.array or torch.Tensor): 形状为 (N, T, 1) 的预测动作标签。

    Returns:
    - returns (list of lists): 每个子列表对应一个batch的动作回报。
    """
    # 转换为 NumPy 并压缩维度
    # prices = prices.squeeze()  # shape: (64,)
    pred_labels = pred_labels.squeeze()  # shape: (64, 16)

    all_returns = []

    for batch_idx in range(pred_labels.shape[1]):
        returns = []
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i - 1]) / prices[i - 1]
            action = pred_labels[i, batch_idx]

            if action == 1:  # Buy
                returns.append(price_change)
            elif action == 2:  # Sell
                returns.append(-price_change)
            else:  # Hold
                returns.append(0)
        all_returns.append(returns)

    return all_returns  # shape: (16, 63)


def calculate_returns_with_probabilities(prices, pred_probs):
    """
    计算给定价格数据和预测标签概率的回报。

    Args:
    - prices (np.array): 价格数据。
    - pred_probs (np.array): 预测的标签概率，shape应为 (n_samples, 3)，
                              每一行分别表示 [hold_prob, buy_prob, sell_prob]。

    Returns:
    - returns (list): 回报列表。
    """
    returns = []
    for i in range(1, len(prices)):
        # 计算价格变化
        price_change = (prices[i] - prices[i - 1]) / prices[i - 1]

        # 获取预测的概率
        hold_prob = pred_probs[i][0]
        buy_prob = pred_probs[i][1]
        sell_prob = pred_probs[i][2]

        # 根据概率加权计算回报
        weighted_return = hold_prob * 0 + buy_prob * price_change + sell_prob * (-price_change)
        returns.append(weighted_return)

    return returns


def calculate_metrics(returns, periods_per_year=252):
    """
    计算财务指标（Sharpe Ratio, Max Drawdown, Annualized Return, Win Rate）

    Args:
    - returns (list): 回报数据。
    - periods_per_year (int): 每年交易日数（默认252）。

    Returns:
    - metrics (dict): 包含财务指标的字典。
    """
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year)
    max_drawdown = calculate_max_drawdown(returns)
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    win_rate = calculate_win_rate(returns)

    return {
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown': round(max_drawdown, 2),
        'annualized_return': round(annualized_return, 2),
        'win_rate': round(win_rate, 2)
    }

def validate_financial_indicator(model, val_x, val_y, val_labels,y_scaler):
    model.eval()
    pred_reg, pred_cls = model(val_x)
    y_inv=inverse_transform(val_y, y_scaler)
    pred_labels = pred_cls.argmax(dim=-1).cpu().numpy()

    returns=calculate_returns(y_inv.flatten(), pred_labels)
    financial_indicator_metrics = calculate_metrics(returns, periods_per_year=252)
    return financial_indicator_metrics



def validate_financial_metric(generators, train_x,  y_scaler):

    financial_indicator_metrics = []
    for i, generator in enumerate(generators):
        generator.eval()
        train_x_i = train_x[i]
        with torch.no_grad():
            train_pred, train_pred_cls = generator(train_x_i)
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_probs = torch.softmax(train_pred_cls, dim=-1).cpu().numpy()
            pred_returns = calculate_returns_with_probabilities(train_pred_inv, train_probs)
            financial_indicator_metrics.append(calculate_metrics(pred_returns, periods_per_year=252))
    return financial_indicator_metrics


def validate_financial_metric_loss(generators, train_x, train_y, LABELS, y_scaler):
    mse_list = []  # 用于保存每个生成器的 MSE

    for i, generator in enumerate(generators):
        generator.eval()

        train_x_i = train_x[i]
        train_y_i = train_y[i]
        Label_i = LABELS[i]

        train_y_inv_i = inverse_transform(train_y_i, y_scaler)

        with torch.no_grad():
            train_pred, train_pred_cls = generator(train_x_i)
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_probs = torch.softmax(train_pred_cls, dim=-1).cpu().numpy()

            pred_returns = calculate_returns_with_probabilities(train_y_inv_i, train_probs)
            true_returns = calculate_returns(train_pred_inv, Label_i.detach().cpu().numpy())

            # ---- 转换 pred_returns ----
            pred_array = np.stack([p.flatten() for p in pred_returns], axis=0)  # (63, 16)

            # ---- 转换 true_returns ----
            true_array = []
            for ts in true_returns:
                row = [float(arr.item()) if isinstance(arr, (np.ndarray, np.generic)) else float(arr) for arr in ts]
                true_array.append(row)
            true_array = np.array(true_array).T  # (63, 16)

            # ---- 计算 MSE ----
            mse = np.mean((pred_array - true_array) ** 2)
            mse_list.append(mse)  # 存入列表

    # 将 list 转为 tensor
    mse_tensor = torch.tensor(mse_list, dtype=torch.float32)

    return mse_tensor

def evaluate_best_solution(y_scaler, train_y, val_y,train_label_y,val_label_y):

    train_y_inv = inverse_transform(train_y, y_scaler)
    val_y_inv = inverse_transform(val_y, y_scaler)
    train_label_y = train_label_y[:,-1].cpu().numpy()  # 转换为 NumPy 数组
    val_label_y = val_label_y[:,-1].cpu().numpy()      # 转换为 NumPy 数组

    # 计算训练集和测试集的回报
    train_returns = calculate_returns(train_y_inv.flatten(), train_label_y)
    val_returns = calculate_returns(val_y_inv.flatten(), val_label_y)

    # 计算财务指标
    train_metrics = calculate_metrics(train_returns)
    val_metrics = calculate_metrics(val_returns)
    print("---------------------------\nPerfect Solution:")
    # 打印并返回结果
    print("Train Metrics:", train_metrics)
    print("Val Metrics:", val_metrics)

    return train_metrics, val_metrics
