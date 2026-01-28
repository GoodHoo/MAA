import numpy as np
import torch
from utils.evaluate_visualization import *
from scipy.special import softmax
def calculate_annualized_return(returns, periods_per_year=252):
    """
    计算年化收益率。
    """
    returns = np.array(returns)
    cumulative_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    return annualized_return * 100


def calculate_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0):
    """
    计算夏普比率，默认无风险利率为0。
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    if std_excess == 0:
        return 0.0
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(returns):
    """
    计算最大回撤（历史最高点之后的最小值之间的跌幅）
    返回单位为百分比（正数）

    Parameters:
    - returns: 收益率序列（线性）

    Returns:
    - 最大回撤（%）
    """
    if len(returns) == 0:
        return np.nan

    returns = np.array(returns)
    cumulative_returns = np.cumprod(1 + returns)

    # 找出历史最高点的位置
    peak_index = np.argmax(cumulative_returns)

    # 如果 peak 是最后一个点，后面没有最小值，回撤为 0
    if peak_index == len(cumulative_returns) - 1:
        return 0.0

    # 在 peak 之后找最小值
    min_after_peak = np.min(cumulative_returns[peak_index + 1:])
    peak_value = cumulative_returns[peak_index]

    # 最大回撤：从 peak 到 min 的百分比跌幅
    max_drawdown = (min_after_peak - peak_value) / peak_value

    return abs(max_drawdown) * 100



def calculate_win_rate(returns):
    """
    计算胜率（盈利交易比例）。
    """
    returns = np.array(returns)
    profitable_trades = np.sum(returns > 0)
    total_trades = np.sum(returns != 0)
    return (profitable_trades / total_trades) * 100 if total_trades > 0 else 0.0


def calculate_returns(prices, pred_reg, pred_cls, use_position=True):
    """
    根据预测和价格计算回报，同时计算平均仓位。

    Args:
        prices (array-like): 价格序列。
        pred_reg (np.array): 连续预测仓位权重。
        pred_cls (torch.Tensor or np.array): 分类预测结果（概率或logits）。
        use_position (bool): 是否根据连续仓位调整收益。

    Returns:
        returns (list): 计算得到的每日收益率。
        avg_position (float): 平均仓位权重（0~1之间）
    """
    prices = np.array(prices)
    if isinstance(pred_cls, torch.Tensor):
        pred_cls = pred_cls.cpu().numpy()
    if isinstance(pred_reg, torch.Tensor):
        pred_reg = pred_reg.cpu().numpy()
    pred_cls = softmax(pred_cls, axis=-1)  # numpy版
    pred_labels = np.argmax(pred_cls, axis=-1)
    returns = []
    position_weights = []

    for i in range(1, len(prices)):
        price_change = (prices[i] - prices[i - 1]) / prices[i - 1]
        action = pred_labels[i]

        predicted_price_change = (pred_reg[i] - prices[i - 1]) / prices[i - 1]
        strength = float(np.abs(predicted_price_change))
        confidence = float(pred_cls[i][action])

        # 计算仓位权重
        position_weight = confidence if use_position else 1
        if action!=0:
            position_weights.append(position_weight)

        if action == 2:  # 多头
            ret = price_change * position_weight
        elif action == 0:  # 空头
            ret = -price_change * position_weight
        else:  # 持有/空仓
            ret = 0.0
        returns.append(ret)

    avg_position = np.mean(position_weights) if position_weights else 0.0

    return returns, avg_position



def get_model_predictions(generator, x_data):
    """
    获取模型的预测结果。
    """
    generator.eval()
    with torch.no_grad():
        pred_reg, pred_cls = generator(x_data)
        pred_reg = pred_reg.cpu().numpy()
        pred_cls = pred_cls.cpu().numpy()
    return pred_reg, pred_cls


def calculate_metrics(returns, periods_per_year=252, risk_free_rate=0.0):
    """
    计算多项财务指标，包括累计收益率。

    返回字典包含：
    - sharpe_ratio: 夏普比率
    - max_drawdown: 最大回撤 (%)
    - annualized_return: 年化收益率 (%)
    - win_rate: 胜率 (%)
    - cumulative_return: 累计收益率 (%)
    """
    returns = np.array(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year, risk_free_rate)
    max_drawdown = calculate_max_drawdown(returns)
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    win_rate = calculate_win_rate(returns)
    cumulative_return = (np.prod(1 + returns) - 1) * 100

    return {
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown': round(max_drawdown, 2),
        'annualized_return': round(annualized_return, 2),
        'win_rate': round(win_rate, 2),
        'cumulative_return': round(cumulative_return, 2)
    }


def validate_financial_metric(generator, train_x, train_y, val_x, val_y, y_scaler):
    """
    验证模型的财务指标。

    Args:
    - generator (torch.nn.Module): 训练好的模型。
    - train_x (torch.Tensor): 训练特征数据。
    - train_y (torch.Tensor): 训练真实标签。
    - val_x (torch.Tensor): 测试特征数据。
    - val_y (torch.Tensor): 测试真实标签。
    - y_scaler (scaler): 目标数据的标准化器。

    Returns:
    - train_metrics_list (list): 训练集的财务指标列表。
    - val_metrics_list (list): 测试集的财务指标列表。
    """
    # 设置模型为评估模式
    generator.eval()

    # 反向变换目标变量
    train_y_inv = inverse_transform(train_y, y_scaler)
    val_y_inv = inverse_transform(val_y, y_scaler)

    # 获取训练和测试集的预测
    train_pred_reg, train_pred_cls = get_model_predictions(generator, train_x)
    test_pred_reg, test_pred_cls = get_model_predictions(generator, val_x)

    # 计算训练集和测试集的回报
    train_returns ,train_avg_pos = calculate_returns(train_y_inv.flatten(), train_pred_reg, train_pred_cls)
    test_returns ,test_avg_pos = calculate_returns(val_y_inv.flatten(), test_pred_reg, test_pred_cls)

    # 计算财务指标
    train_metrics = calculate_metrics(train_returns)
    val_metrics = calculate_metrics(test_returns)

    train_metrics['avg_position'] = round(train_avg_pos * 100, 2)
    val_metrics['avg_position'] = round(test_avg_pos * 100, 2)

    return train_metrics, val_metrics
