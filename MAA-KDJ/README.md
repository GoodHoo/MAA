# MAA-KDJ: 基于多对抗生成网络的量化交易策略

这是一个基于多对抗生成网络（Multi-Adversarial Generation Network）的量化交易策略项目，主要用于时间序列预测和交易信号生成。

## 项目特点

- 使用多对抗生成网络进行时间序列预测
- 支持多种技术指标作为特征输入
- 包含完整的回测系统
- 提供可视化评估工具
- 支持模型蒸馏和交叉微调

## 环境要求

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- backtrader

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Qmacd/MAA-KDJ.git
cd MAA-KDJ
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
MAA-KDJ/
├── models/          # 模型定义
├── trainer/         # 训练相关代码
├── utils/          # 工具函数
├── database/       # 数据存储
├── sh/             # 脚本文件
├── time_series_maa.py    # 主要策略实现
├── MAA_base.py     # 基础类
├── run_backtrader.py     # 回测运行脚本
└── requirements.txt      # 项目依赖
```

## 使用方法

1. 数据准备：
   - 将数据文件放在 `database` 目录下
   - 支持 CSV 格式的数据文件

2. 运行回测：
```bash
python run_backtrader.py
```

3. 训练模型：
```bash
python run_multi_gan.py
```

## 主要功能

- 时间序列数据处理和特征工程
- 多对抗生成网络训练
- 模型蒸馏和交叉微调
- 回测系统集成
- 性能评估和可视化

## 注意事项

- 请确保数据格式正确，包含必要的技术指标
- 建议先使用小规模数据测试系统
- 可以根据需要调整模型参数和训练参数

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License 