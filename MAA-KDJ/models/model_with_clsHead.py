import torch
import torch.nn as nn
import math

# --- PositionalEncoding 模型 ---
# 用于 Transformer 模型的通用位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """
        Args:
            model_dim (int): 模型的特征向量维度 (即 Transformer 的 d_model)。
            max_len (int): 支持的最大序列长度。
        """
        super().__init__()
        encoding = torch.zeros(max_len, model_dim)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        # 增加 batch 维度，并注册为 buffer，使其随模型保存加载，但不作为可训练参数
        self.register_buffer('encoding', encoding.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, seq_len, model_dim)。

        Returns:
            torch.Tensor: 添加了位置编码的输入特征。
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# --- Generator_gru 模型 ---
# 基于 GRU 的生成器，支持多层 GRU，回归分支包含非线性激活，分类头输出 Logits。
class Generator_gru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim=128, num_layers=2):
        """
        Args:
            input_size (int): 输入特征的维度。
            out_size (int): 回归任务的输出维度。
            hidden_dim (int): GRU 隐藏状态的维度。
            num_layers (int): GRU 层的数量，默认为 2。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True, num_layers=num_layers)

        # 回归分支：线性层之间加入激活函数以增强非线性
        self.linear_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.linear_3 = nn.Linear(hidden_dim // 4, out_size)
        self.dropout = nn.Dropout(0.2)

        # 分类头：统一输出 Logits (不含 Softmax)，结构与其他生成器保持一致
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 3) # 输出 3 类别的 Logits
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_size)。

        Returns:
            tuple: 包含回归预测 (gen) 和分类预测 Logits (cls) 的元组。
                   gen 形状为 (batch_size, out_size)。
                   cls 形状为 (batch_size, 3)。
        """
        device = x.device
        # 初始化 GRU 隐藏状态，与 num_layers 匹配
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, _ = self.gru(x, h0)
        last_feature = self.dropout(out[:, -1, :]) # 取序列最后一个时间步的输出

        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)

        cls = self.classifier(last_feature)

        return gen, cls

# --- Generator_lstm 模型 ---
class Generator_lstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            input_size (int): 输入特征数。
            out_size (int): 回归任务的输出维度。
            hidden_size (int): LSTM 的隐藏单元数。
            num_layers (int): LSTM 层数。注意：如果 num_layers = 1，LSTM 的 dropout 参数不生效。
            dropout (float): LSTM 内部 dropout 系数。
        """
        super().__init__()
        self.hidden_size = hidden_size

        # 移除卷积层，直接将输入送入 LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # 回归头：直接使用最后一个时间步的输出进行线性映射
        self.linear = nn.Linear(hidden_size, out_size)

        # 分类头：统一输出 Logits (不含 Softmax)，结构与 Generator_gru 一致
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size // 2, 3) # 输出 3 类别的 Logits
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): 输入，形状 (batch_size, seq_len, input_size)。
            hidden: 可选的 LSTM 初始状态 (h_0, c_0)。

        Returns:
            tuple: 包含回归预测 (out) 和分类预测 Logits (cls) 的元组。
                   out 形状为 (batch_size, out_size)。
                   cls 形状为 (batch_size, 3)。
        """
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :] # 取最后一个时间步的输出

        out = self.linear(last_out)
        cls = self.classifier(last_out)

        return out, cls

# --- Generator_transformer 模型 ---
# 基于 Transformer 的生成器，使用平均池化聚合序列特征，分类头输出 Logits。
class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2, num_heads=8, dropout=0.1, output_len=1):
        """
        Args:
            input_dim (int): 输入数据特征的维度。
            feature_size (int): 模型特征维度 (即 Transformer 的 d_model)。
            num_layers (int): Transformer 编码器层的数量。
            num_heads (int): 注意力头的数目。
            dropout (float): dropout 概率。
            output_len (int): 回归任务的预测时间步长度（输出维度）。
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len

        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                         batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, output_len)

        # 分类头：统一输出 Logits (不含 Softmax)，结构与 GRU/LSTM 一致
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feature_size // 2, 3) # 输出 3 类别的 Logits
        )

        self._init_weights() # 自定义权重初始化
        self.src_mask = None # 用于存储因果掩码

    def _init_weights(self):
        """
        自定义权重初始化：对所有线性层使用 Xavier 均匀初始化，偏置初始化为 0。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src, src_mask=None):
        """
        Args:
            src (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_dim)。
            src_mask (torch.Tensor, optional): 源序列的注意力掩码。
                                               对于时序预测，通常是因果掩码。
                                               默认为 None，将在内部生成。

        Returns:
            tuple: 包含回归预测 (gen) 和分类预测 Logits (cls) 的元组。
                   gen 形状为 (batch_size, output_len)。
                   cls 形状为 (batch_size, 3)。
        """
        batch_size, seq_len, _ = src.size()

        src = self.input_projection(src)
        src = self.pos_encoder(src)

        # 如果没有提供掩码，则生成一个因果掩码
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        output = self.transformer_encoder(src, src_mask)

        # 使用平均池化来汇总所有时间步的信息
        last_feature = torch.mean(output, dim=1)

        gen = self.decoder(last_feature)
        cls = self.classifier(last_feature)

        return gen, cls

    def _generate_square_subsequent_mask(self, seq_len):
        """
        生成一个方形的因果掩码 (上三角掩码)，用于确保当前时间步只能注意到过去和当前时间步。
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size, hidden_dim):
        """
        input_dim: 每个时间步的特征数，比如你是21
        out_size: 你想输出几个预测值，比如5
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim+1, hidden_dim//4, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, stride=1, padding='same')

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_size)

        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, F] => [B, F, T]
        #x = x.permute(0, 2, 1)

        conv1 = self.leaky(self.conv1(x))  # [B, 32, T]
        conv2 = self.leaky(self.conv2(conv1))  # [B, 64, T]
        conv3 = self.leaky(self.conv3(conv2))  # [B, 128, T]

        # 聚合时间信息，取平均
        pooled = torch.mean(conv3, dim=2)  # [B, 128]

        out = self.leaky(self.linear1(pooled))  # [B, 220]
        out = self.relu(self.linear2(out))     # [B, 220]
        out = self.relu(self.linear3(out))  # [B, out_size]

        return out