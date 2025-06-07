import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Time2Vec(nn.Module):
    def __init__(self, kernel_size):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(kernel_size - 1))
        self.b = nn.Parameter(torch.randn(kernel_size - 1))

    def forward(self, t):
        # t: (B, T, 1)
        v1 = self.w0 * t + self.b0  # 线性趋势部分 (B, T, 1)
        v2 = torch.sin(t * self.w + self.b)  # 周期性部分 (B, T, kernel_size-1)
        return torch.cat([v1, v2], dim=-1)  # (B, T, kernel_size)


def moving_average_pooling(x, kernel_size=5):
    # x: (B, T, 1)
    x = x.permute(0, 2, 1)  # (B, 1, T)
    padding = kernel_size // 2
    x_smooth = F.avg_pool1d(F.pad(x, (padding, padding), mode='replicate'), kernel_size, stride=1)
    x_smooth = x_smooth.permute(0, 2, 1)  # (B, T, 1)
    return x_smooth


class Encoder(nn.Module):
    def __init__(self, input_dim=5, time2vec_dim=7, d_model=64, nhead=4, num_layers=2,
                 ma_kernel_size=11, decompose_dim=32):
        """
        可分解的SOC特征编码器

        Args:
            input_dim: 输入特征维度 (电流、电压、滤波后电流、滤波后电压、温度)
            time2vec_dim: 时间编码维度
            d_model: 模型隐藏维度
            nhead: 注意力头数
            num_layers: Transformer层数
            ma_kernel_size: 移动平均核大小
            decompose_dim: 分解后每个组件的维度
        """
        super().__init__()
        self.time2vec = Time2Vec(time2vec_dim)
        self.ma_kernel_size = ma_kernel_size
        self.decompose_dim = decompose_dim

        # 特征投影层
        self.feature_proj = nn.Linear(input_dim, d_model)

        # 时间信息投影层
        self.time_proj = nn.Linear(time2vec_dim + 1, d_model)  # +1 for trend

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分解投影层
        self.trend_proj = nn.Linear(d_model, decompose_dim)  # 趋势分量
        self.periodic_proj = nn.Linear(d_model, decompose_dim)  # 周期分量
        self.residual_proj = nn.Linear(d_model, decompose_dim)  # 残差分量

        # 融合层
        self.fusion_layer = nn.Linear(decompose_dim * 3, d_model)

        # 注意力权重层用于分解
        self.decompose_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

    def forward(self, x, t):
        """
        前向传播

        Args:
            x: (B, T, input_dim) - 特征序列 [电流, 电压, 滤波电流, 滤波电压, 温度]
            t: (B, T, 1) - 时间序列

        Returns:
            encoded: (B, T, d_model) - 编码结果
            decomposed: dict - 分解的组件
        """
        batch_size, seq_len, _ = x.shape

        # 1. 时间编码
        t_trend = moving_average_pooling(t, kernel_size=self.ma_kernel_size)  # 趋势
        t_periodic = self.time2vec(t)  # 周期性
        t_combined = torch.cat([t_trend, t_periodic], dim=-1)

        # 2. 特征编码
        x_encoded = self.feature_proj(x)  # (B, T, d_model)
        t_encoded = self.time_proj(t_combined)  # (B, T, d_model)

        # 3. 特征和时间信息融合
        # 使用注意力机制让特征关注时间信息
        fused_features, _ = self.decompose_attention(
            query=x_encoded,
            key=t_encoded,
            value=t_encoded
        )

        # 残差连接
        fused_features = fused_features + x_encoded

        # 4. Transformer编码
        encoded = self.encoder(fused_features)  # (B, T, d_model)

        # 5. 分解编码结果
        trend_component = self.trend_proj(encoded)  # 趋势分量
        periodic_component = self.periodic_proj(encoded)  # 周期分量
        residual_component = self.residual_proj(encoded)  # 残差分量

        # 6. 重新融合分解的组件
        decomposed_concat = torch.cat([
            trend_component,
            periodic_component,
            residual_component
        ], dim=-1)

        final_encoded = self.fusion_layer(decomposed_concat)

        # 返回编码结果和分解组件
        decomposed = {
            'trend': trend_component,  # 趋势分量 - 长期变化
            'periodic': periodic_component,  # 周期分量 - 周期性波动
            'residual': residual_component,  # 残差分量 - 随机波动
            'time_trend': t_trend,  # 时间趋势
            'time_periodic': t_periodic  # 时间周期信息
        }
        return final_encoded, decomposed


class SOCPredictor(nn.Module):
    """
    完整的SOC预测模型，包含编码器和预测头
    """

    def __init__(self, input_dim=5, time2vec_dim=7, d_model=64, nhead=4, num_layers=2,
                 ma_kernel_size=11, decompose_dim=32, hidden_dim=128, output_dim=1):
        super().__init__()

        # 可分解编码器
        self.encoder = Encoder(
            input_dim=input_dim,
            time2vec_dim=time2vec_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ma_kernel_size=ma_kernel_size,
            decompose_dim=decompose_dim
        )

        self.component_attention = nn.MultiheadAttention(
            embed_dim=decompose_dim,
            num_heads=4,
            batch_first=True
        )
        fusion_input_dim = decompose_dim

        # SOC预测头
        self.soc_predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # SOC值通常在0-1之间
        )

    def integrate_components(self, decomposed):
        """
        整合趋势、周期、残差分量
        """
        trend = decomposed['trend']  # (B, T, decompose_dim)
        periodic = decomposed['periodic']  # (B, T, decompose_dim)
        residual = decomposed['residual']  # (B, T, decompose_dim)

        components = torch.stack([trend, periodic, residual], dim=2)  # (B, T, 3, decompose_dim)
        B, T, C, D = components.shape
        components = components.view(B * T, C, D)  # (B*T, 3, decompose_dim)

        # 使用注意力融合
        fused, _ = self.component_attention(
            query=components.mean(dim=1, keepdim=True),  # (B*T, 1, decompose_dim)
            key=components,
            value=components
        )
        integrated = fused.squeeze(1).view(B, T, D)  # (B, T, decompose_dim)
        return integrated

    def aggregate_temporal(self, integrated):
        """
        将时间序列聚合为单一表示
        """
        aggregated = integrated[:, -1, :]  # (B, fusion_input_dim)
        return aggregated

    def forward(self, x, t):
        """
        完整的前向传播

        Args:
            x: (B, T, 5) - [电流, 电压, 滤波电流, 滤波电压, 温度]
            t: (B, T, 1) - 时间序列

        Returns:
            soc_pred: (B, 1) - SOC预测值
            decomposed: dict - 分解的组件（用于分析）
        """
        # 1. 编码和分解
        encoded, decomposed = self.encoder(x, t)

        # 2. 整合分量
        integrated = self.integrate_components(decomposed)

        # 3. 时序聚合
        aggregated = self.aggregate_temporal(integrated)

        # 4. SOC预测
        soc_pred = self.soc_predictor(aggregated)

        return soc_pred, decomposed


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReverseFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 二分类：源域/目标域
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        logits = self.classifier(x)
        return self.tanh(logits) * 5