import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from preprocess import *
from Modules import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class DecomposableEncoder(nn.Module):
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

    def get_feature_importance(self, x, t):
        """
        获取不同特征对各个分量的重要性
        """
        with torch.no_grad():
            encoded, decomposed = self.forward(x, t)

            # 计算各特征对不同分量的贡献
            feature_names = ['current', 'voltage', 'current_smooth', 'voltage_smooth', 'temperature']
            importance = {}

            for i, name in enumerate(feature_names):
                # 单独输入每个特征
                single_feature = torch.zeros_like(x)
                single_feature[:, :, i] = x[:, :, i]

                _, single_decomposed = self.forward(single_feature, t)

                importance[name] = {
                    'trend_contribution': torch.norm(single_decomposed['trend'], dim=-1).mean(),
                    'periodic_contribution': torch.norm(single_decomposed['periodic'], dim=-1).mean(),
                    'residual_contribution': torch.norm(single_decomposed['residual'], dim=-1).mean()
                }

            return importance


# 辅助函数
def moving_average_pooling(x, kernel_size):
    """移动平均池化"""
    if kernel_size <= 1:
        return x

    # 使用1D卷积实现移动平均
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size) / kernel_size
    kernel = kernel.to(x.device)

    # x: (B, T, 1) -> (B, 1, T)
    x_conv = x.transpose(1, 2)
    smoothed = F.conv1d(x_conv, kernel, padding=padding)

    return smoothed.transpose(1, 2)  # (B, T, 1)


class SOCPredictor(nn.Module):
    """
    完整的SOC预测模型，包含编码器和预测头
    """

    def __init__(self, input_dim=5, time2vec_dim=7, d_model=64, nhead=4, num_layers=2,
                 ma_kernel_size=11, decompose_dim=32, hidden_dim=128, output_dim=1):
        super().__init__()

        # 可分解编码器
        self.encoder = DecomposableEncoder(
            input_dim=input_dim,
            time2vec_dim=time2vec_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ma_kernel_size=ma_kernel_size,
            decompose_dim=decompose_dim
        )

        # 分量整合策略
        self.integration_method = 'attention_fusion'  # 可选: 'concat', 'weighted_fusion', 'attention_fusion'

        if self.integration_method == 'concat':
            # 方法1: 直接拼接
            fusion_input_dim = decompose_dim * 3  # trend + periodic + residual

        elif self.integration_method == 'weighted_fusion':
            # 方法2: 加权融合
            self.trend_weight = nn.Parameter(torch.tensor(0.5))
            self.periodic_weight = nn.Parameter(torch.tensor(0.3))
            self.residual_weight = nn.Parameter(torch.tensor(0.2))
            fusion_input_dim = decompose_dim

        elif self.integration_method == 'attention_fusion':
            # 方法3: 注意力融合
            self.component_attention = nn.MultiheadAttention(
                embed_dim=decompose_dim,
                num_heads=4,
                batch_first=True
            )
            fusion_input_dim = decompose_dim

        # 时序聚合层 (将时间序列聚合为单一表示)
        self.temporal_aggregation = 'last'  # 可选: 'last', 'mean', 'attention', 'lstm'

        if self.temporal_aggregation == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=fusion_input_dim,
                num_heads=4,
                batch_first=True
            )
        elif self.temporal_aggregation == 'lstm':
            self.temporal_lstm = nn.LSTM(
                input_size=fusion_input_dim,
                hidden_size=fusion_input_dim,
                batch_first=True
            )

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

        if self.integration_method == 'concat':
            # 直接拼接
            integrated = torch.cat([trend, periodic, residual], dim=-1)

        elif self.integration_method == 'weighted_fusion':
            # 加权融合
            weights = F.softmax(torch.stack([
                self.trend_weight,
                self.periodic_weight,
                self.residual_weight
            ]), dim=0)

            integrated = (weights[0] * trend +
                          weights[1] * periodic +
                          weights[2] * residual)

        elif self.integration_method == 'attention_fusion':
            # 注意力融合
            # 将三个分量stack作为序列
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
        if self.temporal_aggregation == 'last':
            # 使用最后一个时间步
            aggregated = integrated[:, -1, :]  # (B, fusion_input_dim)

        elif self.temporal_aggregation == 'mean':
            # 使用平均值
            aggregated = integrated.mean(dim=1)  # (B, fusion_input_dim)

        elif self.temporal_aggregation == 'attention':
            # 使用注意力加权平均
            attended, attention_weights = self.temporal_attention(
                query=integrated.mean(dim=1, keepdim=True),  # (B, 1, fusion_input_dim)
                key=integrated,
                value=integrated
            )
            aggregated = attended.squeeze(1)  # (B, fusion_input_dim)

        elif self.temporal_aggregation == 'lstm':
            # 使用LSTM的最后隐状态
            lstm_out, (hidden, _) = self.temporal_lstm(integrated)
            aggregated = hidden[-1]  # (B, fusion_input_dim)

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

    def get_component_contributions(self, x, t):
        """
        分析各个分量对SOC预测的贡献
        """
        with torch.no_grad():
            encoded, decomposed = self.encoder(x, t)

            # 分别使用单一分量预测
            trend_only = self.integrate_components({
                'trend': decomposed['trend'],
                'periodic': torch.zeros_like(decomposed['periodic']),
                'residual': torch.zeros_like(decomposed['residual'])
            })

            periodic_only = self.integrate_components({
                'trend': torch.zeros_like(decomposed['trend']),
                'periodic': decomposed['periodic'],
                'residual': torch.zeros_like(decomposed['residual'])
            })

            residual_only = self.integrate_components({
                'trend': torch.zeros_like(decomposed['trend']),
                'periodic': torch.zeros_like(decomposed['periodic']),
                'residual': decomposed['residual']
            })

            # 分别预测SOC
            trend_soc = self.soc_predictor(self.aggregate_temporal(trend_only))
            periodic_soc = self.soc_predictor(self.aggregate_temporal(periodic_only))
            residual_soc = self.soc_predictor(self.aggregate_temporal(residual_only))

            return {
                'trend_contribution': trend_soc,
                'periodic_contribution': periodic_soc,
                'residual_contribution': residual_soc
            }


# 使用示例

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# 假设你已经有了前面定义的模型类
# from your_model_file import SOCPredictor

def train_soc_model():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载
    print("加载数据...")
    train_loader, val_loader, scalers = get_dataloader('10degC/10degC_UDDS.csv', flag='train')

    # 获取数据维度信息
    sample_times, sample_features, sample_labels = next(iter(train_loader))
    seq_len = sample_features.shape[1]  # 时间序列长度
    feature_dim = sample_features.shape[2]  # 特征维度 (应该是5)

    print(f"数据信息:")
    print(f"  批次大小: {sample_features.shape[0]}")
    print(f"  序列长度: {seq_len}")
    print(f"  特征维度: {feature_dim}")
    print(f"  标签维度: {sample_labels.shape}")

    # 2. 模型初始化
    print("初始化模型...")
    model = SOCPredictor(
        input_dim=feature_dim,  # 5个特征 [电流, 电压, 滤波电流, 滤波电压, 温度]
        time2vec_dim=8,  # 时间编码维度
        d_model=64,  # 模型隐藏维度
        nhead=8,  # 注意力头数
        num_layers=3,  # Transformer层数
        ma_kernel_size=11,  # 移动平均核大小
        decompose_dim=32,  # 分解维度
        hidden_dim=128,  # 预测头隐藏维度
        output_dim=1  # SOC输出维度
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 3. 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练参数
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 记录训练过程
    train_losses = []
    val_losses = []

    # TensorBoard记录（可选）
    # writer = SummaryWriter('runs/soc_prediction')

    print(f"开始训练，共{num_epochs}个epoch...")

    # 4. 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for times, features, labels in pbar:
                # 数据转移到设备
                times = times.to(device).float()  # (B, T, 1)
                features = features.to(device).float()  # (B, T, 5)
                labels = labels.to(device).float()  # (B, 1)

                # 前向传播
                optimizer.zero_grad()
                soc_pred, decomposed = model(features, times)

                # 计算损失
                loss = criterion(soc_pred, labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # 累计损失
                train_loss += loss.item()
                train_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{train_loss / train_batches:.6f}'
                })

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # # 验证阶段
        # model.eval()
        # val_loss = 0.0
        # val_batches = 0
        #
        # with torch.no_grad():
        #     for times, features, labels in val_loader:
        #         times = times.to(device).float()
        #         features = features.to(device).float()
        #         labels = labels.to(device).float()
        #
        #         soc_pred, decomposed = model(features, times)
        #         loss = criterion(soc_pred, labels)
        #
        #         val_loss += loss.item()
        #         val_batches += 1
        #
        # avg_val_loss = val_loss / val_batches
        # val_losses.append(avg_val_loss)

        # 学习率调度
        # scheduler.step(avg_val_loss)
        scheduler.step(avg_train_loss)

        # 打印训练信息
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.6f}")
        # print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停检查
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     # 保存最佳模型
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'val_loss': avg_val_loss,
        #         'scalers': scalers
        #     }, 'best_soc_model.pth')
        #     print(f"  保存最佳模型 (验证损失: {best_val_loss:.6f})")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"早停触发! 验证损失连续{patience}个epoch未改善")
        #         break

        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                # 'val_loss': avg_val_loss,
                'scalers': scalers
            }, 'best_soc_model.pth')
            print(f"  保存最佳模型 (验证损失: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发! 验证损失连续{patience}个epoch未改善")
                break

        print("-" * 50)

    # 5. 训练结果分析
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SOC预测模型训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

    return model, scalers


def evaluate_model():
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型
    checkpoint = torch.load('best_soc_model.pth', weights_only=False)
    scalers = checkpoint['scalers']

    # 重新初始化模型
    model = SOCPredictor(
        input_dim=5,
        time2vec_dim=8,
        d_model=64,
        nhead=8,
        num_layers=3,
        ma_kernel_size=11,
        decompose_dim=32,
        hidden_dim=128,
        output_dim=1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载验证数据
    val_loader = get_dataloader('10degC/10degC_LA92.csv', flag='val', scalers=scalers)

    predictions = []
    true_values = []
    component_contributions = []

    print("评估模型性能...")
    with torch.no_grad():
        for times, features, labels in tqdm(val_loader):
            times = times.to(device).float()
            features = features.to(device).float()
            labels = labels.to(device).float()

            # 预测
            soc_pred, decomposed = model(features, times)

            # 分析分量贡献
            contributions = model.get_component_contributions(features, times)

            predictions.extend(soc_pred.cpu().numpy())
            true_values.extend(labels.cpu().numpy())
            component_contributions.extend([{
                'trend': contributions['trend_contribution'].cpu().numpy(),
                'periodic': contributions['periodic_contribution'].cpu().numpy(),
                'residual': contributions['residual_contribution'].cpu().numpy()
            }])

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # 计算评估指标
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    smape = 100 * np.mean(2 * np.abs(predictions - true_values) / (np.abs(predictions) + np.abs(true_values) + 1e-8))
    r2 = r2_score(true_values, predictions)

    print(f"模型评估结果:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"R²: {r2:.4f}")

    # 绘制预测结果
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(true_values, label='True Values', linewidth=2)
    plt.plot(predictions, label='Predicted Values', linewidth=2)
    # plt.scatter(true_values, predictions, alpha=0.6)
    # plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')
    plt.xlabel('真实SOC')
    plt.ylabel('预测SOC')
    plt.title('SOC预测图')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    errors = predictions - true_values
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(true_values[:1000], label='真实值', marker='o', markersize=3)
    plt.plot(predictions[:1000], label='预测值', marker='s', markersize=3)
    plt.xlabel('样本索引')
    plt.ylabel('SOC值')
    plt.title('SOC预测对比 (前100个样本)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    # 分量贡献分析
    trend_contrib = np.mean([c['trend'] for c in component_contributions[:100]], axis=0)
    periodic_contrib = np.mean([c['periodic'] for c in component_contributions[:100]], axis=0)
    residual_contrib = np.mean([c['residual'] for c in component_contributions[:100]], axis=0)

    components = ['趋势', '周期', '残差']
    contributions = [np.mean(trend_contrib), np.mean(periodic_contrib), np.mean(residual_contrib)]

    plt.bar(components, contributions)
    plt.ylabel('平均贡献度')
    plt.title('各分量对SOC预测的贡献')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()


if __name__ == "__main__":
    # 训练模型
    model, scalers = train_soc_model()

    # 评估模型
    evaluate_model()