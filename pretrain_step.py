import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from preprocess import get_dataloader
from Modules import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def visualize_periodic_features(src_periodic, sample_idx=0, channel_idx=0):
    """
    可视化周期特征的时域波形和频谱
    参数:
        src_periodic: torch.Tensor, shape (B, T, D)
        tgt_periodic: torch.Tensor, shape (B, T, D)
        sample_idx: int, 选择第几个样本绘制
        channel_idx: int, 选择第几个通道绘制
    """

    # 转为numpy，取对应样本和通道
    src_signal = src_periodic[sample_idx, :, channel_idx].detach().cpu().numpy()

    # 画时域波形
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(src_signal, label='Source Periodic')
    plt.title('Periodic Feature - Time Domain')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # 画频域频谱
    src_freq = np.abs(np.fft.fft(src_signal))
    freqs = np.fft.fftfreq(len(src_signal))

    plt.subplot(1, 2, 2)
    plt.plot(freqs[:len(freqs)//2], src_freq[:len(freqs)//2], label='Source Periodic')
    plt.title('Periodic Feature - Frequency Domain')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.show()



def pretrain():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载
    print("加载数据...")
    train_loader, _, scalers = get_dataloader('10degC/10degC_US06.csv', flag='train')

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

    # TensorBoard记录
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
                # periodic = decomposed['periodic']
                # if epoch in [10]:
                #     visualize_periodic_features(periodic, sample_idx=0, channel_idx=0)

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

        # 学习率调度
        scheduler.step(avg_train_loss)

        # 打印训练信息
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'scalers': scalers
            }, 'pretrain_model.pth')
            print(f"  保存最佳模型 (训练损失: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发! 训练损失连续{patience}个epoch未改善")
                break

        print("-" * 50)

    # 5. 训练结果分析
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SOC预测模型训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

    return model, scalers


if __name__ == "__main__":
    # 预训练模型
    model, scalers = pretrain()