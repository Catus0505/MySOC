import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
from itertools import cycle

from preprocess import get_dataloader
from Modules import *
from mmd import *
from configs import TransferConfig
TransCfg = TransferConfig()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def visualize_periodic_features(src_periodic, tgt_periodic, sample_idx=0, channel_idx=0):
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
    tgt_signal = tgt_periodic[sample_idx, :, channel_idx].detach().cpu().numpy()

    # 画时域波形
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(src_signal, label='Source Periodic')
    plt.plot(tgt_signal, label='Target Periodic')
    plt.title('Periodic Feature - Time Domain')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # 画频域频谱
    src_freq = np.abs(np.fft.fft(src_signal))
    tgt_freq = np.abs(np.fft.fft(tgt_signal))
    freqs = np.fft.fftfreq(len(src_signal))

    plt.subplot(1, 2, 2)
    plt.plot(freqs[:len(freqs)//2], src_freq[:len(freqs)//2], label='Source Periodic')
    plt.plot(freqs[:len(freqs)//2], tgt_freq[:len(freqs)//2], label='Target Periodic')
    plt.title('Periodic Feature - Frequency Domain')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.show()


def transfer():
    # 设备设置
    device = TransCfg.device
    print(f"使用设备: {device}")

    # 1. 数据加载
    print("加载数据...")
    checkpoint = torch.load(TransCfg.checkpoint, weights_only=False)
    scalers = checkpoint['scalers']
    src_loader, _, _ = get_dataloader(TransCfg.source, flag='transfer', scalers=scalers)
    tgt_loader, _, _ = get_dataloader(TransCfg.target, flag='transfer', scalers=scalers)

    # 获取数据维度信息
    src_times, src_features, src_labels = next(iter(src_loader))
    src_seq_len = src_features.shape[1]  # 时间序列长度
    src_feature_dim = src_features.shape[2]  # 特征维度 (应该是5)

    print(f"数据信息:")
    print(f"  批次大小: {src_features.shape[0]}")
    print(f"  序列长度: {src_seq_len}")
    print(f"  特征维度: {src_feature_dim}")
    print(f"  标签维度: {src_labels.shape}")
    print(f"Source Loader 总 batch 数: {len(src_loader)}")

    tgt_times, tgt_features, tgt_labels = next(iter(tgt_loader))
    tgt_seq_len = tgt_features.shape[1]  # 时间序列长度
    tgt_feature_dim = tgt_features.shape[2]  # 特征维度 (应该是5)

    print(f"数据信息:")
    print(f"  批次大小: {tgt_features.shape[0]}")
    print(f"  序列长度: {tgt_seq_len}")
    print(f"  特征维度: {tgt_feature_dim}")
    print(f"  标签维度: {tgt_labels.shape}")
    print(f"Target Loader 总 batch 数: {len(tgt_loader)}")

    # 2. 模型初始化
    print("初始化模型...")
    src_model = SOCPredictor(
        input_dim=src_feature_dim,  # 5个特征 [电流, 电压, 滤波电流, 滤波电压, 温度]
        time2vec_dim=8,  # 时间编码维度
        d_model=64,  # 模型隐藏维度
        nhead=8,  # 注意力头数
        num_layers=3,  # Transformer层数
        ma_kernel_size=11,  # 移动平均核大小
        decompose_dim=32,  # 分解维度
        hidden_dim=128,  # 预测头隐藏维度
        output_dim=1  # SOC输出维度
    ).to(device)

    tgt_model = SOCPredictor(
        input_dim=tgt_feature_dim,  # 5个特征 [电流, 电压, 滤波电流, 滤波电压, 温度]
        time2vec_dim=8,  # 时间编码维度
        d_model=64,  # 模型隐藏维度
        nhead=8,  # 注意力头数
        num_layers=3,  # Transformer层数
        ma_kernel_size=11,  # 移动平均核大小
        decompose_dim=32,  # 分解维度
        hidden_dim=128,  # 预测头隐藏维度
        output_dim=1  # SOC输出维度
    ).to(device)

    src_model.load_state_dict(checkpoint['model_state_dict'])
    tgt_model.load_state_dict(checkpoint['model_state_dict'])

    # 冻结 src_model
    for param in src_model.parameters():
        param.requires_grad = False
    # 冻结 tgt_model 的 trend_component
    # for param in tgt_model.encoder.trend_proj.parameters():
    #     param.requires_grad = False

    # 3. 训练设置
    optimizer = optim.AdamW(tgt_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练参数
    num_epochs = TransCfg.num_epochs
    best_train_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 记录训练过程
    train_losses = []

    # TensorBoard记录
    # writer = SummaryWriter('runs/soc_prediction')

    criterion = nn.MSELoss()

    print(f"开始训练，共{num_epochs}个epoch...")

    # 4. 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        src_model.train()
        tgt_model.train()

        train_loss = 0.0
        train_batches = 0

        with tqdm(zip(cycle(src_loader), tgt_loader), total=len(tgt_loader),
                  desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for (src_times, src_features, src_labels), (tgt_times, tgt_features, tgt_labels) in pbar:
                optimizer.zero_grad()

                src_features, src_times = src_features.to(device), src_times.to(device)
                tgt_features, tgt_times = tgt_features.to(device), tgt_times.to(device)

                ###### 1. 提取源/目标域特征 ######
                with torch.no_grad():
                    src_feats, src_decomposed = src_model.encoder(src_features, src_times)
                with torch.enable_grad():
                    tgt_feats, tgt_decomposed = tgt_model.encoder(tgt_features, tgt_times)

                mmd_loss= compute_mmd(src_decomposed['periodic'], tgt_decomposed['periodic'])

                # 总损失
                loss = mmd_loss

                loss.backward()
                # for name, param in tgt_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} grad norm:", param.grad.norm().item())
                grad_norm = torch.nn.utils.clip_grad_norm_(tgt_model.parameters(), max_norm=10.0)
                # print("Gradient norm:", grad_norm.item())
                optimizer.step()

                # 累积loss和批次数
                train_loss += loss.item()
                train_batches += 1

                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MMD': f'{mmd_loss.item():.6f}',
                    'Avg': f'{train_loss / train_batches:.6f}'
                })

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # 学习率调度
        scheduler.step(avg_train_loss)

        # 打印训练信息
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': tgt_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'scalers': scalers
            }, 'transfer_model.pth')
            print(f"  保存最佳模型 (验证损失: {best_train_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发! 验证损失连续{patience}个epoch未改善")
                break

        print("-" * 50)


if __name__ == "__main__":
    # 预训练模型
    transfer()