import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch import nn
from tqdm import tqdm

from Modules import SOCPredictor
from preprocess import get_dataloader

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid')


def compute_rmse(y_pred, y_true):
    mse = nn.MSELoss()(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse


def compute_r2(y_pred, y_true):
    var_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (residual / var_total)


def evaluate_model():
    """评估模型性能"""
    criterion = nn.MSELoss()
    total_loss = 0
    all_preds, all_labels = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型
    checkpoint = torch.load('transfer_model.pth', weights_only=False)
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
    val_loader = get_dataloader('10degC/10degC_LA92.csv', flag='test', scalers=scalers)

    predictions = []
    true_values = []

    print("评估模型性能...")
    with torch.no_grad():
        for times, features, labels in tqdm(val_loader):
            times = times.to(device).float()
            features = features.to(device).float()
            labels = labels.to(device).float()

            # 预测
            soc_pred, _ = model(features, times)

            loss = criterion(soc_pred, labels)
            total_loss += loss.item() * labels.size(0)

            all_preds.append(soc_pred.cpu())
            all_labels.append(labels.cpu())

            # predictions.extend(soc_pred.cpu().numpy())
            # true_values.extend(labels.cpu().numpy())

    # 聚合结果
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    preds = all_preds.squeeze(1).cpu().numpy()
    # preds = moving_average(preds, w=50)
    labels = all_labels.squeeze(1).cpu().numpy()

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Values', linewidth=2)
    plt.plot(preds, label='Predicted Values', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Predicted vs. True Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 计算全局指标
    metrics = {
        'mae': torch.mean(torch.abs(all_preds - all_labels)),
        'rmse': compute_rmse(all_preds, all_labels),
        'r2': compute_r2(all_preds, all_labels)
    }
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")

    # predictions = np.array(predictions)
    # true_values = np.array(true_values)

    # # 计算评估指标
    # mse = np.mean((predictions - true_values) ** 2)
    # rmse = np.sqrt(mse)
    # mae = np.mean(np.abs(predictions - true_values))
    # smape = 100 * np.mean(2 * np.abs(predictions - true_values) / (np.abs(predictions) + np.abs(true_values) + 1e-8))
    # r2 = r2_score(true_values, predictions)

    # print(f"模型评估结果:")
    # print(f"  MSE: {mse:.6f}")
    # print(f"  RMSE: {rmse:.6f}")
    # print(f"  MAE: {mae:.6f}")
    # print(f"  SMAPE: {smape:.2f}%")
    # print(f"R²: {r2:.4f}")
    #
    # # 绘制预测结果
    # plt.figure(figsize=(12, 8))
    #
    # plt.plot(true_values, label='True Values', linewidth=2)
    # plt.plot(predictions, label='Predicted Values', linewidth=2)
    # plt.xlabel('真实SOC')
    # plt.ylabel('预测SOC')
    # plt.title('SOC预测图')
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # 评估模型
    evaluate_model()