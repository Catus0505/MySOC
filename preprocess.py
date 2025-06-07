import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from configs import PreprocessConfig
PreprocCfg = PreprocessConfig()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_voltage_current_raw_vs_filtered_after_normalize(data):
    # 归一化后作图
    time = data[:, 0]
    orig_current = data[:, 1]
    orig_voltage = data[:, 2]
    smooth_current = data[:, 3]
    smooth_voltage = data[:, 4]

    plt.figure(figsize=(14, 6))

    # 电流对比
    plt.subplot(1, 2, 1)
    plt.plot(time, orig_current, label='Original Current', alpha=0.5)
    plt.plot(time, smooth_current, label='Smoothed Current', linewidth=2)
    plt.title('Current: Original vs Smoothed')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)

    # 电压对比
    plt.subplot(1, 2, 2)
    plt.plot(time, orig_voltage, label='Original Voltage', alpha=0.5)
    plt.plot(time, smooth_voltage, label='Smoothed Voltage', linewidth=2)
    plt.title('Voltage: Original vs Smoothed')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_dataloader(data_arrays, batch_size, shuffle=True, seed=42):
    dataset = data.TensorDataset(*data_arrays)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)  # 设置种子
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    else:
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def slide_window_cut(scaled_data, time_step=PreprocCfg.time_step,
                     stride=PreprocCfg.stride):
    # 样本数量、每个样本的时间步数、每个时间步的特征数量
    # 对时间序列进行滑动窗口切分
    times_list = []
    feats_list = []
    labels_list = []

    # 滑动窗口采样
    for i in range(0, len(scaled_data) - time_step + 1, stride):
        time = scaled_data[i:i + time_step, 0:1]
        feature = scaled_data[i:i + time_step, 1:6]  # 取1-5列作为特征
        label = scaled_data[i + time_step - 1, -1]  # 取最后一个时间步作为标签

        times_list.append(time)
        feats_list.append(feature)
        labels_list.append(label)

    # 转换为 numpy 数组
    times = np.stack(times_list)
    feats = np.stack(feats_list)
    labels = np.stack(labels_list).reshape(-1, 1)
    return times, feats, labels


def apply_filter(df):
    # 对归一化后的电压、电流进行Savitzky-Golay滤波
    window_length = 301
    polyorder = 2
    df['voltage_smooth'] = savgol_filter(df.iloc[:, 1], window_length, polyorder)
    df['current_smooth'] = savgol_filter(df.iloc[:, 2], window_length, polyorder)
    return df


def normalize(df, flag='train', scalers=None):
    """
    对数据进行归一化。训练阶段创建并fit scaler，测试阶段使用已有scaler。
    :param df: 输入的 DataFrame
    :param flag: 'train' or 'test'
    :param scalers: dict，用于测试阶段传入训练好的scaler
    :return: 归一化后的 df，以及 scalers（仅训练阶段返回）
    """
    if flag == 'train':
        scalers = {
            'time': MinMaxScaler(),
            'current': MinMaxScaler(),
            'voltage': MinMaxScaler(),
            'temp': MinMaxScaler()
        }
        df.iloc[:, 0] = scalers['time'].fit_transform(df.iloc[:, 0].values.reshape(-1, 1)).flatten()
        df.iloc[:, 1] = scalers['current'].fit_transform(df.iloc[:, 1].values.reshape(-1, 1)).flatten()
        df.iloc[:, 2] = scalers['voltage'].fit_transform(df.iloc[:, 2].values.reshape(-1, 1)).flatten()
        df.iloc[:, 3] = scalers['temp'].fit_transform(df.iloc[:, 3].values.reshape(-1, 1)).flatten()
        return df, scalers

    elif flag == 'test':
        assert scalers is not None, "传入训练阶段得到的scalers"
        df.iloc[:, 0] = scalers['time'].transform(df.iloc[:, 0].values.reshape(-1, 1)).flatten()
        df.iloc[:, 1] = scalers['current'].transform(df.iloc[:, 1].values.reshape(-1, 1)).flatten()
        df.iloc[:, 2] = scalers['voltage'].transform(df.iloc[:, 2].values.reshape(-1, 1)).flatten()
        df.iloc[:, 3] = scalers['temp'].transform(df.iloc[:, 3].values.reshape(-1, 1)).flatten()
        return df

    else:
        raise ValueError("Invalid flag or scalers not provided for test mode")


def get_dataloader(dataset, flag, scalers=None, validation_split=0, seed=42):
    """
    获取数据加载器，使用顺序划分

    Args:
        dataset: 数据集文件名
        flag: 'train', 'val', 'test'
        scalers: 归一化器
        validation_split: 验证集比例
    """
    batch_size = PreprocCfg.batch_size
    data_path = os.path.join('dataset/', dataset)

    df = pd.read_csv(data_path)

    # 根据flag进行归一化
    if flag == 'train':
        df, scalers = normalize(df, flag='train')
    else:
        df = normalize(df, flag='test', scalers=scalers)

    df = apply_filter(df)

    # 重新排列列顺序
    df = df[[df.columns[0], df.columns[1], df.columns[2], 'current_smooth', 'voltage_smooth', df.columns[3],
             df.columns[4]]].values

    # plot_voltage_current_raw_vs_filtered_after_normalize(df)

    times, feats, labels = slide_window_cut(df)

    times = torch.tensor(times).float()
    feats = torch.tensor(feats).float()
    labels = torch.tensor(labels).float()

    if flag in ['train', 'transfer']:
        dataset_size = len(feats)
        train_size = int(dataset_size * (1 - validation_split))

        train_times = times[:train_size]
        train_feats = feats[:train_size]
        train_labels = labels[:train_size]

        val_times = times[train_size:]
        val_feats = feats[train_size:]
        val_labels = labels[train_size:]

        # 创建数据加载器
        train_dataloader = create_dataloader((train_times, train_feats, train_labels), batch_size, shuffle=True, seed=seed)
        val_dataloader = create_dataloader((val_times, val_feats, val_labels), batch_size, shuffle=False)
        return train_dataloader, val_dataloader, scalers


    else:
        dataloader = create_dataloader((times, feats, labels), len(feats), shuffle=False)
        return dataloader