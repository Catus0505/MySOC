import torch

from Modules import *
from preprocess import *
B, T = 32, 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)

loss_fn = nn.MSELoss()

train_loader, val_loader, scalers = get_dataloader('10degC/10degC_HWFET.csv', flag='train')

for times, features, labels in train_loader:
    times = times.to(device).float()        # (B, T, 1)
    features = features.to(device).float()  # (B, T, D)
    labels = labels.to(device).float()      # (B, 1)

    # encoder
    memory, decomposed = encoder(features, times)       # (B, T, D')

    print(memory.shape)
    trend = decomposed['trend']  # 趋势信息
    periodic = decomposed['periodic']  # 周期信息

    print(trend.shape, periodic.shape)

    # # decoder
    # last_time = times[:, -1, :]             # (B, 1)
    # delta_t = 1.0 / times.shape[1]
    # t_last = last_time + delta_t            # (B, 1)
    # t_last = t_last.unsqueeze(1)            # (B, 1, 1)
    # tgt_init = labels.unsqueeze(1)          # (B, 1, 1)
    #
    # preds = decoder(memory, t_last, tgt_init)
    # loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))
