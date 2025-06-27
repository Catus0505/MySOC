import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def temporal_mean_std_pool(features):
    """
    Mean + Std pooling over time, for better domain representation
    Input:  [B, T, D]
    Output: [B, 2D]
    """
    mean = features.mean(dim=1)
    std = features.std(dim=1)
    return torch.cat([mean, std], dim=-1)


def compute_disc(domain_disc, src_encoded, tgt_encoded):
    # 对时间序列维度做平均池化，变成 (batch_size, feature_dim)
    src_pooled = temporal_mean_std_pool(src_encoded)  # (B, D)
    tgt_pooled = temporal_mean_std_pool(tgt_encoded) # (B, D)

    # src_encoded_mean = src_pooled / (src_pooled.norm(dim=1, keepdim=True) + 1e-8)  # L2 归一化
    # tgt_encoded_mean = tgt_pooled / (tgt_pooled.norm(dim=1, keepdim=True) + 1e-8)  # L2 归一化

    # 判别器前向
    src_preds = domain_disc(src_pooled)  # (B, 2)
    tgt_preds = domain_disc(tgt_pooled)  # (B, 2)

    # probs_src = torch.softmax(src_preds, dim=1)  # 转成概率
    # pred_labels_src = torch.argmax(probs_src, dim=1)  # 预测标签
    # print("src preds probabilities:", probs_src[0])
    # print("src predicted label:", pred_labels_src[0].item())
    #
    # probs_tgt = torch.softmax(tgt_preds, dim=1)  # 转成概率
    # pred_labels_tgt = torch.argmax(probs_tgt, dim=1)  # 预测标签
    # print("tgt preds probabilities:", probs_tgt[0])
    # print("tgt predicted label:", pred_labels_tgt[0].item())

    # 构造标签
    src_labels = torch.zeros(src_pooled.size(0), dtype=torch.long).to(device)  # 全0，表示源域
    tgt_labels = torch.ones(tgt_pooled.size(0), dtype=torch.long).to(device)   # 全1，表示目标域

    # 合并预测和标签
    preds = torch.cat([src_preds, tgt_preds], dim=0)   # (2B, 2)
    labels = torch.cat([src_labels, tgt_labels], dim=0)  # (2B,)

    # 计算交叉熵损失
    domain_loss = F.cross_entropy(preds, labels)

    return domain_loss
