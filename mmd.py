import torch
import torch.nn as nn
import torch.nn.functional as F

def temporal_avg_pool(features):
    """
    Flatten time and feature dimensions into one dimension
    Input:  [B, T, D]
    Output: [B, T*D]
    """
    B, T, D = features.shape
    return features.view(B, T * D)


def temporal_mean_std_pool(features):
    """
    Mean + Std pooling over time, for better domain representation
    Input:  [B, T, D]
    Output: [B, 2D]
    """
    mean = features.mean(dim=1)
    std = features.std(dim=1)
    return torch.cat([mean, std], dim=-1)


def gaussian_kernel(x, y, gammas):
    """
    Multi-kernel RBF computation
    Input: x: [B, D], y: [B, D]
    Output: [B, B]
    """
    xx = x.unsqueeze(1)  # (N, 1, D)
    yy = y.unsqueeze(0)  # (1, M, D)
    dist_sq = ((xx - yy) ** 2).sum(2)  # (N, M)
    kernels = [torch.exp(-gamma * dist_sq) for gamma in gammas]
    return sum(kernels) / len(kernels)


def compute_mmd(source, target, gammas=None, pool_fn=temporal_mean_std_pool, normalize=True):
    """
    Compute MK-MMD loss between source and target
    - source, target: [B, T, D]
    - Returns scalar MMD loss
    """
    source_2d = pool_fn(source)  # e.g. [B, D] or [B, 2D]
    target_2d = pool_fn(target)

    if normalize:
        source_2d = F.normalize(source_2d, dim=1)
        target_2d = F.normalize(target_2d, dim=1)

    if gammas is None:
        with torch.no_grad():
            combined = torch.cat([source_2d, target_2d], dim=0)
            dists = torch.cdist(combined, combined, p=2).pow(2)
            median_sq = dists.median()
            eps = 1e-8
            gammas = [1 / (median_sq / f + eps) for f in [2, 5, 10]]

    K_ss = gaussian_kernel(source_2d, source_2d, gammas)
    K_tt = gaussian_kernel(target_2d, target_2d, gammas)
    K_st = gaussian_kernel(source_2d, target_2d, gammas)

    m, n = source_2d.size(0), target_2d.size(0)

    loss = (K_ss.sum() - torch.diag(K_ss).sum()) / (m * (m - 1)) \
         + (K_tt.sum() - torch.diag(K_tt).sum()) / (n * (n - 1)) \
         - 2 * K_st.mean()

    return loss