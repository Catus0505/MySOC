import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def dis(src_periodic, tgt_periodic, time_step=-1):
    """
    Visualize periodic features using PCA and t-SNE.

    Args:
        src_periodic: (B, T, F) torch.Tensor
        tgt_periodic: (B, T, F) torch.Tensor
        time_step: int, which time step to visualize (default: -1 for last)
    """
    B, T, F = src_periodic.shape

    # Convert to numpy
    src_np = src_periodic.detach().cpu().numpy().reshape(-1, F)  # (B*T, F)
    tgt_np = tgt_periodic.detach().cpu().numpy().reshape(-1, F)  # (B*T, F)

    # === PCA Visualization ===
    pca = PCA(n_components=2)
    combined_np = np.concatenate([src_np, tgt_np], axis=0)
    combined_pca = pca.fit_transform(combined_np)

    src_pca = combined_pca[:B*T]
    tgt_pca = combined_pca[B*T:]

    # 只画指定 time_step 的降维结果
    if 0 <= time_step < T:
        src_plot = src_pca.reshape(B, T, -1)[:, time_step, :]
        tgt_plot = tgt_pca.reshape(B, T, -1)[:, time_step, :]
        title_suffix = f"(PCA at time step {time_step})"
    else:
        src_plot = src_pca
        tgt_plot = tgt_pca
        title_suffix = "(PCA over all time steps)"

    plt.figure()
    plt.scatter(src_plot[:, 0], src_plot[:, 1], label="Source", alpha=0.5)
    plt.scatter(tgt_plot[:, 0], tgt_plot[:, 1], label="Target", alpha=0.5)
    plt.legend()
    plt.title("Periodic Feature Distributions " + title_suffix)
    plt.grid(True)
    plt.show()

    # === t-SNE Visualization ===
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, init='pca')
    tsne_result = tsne.fit_transform(combined_np)

    src_tsne = tsne_result[:B*T]
    tgt_tsne = tsne_result[B*T:]

    if 0 <= time_step < T:
        src_plot = src_tsne.reshape(B, T, -1)[:, time_step, :]
        tgt_plot = tgt_tsne.reshape(B, T, -1)[:, time_step, :]
        title_suffix = f"(t-SNE at time step {time_step})"
    else:
        src_plot = src_tsne
        tgt_plot = tgt_tsne
        title_suffix = "(t-SNE over all time steps)"

    plt.figure()
    plt.scatter(src_plot[:, 0], src_plot[:, 1], label="Source", alpha=0.6)
    plt.scatter(tgt_plot[:, 0], tgt_plot[:, 1], label="Target", alpha=0.6)
    plt.legend()
    plt.title("t-SNE Visualization of Periodic Features " + title_suffix)
    plt.grid(True)
    plt.show()
