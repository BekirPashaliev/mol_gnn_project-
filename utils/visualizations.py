# utils/visualizations.py

from __future__ import annotations

"""Набор функций визуализации для Graph VAE."""

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

__all__ = ["plot_latent_tsne"]

def plot_latent_tsne(
        z_vectors: torch.Tensor,
        labels: Sequence[int] | None = None,
        save_path: str | Path | None = None):
    """Рисует t‑SNE 2D‑проекцию латент‑векторов.

    * **z_vectors** – Tensor [N, latent_dim]
    * **labels** – опциональные категориальные метки (цвета).
    """
    z_np = z_vectors.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto")
    z_2d = tsne.fit_transform(z_np)

    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=8)
    else:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=8, c=labels, cmap="tab10")
    plt.title("Latent space (t‑SNE)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
