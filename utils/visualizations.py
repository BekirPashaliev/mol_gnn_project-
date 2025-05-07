# utils/visualizations.py

from __future__ import annotations

"""Набор функций визуализации для Graph VAE."""

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from typing import Dict, List

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


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str],
    save_dir: Path | str
):
    """
    Рисует кривые train/val по списку метрик.
      - history: { 'loss_train': [...], 'loss_val': [...], 'ELBO': [...], … }
      - metrics: список базовых имён метрик (например ['loss','recon','kld','ELBO'])
      - save_dir: куда сохранять png-файлы
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history[f"{metrics[0]}_train"]) + 1)
    for m in metrics:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history[f"{m}_train"],  label="train")
        plt.plot(epochs, history[f"{m}_val"],    label="val")
        plt.title(f"{m} per epoch")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{m}.png", dpi=200)
        plt.close()

def plot_times(
    train_times: List[float],
    val_times:   List[float],
    save_dir:    Path | str
):
    """
    Рисует время train/val по эпохам.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_times) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_times, label="train time")
    plt.plot(epochs, val_times,   label="val time")
    plt.title("Time per epoch, s")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "times.png", dpi=200)
    plt.close()
