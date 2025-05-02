# models/losses.py

from __future__ import annotations

"""Набор функций лоссов для проекта Graph VAE.

* **recon_loss**  – восстанавливает узловые признаки (MSE или CE).
* **kld_loss**    – KL‑дивергенция между q(z|x) и N(0,1).
* **ic50_loss**   – MAE‑регрессия (понадобится на Stage B).

Все функции возвращают одно число‑тензор (scalar Tensor).
"""

from typing import Literal, Optional
import torch
import torch.nn.functional as F

__all__ = [
    "recon_loss",
    "kld_loss",
    "ic50_loss",
]

# -----------------------------------------------------------------------------

def recon_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    mode: Literal["mse", "ce"] = "mse",
) -> torch.Tensor:
    """Считает ошибку реконструкции узловых признаков.

    Параметры
    ----------
    x_hat : Tensor  – предсказанные признаки (N×F).
    x_true: Tensor  – правдивые признаки  (N×F).
    mode  : "mse" – непрерывные признаки,
            "ce"  – x_true — one‑hot: превращаем в индексы и считаем CrossEntropy.
    """
    if mode == "mse":
        return F.mse_loss(x_hat, x_true, reduction="mean")
    if mode == "ce":
        cls = x_true.argmax(dim=-1)
        return F.cross_entropy(x_hat, cls, reduction="mean")
    raise ValueError(f"Unsupported recon mode: {mode}")

# -----------------------------------------------------------------------------

def kld_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL‑дивергенция *в среднем по батчу*.
    D_KL( q(z|x) || N(0,1) ) = −½ Σ(1 + logσ² − μ² − σ²).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# -----------------------------------------------------------------------------
# IC50 loss (MAE) – пригодится на Stage B
# -----------------------------------------------------------------------------

def ic50_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MAE по доступным значениям.

    Если *mask* — булев (T) или {0,1}, берём только присутствующие элементы.
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return F.l1_loss(pred, target, reduction="mean")