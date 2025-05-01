# models/regressor.py

from __future__ import annotations

"""IC50Regressor – supervised head mapping latent vectors → activity.
Skeleton MLP, to be extended (dropout, batch‑norm, etc.).
"""

from typing import Sequence

import torch
from torch import nn

__all__ = ["IC50Regressor"]


class IC50Regressor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int] = (128, 64)) -> None:
        super().__init__()
        dims = [latent_dim, *hidden_dims, 1]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # → predicted IC50 (log µM)
        return self.mlp(z).squeeze(-1)
