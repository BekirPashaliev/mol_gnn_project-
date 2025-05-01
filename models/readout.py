# models/readout.py

from __future__ import annotations

"""Graph-level pooling operators."""

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

__all__ = ["GraphReadout"]


class GraphReadout(nn.Module):
    """Wrapper to choose pooling strategy by name."""

    _POOLERS = {
        "mean": global_mean_pool,
        "max": global_max_pool,
        "sum": global_add_pool,
    }

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in self._POOLERS:
            raise ValueError(f"Unknown pooling mode: {mode}")
        self.pool = self._POOLERS[mode]

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.pool(node_emb, batch)
