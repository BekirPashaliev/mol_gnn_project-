# models/encoder.py

from __future__ import annotations

"""Графовый энкодер: GCN / GIN / GINE / GraphSAGE.

* Поддерживает выбор слоя через аргумент `gnn_type` (`gcn`, `gin`, `gine`, `sage`).
* Конфигурируется количеством слоёв, шириной скрытого слоя, dropout и
  опцией residual.
* Вычисляет **пер‑узловые** эмбеддинги; глобальный пуллинг реализуется
  отдельно в `models/readout.py`.

Почему так:
-----------
* GCN — быстрая база.
* GIN / GINE показывают лучшее качество на «химии» (W‑L‑экспрессивность).
* GraphSAGE полезен при очень больших графах, т.к. усредняет соседей.

Edge‑фичи (`edge_attr`) поддерживаются только у GINE; для других типов слоёв
передаётся `None`.
"""

from typing import Literal, List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, GINEConv, SAGEConv

__all__ = ["GraphEncoder"]


_LAYER_MAP = {
    "gcn": GCNConv,
    "gin": GINConv,
    "gine": GINEConv,
    "sage": SAGEConv,
}

class GraphEncoder(nn.Module):
    """Стэк одинаковых GNN‑слоёв (GCN/GIN/…)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        gnn_type: Literal["gcn", "gin", "gine", "sage"] = "gcn",
        dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if gnn_type not in _LAYER_MAP:
            raise ValueError(f"Unsupported gnn_type={gnn_type}")

        self.gnn_type = gnn_type
        self.residual = residual
        self.dropout = dropout

        dims: List[int] = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if batch_norm else None
        act = nn.ReLU()

        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            # GIN/GINE ожидают nn.Sequential в качестве MLP-агрегатора
            if gnn_type in {"gin", "gine"}:
                mlp = nn.Sequential(
                    nn.Linear(d_in, hidden_dim), act, nn.Linear(hidden_dim, d_out)
                )
            layer_cls = _LAYER_MAP[gnn_type]
            if gnn_type == "gine":
                conv = layer_cls(mlp, edge_dim=None)  # edge_attr передаётся в forward
            elif gnn_type == "gin":
                conv = layer_cls(mlp)
            else:
                conv = layer_cls(d_in, d_out)
            self.convs.append(conv)
            if batch_norm and i != num_layers - 1:
                self.norms.append(nn.BatchNorm1d(d_out))

        self.act = act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None, batch=None):
        """Проброс через стэк слоёв.

        Параметр *batch* не используется, но оставлен для совместимости с
        readout, чтобы можно было вызвать encoder внутри единой модели.
        """
        for idx, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_attr) if self.gnn_type == "gine" else conv(x, edge_index)
            if idx != len(self.convs) - 1:  # все кроме последнего
                if self.norms is not None:
                    h = self.norms[idx](h)
                h = self.act(h)
                h = self.drop(h)
                if self.residual and h.shape == x.shape:
                    h = h + x
            x = h
        return x

