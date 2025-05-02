# models/vae.py

from __future__ import annotations

"""GraphVAE – вариационный автоэнкодер для молекулярных графов.

* **Encoder**: GraphEncoder → global pool (mean) → μ, logσ² (dim = latent_dim).
* **Decoder**:
    * Узловые признаки – MLP («broadcast» глобального z каждому узлу).
    * (Опционально) рёбра – InnerProductDecoder (можно включить флагом).

Структура позволяет гибко расширять декодер, не ломая интерфейс.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import InnerProductDecoder

from mol_gnn_project.models.encoder import GraphEncoder  # local import
from mol_gnn_project.models.readout import GraphReadout

__all__ = ["GraphVAE"]


class GraphVAE(nn.Module):
    """Вариационный автоэнкодер для молекул.

    Параметры
    ----------
    node_dim     : размер вектора признаков атома.
    hidden_dim   : размер скрытых слоёв в энкодере.
    latent_dim   : размер латентного пространства (z).
    edge_decode  : bool, если True – декодируем смежность.
    """

    def __init__(
            self,
            node_dim: int,
            hidden_dim: int,
            latent_dim: int,
            num_layers: int = 3,
            edge_decode: bool = False,
            edge_dim: Optional[int] = None,
    ) -> None:

        super().__init__()
        # Энкодер выводит 2×latent_dim (μ и logσ²)
        self.encoder = GraphEncoder(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim * 2,
            num_layers=num_layers,
            gnn_type="gine",
            dropout=0.1,
            residual=True,
            edge_dim=edge_dim,
        )
        self.readout = GraphReadout("mean")
        # Декодер узловых признаков: z → x_hat
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # (Опционально) декодер рёбер
        self.edge_decode = edge_decode
        if edge_decode:
            self.edge_decoder = InnerProductDecoder()

        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # *** ЭНКОДЕР ***
    # ------------------------------------------------------------------
    def encode_graph(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (μ, logσ²) для каждого графа в батче."""
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        g = self.readout(h, batch.batch)  # [B, 2·latent]
        mu, logvar = torch.chunk(g, chunks=2, dim=-1)  # обе [B, latent]
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # *** ДЕКОДЕР ***
    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor, batch: Batch):
        """Декодируем узлы (и рёбра, если включено)."""
        # — Узлы —
        z_expanded = z[batch.batch]  # (N_nodes, latent_dim)
        x_hat = self.node_decoder(z_expanded)  # (N_nodes, node_dim)

        # — Рёбра —
        adj_hat = None
        if self.edge_decode:
            # InnerProductDecoder ждёт z размера (B, latent)
            adj_hat = self.edge_decoder(z)
        return x_hat, adj_hat

    # ------------------------------------------------------------------
    def forward(self, batch: Batch):
        """Полный проход: (recon_x, recon_adj, μ, logσ²)."""
        mu, logvar = self.encode_graph(batch)
        z = self.reparameterize(mu, logvar)
        x_hat, adj_hat = self.decode(z, batch)
        return x_hat, adj_hat, mu, logvar
