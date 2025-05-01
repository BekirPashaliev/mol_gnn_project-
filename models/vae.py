# models/vae.py

from __future__ import annotations

"""GraphVAE – unsupervised encoder–decoder for molecular graphs.

Skeleton implementation: fill in *encode_graph* and *decode* logic later.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from mol_gnn_project.models.encoder import GraphEncoder  # local import
from mol_gnn_project.models.readout import GraphReadout

__all__ = ["GraphVAE"]


class GraphVAE(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(node_dim, hidden_dim, latent_dim * 2, num_layers)
        self.readout = GraphReadout("mean")
        # simple MLP decoder placeholder: z → (reconstructed node feats)
        self.node_mlp = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
        )
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    def encode_graph(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for each graph in *batch*."""
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        g = self.readout(h, batch.batch)  # [n_graphs, hidden]
        mu, logvar = torch.chunk(g, chunks=2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        """
        Расклеиваем z по узлам: каждому атому графа
        скармливаем его «глобальный» z (broadcast).
        """
        z_expanded = z[batch.batch]  # (N_nodes, latent_dim)
        x_hat = self.node_mlp(z_expanded)  # (N_nodes, F_atom)
        return x_hat

    # ------------------------------------------------------------------
    def forward(self, batch: Batch):
        mu, logvar = self.encode_graph(batch)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    @staticmethod
    def loss_function(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.1):
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld, recon_loss.item(), kld.item()
