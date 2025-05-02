# tests/test_training_loops.py

from __future__ import annotations

"""PyTest: проверяем, что VAE способен переобучиться на маленьком наборе."""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.models.losses import recon_loss, kld_loss

def random_graph(num_nodes:int=10)->Data:
    # toy граф: полный без признаков рёбер
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    x = torch.randn(num_nodes, 8)
    return Data(x=x, edge_index=edge_index, edge_attr=None)


def test_vae_overfit_small():
    dataset = [random_graph() for _ in range(8)]
    loader = DataLoader(dataset, batch_size=8)
    model = GraphVAE(node_dim=8, hidden_dim=32, latent_dim=16)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        for batch in loader:
            optim.zero_grad()
            x_hat, _adj_hat, mu, logvar = model(batch)
            loss = recon_loss(x_hat, batch.x) + 0.1 * kld_loss(mu, logvar)
            loss.backward()
            optim.step()
    assert loss.item() < 0.05
