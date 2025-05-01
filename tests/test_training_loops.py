# tests/test_training_loops.py
import torch
from mol_gnn_project.models.regressor import IC50Regressor


def test_regressor_forward():
    z = torch.randn(8, 64)
    model = IC50Regressor(latent_dim=64)
    out = model(z)
    assert out.shape == (8,)
