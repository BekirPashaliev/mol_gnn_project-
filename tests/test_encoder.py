# tests/test_encoder.py

from rdkit import Chem

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch

from mol_gnn_project.models.encoder import GraphEncoder
from mol_gnn_project.graphs.converter import mol_to_graph_data


# ---------- утилита: «игрушечный» граф -----------
def get_dummy_graph():
    # CCO  ─ маленькая молекула: 3 атома, 2 связи
    mol = Chem.MolFromSmiles("CCO")          # 3 атома, 2 связи
    data = mol_to_graph_data(mol)            # Data.x, edge_index, edge_attr
    return data


# ---------- 1. Shape-тест ---------------------------------
def test_encoder_shape():
    g = get_dummy_graph()
    F = g.x.shape[1]
    enc = GraphEncoder(in_dim=F, hidden_dim=64, out_dim=128,
                       num_layers=4, gnn_type="gcn")
    out = enc(g.x, g.edge_index)          # (3, 128)
    assert out.shape == (3, 128)


# ---------- 2. Overfit-тест -------------------------------
def test_encoder_overfit():
    #  Cоберём батч из 8 копий dummy-графа
    batch = Batch.from_data_list([get_dummy_graph() for _ in range(8)])
    F = batch.x.shape[1]

    enc = GraphEncoder(in_dim=F, hidden_dim=64, out_dim=64,
                       num_layers=3, gnn_type="gin")

    # «Декодер» — один Linear, восстанавливает исходные atom-фичи
    dec = nn.Linear(64, F)

    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-2)
    mse = nn.MSELoss()

    torch.manual_seed(0)
    for _ in range(300):
        opt.zero_grad()
        h = enc(batch.x, batch.edge_index)          # (N, 64)
        x_hat = dec(h)                              # реконструкция
        loss = mse(x_hat, batch.x)
        loss.backward()
        opt.step()

    # sanity-check: должно переобучиться < 1e-3
    assert loss.item() < 1e-3


def test_permutation_invariance():
    from torch_geometric.utils import to_undirected

    data = get_dummy_graph()
    enc = GraphEncoder(data.x.size(1), 64, 64, gnn_type="gcn")

    # baseline
    z1 = enc(data.x, data.edge_index).mean(dim=0)

    # permute nodes
    perm = torch.randperm(data.num_nodes)
    data_p = data.clone()
    data_p.x = data.x[perm]

    # переименуем индексы в edge_index
    idx_map = {old.item(): new for new, old in enumerate(perm)}
    ei = data.edge_index.t()                                # (E,2)
    ei = torch.tensor([[idx_map[i.item()] for i in pair] for pair in ei],
                      dtype=torch.long)
    data_p.edge_index = ei.t().contiguous()

    # прогоняем
    z2 = enc(data_p.x, data_p.edge_index).mean(dim=0)
    assert torch.allclose(z1, z2, atol=1e-5)

