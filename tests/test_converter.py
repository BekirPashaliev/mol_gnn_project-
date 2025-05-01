# tests/test_converter.py

import torch
from mol_gnn_project.graphs.converter import mol_to_graph_data, mol_from_smiles


def test_converter_shapes():
    mol = mol_from_smiles("CCO")  # ethanol
    data = mol_to_graph_data(mol)
    assert data.x.shape[0] == mol.GetNumAtoms()
    assert data.edge_index.shape[0] == 2
    # undirected edges → even number
    assert data.edge_index.shape[1] % 2 == 0
    # edge attributes length matches edges
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.x.dtype == torch.float32
