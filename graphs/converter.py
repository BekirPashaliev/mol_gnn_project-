# graphs/converter.py

from __future__ import annotations

"""Converter: SMILES ⇄ RDKit Mol ⇄ PyG Data."""

from typing import List

import torch
from rdkit import Chem
from torch_geometric.data import Data

from mol_gnn_project.graphs.featurizer import atom_features, bond_features

__all__ = [
    "mol_from_smiles",
    "mol_to_graph_data",
    "smiles_batch_to_graphs",
]


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Return RDKit Mol object or *None* if invalid."""
    return Chem.MolFromSmiles(smiles)


def mol_to_graph_data(mol: Chem.Mol) -> Data:
    """RDKit Mol → torch_geometric.data.Data with node/edge features."""
    node_feats: List[List[float]] = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    edge_index: List[List[int]] = []
    edge_attr_feats: List[List[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # undirected (i→j, j→i)
        edge_index.extend([[i, j], [j, i]])
        edge_attr_feats.extend([feat, feat])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_batch_to_graphs(smiles_list: List[str]) -> List[Data]:
    """Utility: list of SMILES → list of Data graphs."""
    out: List[Data] = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        out.append(mol_to_graph_data(mol))
    return out
