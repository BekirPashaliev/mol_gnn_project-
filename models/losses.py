# models/losses.py

from __future__ import annotations

"""GraphDataset – wraps molecular graphs for PyTorch Geometric."""

from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset

from mol_gnn_project.graphs.converter import mol_from_smiles, mol_to_graph_data

__all__ = ["GraphDataset", "build_graph_dataset"]


def build_graph_dataset(records: List[Tuple[str, str, float]]):
    """Utility to create a *GraphDataset* on-the-fly."""
    return GraphDataset(records)


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        records: List[Tuple[str, str, float]],
        root: str | Path = "./",
        transform=None,
        pre_transform=None,
    ):
        self.records = records
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_records(records)

    # pylint: disable=arguments-differ,unused-argument
    def process(self):
        # unused – we bypass PyG's disk processing, keeping everything in memory
        pass

    def process_records(self, records):
        data_list = []
        for mol_id, smi, ic50 in records:
            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            data = mol_to_graph_data(mol)
            data.y = torch.tensor([ic50], dtype=torch.float)
            data.mol_id = mol_id
            data_list.append(data)
        return self.collate(data_list)
