# tests/test_graph_dataset.py
from mol_gnn_project.data.graph_dataset import build_graph_dataset


def test_graph_dataset_length():
    records = [
        ("mol1", "CC", 10.0),
        ("mol2", "O=C=O", 50.0),
    ]
    ds = build_graph_dataset(records)
    assert len(ds) == 2
    # each item has y attribute
    sample = ds[0]
    assert hasattr(sample, "y") and sample.y.numel() == 1
