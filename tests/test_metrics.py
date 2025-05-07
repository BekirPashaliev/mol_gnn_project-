# tests/test_metrics.py

"""tests/test_metrics.py: Юнит-тесты для всего набора метрик utils/metrics.py"""

import pytest
import torch
import numpy as np

torch.manual_seed(0)

from mol_gnn_project.utils.metrics import (
    compute_elbo,
    compute_node_mse,
    compute_edge_bce,
    compute_kld,
    compute_auc,
    compute_pr,
    compute_mae,
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_avg_tanimoto,
    compute_avg_ged,
    compute_atom_count_diff,
    compute_bond_count_diff,
    compute_degree_js,
    compute_internal_diversity,
    compute_generation_time,
)

# === VAE-loss tests ===

def test_compute_elbo():
    assert compute_elbo(1.0, 2.0, beta=0.5) == pytest.approx(1.0 + 0.5 * 2.0)


def test_node_mse_zero():
    x = torch.ones((5, 3))
    x_rec = x.clone()
    assert compute_node_mse(x_rec, x) == pytest.approx(0.0)


def test_edge_bce_perfect():
    # граф: 2 узла, одно ребро 0->1
    edge_index = torch.tensor([[0], [1]])
    logits = torch.tensor([[10.0]])  # сильная уверенность в наличии ребра
    loss = compute_edge_bce(logits, edge_index, num_nodes=2)
    assert loss >= 0.0  # должно быть неотрицательным
    assert loss < 1.0  # слишком большой loss сигнализирует об ошибке в реализации


def test_kld_zero():
    mu = torch.zeros((2, 4))
    logvar = torch.zeros_like(mu)
    assert compute_kld(mu, logvar) == pytest.approx(0.0)

# === Classic metrics ===

def test_classic_auc_pr_mae():
    y_true = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float32)
    auc = compute_auc(y_true, y_pred)
    pr = compute_pr(y_true, y_pred)
    mae = compute_mae(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 1.5]))
    assert 0.9 < auc <= 1.0
    assert 0.8 < pr <= 1.0
    assert mae == pytest.approx(0.5)

# === Chemical metrics ===
_VALID = ["CCO", "c1ccccc1"]  # этанол, бензол
_DUPL = ["CCO", "CCO", "c1ccccc1"]

def test_validity_uniqueness_novelty():
    assert compute_validity(_VALID) == 1.0
    assert compute_validity(["BAD"]) == 0.0
    assert compute_uniqueness(_DUPL) == pytest.approx(2/3)
    assert compute_novelty(["c1ccccc1"], ["CCO"]) == 1.0
    assert compute_novelty(["CCO"], ["CCO"]) == 0.0


def test_graph_metrics_identity():
    orig = ["CCO", "c1ccccc1"]
    pred = ["CCO", "c1ccccc1"]
    assert compute_avg_tanimoto(orig, pred) == pytest.approx(1.0)
    assert compute_avg_ged(orig, pred) == pytest.approx(0.0)
    assert compute_atom_count_diff(orig, pred) == pytest.approx(0.0)
    assert compute_bond_count_diff(orig, pred) == pytest.approx(0.0)
    assert compute_degree_js(orig, pred) == pytest.approx(0.0)


def test_internal_diversity_range():
    # разные молекулы дают 0 <= diversity <= 1
    div = compute_internal_diversity(_VALID)
    assert 0.0 <= div <= 1.0


def test_generation_time():
    dur, res = compute_generation_time(lambda x: x + 1, 10)
    assert res == 11
    assert dur >= 0.0
