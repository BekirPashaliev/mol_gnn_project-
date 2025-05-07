# utils/metrics.py

"""utils/metrics.py – расширенный набор метрик.

Содержит  ➜  *классические* метрики (`compute_auc`, `compute_pr`, `compute_mae`) и
          ➜  *химические* метрики для Graph VAE.
Все функции возвращают **float**.
"""
from typing import Sequence, Tuple
import math
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.DataStructs import TanimotoSimilarity

import networkx as nx
import numpy as np
import scipy.stats as st

__all__ = [
    # VAE‑loss functions
    "compute_elbo",
    "compute_node_mse",
    "compute_edge_bce",
    "compute_kld",
    # Classic metrics
    "compute_auc",
    "compute_pr",
    "compute_mae",
    # Chemical metrics
    "compute_validity",
    "compute_uniqueness",
    "compute_novelty",
    "compute_avg_tanimoto",
    "compute_avg_ged",
    "compute_atom_count_diff",
    "compute_bond_count_diff",
    "compute_degree_js",
    "compute_internal_diversity",
    "compute_generation_time",
    # Registry
    "ALL_METRICS",
]

# ------------------------------------------------------------------------
# VAE‑loss functions
# ------------------------------------------------------------------------

def compute_elbo(recon_loss: float, kld_loss: float, beta: float = 1.0) -> float:
    """ELBO = recon + β·KLD."""
    return recon_loss + beta * kld_loss


def compute_node_mse(x_rec: torch.Tensor, x: torch.Tensor) -> float:
    """MSE восстановления фич узлов."""
    return F.mse_loss(x_rec, x).item()


def compute_edge_bce(edge_logits: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> float:
    """BCE‑лосс для восстановления рёбер графа"""
    device = edge_logits.device
    target = torch.zeros((num_nodes, num_nodes), device=device)
    target[edge_index[0], edge_index[1]] = 1.0
    logits = torch.zeros_like(target)
    logits[edge_index[0], edge_index[1]] = edge_logits.squeeze()
    return F.binary_cross_entropy_with_logits(logits, target).item()


def compute_kld(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """KL‑дивергенция q(z|x) || N(0, I)."""
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()


# ------------------------------------------------------------------------
# Classic metrics
# ------------------------------------------------------------------------

def compute_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def compute_pr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return average_precision_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return mean_absolute_error(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

# =========================================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ для Chemical metrics
# =========================================================================

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Конвертация Tensor → numpy без градиентов."""
    return t.detach().cpu().numpy()


def _mol_from_smiles(smi: str):
    return Chem.MolFromSmiles(smi)


def _to_canonical(smi: str):
    m = _mol_from_smiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def _morgan_fp(mol: Chem.Mol, *, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _graph_from_mol(mol: Chem.Mol):
    adj = rdmolops.GetAdjacencyMatrix(mol)
    return nx.from_numpy_array(adj)

# =========================================================================
#  Chemical metrics
# =========================================================================

def compute_validity(smiles: Sequence[str]) -> float:
    """Доля валидных молекул."""
    mols = [_mol_from_smiles(s) for s in smiles]
    return sum(m is not None for m in mols) / max(len(smiles), 1)


def compute_uniqueness(smiles: Sequence[str]) -> float:
    """Доля уникальных канонических SMILES среди валидных."""
    canon = [_to_canonical(s) for s in smiles if _to_canonical(s)]
    return len(set(canon)) / max(len(canon), 1)


def compute_novelty(pred_smiles: Sequence[str], train_smiles: Sequence[str]) -> float:
    """Доля предсказанных SMILES вне train-набора."""
    train_set = set(train_smiles)
    return sum(s not in train_set for s in pred_smiles) / max(len(pred_smiles), 1)


def compute_avg_tanimoto(orig: Sequence[str], pred: Sequence[str]) -> float:
    """Средняя Tanimoto similarity Morgan fp."""
    sims = []
    for o, p in zip(orig, pred):
        m1, m2 = _mol_from_smiles(o), _mol_from_smiles(p)
        if m1 and m2:
            sims.append(TanimotoSimilarity(_morgan_fp(m1), _morgan_fp(m2)))
    return float(np.mean(sims)) if sims else 0.0


def compute_avg_ged(orig: Sequence[str], pred: Sequence[str]) -> float:
    """Среднее graph edit distance (approx)."""
    dists = []
    for o, p in zip(orig, pred):
        m1, m2 = _mol_from_smiles(o), _mol_from_smiles(p)
        if m1 and m2:
            try:
                d = nx.graph_edit_distance(_graph_from_mol(m1), _graph_from_mol(m2))
            except Exception:
                d = math.inf
            dists.append(d)
    return float(np.mean(dists)) if dists else math.nan

def compute_atom_count_diff(orig: Sequence[str], pred: Sequence[str]) -> float:
    """Средняя разница в числе атомов."""
    diffs = []
    for o, p in zip(orig, pred):
        m1, m2 = _mol_from_smiles(o), _mol_from_smiles(p)
        if m1 and m2:
            diffs.append(abs(m1.GetNumAtoms() - m2.GetNumAtoms()))
    return float(np.mean(diffs)) if diffs else 0.0


def compute_bond_count_diff(orig: Sequence[str], pred: Sequence[str]) -> float:
    """Средняя разница в числе связей."""
    diffs = []
    for o, p in zip(orig, pred):
        m1, m2 = _mol_from_smiles(o), _mol_from_smiles(p)
        if m1 and m2:
            diffs.append(abs(m1.GetNumBonds() - m2.GetNumBonds()))
    return float(np.mean(diffs)) if diffs else 0.0


def compute_degree_js(orig: Sequence[str], pred: Sequence[str]) -> float:
    """JS‑дивергенция распределения степеней."""
    vals = []
    for o, p in zip(orig, pred):
        m1, m2 = _mol_from_smiles(o), _mol_from_smiles(p)
        if m1 and m2:
            deg1 = [d for _, d in _graph_from_mol(m1).degree()]
            deg2 = [d for _, d in _graph_from_mol(m2).degree()]
            maxd = max(max(deg1, default=0), max(deg2, default=0))
            p1, _ = np.histogram(deg1, bins=range(maxd+2), density=True)
            p2, _ = np.histogram(deg2, bins=range(maxd+2), density=True)
            m = 0.5*(p1+p2)
            js = 0.5*(st.entropy(p1,m)+st.entropy(p2,m))
            vals.append(js)
    return float(np.mean(vals)) if vals else math.nan


def compute_internal_diversity(smiles: Sequence[str]) -> float:
    """1 – avg Tanimoto over all valid pairs."""
    fps = []
    for s in smiles:
        m = _mol_from_smiles(s)
        if m:
            fps.append(_morgan_fp(m))
    if len(fps)<2:
        return 0.0
    sims = [TanimotoSimilarity(fps[i],fps[j]) for i in range(len(fps)) for j in range(i+1,len(fps))]
    return 1.0 - float(np.mean(sims))


def compute_generation_time(func, *args, **kwargs) -> Tuple[float, any]:
    """Замер времени выполнения функции."""
    start = time.perf_counter()
    res = func(*args, **kwargs)
    return time.perf_counter()-start, res

ALL_METRICS = {
    # VAE-loss functions
    "ELBO": compute_elbo,
    "Node_MSE": compute_node_mse,
    "Edge_BCE": compute_edge_bce,
    "KLD": compute_kld,

    # Classic metrics
    "AUC": compute_auc,
    "PR": compute_pr,
    "MAE": compute_mae,

    # Chemical metrics
    "Validity": compute_validity,
    "Uniqueness": compute_uniqueness,
    "Novelty": compute_novelty,
    "Avg_Tanimoto": compute_avg_tanimoto,
    "Avg_GED": compute_avg_ged,
    "Atom_Count_Diff": compute_atom_count_diff,
    "Bond_Count_Diff": compute_bond_count_diff,
    "Degree_JS": compute_degree_js,
    "Internal_Diversity": compute_internal_diversity,
}