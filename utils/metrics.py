# utils/metrics.py
from __future__ import annotations

"""Common metric calculations."""

import torch
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

__all__ = ["compute_auc", "compute_pr", "compute_mae"]


def _to_numpy(t: torch.Tensor):
    if t.is_cuda:
        t = t.cpu()
    return t.detach().numpy()


def compute_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return roc_auc_score(_to_numpy(y_true), _to_numpy(y_pred))


def compute_pr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return average_precision_score(_to_numpy(y_true), _to_numpy(y_pred))


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return mean_absolute_error(_to_numpy(y_true), _to_numpy(y_pred))
