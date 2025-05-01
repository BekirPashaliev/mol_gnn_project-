# utils/visualizations.py
from __future__ import annotations

"""Plot utilities – uses matplotlib for ROC / PR curves."""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

__all__ = ["plot_roc_curve", "plot_pr_curve"]


def plot_roc_curve(y_true, y_scores, save_path: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true, y_scores, save_path: str | None = None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
