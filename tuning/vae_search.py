# tuning/vae_search.py

from __future__ import annotations

"""Optuna‑search конфигов для Graph VAE."""

from pathlib import Path
from typing import Dict
import optuna
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import Subset

from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.training.vae_trainer import run_training_vae

__all__ = ["objective_vae", "define_search_space_vae"]



def define_search_space_vae(trial: optuna.Trial) -> Dict:
    return {
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128, 256]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "lr": trial.suggest_loguniform("lr", 1e-4, 5e-3),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.3),
    }

def objective_vae(trial:optuna.Trial,dataset,cfg_base:dict,log_root:Path):
    cfg=cfg_base.copy();cfg.update(define_search_space_vae(trial))
    train_idx,val_idx=train_test_split(range(len(dataset)),test_size=0.15,random_state=42)
    train_ds=Subset(dataset,train_idx);val_ds=Subset(dataset,val_idx)
    # определяем dims
    node_dim=train_ds[0].x.size(1)
    edge_dim=None
    if hasattr(train_ds[0],"edge_attr") and train_ds[0].edge_attr is not None:
        edge_dim=train_ds[0].edge_attr.size(1)
    model=GraphVAE(
        node_dim=node_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_layers"],
        edge_decode=False,
        edge_dim=edge_dim,
    )

    run_training_vae(
        model,
        train_ds,
        val_ds,
        cfg,
        Path(log_root) / f"trial_{trial.number}")

    best = torch.load(Path(log_root) / f"trial_{trial.number}" / "vae_best.ckpt", map_location="cpu")
    return best.get("val_loss", 0.0)