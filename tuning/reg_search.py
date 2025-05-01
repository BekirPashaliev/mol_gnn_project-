# tuning/reg_search.py
from __future__ import annotations

"""Optuna search for IC50Regressor hyperparameters."""

from pathlib import Path
from typing import Dict

import optuna
import yaml

from mol_gnn_project.training.reg_trainer import run_training_reg

__all__ = ["define_search_space_reg", "objective_reg", "run_optuna_reg"]


def define_search_space_reg(trial: optuna.Trial) -> Dict:
    return {
        "hidden_dims": [
            trial.suggest_categorical("h1", [64, 128, 256]),
            trial.suggest_categorical("h2", [32, 64, 128]),
        ],
        "lr": trial.suggest_loguniform("lr", 1e-4, 5e-3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def _patch_config(base_cfg: Dict, patch: Dict) -> Dict:
    cfg = yaml.safe_load(yaml.dump(base_cfg))
    cfg["regressor"].update(patch)
    return cfg


def objective_reg(trial: optuna.Trial, base_cfg: Path, encoder_ckpt: Path) -> float:
    with base_cfg.open("r", encoding="utf-8") as f:
        base_cfg_dict = yaml.safe_load(f)
    cfg = _patch_config(base_cfg_dict, define_search_space_reg(trial))
    cfg["regressor"]["epochs"] = 10
    tmp_cfg = base_cfg.parent / f"_optuna_reg_{trial.number}.yaml"
    tmp_cfg.write_text(yaml.dump(cfg))

    run_training_reg(tmp_cfg, encoder_ckpt)
    return float(trial.number)  # stub


def run_optuna_reg(cfg_path: str | Path, encoder_ckpt: str | Path, n_trials: int = 20):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective_reg(t, Path(cfg_path), Path(encoder_ckpt)), n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
