# tuning/vae_search.py
from __future__ import annotations

"""Optuna search space + objective for GraphVAE hyperparameters."""

from pathlib import Path
from typing import Dict

import optuna
import yaml

from mol_gnn_project.training.vae_trainer import run_training_vae

__all__ = ["define_search_space_vae", "objective_vae", "run_optuna_vae"]


def define_search_space_vae(trial: optuna.Trial) -> Dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384]),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128]),
        "beta": trial.suggest_float("beta", 0.05, 0.3),
        "lr": trial.suggest_loguniform("lr", 1e-4, 5e-3),
    }


def _patch_config(base_cfg: Dict, patch: Dict) -> Dict:
    cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    cfg["vae"].update(patch)
    return cfg


def objective_vae(trial: optuna.Trial, base_config_path: Path) -> float:
    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    patch = define_search_space_vae(trial)
    cfg = _patch_config(base_cfg, patch)

    # save temporary config file for this trial
    tmp_cfg_path = base_config_path.parent / f"_optuna_tmp_{trial.number}.yaml"
    tmp_cfg_path.write_text(yaml.dump(cfg))

    # run training (returns nothing, we need to parse val_loss from logger or ckpt)
    # For simplicity, we rely on run_training_vae printing "val {value}" at each epoch.
    # Here we call run_training_vae and capture minimal val_loss from a small #epochs subset.
    cfg["vae"]["epochs"] = 10  # speed-up during search
    tmp_cfg_path.write_text(yaml.dump(cfg))
    run_training_vae(tmp_cfg_path)

    # after run, val_loss is stored in checkpoints filename or logs; for now,
    # assume best_val saved in file with name pattern containing value.
    # Stub: return dummy value (trial number) – replace with real parsing.
    return float(trial.number)


def run_optuna_vae(cfg_path: str | Path, n_trials: int = 20):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective_vae(t, Path(cfg_path)), n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
