# scripts/train_regressor.py

#!/usr/bin/env python3
"""CLI wrapper to train IC50Regressor (Block B).

Example:
    python scripts/train_regressor.py --cfg configs/default.yaml \
        --encoder_ckpt checkpoints/vae_best.ckpt
"""
from __future__ import annotations

import argparse

from mol_gnn_project.training.reg_trainer import run_training_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        required=True,
        help="Path to trained VAE encoder checkpoint (vae_best.ckpt)",
    )
    args = parser.parse_args()
    run_training_reg(args.cfg, args.encoder_ckpt)


if __name__ == "__main__":
    main()
