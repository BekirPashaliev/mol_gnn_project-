# scripts/train_vae.py

#!/usr/bin/env python3
"""CLI wrapper to train GraphVAE (Block A).

Example:
    python scripts/train_vae.py --cfg configs/default.yaml
"""
from __future__ import annotations

import argparse

from mol_gnn_project.training.vae_trainer import run_training_vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    run_training_vae(args.cfg)


if __name__ == "__main__":
    main()
