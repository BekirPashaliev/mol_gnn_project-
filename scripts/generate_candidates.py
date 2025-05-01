# scripts/generate_candidates.py

#!/usr/bin/env python3
"""Generate new molecule SMILES via latent-space search.

Example:
    python scripts/generate_candidates.py \
        --cfg configs/default.yaml \
        --vae_ckpt checkpoints/vae_best.ckpt \
        --reg_ckpt checkpoints/reg_best.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import yaml

from mol_gnn_project.data.graph_dataset import build_graph_dataset
from mol_gnn_project.generation.beam_search import beam_search_latent
from mol_gnn_project.graphs.converter import smiles_batch_to_graphs
from mol_gnn_project.models.regressor import IC50Regressor
from mol_gnn_project.models.vae import GraphVAE
from torch_geometric.loader import DataLoader


def main(cfg_path: str, vae_ckpt: str, reg_ckpt: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained models
    node_dim = 16  # placeholder; replace by dataset num_node_features
    vae = GraphVAE(
        node_dim=node_dim,
        hidden_dim=cfg["vae"]["hidden_dim"],
        latent_dim=cfg["vae"]["latent_dim"],
        num_layers=cfg["vae"]["num_layers"],
    ).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()

    regressor = IC50Regressor(cfg["vae"]["latent_dim"], cfg["regressor"]["hidden_dims"]).to(device)
    regressor.load_state_dict(torch.load(reg_ckpt, map_location=device))
    regressor.eval()

    # Seed latent codes from random normals
    n_seeds = cfg["generation"]["n_seeds"]
    seeds = torch.randn(n_seeds, cfg["vae"]["latent_dim"], device=device)

    # Beam search
    beam = beam_search_latent(
        regressor,
        seeds,
        beam_width=cfg["generation"].get("beam_width", 20),
        depth=cfg["generation"].get("depth", 5),
        n_children=cfg["generation"].get("n_children", 30),
        step_size=cfg["generation"].get("step_size", 0.1),
    )

    # Decode top K
    smiles_set: List[str] = []
    for z, pred_ic50 in beam[: cfg["generation"]["top_k"]]:
        smi_list = []  # placeholder until decoder implemented
        # smi_list = latent_to_smiles(z, vae)
        for smi in smi_list:
            smiles_set.append(smi)
    smiles_set = list(dict.fromkeys(smiles_set))  # unique preserve order

    out_path = Path("outputs/candidates.smi")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(smiles_set))
    print(f"Saved {len(smiles_set)} SMILES → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--reg_ckpt", required=True)
    args = parser.parse_args()
    main(args.cfg, args.vae_ckpt, args.reg_ckpt)
