# training/vae_trainer.py
from __future__ import annotations

"""Training loop for GraphVAE (Block A)."""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch_geometric.loader import DataLoader
import yaml

from mol_gnn_project.data.graph_dataset import GraphDataset
from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.models.losses import recon_loss, kld_loss
from mol_gnn_project.models.encoder import GraphEncoder  # ensure module import works
from mol_gnn_project.models.readout import GraphReadout
from mol_gnn_project.training.optimizer import init_optimizer

from mol_gnn_project.utils.logger import Logger

__all__ = [
    "train_epoch_vae",
    "validate_epoch_vae",
    "run_training_vae",
]


def _move_batch(batch, device):
    batch = batch.to(device)
    return batch


def _step_vae(model: GraphVAE, batch, beta: float, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(train):
        recon, mu, logvar = model(batch)
        loss = recon_loss(recon, batch.x) + beta * kld_loss(mu, logvar)
    return loss


def train_epoch_vae(model: GraphVAE, loader: DataLoader, optimizer, device, beta: float):
    total_loss = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        loss = _step_vae(model, batch, beta, train=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def validate_epoch_vae(model: GraphVAE, loader: DataLoader, device, beta: float):
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            loss = _step_vae(model, batch, beta, train=False)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def run_training_vae(config_path: str | Path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load processed bundle
    data_bundle: Dict[str, GraphDataset] = torch.load(cfg["data"]["processed_dataset"])
    train_dl = DataLoader(data_bundle["train"], batch_size=cfg["vae"]["batch_size"], shuffle=True)
    val_dl = DataLoader(data_bundle["val"], batch_size=cfg["vae"]["batch_size"], shuffle=False)

    node_dim = data_bundle["train"].num_node_features
    model = GraphVAE(
        node_dim=node_dim,
        hidden_dim=cfg["vae"]["hidden_dim"],
        latent_dim=cfg["vae"]["latent_dim"],
        num_layers=cfg["vae"]["num_layers"],
    ).to(device)

    opt, sch = init_optimizer(
        model,
        lr=cfg["vae"]["lr"],
        weight_decay=cfg["vae"]["weight_decay"],
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
    )

    logger = Logger(cfg["logging"]["log_dir"], run_name="vae")

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, cfg["vae"]["epochs"] + 1):
        train_loss = train_epoch_vae(model, train_dl, opt, device, cfg["vae"]["beta"])
        val_loss = validate_epoch_vae(model, val_dl, device, cfg["vae"]["beta"])
        sch.step(val_loss)

        logger.log_scalars({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        print(f"Epoch {epoch:03d}: train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "vae_best.ckpt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved new best model → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping – no val improvement for 10 epochs.")
                break

    logger.close()
