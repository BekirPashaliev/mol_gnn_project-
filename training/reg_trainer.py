# training/reg_trainer.py
from __future__ import annotations

"""Training loop for IC50Regressor (Block B)."""

from pathlib import Path
from typing import Dict

import torch
from torch_geometric.loader import DataLoader
import yaml

from mol_gnn_project.data.graph_dataset import GraphDataset
from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.models.regressor import IC50Regressor
from mol_gnn_project.training.optimizer import init_optimizer
from mol_gnn_project.utils.logger import Logger

__all__ = [
    "train_epoch_reg",
    "validate_epoch_reg",
    "run_training_reg",
]


def _encode_dataset(encoder: GraphVAE, loader: DataLoader, device) -> torch.Tensor:
    """Return latent matrix Z (N × latent_dim) and IC50 vector y."""
    encoder.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = encoder.encode_graph(batch)
            zs.append(mu.cpu())
            ys.append(batch.y.cpu())
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def train_epoch_reg(model, z, y, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(z)
    loss = torch.nn.functional.l1_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_epoch_reg(model, z, y):
    model.eval()
    with torch.no_grad():
        pred = model(z)
        loss = torch.nn.functional.l1_loss(pred, y)
    return loss.item()


def run_training_reg(config_path: str | Path, encoder_ckpt: str | Path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    bundle: Dict[str, GraphDataset] = torch.load(cfg["data"]["processed_dataset"])
    train_dl = DataLoader(bundle["train"], batch_size=cfg["regressor"]["batch_size"], shuffle=False)
    val_dl = DataLoader(bundle["val"], batch_size=cfg["regressor"]["batch_size"], shuffle=False)

    # load frozen encoder
    node_dim = bundle["train"].num_node_features
    encoder = GraphVAE(
        node_dim=node_dim,
        hidden_dim=cfg["vae"]["hidden_dim"],
        latent_dim=cfg["vae"]["latent_dim"],
        num_layers=cfg["vae"]["num_layers"],
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.encoder.eval()  # freeze weights

    # pre-compute latent matrices
    z_train, y_train = _encode_dataset(encoder, train_dl, device)
    z_val, y_val = _encode_dataset(encoder, val_dl, device)

    z_train, y_train = z_train.to(device), y_train.to(device)
    z_val, y_val = z_val.to(device), y_val.to(device)

    # regressor
    model = IC50Regressor(cfg["vae"]["latent_dim"], cfg["regressor"]["hidden_dims"]).to(device)
    opt, sch = init_optimizer(
        model,
        lr=cfg["regressor"]["lr"],
        weight_decay=cfg["regressor"]["weight_decay"],
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
    )

    logger = Logger(cfg["logging"]["log_dir"], run_name="regressor")

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, cfg["regressor"]["epochs"] + 1):
        train_loss = train_epoch_reg(model, z_train, y_train, opt)
        val_loss = validate_epoch_reg(model, z_val, y_val)
        sch.step(val_loss)

        logger.log_scalars({"train_l1": train_loss, "val_l1": val_loss}, step=epoch)
        print(f"Epoch {epoch:03d}: train L1 {train_loss:.4f}  val L1 {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "reg_best.ckpt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved new best model → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping – no val improvement for 10 epochs.")
                break

    logger.close()
