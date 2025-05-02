# scripts/train_vae.py

from __future__ import annotations

"""CLI‑скрипт обучения GraphVAE.

Шаги:
1. Загружаем YAML‑конфиг.
2. Загружаем готовый **dataset.pt** (если путь задан) *или* строим на лету.
3. Делим на train/val.
4. Создаём модель и запускаем `run_training_vae`.
"""

import argparse
from pathlib import Path
import yaml
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.training.vae_trainer import run_training_vae
from mol_gnn_project.data.graph_dataset import GraphDataset
from mol_gnn_project.data.loader import load_raw_data, clean_smiles, split_dataset
from mol_gnn_project.data.graph_dataset import build_graph_dataset

# ---------------------------------------------------------------------

def load_dataset(cfg: dict):
    """Загрузка GraphDataset: сначала пытаемся по готовому .pt, иначе
    переходим к «сырому» CSV и собираем графы на лету.
    """
    from mol_gnn_project.data.graph_dataset import GraphDataset  # для allow‑list

    ds_path = Path(cfg["data"]["dataset_pt"])
    if ds_path.exists():
        try:
            print(f"[INFO] Loading cached dataset from {ds_path}")
            # Важно! weights_only=False — разрешаем полную распаковку объекта
            return torch.load(ds_path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: broad-except
            print(f"[WARN] Couldn't unpickle dataset ({e}). Rebuilding from raw CSV…")

    # --- Фолбэк: строим с нуля ------------------------------------------------
    print(f"[INFO] Reading raw CSV {cfg['data']['csv_path']}...")
    df = load_raw_data(cfg["data"]["csv_path"])

    # Сплитим на train/val/test
    bundle = split_dataset(
        df,
        ratios=(cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]),
        random_state=cfg["split"]["random_state"],
    )
    print("[INFO] Building GraphDataset and caching...")

    train_ds = build_graph_dataset(bundle[0], root="cache/train", use_cache=cfg["data"]["use_cache"])
    val_ds = build_graph_dataset(bundle[1], root="cache/val", use_cache=cfg["data"]["use_cache"])
    test_ds = build_graph_dataset(bundle[2], root="cache/test", use_cache=cfg["data"]["use_cache"])

    result = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, ds_path)
    return result

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML‑конфиг с гиперпараметрами")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))

    # Устройство (GPU/CPU)
    use_gpu = cfg['training'].get('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device set to: {device}")

    dataset = load_dataset(cfg)

    # -------- split train/val ----------------------------------------
    if isinstance(dataset, dict):  # dataset.pt = {train, val, test}
        train_ds = dataset["train"]
        val_ds = dataset["val"]
    else:  # единый датасет – делим на лету
        idx_tr, idx_va = train_test_split(
            range(len(dataset)),
            test_size=cfg['split']['val'],
            random_state=cfg['split']['random_state']
        )
        train_ds = Subset(dataset, idx_tr)
        val_ds = Subset(dataset, idx_va)

    # -------- модель --------------------------------------------------
    node_dim = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1) if train_ds[0].edge_attr is not None else None
    print(f"[INFO] node_dim={node_dim}, edge_dim={edge_dim}")

    model = GraphVAE(
        node_dim=node_dim,
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        num_layers=cfg['model']['num_layers'],
        edge_decode=False,
        edge_dim=edge_dim,
    )

    print("[INFO] Model instantiated.")

    # -------- запуск тренировки --------------------------------------
    run_training_vae(
        model,
        train_ds,
        val_ds,
        cfg['training'],
        Path("runs")/Path(args.cfg).stem
    )


if __name__ == "__main__":
    main()
