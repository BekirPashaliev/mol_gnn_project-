# scripts/preprocess_data.py

#!/usr/bin/env python3
"""
Пред‑обработка CSV «SMILES → био‑активности» → сериализованный GraphDataset.

Запуск из корня репозитория::

    python scripts/preprocess_data.py --cfg configs/default.yaml

Ожидаемая структура YAML‑конфига::

    data:
        raw_csv: data/compounds.csv          # исходный CSV
        processed_dataset: data/dataset.pt   # куда сохранять *.pt
    split:
        train: 0.7
        val:   0.15
        test:  0.15
        random_state: 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mol_gnn_project.data.loader import load_raw_data, split_dataset
from mol_gnn_project.data.graph_dataset import build_graph_dataset


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------


def main(cfg_path: str | Path) -> None:
    # ---------- 1. Читаем конфиг ----------
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv = Path(cfg["data"]["raw_csv"])
    out_path = Path(cfg["data"]["processed_dataset"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 2. Загружаем и очищаем CSV ----------
    print(f"[Preprocess] Чтение {raw_csv} ...")
    df_full = load_raw_data(raw_csv)  # включает очистку SMILES + маски

    # ---------- 3. Train/Val/Test сплит ----------
    ratios = (
        cfg["split"]["train"],
        cfg["split"]["val"],
        cfg["split"]["test"],
    )
    train_df, val_df, test_df = split_dataset(
        df_full,
        ratios=ratios,
        random_state=cfg["split"]["random_state"],
    )

    # ---------- 4. GraphDataset для каждой части ----------
    print("[Preprocess] Построение GraphDataset ...")
    use_cache = cfg["data"].get("use_cache", True)
    train_ds = build_graph_dataset(train_df, root="mol_gnn_project/cache/train", use_cache=use_cache)
    val_ds = build_graph_dataset(val_df, root="mol_gnn_project/cache/val", use_cache=use_cache)
    test_ds = build_graph_dataset(test_df, root="mol_gnn_project/cache/test", use_cache=use_cache)

    bundle = {"train": train_ds, "val": val_ds, "test": test_ds}
    torch.save(bundle, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[Preprocess] Сохранено → {out_path} ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# CLI‑обёртка
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre‑process SMILES CSV → dataset.pt")
    parser.add_argument("--cfg", type=str, required=True, help="Путь до YAML‑конфига")
    args = parser.parse_args()
    main(args.cfg)
