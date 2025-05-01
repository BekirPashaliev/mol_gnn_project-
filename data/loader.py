# data/loader.py

from __future__ import annotations
"""
Утилиты для загрузки и первичной обработки данных в проекте **mol_gnn_project**.

Этапы, реализуемые в этом модуле
---------------------------------
0.1  **Загрузка** файла `compounds.csv` с минимальным набором колонок
     `ID, SMILES, Target, Organism, pKi, pIC50, pEC50, pKd`.
0.2  **Очистка SMILES**
     * удаляем соли (оставляем самый тяжёлый фрагмент);
     * канонизируем строку (`RDKit` → canonical SMILES).
0.3  **Категории**
     * 533 различных белковых целей   → `target_idx` (int64);
     * 4 организма (Human/Mice/…)     → `org_idx`   (int64).
0.4  **Маски пропусков** для био‑активностей
     `mask_pki`, `mask_pic50`, `mask_pec50`, `mask_pkd` (int8).
0.5  **Стратифицированный сплит** `train/val/test` по 4‑битному паттерну
     наличия активностей (`0000` … `1111`).

Публичный API
-------------
``load_raw_data(path)`` → полностью очищенный `pd.DataFrame`
``clean_smiles(text)``  → канонический SMILES **или** `None`
``split_dataset(df)``   → три `DataFrame` с теми же колонками

> Словари `TARGET2IDX` и `ORG2IDX` заполняются при первом вызове
> `load_raw_data` и могут быть переиспользованы в других частях кода.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from rdkit import Chem

from rdkit.Chem import rdchem, SanitizeFlags
from rdkit import RDLogger
from tqdm.auto import tqdm

__all__ = [
    "load_raw_data",
    "clean_smiles",
    "split_dataset",
    "TARGET2IDX",
    "ORG2IDX",
]

# ---------------------------------------------------------------------------
# Глобальные словари «категория → индекс»
# ---------------------------------------------------------------------------

TARGET2IDX: Dict[str, int] = {}
ORG2IDX: Dict[str, int] = {}

# Колонки с био‑активностями (log‑scale).  Измените здесь → обновится везде.
BIO_COLS: Tuple[str, str, str, str] = ("pKi", "pIC50", "pEC50", "pKd")

# ---------------------------------------------------------------------------
# Настройки RDKit: глушим спам‑сообщения
# ---------------------------------------------------------------------------

RDLogger.DisableLog("rdApp.error")  # explicit valence, kekulize …
RDLogger.DisableLog("rdApp.warning")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Привести названия колонок к единому формату (snake_case)."""
    rename_map = {
        "ID": "id",
        "Smiles": "smiles",
        "SMILES": "smiles",
        "Target": "target",
        "Organism": "organism",
        # био‑активности оставляем без изменений (важен регистр)
        "pKi": "pKi",
        "pIC50": "pIC50",
        "pEC50": "pEC50",
        "pKd": "pKd",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=rename_map)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Вернуть фрагмент с максимальным числом тяжёлых атомов."""
    try:
        if mol.GetNumAtoms() == 0:
            return mol
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            return frags[0]
        frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
        return frags[0]
    except Exception:
        return None


# ---------- автоматическая санация ----------

def _sanitize_mol_auto(mol: Chem.Mol) -> Chem.Mol | None:
    """Пытаемся санитизировать молекулу, автоматически пропуская сбойные этапы."""
    try:
        # быстрая ветка — всё хорошо
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass  # перейдём к адаптивному режиму

    # catchErrors=True → RDKit *не* кидает исключение, а возвращает флаги ошибок
    error_flags = Chem.SanitizeMol(mol, catchErrors=True)
    if error_flags == 0:
        return mol  # неожиданно, но всё в порядке

    try:
        Chem.SanitizeMol(mol, sanitizeOps=int(SanitizeFlags.SANITIZE_ALL ^ error_flags))
        return mol
    except Exception:
        return None  # спасти не удалось

# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------

def clean_smiles(smiles: str | None) -> str | None:
    """Канонизировать *smiles*; вернуть `None`, если строка невалидна."""
    if smiles is None or not isinstance(smiles, str) or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    mol = _sanitize_mol_auto(mol)
    if mol is None:
        return None

    mol = _largest_fragment(mol)
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Загрузить CSV и выполнить шаги 0.1–0.4."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # ---------- 0.1 Загрузка ----------
    df = pd.read_csv(path)  # [:10000] # для небольших тестов
    df = _standardise_column_names(df)

    required = {"id", "smiles", "target", "organism", *BIO_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {sorted(missing)}")

    # ---------- 0.2 Очистка SMILES ----------
    tqdm.pandas(desc="[Loader] Санация SMILES")
    df["smiles"] = df["smiles"].progress_apply(clean_smiles)

    n_before = len(df)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        pct = n_dropped / n_before * 100
        print(f"[Loader] Отброшено невалидных SMILES: {n_dropped} ({pct:.1f}%)")

    # ---------- 0.3 Категориальные индексы ----------
    global TARGET2IDX, ORG2IDX
    TARGET2IDX = {t: i for i, t in enumerate(sorted(df["target"].unique()))}
    ORG2IDX   = {o: i for i, o in enumerate(sorted(df["organism"].unique()))}

    df["target_idx"] = df["target"].map(TARGET2IDX).astype("int64")
    df["org_idx"]    = df["organism"].map(ORG2IDX).astype("int64")

    # ---------- 0.4 Маски био‑активностей ----------
    for col in BIO_COLS:
        mask = f"mask_{col.lower()}"
        df[mask] = df[col].notna().astype("int8")

    return df

def split_dataset(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разделение по паттерну масок (4 бита)."""
    from sklearn.model_selection import train_test_split

    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"Сумма ratios должна быть 1.0, получено {ratios}")

    # формируем строку‑паттерн, например '1101'
    strat_col = df[[f"mask_{c.lower()}" for c in BIO_COLS]].astype(str).agg("".join, axis=1)

    # --- первый сплит: train vs (val+test) ---
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - ratios[0],
        stratify=strat_col,
        random_state=random_state,
    )

    # --- второй сплит: val vs test ---
    val_rel = ratios[1] / (ratios[1] + ratios[2])
    strat_temp = temp_df[[f"mask_{c.lower()}" for c in BIO_COLS]].astype(str).agg("".join, axis=1)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_rel,
        stratify=strat_temp,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
