# data/graph_dataset.py

from __future__ import annotations

"""Графовое представление молекул для **PyTorch Geometric** с адаптивным кэшем..

Файл реализует два уровня абстракции
===============================
* **`mol_to_graph_data`** – перевод одного `rdkit.Chem.Mol` в `Data`‑объект
  (узловые, рёберные фичи; дополнительные атрибуты).
* **`GraphDataset`** – обёртка над списком (или `DataFrame`) записей, которая
  коллатирует все объекты в единый in‑memory датасет (PyG `InMemoryDataset`).

Требования этапа 0.6
--------------------
Для каждой молекулы нужно сохранить:
* `x` – матрица атомных признаков         *(N_atoms × F_atom)*.
* `edge_index`, `edge_attr` – список рёбер и их признаков.
* **Доп. поля**
    * `target_idx  : int64`  – индекс белковой цели.
    * `org_idx     : int64`  – индекс организма.
    * `y           : float32[4]` – вектор био‑активностей `[pKi, pIC50, pEC50, pKd]`.
    * `mask        : bool[4]`   – индикаторы наличия каждой активности
      (`mask_pki`, `mask_pic50`, …).
    * `mol_id`     : str        – исходный идентификатор (для отладочных логов).

Этот формат полностью совместим с остальной архитектурой (VAE‑энкодер,
мульти‑головый регрессор и т. д.).

Теперь `GraphDataset` умеет:
* **Автоматический кеш**: если в каталоге `root/processed/` уже лежит
  `graphs.pt`, датасет мгновенно загружается; иначе строится «на лету» и
  сохраняется для будущих запусков.
* Методы удобства:
  * `get_smiles(idx)` — исходный SMILES по индексу.
  * `set_target_scaler(scaler)` — привязать `StandardScaler` (или любой
    объект с `.transform/.inverse_transform`) для целей.
  * `to_device(device)` — перенести весь датасет в GPU/CPU одним вызовом.

Поддержка `transform/pre_transform` остаётся: если переданы функции, они
применяются (в RAM) *после* загрузки, независимо от того, пришли данные из
кеша или были построены заново.

"""

from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from tqdm.auto import tqdm

from mol_gnn_project.graphs.featurizer import atom_features, bond_features

__all__ = ["GraphDataset", "build_graph_dataset", "mol_to_graph_data"]

# ---------------------------------------------------------------------------
# ---------- низкоуровневое преобразование Mol → Data -----------------------
# ---------------------------------------------------------------------------

def mol_to_graph_data(
    mol: Chem.Mol,
    target_idx: int,
    org_idx: int,
    y: Sequence[float],
    mask: Sequence[int],
    mol_id: str | int,
) -> Data:

    """Построить `torch_geometric.data.Data` из RDKit‑молекулы и метаданных."""

    # --- узловые признаки ---
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    # --- рёбра (неорграф → дублируем оба направления) ---
    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        y=torch.tensor(y, dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.bool),
        target_idx=torch.tensor([target_idx], dtype=torch.int64),
        org_idx=torch.tensor([org_idx], dtype=torch.int64),
        mol_id=str(mol_id),
    )
    return data


# ---------------------------------------------------------------------------
# ---------- build_graph_dataset -------------------------------------------
# ---------------------------------------------------------------------------

def build_graph_dataset(records: pd.DataFrame | List[dict | tuple], **kwargs):
    """Создать `GraphDataset` *на лету* из DataFrame или списка словарей.

    Делегируем всё в конструктор, лишь приводим к единообразному формату.

    Thin‑wrapper вокруг ``GraphDataset``.

    Передаёт все позиционные/именованные аргументы дальше, поэтому можно
    писать как
    ```python
    build_graph_dataset(train_df, root="cache/train", transform=my_tf)
    ```
    а не заботиться о сигнатуре.
    """
    return GraphDataset(records, **kwargs)

# ---------------------------------------------------------------------------
# ---------- основной класс PyG‑датасета ------------------------------------
# ---------------------------------------------------------------------------
# ---------- In‑memory датасет c адаптивным кешем ---------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class GraphDataset(InMemoryDataset):
    """In‑memory датасет молекулярных графов.

    Parameters
    ----------
    records : pandas.DataFrame | Sequence
        Должен содержать (как колонки или позиции):
        `id, smiles, target_idx, org_idx, pKi, pIC50, pEC50, pKd,`
        `mask_pki, mask_pic50, mask_pec50, mask_pkd`.
    """

    @property
    def raw_file_names(self):  # noqa: D401
        return []  # мы не используем raw-файлы

    @property
    def processed_file_names(self):  # noqa: D401
        return ["graphs.pt"]

    def __init__(
        self,
        records: pd.DataFrame | List[tuple | dict],
        root: str | Path = "./",
        use_cache: bool = True,
        transform=None,
        pre_transform=None,
    ) -> None:

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache_path = self.root / "processed" / "graphs.pt"
        self._cache_path.parent.mkdir(exist_ok=True)

        # meta для вспомогательных методов (smiles и т.п.)
        self._smiles: List[str] = []

        self._scaler = None  # type: ignore

        # базовый конструктор создаёт служебные поля
        super().__init__(str(root), transform, pre_transform)

        # ---------- загрузка или построение ----------

        if use_cache and self._cache_path.exists():
            obj = torch.load(self._cache_path, weights_only=False)
            self._data, self.slices = obj["data"], obj["slices"]
            self._smiles = obj["smiles"]
        else:
            if records is None:
                raise ValueError("Первый запуск: необходимо передать records")
            df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
            self._data, self.slices = self._process_records(df, pre_transform)
            self._smiles = df["smiles"].tolist()
            if use_cache:
                torch.save(
                    {"data": self._data, "slices": self.slices, "smiles": self._smiles},
                    self._cache_path,
                    pickle_protocol=4,
                )

    # ------------------------------------------------------------------
    # Дополнительные утилиты
    # ------------------------------------------------------------------

    def get_smiles(self, idx: int) -> str:
        """Вернуть исходный SMILES по индексу графа."""
        return self._smiles[idx]

    def set_target_scaler(self, scaler):
        """Привязать внешний scaler (StandardScaler, MinMaxScaler…)"""
        self._scaler = scaler

    def to_device(self, device):
        """Переместить *всю* батч‑структуру в `device` (GPU/CPU)."""
        self._data = self._data.to(device)
        return self



    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _process_records(df: pd.DataFrame, pre_transform=None):
        data_list: List[Data] = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="[Graph] Mol→Graph"):
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None:
                continue  # уже не должно случиться после loader, но на всякий

            y = [row.pKi, row.pIC50, row.pEC50, row.pKd]
            mask = [row.mask_pki, row.mask_pic50, row.mask_pec50, row.mask_pkd]

            data = mol_to_graph_data(
                mol=mol,
                target_idx=int(row.target_idx),
                org_idx=int(row.org_idx),
                y=[v if pd.notna(v) else 0.0 for v in y],  # NaN → 0 (маска скажет, что нет значения)
                mask=mask,
                mol_id=row.id,
            )
            if pre_transform is not None:
                data = pre_transform(data)
            data_list.append(data)

        if not data_list:
            raise ValueError("Нет валидных молекул для создания GraphDataset")

        return InMemoryDataset.collate(data_list)

    # `process` не нужен: логику вынесли вручную (PyG всё равно проверяет файл‑кеш)
    def process(self):  # noqa: D401, pylint: disable=arguments-differ
        pass



















