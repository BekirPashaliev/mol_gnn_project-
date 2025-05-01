# utils/checkpoint.py
from __future__ import annotations

"""Checkpoint helpers – thin wrappers over torch.save / load with extra info."""

from pathlib import Path
from typing import Any, Dict

import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(state_dict: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path: str | Path):
    return torch.load(path, map_location="cpu")
