# training/optimizer.py
from __future__ import annotations

"""Optimizer and scheduler helpers."""

from typing import Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

__all__ = ["init_optimizer"]


def init_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    scheduler_type: str = "plateau",
    scheduler_kwargs: dict | None = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Return (optimizer, scheduler)."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_kwargs = scheduler_kwargs or {}
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, **scheduler_kwargs)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
    return optimizer, scheduler
