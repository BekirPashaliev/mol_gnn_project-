# generation/beam_search.py

from __future__ import annotations

"""Beam-search exploration of latent space.

For each seed latent vector z₀ we generate *k* perturbations, keep top *beam_width*
by lowest predicted IC50, and iterate *depth* levels.  Perturbations are sampled
from N(0, step_size² I).
"""

from typing import List, Tuple

import torch

from mol_gnn_project.models.regressor import IC50Regressor

__all__ = ["beam_search_latent"]


def _perturb(z: torch.Tensor, n_samples: int, step_size: float) -> torch.Tensor:
    noise = torch.randn(n_samples, z.size(-1), device=z.device) * step_size
    return z.unsqueeze(0) + noise  # (n_samples, latent_dim)


def beam_search_latent(
    regressor: IC50Regressor,
    seeds: torch.Tensor,  # (n_seeds, latent_dim)
    beam_width: int = 10,
    depth: int = 5,
    n_children: int = 20,
    step_size: float = 0.1,
) -> List[Tuple[torch.Tensor, float]]:
    """Return list of (latent_vector, predicted_ic50) sorted by activity."""
    device = next(regressor.parameters()).device
    seeds = seeds.to(device)
    beam = [(z, regressor(z.unsqueeze(0)).item()) for z in seeds]
    beam.sort(key=lambda t: t[1])  # ascending IC50
    beam = beam[:beam_width]

    for _ in range(depth):
        candidates: List[Tuple[torch.Tensor, float]] = []
        for z, _ in beam:
            children = _perturb(z, n_children, step_size)
            preds = regressor(children).detach().cpu()
            candidates.extend(list(zip(children.cpu(), preds.tolist())))
        # keep top beam_width
        candidates.sort(key=lambda t: t[1])
        beam = candidates[:beam_width]
    return beam
