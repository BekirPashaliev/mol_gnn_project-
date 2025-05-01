# generation/gradient_search.py

from __future__ import annotations

"""Gradient-based search in latent space to minimise predicted IC50.

This is a *stub*; more advanced techniques (e.g. constrained optimisation,
validity filters) can be layered later.
"""

from typing import List

import torch
from rdkit import Chem

from mol_gnn_project.models.vae import GraphVAE
from mol_gnn_project.models.regressor import IC50Regressor

__all__ = ["optimize_latent", "latent_to_smiles"]


def optimize_latent(
    z0: torch.Tensor,
    regressor: IC50Regressor,
    n_steps: int = 50,
    lr: float = 0.05,
) -> torch.Tensor:
    """Return *new* latent vector with lower predicted IC50."""
    z = z0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([z], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = regressor(z)
        pred.backward()  # gradient w.r.t z
        optimizer.step()
    return z.detach()


# ------------------------------------------------------------------
# NOTE: A proper decoder must output a *graph* that maps back to RDKit Mol.
# For now we assume GraphVAE.decode(z) → node feature matrix *and* we combine
# with a simple scaffold (placeholder). In practice you'd implement a
# graph-generative decoder (e.g. GraphVAE with edge prediction) or call an external
# graph builder. Here we return dummy SMILES to keep pipeline coherent.
# ------------------------------------------------------------------

def _dummy_decode_to_mol(vae: GraphVAE, z: torch.Tensor) -> Chem.Mol | None:
    """Placeholder: always returns None (to be replaced)."""
    _ = vae  # unused for now
    return None


def latent_to_smiles(z: torch.Tensor, vae: GraphVAE) -> List[str]:
    mol = _dummy_decode_to_mol(vae, z)
    if mol is None:
        return []
    smi = Chem.MolToSmiles(mol)
    return [smi]
