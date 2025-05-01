# graphs/featurizer.py

from __future__ import annotations

"""Atom / bond featurisation for molecular graphs.

Each function returns a *fixed‑length list* of floats (0/1 or scaled).
These are deliberately simple & can be extended later.
"""

from typing import List

from rdkit import Chem

# ------------------------ atom features ------------------------- #

_ATOM_HYBRIDIZATION = {
    Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0],
    Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],
    Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1],
}


def atom_features(atom: Chem.Atom) -> List[float]:
    """Return a 1D feature list for a single atom."""
    # atomic number (scaled / 100 for numeric stability)
    atomic_num = atom.GetAtomicNum() / 100.0
    degree = atom.GetTotalDegree() / 4.0  # max ~4 for organic chemistry
    formal_charge = atom.GetFormalCharge() / 3.0  # -3…+3 normalised
    hybrid = _ATOM_HYBRIDIZATION.get(atom.GetHybridization(), [0, 0, 0, 0, 0])
    aromatic = [1.0] if atom.GetIsAromatic() else [0.0]
    implicit_h = atom.GetTotalNumHs(includeNeighbors=True) / 8.0

    return [
        atomic_num,
        degree,
        formal_charge,
        implicit_h,
        *hybrid,
        *aromatic,
    ]


# ------------------------- bond features ------------------------ #

_BOND_TYPES = {
    Chem.BondType.SINGLE: [1, 0, 0, 0],
    Chem.BondType.DOUBLE: [0, 1, 0, 0],
    Chem.BondType.TRIPLE: [0, 0, 1, 0],
    Chem.BondType.AROMATIC: [0, 0, 0, 1],
}


def bond_features(bond: Chem.Bond) -> List[float]:
    """Return a 1D feature list for a bond (edge)."""
    bt = _BOND_TYPES.get(bond.GetBondType(), [0, 0, 0, 0])
    conjugated = [1.0] if bond.GetIsConjugated() else [0.0]
    in_ring = [1.0] if bond.IsInRing() else [0.0]
    stereo = [bond.GetStereo()]  # Enum → int 0‑5
    stereo = [stereo[0] / 5.0]
    return [*bt, *conjugated, *in_ring, *stereo]
