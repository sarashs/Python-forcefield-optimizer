"""QM backend interface + result types.

Two operations cover every objective today (see DEV.md §11.2):

- `single_point(structure)` → `QmSinglePoint(energy, forces, charges?)`
- `relax(structure, constraint?)` → `QmRelaxResult(structure, energy)`

Energies are returned in **kcal/mol** and forces in **kcal/mol/Å** — the
backend does the unit conversion from whatever its native units are.

A `ConstraintSpec` is the geometric constraint enforced during a relax
— distance / angle / dihedral on a small atom tuple. The same spec
drives the FF-side `fix restrain` so QM and FF agree on what's held
fixed (and what's free to relax) at each scan point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict

import numpy as np

from pyfield.config.schema import StructureCfg


class ConstraintSpec(TypedDict):
    """A relax-time geometric constraint, shared by QM and FF sides."""
    kind: Literal["distance", "angle", "dihedral"]
    atoms: List[int]      # 1-based; len 2 for distance, 3 for angle, 4 for dihedral
    value: float          # Å for distance, degrees for angle / dihedral


@dataclass
class QmSinglePoint:
    energy_kcal_mol: float
    forces_kcal_mol_per_A: Optional[np.ndarray] = None     # shape (N, 3) or None
    charges: Optional[Dict[int, float]] = None             # 1-based atom id → charge


@dataclass
class QmRelaxResult:
    structure: StructureCfg                                # with relaxed `atoms:`
    energy_kcal_mol: float


class QmBackend:
    """Interface every QM backend implements. Per-backend modules
    (`pyfield.qm.pyscf_backend`, `.xtb_backend`, …) subclass this and
    register with the `make_backend(qm_cfg)` factory."""

    name: str = "abstract"

    def single_point(self, structure: StructureCfg) -> QmSinglePoint:  # pragma: no cover
        raise NotImplementedError

    def relax(
        self,
        structure: StructureCfg,
        constraint: Optional[ConstraintSpec] = None,
    ) -> QmRelaxResult:  # pragma: no cover
        """Relax `structure`. If `constraint` is given, hold that
        coordinate fixed during the optimisation (for relaxed scan
        points). Backends that can't enforce a constraint raise
        `NotImplementedError` with a clear pointer."""
        raise NotImplementedError

    def settings_fingerprint(self) -> str:
        """Return a deterministic, content-stable string of the QM settings.
        Used by `QmCache` to key results. Backend-specific."""
        raise NotImplementedError


def make_backend(qm_cfg) -> QmBackend:
    """Instantiate the right QmBackend for the configured `qm.code`.

    Free / open-source backends only. Commercial codes (Gaussian, VASP,
    Molpro) are intentionally absent — DEV.md §11.3 has the rationale.
    """
    if qm_cfg.code == "pyscf":
        from pyfield.qm.pyscf_backend import PySCFBackend
        return PySCFBackend(qm_cfg)
    raise NotImplementedError(
        f"qm.code={qm_cfg.code!r} reserved but not yet wired. Today the only "
        "implemented backend is `pyscf`. xtb / qe / gpaw etc. are one new "
        "file under pyfield/qm/ each — see DEV.md §11.13."
    )
