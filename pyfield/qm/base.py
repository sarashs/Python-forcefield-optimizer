"""QM backend interface + result types.

Two operations cover every objective today (see DEV.md §11.2):

- `single_point(structure)` → `QmSinglePoint(energy, forces, charges?)`
- `relax(structure)` → `QmRelaxResult(structure, energy)`

Energies are returned in **kcal/mol** and forces in **kcal/mol/Å** — the
backend does the unit conversion from whatever its native units are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pyfield.config.schema import StructureCfg


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

    def relax(self, structure: StructureCfg) -> QmRelaxResult:  # pragma: no cover
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
