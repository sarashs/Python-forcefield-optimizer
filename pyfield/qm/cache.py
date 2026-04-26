"""Content-keyed QM cache.

Keyed on `sha256(canonical_atoms || backend_fingerprint || op)` so:

- Re-running `pyfield qm-prep` is a no-op when nothing changed.
- Adding a target only runs the new structure (if it shares
  the same QM settings).
- Switching `qm.functional: pbe → pbe0` invalidates everything; switch
  back, hits cache.

Each cache entry is a directory under `cache_dir/<hash>/` containing a
`result.json` with the energy / forces / relaxed atoms. The directory
also receives a copy of the inputs (the canonical atom list) so a
`grep` in the cache directory tells you exactly which structure that
hash maps to.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from pyfield.config.schema import StructureCfg
from pyfield.qm.base import QmRelaxResult, QmSinglePoint


def _canonical_atoms(structure: StructureCfg) -> str:
    if structure.atoms is None:
        raise ValueError("cache requires inline atoms (path-loading lands later)")
    rows = [
        [a.element, round(a.x, 6), round(a.y, 6), round(a.z, 6), round(a.charge, 6)]
        for a in structure.atoms
    ]
    return json.dumps({
        "atoms": rows,
        "box": [round(b, 6) for b in structure.box],
    }, sort_keys=True)


def _key(structure: StructureCfg, fingerprint: str, op: str) -> str:
    h = hashlib.sha256()
    h.update(_canonical_atoms(structure).encode())
    h.update(fingerprint.encode())
    h.update(op.encode())
    return h.hexdigest()[:16]


@dataclass
class QmCache:
    cache_dir: Path

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- entry shape -------------------------------------------------

    def _entry_dir(self, key: str) -> Path:
        return self.cache_dir / key

    def has(self, key: str) -> bool:
        return (self._entry_dir(key) / "result.json").exists()

    def _load_single_point(self, key: str) -> QmSinglePoint:
        d = json.loads((self._entry_dir(key) / "result.json").read_text())
        forces = np.asarray(d["forces"]) if d.get("forces") is not None else None
        return QmSinglePoint(
            energy_kcal_mol=float(d["energy_kcal_mol"]),
            forces_kcal_mol_per_A=forces,
            charges={int(k): float(v) for k, v in (d.get("charges") or {}).items()} or None,
        )

    def _store_single_point(self, key: str, result: QmSinglePoint, structure: StructureCfg) -> None:
        d = self._entry_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "single_point",
            "energy_kcal_mol": result.energy_kcal_mol,
            "forces": (
                result.forces_kcal_mol_per_A.tolist()
                if result.forces_kcal_mol_per_A is not None else None
            ),
            "charges": result.charges,
        }
        (d / "result.json").write_text(json.dumps(payload, indent=2))
        (d / "structure.json").write_text(_canonical_atoms(structure))

    def _load_relax(self, key: str) -> QmRelaxResult:
        d = json.loads((self._entry_dir(key) / "result.json").read_text())
        from pyfield.config.schema import AtomCfg, StructureCfg as _S
        atoms = [AtomCfg(**row) for row in d["atoms"]]
        struct = _S(box=tuple(d["box"]), atoms=atoms, qm_relax=False)
        return QmRelaxResult(structure=struct, energy_kcal_mol=float(d["energy_kcal_mol"]))

    def _store_relax(self, key: str, result: QmRelaxResult) -> None:
        d = self._entry_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "relax",
            "energy_kcal_mol": result.energy_kcal_mol,
            "atoms": [a.model_dump() for a in result.structure.atoms],
            "box": list(result.structure.box),
        }
        (d / "result.json").write_text(json.dumps(payload, indent=2))

    # ---------- public memoise helpers --------------------------------------

    def memoise_single_point(
        self,
        structure: StructureCfg,
        fingerprint: str,
        op: str,
        compute: Callable[[], QmSinglePoint],
        *,
        force: bool = False,
    ) -> tuple[QmSinglePoint, str, bool]:
        """Returns (result, cache_key, was_hit)."""
        key = _key(structure, fingerprint, op)
        if not force and self.has(key):
            return self._load_single_point(key), key, True
        result = compute()
        self._store_single_point(key, result, structure)
        return result, key, False

    def memoise_relax(
        self,
        structure: StructureCfg,
        fingerprint: str,
        op: str,
        compute: Callable[[], QmRelaxResult],
        *,
        force: bool = False,
    ) -> tuple[QmRelaxResult, str, bool]:
        key = _key(structure, fingerprint, op)
        if not force and self.has(key):
            return self._load_relax(key), key, True
        result = compute()
        self._store_relax(key, result)
        return result, key, False
