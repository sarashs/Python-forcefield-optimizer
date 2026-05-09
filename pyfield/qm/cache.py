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
    extras = getattr(structure, "__pydantic_extra__", {}) or {}
    return json.dumps({
        "atoms": rows,
        "box": [round(b, 6) for b in structure.box],
        # `pbc` flips the QM mode (cluster vs periodic). Fold it into
        # the key so the same atoms at the same coords give different
        # cache entries depending on which mode was used.
        "pbc": bool(getattr(structure, "pbc", False)),
        # Per-structure QM overrides — different code/functional/basis
        # → different result → different cache entry.
        "qm_code": extras.get("qm_code"),
        "qm_functional": extras.get("qm_functional"),
        "qm_basis": extras.get("qm_basis"),
    }, sort_keys=True)


def _merge_relax_with_input(cached: StructureCfg, original: StructureCfg) -> StructureCfg:
    """Fold the input's dispatch metadata onto a cache-loaded structure.

    The cache holds atoms / box / energy. Everything else that affects
    downstream backend dispatch (pbc, qm_code, qm_functional, qm_basis,
    constraint) is taken from the original input structure — the cache
    hit means the same chemistry as a fresh relax, so the dispatch
    profile must match too.
    """
    update = {
        "pbc": getattr(original, "pbc", False),
        "qm_relax": False,
    }
    extras = getattr(original, "__pydantic_extra__", {}) or {}
    for key in ("qm_code", "qm_functional", "qm_basis", "constraint"):
        if key in extras:
            update[key] = extras[key]
    return cached.model_copy(update=update)


def _canonical_constraint(constraint) -> str:
    """Stable JSON of a constraint spec for cache keying. None → ''."""
    if not constraint:
        return ""
    return json.dumps({
        "kind": constraint["kind"],
        "atoms": [int(a) for a in constraint["atoms"]],
        "value": round(float(constraint["value"]), 6),
    }, sort_keys=True)


def _key(structure: StructureCfg, fingerprint: str, op: str,
         constraint=None) -> str:
    h = hashlib.sha256()
    h.update(_canonical_atoms(structure).encode())
    h.update(fingerprint.encode())
    h.update(op.encode())
    h.update(_canonical_constraint(constraint).encode())
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
        from pyfield.config.schema import StructureCfg as _S
        # Newer cache entries roundtrip the *full* StructureCfg (so
        # `pbc`, `qm_code`, `qm_functional`, `constraint`, … survive
        # the cache hit). Older entries only stored `box / atoms /
        # qm_relax`; fall back to that shape so existing caches
        # don't have to be invalidated.
        if "structure" in d:
            struct = _S.model_validate(d["structure"])
        else:
            from pyfield.config.schema import AtomCfg
            atoms = [AtomCfg(**row) for row in d["atoms"]]
            struct = _S(box=tuple(d["box"]), atoms=atoms, qm_relax=False)
        return QmRelaxResult(structure=struct, energy_kcal_mol=float(d["energy_kcal_mol"]))

    def _store_relax(self, key: str, result: QmRelaxResult) -> None:
        d = self._entry_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        # Dump the full StructureCfg (extras included) so cache hits
        # produce the same object as a fresh relax — earlier we only
        # stored `box / atoms`, which dropped `pbc`, `qm_code`,
        # `qm_functional`, and the per-scan-point `constraint`,
        # silently routing PBC structures through PySCF on cache hits.
        payload = {
            "kind": "relax",
            "energy_kcal_mol": result.energy_kcal_mol,
            "structure": result.structure.model_dump(mode="json"),
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
        constraint=None,
    ) -> tuple[QmRelaxResult, str, bool]:
        key = _key(structure, fingerprint, op, constraint=constraint)
        if not force and self.has(key):
            cached = self._load_relax(key)
            # Always re-attach the input structure's dispatch metadata
            # (pbc, qm_code, qm_functional, qm_basis, constraint, …) on
            # cache hits. New cache entries already contain these fields,
            # but legacy entries don't — and downstream code routes
            # backends by reading these off the result.structure.
            merged = _merge_relax_with_input(cached.structure, structure)
            return (
                QmRelaxResult(structure=merged, energy_kcal_mol=cached.energy_kcal_mol),
                key,
                True,
            )
        result = compute()
        self._store_relax(key, result)
        return result, key, False
