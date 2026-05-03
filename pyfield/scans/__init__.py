"""`pyfield make-scan` — turn a `scans:` block into structures + sims + targets.

Each `ScanCfg` entry expands to N points (`values` or linspace from
`range`). For every point we:

- copy the reference structure, apply the per-kind geometric transform,
  and add it under the name `{prefix}_{i}`;
- emit a `single_point` simulation `{prefix}_{i}_sp` against that
  structure;
- emit an `energy_combination` target with `terms: {<sp>: +1, <ref_sp>: -1}`
  and `target: { from: dft }`.

A `single_point` against the reference is added once (named
`{reference}_sp` if not already present) so all the per-point targets
can reference it.

The reference structure must have inline `atoms:` (no xyz `path:` yet).
xyz inputs land when the same xyz reader is shared with the FF-side
simulations — until then, run `qm-relax` to write the relaxed coords
back into the YAML inline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from pyfield.config.schema import (
    PyFieldConfig,
    ScanCfg,
    SimulationCfg,
    StructureCfg,
    TargetCfg,
)
from pyfield.scans import transforms as T


# ---------------------------------------------------------------------------
# value-grid resolution
# ---------------------------------------------------------------------------

def _grid(scan: ScanCfg) -> List[float]:
    if scan.values is not None:
        return [float(v) for v in scan.values]
    start, stop, num = scan.range
    return [float(v) for v in np.linspace(start, stop, int(num))]


# ---------------------------------------------------------------------------
# transform dispatch — extra fields read out of __pydantic_extra__
# ---------------------------------------------------------------------------

def _apply(scan: ScanCfg, structure: StructureCfg, value: float) -> StructureCfg:
    extras = scan.__pydantic_extra__ or {}
    if scan.type == "bond_stretch":
        return T.bond_stretch(structure, extras["atoms"], value)
    if scan.type == "angle_bend":
        return T.angle_bend(structure, extras["atoms"], value)
    if scan.type == "dihedral":
        return T.dihedral(structure, extras["atoms"], value)
    if scan.type == "atom_displacement":
        return T.atom_displacement(
            structure, extras["atom"], extras["direction"], value
        )
    if scan.type == "dimer_separation":
        return T.dimer_separation(
            structure, extras["fragments"], extras.get("direction", "auto"), value
        )
    if scan.type == "isotropic_scale":
        return T.isotropic_scale(structure, value)
    raise NotImplementedError(f"unknown scan.type={scan.type!r}")


# ---------------------------------------------------------------------------
# xyz dump
# ---------------------------------------------------------------------------

def _write_xyz(path: Path, structure: StructureCfg, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(structure.atoms)), comment or "pyfield make-scan"]
    for a in structure.atoms:
        lines.append(f"{a.element} {a.x:.6f} {a.y:.6f} {a.z:.6f}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------

def expand_scans(
    cfg: PyFieldConfig,
    *,
    xyz_dir: Path = None,
) -> Tuple[PyFieldConfig, List[str]]:
    """Expand `cfg.scans` into structures + sims + targets.

    Returns `(expanded_cfg, summary_lines)`. The output config has its
    `scans:` block dropped — re-running `make-scan` on the output is a
    no-op rather than a duplicate-name explosion.
    """
    scans = cfg.scans or []
    if not scans:
        return cfg.model_copy(update={"scans": None}), []

    new_structures: Dict[str, StructureCfg] = dict(cfg.structures)
    new_simulations: Dict[str, SimulationCfg] = dict(cfg.simulations)
    new_targets: List[TargetCfg] = list(cfg.targets)
    summary: List[str] = []

    # Make sure each referenced structure has a single_point sim we can
    # subtract against. Naming convention: `{ref}_sp`. If the user
    # already has one with that name targeting the right structure,
    # reuse it; otherwise create.
    def _ensure_ref_sp(ref_name: str) -> str:
        candidate = f"{ref_name}_sp"
        existing = new_simulations.get(candidate)
        if existing and existing.structure == ref_name and existing.type == "single_point":
            return candidate
        if candidate in new_simulations:
            raise ValueError(
                f"make-scan: simulation name {candidate!r} already exists but "
                f"doesn't match the expected reference single_point. Rename it "
                f"or pick a different `name_prefix`."
            )
        new_simulations[candidate] = SimulationCfg(
            structure=ref_name, type="single_point"
        )
        return candidate

    for scan in scans:
        ref_name = scan.reference
        if ref_name not in new_structures:
            raise ValueError(
                f"make-scan: scan references unknown structure {ref_name!r}"
            )
        ref_struct = new_structures[ref_name]
        if ref_struct.atoms is None:
            raise ValueError(
                f"make-scan: structure {ref_name!r} uses `path:` (xyz) — inline "
                "`atoms:` is required for now. Run `pyfield qm-relax` first to "
                "materialise relaxed coords into the YAML."
            )

        ref_sp = _ensure_ref_sp(ref_name)
        grid = _grid(scan)

        for idx, value in enumerate(grid):
            struct_name = f"{scan.name_prefix}_{idx}"
            sim_name = f"{struct_name}_sp"
            if struct_name in new_structures:
                raise ValueError(
                    f"make-scan: generated structure name {struct_name!r} "
                    f"collides with an existing structure"
                )
            if sim_name in new_simulations:
                raise ValueError(
                    f"make-scan: generated simulation name {sim_name!r} "
                    f"collides with an existing simulation"
                )
            new_struct = _apply(scan, ref_struct, value)
            # Scan points are evaluated at the perturbed geometry — never
            # re-relaxed, even if the reference itself was flagged.
            if new_struct.qm_relax:
                new_struct = new_struct.model_copy(update={"qm_relax": False})
            new_structures[struct_name] = new_struct
            new_simulations[sim_name] = SimulationCfg(
                structure=struct_name, type="single_point"
            )
            new_targets.append(TargetCfg.model_validate({
                "kind": "energy_combination",
                "weight": float(scan.target_weight),
                "terms": {sim_name: +1, ref_sp: -1},
                "target": {"from": "dft"},
            }))
            if xyz_dir is not None:
                _write_xyz(
                    Path(xyz_dir) / f"{struct_name}.xyz",
                    new_struct,
                    comment=f"{scan.type} {scan.name_prefix} idx={idx} value={value}",
                )

        summary.append(
            f"{scan.type:<19} ref={ref_name:<10} prefix={scan.name_prefix:<12} "
            f"N={len(grid)}"
        )
        # Reference xyz too (so the viz can show equilibrium alongside scan).
        if xyz_dir is not None:
            _write_xyz(
                Path(xyz_dir) / f"{ref_name}.xyz",
                ref_struct,
                comment=f"reference for {scan.name_prefix}",
            )

    expanded = cfg.model_copy(update={
        "structures": new_structures,
        "simulations": new_simulations,
        "targets": new_targets,
        "scans": None,
    })
    return expanded, summary
