"""`pyfield make-scan` — turn a `scans:` block into structures + sims + targets.

Each `ScanCfg` entry expands to N points (`values` or linspace from
`range`). For every point we:

- copy the reference structure, apply the per-kind geometric transform
  (legs/anchors come along as rigid groups), and add it under the name
  `{prefix}_{i}`;
- emit a simulation against that structure. With `relax_method: rigid`
  it's a `single_point`. With `relax_method: relaxed_constrained` it's
  a `minimize` carrying a `restraints:` string built from the constraint
  spec, which is the FF-side analogue of the QM constrained relax;
- emit an `energy_combination` target with `terms: {<sim>: +1, <ref_sim>: -1}`
  and `target: { from: dft }`;
- attach a `constraint` field on the generated structure so `qm-prep`
  can drive a constrained QM relax. Reference structures don't get
  constraints (they sit at the equilibrium geometry already).

A `single_point` against the reference is added once (named
`{reference}_sp` if not already present) so all the per-point targets
can reference it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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
# transform dispatch — extra fields read from cfg or __pydantic_extra__
# ---------------------------------------------------------------------------

def _apply(scan: ScanCfg, structure: StructureCfg, value: float) -> StructureCfg:
    extras = scan.__pydantic_extra__ or {}
    legs = scan.legs
    if scan.type == "bond_stretch":
        return T.bond_stretch(structure, extras["atoms"], value, legs=legs)
    if scan.type == "angle_bend":
        return T.angle_bend(structure, extras["atoms"], value, legs=legs)
    if scan.type == "dihedral":
        return T.dihedral(structure, extras["atoms"], value, legs=legs)
    if scan.type == "atom_displacement":
        return T.atom_displacement(
            structure, extras["atom"], extras["direction"], value
        )
    if scan.type == "dimer_separation":
        return T.dimer_separation(
            structure, scan.anchors, scan.fragments, value,
            direction=extras.get("direction", "auto"),
        )
    if scan.type == "isotropic_scale":
        return T.isotropic_scale(structure, value)
    raise NotImplementedError(f"unknown scan.type={scan.type!r}")


# ---------------------------------------------------------------------------
# constraint spec — describes "what's held fixed during the relax"
# ---------------------------------------------------------------------------

def _constraint_spec(scan: ScanCfg, value: float) -> Dict | None:
    """Return a {kind, atoms, value} dict describing the constraint that
    QM (geomeTRIC `$set`) and FF (`fix restrain`) must enforce, or None
    when no constraint applies (rigid scans, atom_displacement,
    isotropic_scale)."""
    if scan.relax_method != "relaxed_constrained":
        return None
    extras = scan.__pydantic_extra__ or {}
    if scan.type == "bond_stretch":
        return {"kind": "distance", "atoms": list(extras["atoms"]), "value": float(value)}
    if scan.type == "angle_bend":
        return {"kind": "angle", "atoms": list(extras["atoms"]), "value": float(value)}
    if scan.type == "dihedral":
        return {"kind": "dihedral", "atoms": list(extras["atoms"]), "value": float(value)}
    if scan.type == "dimer_separation":
        return {"kind": "distance", "atoms": list(scan.anchors), "value": float(value)}
    return None


def _restraint_string(constraint: Dict, K: float) -> str:
    """Render a LAMMPS `fix restrain` argument for one constraint."""
    kind = constraint["kind"]
    atoms = constraint["atoms"]
    v = constraint["value"]
    if kind == "distance":
        i, j = atoms
        return f"bond {i} {j} {K} {K} {v}"
    if kind == "angle":
        i, j, k = atoms
        return f"angle {i} {j} {k} {K} {K} {v}"
    if kind == "dihedral":
        i, j, k, l = atoms
        return f"dihedral {i} {j} {k} {l} {K} {K} {v}"
    raise NotImplementedError(f"unknown constraint kind: {kind!r}")


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
            sim_name = f"{struct_name}_sp"  # name kept stable across relax_method;
                                            # actual sim `type:` is single_point or
                                            # minimize depending on the scan
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
            if new_struct.qm_relax:
                # Don't re-relax scan points unconditionally — qm_relax
                # makes sense on the reference, not on perturbed copies
                # whose whole point is to sample off-equilibrium geometry.
                new_struct = new_struct.model_copy(update={"qm_relax": False})

            constraint = _constraint_spec(scan, value)
            if constraint is not None:
                # Stash the constraint on the generated structure for the
                # QM populator to pick up. `extra="allow"` on StructureCfg
                # round-trips it through YAML cleanly.
                new_struct = new_struct.model_copy(update={
                    "qm_relax": True, "constraint": constraint,
                })

            new_structures[struct_name] = new_struct

            # Build the FF-side simulation. relaxed_constrained → minimize
            # with a `restraints:` string holding the same constraint as
            # the QM side. rigid → single_point at the perturbed geometry.
            if scan.relax_method == "relaxed_constrained":
                if constraint is None:
                    raise RuntimeError(
                        f"scan {scan.name_prefix!r}: relax_method="
                        "relaxed_constrained but no constraint could be "
                        "derived from this scan kind."
                    )
                restraint_str = _restraint_string(constraint, scan.restraint_k)
                sim_cfg = SimulationCfg.model_validate({
                    "structure": struct_name,
                    "type": "minimize",
                    "restraints": [restraint_str],
                })
            else:
                sim_cfg = SimulationCfg(structure=struct_name, type="single_point")
            new_simulations[sim_name] = sim_cfg

            # Reference simulation matches: rigid scans use single_point
            # on the reference; relaxed scans need the reference at its
            # equilibrium too — which is also a single_point (no
            # constraint, ref already at equilibrium).
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
                    comment=f"{scan.type} {scan.name_prefix} idx={idx} "
                            f"value={value} relax={scan.relax_method}",
                )

        summary.append(
            f"{scan.type:<19} ref={ref_name:<10} prefix={scan.name_prefix:<12} "
            f"N={len(grid)}  relax={scan.relax_method}"
        )
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
