"""`pyfield qm-prep` populator.

Walks a `PyFieldConfig`, executes every QM job implied by `qm_relax: true`
structures or `from: dft` placeholders, and returns a populated
`PyFieldConfig` whose YAML round-trip is the canonical input to
`pyfield run`.

What gets populated, by `kind`:

- `energy_combination`: every structure named in `terms:` runs a
  single-point. The target value is `Σ coeff_i · E_i_kcal_mol` minus
  the same combination evaluated on the reference geometry of each
  term — actually no: we simply compute `Σ coeff_i · E_i` because
  energy_combination already encodes the signed sum.
- `charges`: single-point on the named simulation's structure; charges
  come from the backend (currently PySCF returns nothing — we leave a
  TODO). For MVP, `charges` targets must still be hand-typed.
- `forces`: single-point on the named structure with forces returned.
- `structural_match`: relax the named simulation's structure (which
  must also be flagged `qm_relax: true` on the structure side) and
  point `reference:` at an xyz emitted into `output:` (or default
  `<structure_name>_dft.xyz`).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyfield.config.schema import (
    AtomCfg,
    PyFieldConfig,
    StructureCfg,
    is_qm_placeholder,
)
from pyfield.qm.base import QmBackend, QmRelaxResult, QmSinglePoint, make_backend
from pyfield.qm.cache import QmCache


log = logging.getLogger(__name__)


def _structure_for_simulation(cfg: PyFieldConfig, sim_id: str) -> StructureCfg:
    sim = cfg.simulations[sim_id]
    return cfg.structures[sim.structure]


def _write_xyz(path: Path, structure: StructureCfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(structure.atoms)), "pyfield qm-prep relaxed geometry"]
    for a in structure.atoms:
        lines.append(f"{a.element} {a.x:.6f} {a.y:.6f} {a.z:.6f}")
    path.write_text("\n".join(lines) + "\n")


def relax_structures(
    cfg: PyFieldConfig,
    *,
    backend: QmBackend = None,
    force: bool = False,
    only: Optional[List[str]] = None,
) -> Tuple[PyFieldConfig, List[Tuple[str, bool, str]]]:
    """Run QM geom-opt for every `qm_relax: true` structure (or `only`),
    return a config whose structures carry the relaxed coordinates and
    whose `qm_relax` flags have been cleared.

    Targets / placeholders are left untouched — this is the input to
    `make-scan` (which needs equilibrium geometries) before `qm-prep`
    fills in target values.
    """
    if cfg.qm is None:
        raise ValueError("relax_structures called on a config without a `qm:` block")
    backend = backend or make_backend(cfg.qm)
    cache = QmCache(cfg.qm.cache_dir)
    fingerprint = backend.settings_fingerprint()
    journal: List[Tuple[str, bool, str]] = []

    populated_structures: Dict[str, StructureCfg] = dict(cfg.structures)
    requested = set(only) if only else None
    if requested:
        unknown = requested - set(cfg.structures)
        if unknown:
            raise ValueError(f"relax_structures: unknown structures {sorted(unknown)!r}")

    for name, struct in cfg.structures.items():
        if requested is None and not struct.qm_relax:
            continue
        if requested is not None and name not in requested:
            continue
        result, key, hit = cache.memoise_relax(
            struct, fingerprint, "relax",
            lambda s=struct: backend.relax(s),
            force=force,
        )
        populated_structures[name] = result.structure
        journal.append((f"relax {name}", hit, key))

    populated = cfg.model_copy(update={"structures": populated_structures})
    return populated, journal


def populate_qm(
    cfg: PyFieldConfig,
    *,
    backend: QmBackend = None,
    force: bool = False,
    output_dir: Path = None,
) -> Tuple[PyFieldConfig, List[Tuple[str, bool, str]]]:
    """Populate every `from: dft` slot + relax `qm_relax` structures.

    Returns `(populated_cfg, journal)` where `journal` is a list of
    `(action_string, was_cache_hit, cache_key)` tuples that the CLI
    prints so the user can see what was run vs. cached.
    """
    if cfg.qm is None:
        raise ValueError("populate_qm called on a config without a `qm:` block")
    backend = backend or make_backend(cfg.qm)
    cache = QmCache(cfg.qm.cache_dir)
    fingerprint = backend.settings_fingerprint()
    journal: List[Tuple[str, bool, str]] = []

    populated_structures: Dict[str, StructureCfg] = dict(cfg.structures)

    # ------------------------------------------------------------------
    # 1. Relax structures flagged `qm_relax: true`. Their post-relax atoms
    #    end up in the populated YAML; the flag itself is dropped.
    # ------------------------------------------------------------------
    for name, struct in cfg.structures.items():
        if not struct.qm_relax:
            continue
        result, key, hit = cache.memoise_relax(
            struct, fingerprint, "relax",
            lambda s=struct: backend.relax(s),
            force=force,
        )
        populated_structures[name] = result.structure
        journal.append((f"relax {name}", hit, key))

    # ------------------------------------------------------------------
    # 2. Single-points for every simulation referenced by a `from: dft`
    #    energy_combination / forces target. Deduplicate across targets.
    # ------------------------------------------------------------------
    needed_sp: Dict[str, StructureCfg] = {}
    for tgt in cfg.targets:
        extras = getattr(tgt, "__pydantic_extra__", {}) or {}
        if tgt.kind == "energy_combination" and is_qm_placeholder(extras.get("target")):
            for sim_id in (extras.get("terms") or {}):
                struct_name = cfg.simulations[sim_id].structure
                needed_sp[sim_id] = populated_structures[struct_name]
        if tgt.kind == "forces" and is_qm_placeholder(extras.get("reference")):
            sim_id = extras.get("simulation")
            struct_name = cfg.simulations[sim_id].structure
            needed_sp[sim_id] = populated_structures[struct_name]

    sp_results: Dict[str, QmSinglePoint] = {}
    for sim_id, struct in needed_sp.items():
        result, key, hit = cache.memoise_single_point(
            struct, fingerprint, "single_point",
            lambda s=struct: backend.single_point(s),
            force=force,
        )
        sp_results[sim_id] = result
        journal.append((f"single_point {sim_id}", hit, key))

    # ------------------------------------------------------------------
    # 3. Walk targets, fill in the placeholders.
    # ------------------------------------------------------------------
    populated_targets = []
    out_dir = Path(output_dir) if output_dir else Path(cfg.output.dir).parent
    for tgt in cfg.targets:
        new_extras = dict(getattr(tgt, "__pydantic_extra__", {}) or {})

        # energy_combination — sum signed simulation energies
        if tgt.kind == "energy_combination" and is_qm_placeholder(new_extras.get("target")):
            terms = new_extras["terms"]
            value = sum(
                float(coeff) * sp_results[sim_id].energy_kcal_mol
                for sim_id, coeff in terms.items()
            )
            new_extras["target"] = round(value, 6)

        # forces — emit reference dict {atom_id: [fx, fy, fz]}
        if tgt.kind == "forces" and is_qm_placeholder(new_extras.get("reference")):
            sim_id = new_extras["simulation"]
            sp = sp_results[sim_id]
            if sp.forces_kcal_mol_per_A is None:
                raise RuntimeError(
                    f"forces: backend {backend.name!r} returned no forces for "
                    f"simulation {sim_id!r}"
                )
            new_extras["reference"] = {
                i + 1: [float(x) for x in row]
                for i, row in enumerate(sp.forces_kcal_mol_per_A)
            }

        # structural_match — point reference: at an xyz of the relaxed structure
        if tgt.kind == "structural_match":
            ref = new_extras.get("reference")
            if is_qm_placeholder(ref):
                output = ref.get("output") if isinstance(ref, dict) else None
                sim_id = new_extras["simulation"]
                struct_name = cfg.simulations[sim_id].structure
                struct = populated_structures[struct_name]
                if not struct.atoms:
                    raise RuntimeError(
                        f"structural_match: structure {struct_name!r} was not "
                        f"relaxed (set `qm_relax: true` on it)"
                    )
                xyz_path = Path(output) if output else (out_dir / f"{struct_name}_dft.xyz")
                _write_xyz(xyz_path, struct)
                new_extras["reference"] = str(xyz_path)

        # Rebuild the target with the populated extras.
        # pydantic's model_construct preserves the kind/weight + the new extras.
        from pyfield.config.schema import TargetCfg
        populated_targets.append(TargetCfg.model_validate({
            "kind": tgt.kind, "weight": tgt.weight, **new_extras
        }))

    populated = cfg.model_copy(update={
        "structures": populated_structures,
        "targets": populated_targets,
    })
    return populated, journal


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def cfg_to_yaml(cfg: PyFieldConfig) -> str:
    """Serialise a PyFieldConfig to YAML preserving key order.

    `exclude_defaults=True` keeps the output tight — `variables: {}`,
    `qm_relax: false`, `weight: 1.0` etc. are stripped on the way out
    and re-supplied by the schema on reload. Cross-tested by
    `test_populated_yaml_round_trips_through_loader`.
    """
    import yaml
    d = cfg.model_dump(exclude_none=True, exclude_defaults=True, mode="python")
    # Convert Path objects to strings for YAML cleanliness.
    def _coerce(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _coerce(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_coerce(x) for x in v]
        if isinstance(v, tuple):
            return [_coerce(x) for x in v]
        return v
    return yaml.safe_dump(_coerce(d), sort_keys=False, default_flow_style=False, width=10_000)
