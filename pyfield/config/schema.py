"""Pydantic schema for the PyField YAML config.

The top-level object is `PyFieldConfig`. Nested models validate the
sections described in DEV.md §7. SimulationCfg is a discriminated union:
either built-in (`type:`) or user-supplied (`template:`), never both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# forcefield
# ---------------------------------------------------------------------------

class ForceFieldCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    type: Literal["reaxff"] = "reaxff"
    params: Path  # the ReaxFF/LAMMPS-style param-selection file (untouched format)


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------

class AtomCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element: str
    x: float
    y: float
    z: float
    charge: float = 0.0


class StructureCfg(BaseModel):
    """One simulation cell. Either inline `atoms` or external `path` (xyz).

    `extra="allow"` so `pyfield make-scan` can attach a `constraint:`
    dict on relaxed_constrained scan points (consumed by `qm-prep` to
    drive a constrained QM relax). Any extra you set will round-trip
    through the YAML serialiser.
    """
    model_config = ConfigDict(extra="allow")
    box: Tuple[float, float, float]
    atoms: Optional[List[AtomCfg]] = None
    path: Optional[Path] = None
    # When True, `pyfield qm-prep` first relaxes this structure with the
    # configured QM backend and writes the relaxed coordinates back into
    # `atoms:` of the populated YAML (the flag itself is removed).
    qm_relax: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "StructureCfg":
        if (self.atoms is None) == (self.path is None):
            raise ValueError("StructureCfg requires exactly one of `atoms` or `path`.")
        return self


# ---------------------------------------------------------------------------
# simulations
# ---------------------------------------------------------------------------

# Built-in simulation types we know how to render. Phase 1 ships only `minimize`;
# `nvt`, `npt`, `temperature_ramp` are reserved names landing in Phase 3.
BuiltinSimType = Literal["minimize", "nvt", "npt", "temperature_ramp", "single_point"]


class SimulationCfg(BaseModel):
    """Either inline (`type: minimize | nvt | …`) or escape-hatch (`template:`).

    The variable-leakage contract from DEV.md §7 applies to the template path:
    user templates must reference {{ FFIELD_PATH }} exactly once and any
    optimizer-tuned value must come through `variables:`.
    """
    model_config = ConfigDict(extra="allow")  # type-specific kwargs land here

    structure: str
    type: Optional[BuiltinSimType] = None
    template: Optional[Path] = None
    variables: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> "SimulationCfg":
        if (self.type is None) == (self.template is None):
            raise ValueError(
                "SimulationCfg requires exactly one of `type:` (built-in) "
                "or `template:` (user-supplied Jinja file)."
            )
        return self


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

class TargetCfg(BaseModel):
    """Base target. Concrete objectives (energy_combination, charges, …) read
    additional fields off `extras` via the registry; we don't enumerate every
    `kind` here so adding an objective doesn't require schema edits."""
    model_config = ConfigDict(extra="allow")
    kind: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------

class OptimizerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["sa", "ga", "sa+ga", "cma"] = "sa"
    # SA
    T: float = 1.0
    T_min: float = 1e-5
    alpha: float = 0.1
    max_iter: int = 50
    number_of_points: int = 1
    parallel: bool = False              # if true, evaluate candidates concurrently
    processors: int = 0                 # 0 = cpu_count() when parallel; else N workers
    seed: Optional[int] = None
    repelling_weight: float = 0.0
    min_style: str = "cg"
    record_costs: bool = True
    show_progress: bool = True          # tqdm progress bar (auto-suppressed in non-tty)
    # GA
    generations: int = 20
    population_size: int = 16
    mutation_rate: float = 0.2          # per-gene probability of being kicked
    mutation_sigma: float = 0.25        # Gaussian σ as a fraction of (max-min)
    crossover_rate: float = 0.7         # probability the children are crossed (vs cloned)
    tournament_size: int = 3            # selection tournament size
    elitism: int = 1                    # number of best-so-far copied unchanged
    # Hybrid: when method="sa+ga", run this many SA refinement steps on each
    # newly-bred child before the next generation evaluates it.
    sa_refine_steps: int = 5
    # CMA-ES (when method="cma")
    cma_sigma0: float = 0.3       # initial step size as a fraction of (max-min)
    cma_popsize: int = 0          # 0 → cma's default 4 + floor(3*ln(N))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

class OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: Path = Path("runs/")
    record_costs: bool = True


# ---------------------------------------------------------------------------
# qm — optional, only required if any target/reference is `from: dft`
# or any structure has `qm_relax: true`. Free / open-source codes only —
# Gaussian/VASP/Molpro require a licence and are intentionally not bundled.
# ---------------------------------------------------------------------------

class QmCfg(BaseModel):
    """Settings for `pyfield qm-prep`. Per-code knobs land in `extras`."""
    model_config = ConfigDict(extra="allow")
    code: Literal["pyscf", "xtb", "qe", "gpaw", "cp2k", "nwchem", "psi4", "orca"]
    functional: str = "lda"
    basis: str = "sto-3g"            # Gaussian-basis codes (pyscf, psi4, orca)
    cache_dir: Path = Path("qm_cache/")


# ---------------------------------------------------------------------------
# scans — consumed by `pyfield make-scan`, stripped from its output YAML.
# Each entry expands into N structures + N single_point sims + N
# energy_combination targets (Scan_i − reference). See pyfield.scans.
# ---------------------------------------------------------------------------

ScanType = Literal[
    "bond_stretch",
    "angle_bend",
    "dihedral",
    "atom_displacement",
    "dimer_separation",
    "isotropic_scale",
]


RelaxMethod = Literal["rigid", "relaxed_constrained"]


class ScanCfg(BaseModel):
    """One perturbation sweep over a reference structure.

    Per-type fields (`atoms`, `atom`, `direction`, `fragments`) live in
    `extras` so adding a new scan kind is one new file under
    `pyfield/scans/` without schema edits.

    `legs` and `anchors`/`fragments` opt the scan into rigid-group
    perturbation: when bending a Si–O–Si angle, the user can declare
    that the substituents on each Si rotate *with* their Si rather
    than staying fixed (the natural starting geometry for a
    constrained relax). See `pyfield/scans/transforms.py` for the
    per-kind semantics.

    `relax_method`:
    - `rigid` (default) — single-point at the perturbed geometry on
      both QM and FF sides; comparable only when the system has no
      internal degrees of freedom (diatomics, fixed cells).
    - `relaxed_constrained` — QM constrained geom-opt (geomeTRIC `$set`)
      + FF constrained minimization (`fix restrain` at `restraint_k`).
      The reaction coordinate (distance / angle / dihedral) is held
      fixed on both sides; everything else relaxes.
    """
    model_config = ConfigDict(extra="allow")
    type: ScanType
    reference: str                                  # name of the structure to perturb
    name_prefix: str                                # generated structures: {prefix}_0, _1, …
    values: Optional[List[float]] = None            # explicit list of scan points
    range: Optional[Tuple[float, float, int]] = None  # (start, stop, num) → linspace
    target_weight: float = 1.0                      # weight applied to each generated target

    relax_method: RelaxMethod = "rigid"
    restraint_k: float = 2000.0                     # kcal/mol/Å² for FF-side fix restrain
    legs: Optional[Dict[str, List[int]]] = None     # role ("i"/"j"/"k"/"l") → atoms
    anchors: Optional[List[int]] = None             # dimer_separation: [a1, a2]
    fragments: Optional[List[List[int]]] = None     # dimer_separation: atoms per fragment

    @model_validator(mode="after")
    def _exactly_one_grid(self) -> "ScanCfg":
        if (self.values is None) == (self.range is None):
            raise ValueError(
                f"scan {self.name_prefix!r}: provide exactly one of "
                "`values:` (explicit list) or `range:` (start, stop, num)."
            )
        if self.range is not None and self.range[2] < 2:
            raise ValueError(
                f"scan {self.name_prefix!r}: range num must be ≥ 2, got {self.range[2]}"
            )
        return self

    @model_validator(mode="after")
    def _legs_and_anchors_well_formed(self) -> "ScanCfg":
        """Per-type structural validation for `legs` / `anchors` / `fragments`.

        Rejects overlapping legs, atoms in forbidden positions (vertex /
        dihedral middle), and missing required fields for
        `dimer_separation`. The transform code can then assume the
        spec is consistent.
        """
        legs = self.legs or {}
        extras = self.__pydantic_extra__ or {}
        atoms = extras.get("atoms")

        if self.type == "bond_stretch":
            allowed = {"i", "j"}
            unknown = set(legs) - allowed
            if unknown:
                raise ValueError(
                    f"scan {self.name_prefix!r}: bond_stretch.legs keys must be "
                    f"a subset of {sorted(allowed)}, got extra {sorted(unknown)}"
                )
            if atoms is not None and len(atoms) >= 2:
                _reject_overlap_for_pair(self.name_prefix, legs, atoms,
                                         pair_keys=("i", "j"))

        elif self.type == "angle_bend":
            allowed = {"i", "k"}
            unknown = set(legs) - allowed
            if unknown:
                raise ValueError(
                    f"scan {self.name_prefix!r}: angle_bend.legs keys must be "
                    f"a subset of {sorted(allowed)}, got extra {sorted(unknown)}"
                )
            if atoms is not None and len(atoms) == 3:
                vertex = atoms[1]
                _reject_atom_in_legs(self.name_prefix, legs, vertex,
                                     "vertex (atoms[1])")
                _reject_overlap_for_pair(self.name_prefix, legs, atoms,
                                         pair_keys=("i", "k"))

        elif self.type == "dihedral":
            allowed = {"i", "l"}
            unknown = set(legs) - allowed
            if unknown:
                raise ValueError(
                    f"scan {self.name_prefix!r}: dihedral.legs keys must be "
                    f"a subset of {sorted(allowed)}, got extra {sorted(unknown)}"
                )
            if atoms is not None and len(atoms) == 4:
                for forbidden, label in ((atoms[1], "atoms[1] (j)"),
                                         (atoms[2], "atoms[2] (k)")):
                    _reject_atom_in_legs(self.name_prefix, legs, forbidden, label)

        elif self.type == "dimer_separation":
            if not self.anchors or len(self.anchors) != 2:
                raise ValueError(
                    f"scan {self.name_prefix!r}: dimer_separation requires "
                    "`anchors: [atom_i, atom_j]`"
                )
            if not self.fragments or len(self.fragments) != 2:
                raise ValueError(
                    f"scan {self.name_prefix!r}: dimer_separation requires "
                    "`fragments: [[…], […]]` (one list per anchor)"
                )
            a1, a2 = self.anchors
            if a1 == a2:
                raise ValueError(
                    f"scan {self.name_prefix!r}: dimer_separation anchors "
                    f"must be distinct, got {self.anchors}"
                )
            f1, f2 = (set(self.fragments[0]) | {a1}), (set(self.fragments[1]) | {a2})
            overlap = f1 & f2
            if overlap:
                raise ValueError(
                    f"scan {self.name_prefix!r}: dimer_separation fragments "
                    f"overlap on atoms {sorted(overlap)} — each atom must "
                    "belong to exactly one fragment."
                )
            if a2 in f1 - {a2}:
                raise ValueError(
                    f"scan {self.name_prefix!r}: anchor {a2} appears in "
                    "fragments[0]"
                )
            if a1 in f2 - {a1}:
                raise ValueError(
                    f"scan {self.name_prefix!r}: anchor {a1} appears in "
                    "fragments[1]"
                )

        if self.relax_method == "relaxed_constrained" and self.type == "isotropic_scale":
            raise ValueError(
                f"scan {self.name_prefix!r}: relax_method=relaxed_constrained "
                "isn't meaningful for isotropic_scale (no internal DOFs to "
                "relax — the scale factor *is* the constraint)."
            )
        return self


def _reject_atom_in_legs(scan_name: str, legs, atom: int, label: str) -> None:
    for role, members in legs.items():
        if atom in members:
            raise ValueError(
                f"scan {scan_name!r}: atom {atom} ({label}) cannot appear in "
                f"legs.{role!r}; the vertex / middle atoms of the constraint "
                "must stay fixed."
            )


def _reject_overlap_for_pair(scan_name: str, legs, atoms, pair_keys) -> None:
    """For bond_stretch / angle_bend: each leg must include its anchor and
    the two legs must be disjoint."""
    a, b = pair_keys
    set_a = set(legs.get(a) or []) | ({atoms[0]} if a == "i" and len(atoms) >= 1 else set())
    if b == "j":
        set_b = set(legs.get(b) or []) | ({atoms[1]} if len(atoms) >= 2 else set())
    elif b == "k":
        set_b = set(legs.get(b) or []) | ({atoms[2]} if len(atoms) >= 3 else set())
    else:
        set_b = set(legs.get(b) or [])
    overlap = set_a & set_b
    if overlap:
        raise ValueError(
            f"scan {scan_name!r}: legs.{a!r} and legs.{b!r} overlap on atoms "
            f"{sorted(overlap)} — each atom must move with at most one anchor."
        )


# ---------------------------------------------------------------------------
# placeholder predicate — `target: { from: dft }` style mappings
# ---------------------------------------------------------------------------

def is_qm_placeholder(value) -> bool:
    """True if `value` is a `{from: dft}`-style placeholder slot."""
    return isinstance(value, dict) and value.get("from") == "dft"


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

class PyFieldConfig(BaseModel):
    """The complete validated PyField config."""
    model_config = ConfigDict(extra="forbid")
    forcefield: ForceFieldCfg
    structures: Dict[str, StructureCfg]
    simulations: Dict[str, SimulationCfg]
    targets: List[TargetCfg]
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)
    qm: Optional[QmCfg] = None
    # `pyfield make-scan` consumes this and removes it from its output YAML.
    # Other commands (`run`, `qm-prep`) tolerate it being present (a no-op).
    scans: Optional[List[ScanCfg]] = None

    @model_validator(mode="after")
    def _qm_required_when_placeholders_exist(self) -> "PyFieldConfig":
        if self.qm is not None:
            return self
        for name, s in self.structures.items():
            if s.qm_relax:
                raise ValueError(
                    f"structure {name!r} has `qm_relax: true` but no top-level "
                    f"`qm:` block is configured"
                )
        for i, t in enumerate(self.targets):
            extras = getattr(t, "__pydantic_extra__", {}) or {}
            for slot in ("target", "reference"):
                if is_qm_placeholder(extras.get(slot)):
                    raise ValueError(
                        f"target #{i} (kind={t.kind!r}) has {slot}: "
                        f"{{from: dft}} but no top-level `qm:` block is configured"
                    )
        return self

    @model_validator(mode="after")
    def _cross_refs_resolve(self) -> "PyFieldConfig":
        # Every simulation references a known structure.
        for sim_id, sim in self.simulations.items():
            if sim.structure not in self.structures:
                raise ValueError(
                    f"simulation {sim_id!r} references unknown structure "
                    f"{sim.structure!r}"
                )
        # Best-effort: if a target carries a `simulation` field, it must point
        # at a known sim. We don't enforce per-kind shape here; the registry
        # validates kind-specific fields when it instantiates the objective.
        for i, tgt in enumerate(self.targets):
            extras = getattr(tgt, "__pydantic_extra__", {}) or {}
            sim = extras.get("simulation")
            if isinstance(sim, str) and sim not in self.simulations:
                raise ValueError(
                    f"target #{i} (kind={tgt.kind!r}) references unknown "
                    f"simulation {sim!r}"
                )
            sims = extras.get("simulations")
            if isinstance(sims, list):
                for s in sims:
                    if isinstance(s, str) and s not in self.simulations:
                        raise ValueError(
                            f"target #{i} (kind={tgt.kind!r}) references "
                            f"unknown simulation {s!r} in `simulations`"
                        )
            terms = extras.get("terms")
            if isinstance(terms, dict):
                for sim_id in terms:
                    if sim_id not in self.simulations:
                        raise ValueError(
                            f"target #{i} (kind={tgt.kind!r}) references "
                            f"unknown simulation {sim_id!r} in `terms`"
                        )
        # Every scan references a known structure.
        for i, scan in enumerate(self.scans or []):
            if scan.reference not in self.structures:
                raise ValueError(
                    f"scan #{i} ({scan.name_prefix!r}) references unknown "
                    f"structure {scan.reference!r}"
                )
        return self
