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

import datetime as _dt
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from pyfield.config.schema import (
    AtomCfg,
    PyFieldConfig,
    StructureCfg,
    is_qm_placeholder,
)
from pyfield.qm.base import (
    QmBackend,
    QmRelaxResult,
    QmSinglePoint,
    make_backend,
    structure_code,
)
from pyfield.qm.cache import QmCache


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Long-run diagnostics
#
# QE relaxes on bulk PBC cells can take many hours. ASE blocks the Python
# kernel inside `atoms.get_potential_energy()` for the whole run, with no
# visibility into what's happening. If the kernel dies mid-run you lose
# the result AND the diagnostic trail — which has bitten us when 50h
# walks vanished on a stuck cell.
#
# So: every QE compute call (cache miss only — hits are silent and fast)
# goes through `_logged_compute`, which:
#   - Sets the backend's log_dir + run_label so `_relax_vc` / `single_point`
#     write `espresso.pwo` to a stable path the user can `tail -f`.
#   - Prints a stderr heartbeat with that path (Jupyter forwards stderr
#     line-by-line, so this shows up in the notebook before the cell
#     blocks on the long-running pw.x call).
#   - Appends START / END lines to a journal file at
#     `<log_dir>/journal.log`. Survives kernel death; line-buffered +
#     flushed on each write so partial entries are durable.
# ---------------------------------------------------------------------------


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _QeJournal:
    """Append-only journal of QE compute calls in this populator session.

    Survives kernel death (line-buffered + flush per write). One line
    per START, one per END. END includes wall time and final energy on
    success, or the exception type+message on failure — so a missing
    END pinpoints exactly which call was in flight when the kernel
    went down.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # `buffering=1` = line-buffered. Combined with explicit flush()
        # below this means each line is on disk before the next op runs.
        self._fh = self.path.open("a", buffering=1)

    def start(self, *, op: str, name: str, backend: str, key: str,
              workdir: Path, summary: str) -> None:
        self._fh.write(
            f"{_ts()} START {op:<18} {name:<22} [{backend}] "
            f"key={key[:16]} {summary} workdir={workdir}\n"
        )
        self._fh.flush()

    def end(self, *, op: str, name: str, dt_s: float,
            energy: Optional[float], status: str) -> None:
        en = f"{energy:.6f}" if energy is not None else "—"
        self._fh.write(
            f"{_ts()} END   {op:<18} {name:<22} dt={dt_s:8.1f}s "
            f"energy={en} status={status}\n"
        )
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _qm_log_dir(cfg: PyFieldConfig) -> Path:
    """Where QE per-call workdirs and the journal live.

    Sibling of `cfg.output.dir`: for `output.dir = studies/gst_drift/runs/sa`
    this returns `studies/gst_drift/runs/qe_logs/`. Putting it under
    `runs/` keeps logs/cache/optimiser-outputs co-located, and using
    the *sibling* of `output.dir` (rather than `output.dir / qe_logs`)
    means several optimiser runs (sa, cma, ga) sharing one project
    can also share one QE log directory.
    """
    return Path(cfg.output.dir).parent / "qe_logs"


def _struct_summary(structure: StructureCfg) -> str:
    n = len(structure.atoms) if structure.atoms is not None else "?"
    if structure.box is not None:
        a, b, c = structure.box
        box = f"box=({a:.3f},{b:.3f},{c:.3f})"
    else:
        box = "box=—"
    return f"atoms={n} {box}"


def _logged_compute(
    compute: Callable,
    *,
    op: str,
    name: str,
    key: str,
    journal: Optional[_QeJournal],
    backend: QmBackend,
    log_dir: Path,
    structure: StructureCfg,
) -> Callable:
    """Wrap a backend compute lambda with workdir setup + heartbeat + journal.

    Only meaningful for the QE backend (`backend.log_dir`/`run_label` are
    QEBackend-specific attributes). PySCF and other backends are
    transparently passed through by the callers — this helper isn't
    invoked for them.
    """
    def wrapped():
        # Timestamp suffix means force-rerun on the same (structure, op)
        # doesn't clobber the previous workdir — each attempt gets its
        # own directory and the prior espresso.pwo stays around for
        # comparison.
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = f"{op}_{name}_{ts}"
        workdir = log_dir / run_label
        backend.log_dir = log_dir
        backend.run_label = run_label
        summary = _struct_summary(structure)
        # stderr is line-buffered + forwarded by Jupyter kernels per
        # stream message, so this lands in the notebook cell output
        # *before* the cell starts blocking on pw.x.
        print(
            f"[{_ts()}] QE {op} {name}  →  tail -f {workdir/'espresso.pwo'}",
            file=sys.stderr, flush=True,
        )
        if journal is not None:
            journal.start(
                op=op, name=name, backend=backend.name, key=key,
                workdir=workdir, summary=summary,
            )
        t0 = time.time()
        try:
            result = compute()
        except BaseException as e:
            dt_s = time.time() - t0
            if journal is not None:
                journal.end(
                    op=op, name=name, dt_s=dt_s, energy=None,
                    status=f"FAIL: {type(e).__name__}: {e}",
                )
            raise
        dt_s = time.time() - t0
        energy = getattr(result, "energy_kcal_mol", None)
        if journal is not None:
            journal.end(
                op=op, name=name, dt_s=dt_s, energy=energy, status="ok",
            )
        return result
    return wrapped


def _structure_for_simulation(cfg: PyFieldConfig, sim_id: str) -> StructureCfg:
    sim = cfg.simulations[sim_id]
    return cfg.structures[sim.structure]


def _write_xyz(path: Path, structure: StructureCfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(structure.atoms)), "pyfield qm-prep relaxed geometry"]
    for a in structure.atoms:
        lines.append(f"{a.element} {a.x:.6f} {a.y:.6f} {a.z:.6f}")
    path.write_text("\n".join(lines) + "\n")


def _structure_constraint(structure: StructureCfg) -> Optional[Dict]:
    """Pull the optional per-structure `constraint:` from `__pydantic_extra__`.

    `make-scan` attaches one to every relaxed_constrained scan point. The
    populator hands it to the backend so QM holds the same coordinate
    fixed that the FF-side `fix restrain` does.
    """
    extras = getattr(structure, "__pydantic_extra__", {}) or {}
    return extras.get("constraint")


def _relax_op(structure: StructureCfg, constraint: Optional[Dict]) -> str:
    """Cache-key tag distinguishing the three relax modes.

    Different ops → different cache entries, so flipping `qm_relax_cell`
    on doesn't accidentally hit a stale atoms-only result keyed on the
    same input box.
    """
    if constraint is not None:
        return "relax_constrained"
    if getattr(structure, "qm_relax_cell", False):
        return "vc-relax"
    return "relax"


class _BackendCache:
    """Lazy per-code backend factory.

    Lets a single training set mix backends — typically PySCF for
    cluster references and QE for PBC supercells. Each unique code
    builds its backend once on first use; subsequent structures with
    the same code reuse the cached instance.
    """

    def __init__(self, qm_cfg, default: Optional[QmBackend] = None):
        self._qm = qm_cfg
        self._cache: Dict[str, QmBackend] = {}
        if default is not None:
            self._cache[qm_cfg.code] = default

    def for_structure(self, structure: StructureCfg) -> QmBackend:
        code = structure_code(structure, fallback=self._qm.code)
        if code not in self._cache:
            self._cache[code] = make_backend(self._qm, code_override=code)
        return self._cache[code]


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

    Each structure may carry a `constraint:` field (set by `make-scan`
    on relaxed_constrained scan points); if so, the relax holds that
    coordinate fixed.

    Targets / placeholders are left untouched — this is the input to
    `make-scan` (which needs equilibrium geometries) before `qm-prep`
    fills in target values.
    """
    if cfg.qm is None:
        raise ValueError("relax_structures called on a config without a `qm:` block")
    backends = _BackendCache(cfg.qm, default=backend)
    cache = QmCache(cfg.qm.cache_dir)
    journal: List[Tuple[str, bool, str]] = []

    log_dir = _qm_log_dir(cfg)
    qe_journal: Optional[_QeJournal] = None

    populated_structures: Dict[str, StructureCfg] = dict(cfg.structures)
    requested = set(only) if only else None
    if requested:
        unknown = requested - set(cfg.structures)
        if unknown:
            raise ValueError(f"relax_structures: unknown structures {sorted(unknown)!r}")

    try:
        for name, struct in cfg.structures.items():
            if requested is None and not struct.qm_relax:
                continue
            if requested is not None and name not in requested:
                continue
            be = backends.for_structure(struct)
            constraint = _structure_constraint(struct)
            op = _relax_op(struct, constraint)
            raw = lambda s=struct, c=constraint, b=be: b.relax(s, constraint=c)
            if be.name == "qe":
                if qe_journal is None:
                    qe_journal = _QeJournal(log_dir / "journal.log")
                from pyfield.qm.cache import _key as _cache_key
                pre_key = _cache_key(struct, be.settings_fingerprint(), op,
                                     constraint=constraint)
                compute_fn = _logged_compute(
                    raw, op=op, name=name, key=pre_key, journal=qe_journal,
                    backend=be, log_dir=log_dir, structure=struct,
                )
            else:
                compute_fn = raw
            result, key, hit = cache.memoise_relax(
                struct, be.settings_fingerprint(), op,
                compute_fn,
                force=force,
                constraint=constraint,
            )
            populated_structures[name] = result.structure
            journal.append((f"{op} {name} [{be.name}]", hit, key))
    finally:
        if qe_journal is not None:
            qe_journal.close()

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
    backends = _BackendCache(cfg.qm, default=backend)
    cache = QmCache(cfg.qm.cache_dir)
    journal: List[Tuple[str, bool, str]] = []

    log_dir = _qm_log_dir(cfg)
    qe_journal: Optional[_QeJournal] = None

    def _ensure_journal() -> _QeJournal:
        nonlocal qe_journal
        if qe_journal is None:
            qe_journal = _QeJournal(log_dir / "journal.log")
        return qe_journal

    populated_structures: Dict[str, StructureCfg] = dict(cfg.structures)
    try:
        # ------------------------------------------------------------------
        # 1. Relax structures flagged `qm_relax: true`. Their post-relax atoms
        #    end up in the populated YAML; the flag itself is dropped.
        #    A `constraint:` field on the structure (set by make-scan for
        #    relaxed_constrained scan points) is forwarded to the backend
        #    so QM holds the same coordinate the FF-side fix restrain does.
        # ------------------------------------------------------------------
        relax_energies: Dict[str, float] = {}        # name → kcal/mol of the relaxed structure
        for name, struct in cfg.structures.items():
            if not struct.qm_relax:
                continue
            be = backends.for_structure(struct)
            constraint = _structure_constraint(struct)
            op = _relax_op(struct, constraint)
            raw = lambda s=struct, c=constraint, b=be: b.relax(s, constraint=c)
            if be.name == "qe":
                from pyfield.qm.cache import _key as _cache_key
                pre_key = _cache_key(struct, be.settings_fingerprint(), op,
                                     constraint=constraint)
                compute_fn = _logged_compute(
                    raw, op=op, name=name, key=pre_key,
                    journal=_ensure_journal(),
                    backend=be, log_dir=log_dir, structure=struct,
                )
            else:
                compute_fn = raw
            result, key, hit = cache.memoise_relax(
                struct, be.settings_fingerprint(), op,
                compute_fn,
                force=force,
                constraint=constraint,
            )
            populated_structures[name] = result.structure
            relax_energies[name] = result.energy_kcal_mol
            journal.append((f"{op} {name} [{be.name}]", hit, key))

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
            struct_name = cfg.simulations[sim_id].structure
            # If we just relaxed this structure (constrained or not), the
            # relax already produced the energy at that geometry — reuse it
            # instead of paying for a redundant single_point.
            if struct_name in relax_energies:
                sp_results[sim_id] = QmSinglePoint(
                    energy_kcal_mol=relax_energies[struct_name]
                )
                journal.append((f"reuse_relax_energy {sim_id}", True, ""))
                continue
            be = backends.for_structure(struct)
            raw_sp = lambda s=struct, b=be: b.single_point(s)
            if be.name == "qe":
                from pyfield.qm.cache import _key as _cache_key
                pre_key = _cache_key(struct, be.settings_fingerprint(), "single_point")
                compute_sp = _logged_compute(
                    raw_sp, op="single_point", name=sim_id, key=pre_key,
                    journal=_ensure_journal(),
                    backend=be, log_dir=log_dir, structure=struct,
                )
            else:
                compute_sp = raw_sp
            result, key, hit = cache.memoise_single_point(
                struct, be.settings_fingerprint(), "single_point",
                compute_sp,
                force=force,
            )
            sp_results[sim_id] = result
            journal.append((f"single_point {sim_id} [{be.name}]", hit, key))

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
    finally:
        if qe_journal is not None:
            qe_journal.close()

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
