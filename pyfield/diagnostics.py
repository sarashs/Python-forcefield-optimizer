"""One-shot diagnostics: run every required simulation once and report
per-sim LAMMPS energies + per-target (computed value, target, residual).

Pure inspection — no SA, no parameter perturbation. Useful for:

- Sanity-checking that the QM targets and the ReaxFF energies are in
  the same ballpark before paying for an optimisation.
- Comparing the *initial* FF against the *post-optimisation* best FF.
- Diagnosing which target dominates the cost (the one with the largest
  residual is the one to scrutinise).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pyfield.config.schema import PyFieldConfig
from pyfield.forcefield.reax import REAX_FF
from pyfield.io.lammps import LammpsRunner
from pyfield.objectives import build_objective
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.simulations.base import SimResult
from pyfield.simulations.runner import build_simulation


@dataclass
class TargetReport:
    kind: str
    weight: float
    description: str            # e.g. "+1*Cl2_414_min -1*Cl2_Opt_min  vs target=354.33"
    value: float                # what FF computed (kcal/mol, or whatever the objective returns)
    target: Optional[float]     # the target the objective compares against
    residual: float             # weight * (value - target)^2


@dataclass
class CostBreakdown:
    sim_energies: Dict[str, float]               # sim_id → LAMMPS energy (kcal/mol)
    target_reports: List[TargetReport]
    total_cost: float

    def print_table(self) -> None:
        print("simulation energies (LAMMPS):")
        for sim_id, e in self.sim_energies.items():
            print(f"  {sim_id:<25}  E = {e:>12.4f} kcal/mol")
        print()
        print("targets:")
        for r in self.target_reports:
            tgt = f"{r.target:>10.4f}" if r.target is not None else "    n/a   "
            print(
                f"  {r.kind:<22} w={r.weight:<5g}  "
                f"FF={r.value:>10.4f}  target={tgt}  "
                f"residual={r.residual:>12.4f}"
            )
            if r.description:
                print(f"      ({r.description})")
        print()
        print(f"total cost = {self.total_cost:.4f}  (sum of residuals)")


def _describe_target(tgt) -> str:
    extras = getattr(tgt, "__pydantic_extra__", {}) or {}
    if tgt.kind == "energy_combination":
        terms = extras.get("terms") or {}
        terms_str = " ".join(
            f"{'+' if c >= 0 else ''}{c}*{s}" for s, c in terms.items()
        )
        target = extras.get("target")
        return f"{terms_str}  vs target={target}"
    if tgt.kind == "charges":
        return f"sim={extras.get('simulation')!r}  atoms={extras.get('atoms')}"
    if tgt.kind == "structural_match":
        return f"sim={extras.get('simulation')!r}  ref={extras.get('reference')}"
    sim = extras.get("simulation") or extras.get("simulations")
    return f"sim={sim!r}"


def _make_progress(total: int, desc: str):
    """tqdm.auto picks the right backend (notebook widget / terminal /
    plain). Falls back to a no-op shim if tqdm isn't installed."""
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
    except ImportError:
        class _Null:
            def update(self, n=1): pass
            def set_postfix_str(self, s, refresh=True): pass
            def write(self, s): print(s, flush=True)
            def close(self): pass
        return _Null()


def cost_breakdown(
    cfg: PyFieldConfig,
    *,
    ffield_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    verbose: bool = True,
    slow_threshold_s: float = 10.0,
) -> CostBreakdown:
    """Run every required simulation once and report per-sim energies +
    per-target residuals.

    `ffield_path` defaults to writing the FF currently parsed from
    `cfg.forcefield.path` — i.e. the *initial* FF. Pass an explicit path
    to evaluate against a post-SA best FF (`SAResult.best_ffield_path`).

    Progress is reported via `tqdm.auto` (notebook widget in Jupyter,
    terminal bar elsewhere). Each sim's wall-clock is tracked; sims
    exceeding `slow_threshold_s` get an inline log line so a hung
    simulation is visible without `py-spy`. Pass `verbose=False` to
    silence everything except the final table.
    """
    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()
    work_dir = Path(work_dir or cfg.output.dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if ffield_path is None:
        ffield_path = work_dir / "_diagnostic_ffield.reax"
        ff.write_forcefield(str(ffield_path))
    ffield_path = Path(ffield_path)

    objectives: List[Objective] = [build_objective(t) for t in cfg.targets]
    needed_sims = sorted({s for o in objectives for s in o.required_simulations()})

    sim_results: Dict[str, SimResult] = {}
    sim_energies: Dict[str, float] = {}
    bar = _make_progress(len(needed_sims), desc="cost_breakdown") if verbose else _make_progress(0, desc="")
    if verbose:
        print(f"cost_breakdown: {len(needed_sims)} simulations to run", flush=True)
    with LammpsRunner() as runner:
        for i, sim_id in enumerate(needed_sims):
            if verbose:
                # Print BEFORE starting so a sim that hangs is visible.
                # Without this, a hung sim is invisible — `bar.write()`
                # below only fires *after* `sim.run()` returns.
                print(f"  [{i+1}/{len(needed_sims)}] starting {sim_id} ...", flush=True)
                sys.stderr.flush()
                bar.set_postfix_str(sim_id, refresh=True)
            t0 = time.perf_counter()
            sim = build_simulation(sim_id, cfg.simulations[sim_id], cfg)
            sim_results[sim_id] = sim.run(
                ffield_path=ffield_path, work_dir=work_dir, runner=runner,
            )
            sim_energies[sim_id] = sim_results[sim_id].energy
            dt = time.perf_counter() - t0
            if verbose:
                marker = " [slow]" if dt >= slow_threshold_s else ""
                bar.write(f"  [{i+1}/{len(needed_sims)}]{marker} {sim_id}: "
                          f"{dt:.2f}s  E={sim_energies[sim_id]:.3f}")
                bar.update(1)
                sys.stdout.flush()
                sys.stderr.flush()
    if verbose:
        bar.close()

    ctx = ObjectiveContext(sim_results=sim_results)
    reports: List[TargetReport] = []
    for cfg_tgt, obj in zip(cfg.targets, objectives):
        try:
            value = obj.compute(ctx)
        except Exception as exc:                    # objective couldn't compute
            value = float("nan")
        residual = obj.residual(ctx)
        target_value = getattr(obj, "_target", None)
        try:
            target_value = float(target_value) if target_value is not None else None
        except (TypeError, ValueError):
            target_value = None
        reports.append(TargetReport(
            kind=cfg_tgt.kind,
            weight=cfg_tgt.weight,
            description=_describe_target(cfg_tgt),
            value=float(value) if value == value else value,    # NaN-safe
            target=target_value,
            residual=float(residual),
        ))
    total = sum(r.residual for r in reports)
    return CostBreakdown(
        sim_energies=sim_energies,
        target_reports=reports,
        total_cost=total,
    )
