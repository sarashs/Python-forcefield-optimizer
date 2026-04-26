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


def cost_breakdown(
    cfg: PyFieldConfig,
    *,
    ffield_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> CostBreakdown:
    """Run every required simulation once and report per-sim energies +
    per-target residuals.

    `ffield_path` defaults to writing the FF currently parsed from
    `cfg.forcefield.path` — i.e. the *initial* FF. Pass an explicit path
    to evaluate against a post-SA best FF (`SAResult.best_ffield_path`).
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
    with LammpsRunner() as runner:
        for sim_id in needed_sims:
            sim = build_simulation(sim_id, cfg.simulations[sim_id], cfg)
            sim_results[sim_id] = sim.run(
                ffield_path=ffield_path, work_dir=work_dir, runner=runner,
            )
            sim_energies[sim_id] = sim_results[sim_id].energy

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
