"""Simulated-annealing driver for the new YAML pipeline.

Phase-2 swaps two hot-path costs the legacy SA paid every iteration:
- Per-step `deepcopy(ff.params)` → `ParameterSnapshot.copy()` (a length-N
  numpy `np.copy`, where N is the number of selected parameters).
- Per-simulation `lammps()` spawn + close → one long-lived
  `LammpsRunner` reused across all simulations and all SA steps; `clear`
  resets state between input files.

Same Metropolis / cooling semantics as the legacy `SA_REAX_FF.anneal`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pyfield.config.schema import PyFieldConfig
from pyfield.forcefield.reax import REAX_FF
from pyfield.forcefield.snapshot import ParameterSnapshot
from pyfield.io.lammps import LammpsRunner
from pyfield.objectives import build_objective
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.simulations.base import SimResult
from pyfield.simulations.runner import build_simulation


@dataclass
class SAResult:
    final_cost: float
    cost_trace: List[float] = field(default_factory=list)
    best_ffield_path: Optional[Path] = None


def _perturb(ff: REAX_FF) -> None:
    """One in-place SA move: kick every selected param by U(-1,1)·delta, clamp."""
    for sec, entry, item in ff.param_min_max_delta:
        bounds = ff.param_min_max_delta[(sec, entry, item)]
        while True:
            new_val = round(
                ff.params[sec][entry][item] + random.uniform(-1, 1) * bounds["delta"],
                4,
            )
            if bounds["min"] <= new_val <= bounds["max"]:
                ff.params[sec][entry][item] = new_val
                break


def run_sa(cfg: PyFieldConfig) -> SAResult:
    """Run SA on a validated config; returns final cost + per-iter trace."""
    if cfg.optimizer.seed is not None:
        random.seed(cfg.optimizer.seed)
        np.random.seed(cfg.optimizer.seed)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()

    ffield_iter_path = out_dir / "annealer_0.reax"
    ff.write_forcefield(str(ffield_iter_path))

    objectives: List[Objective] = [build_objective(t) for t in cfg.targets]
    needed_sims: List[str] = sorted({s for o in objectives for s in o.required_simulations()})

    with LammpsRunner() as runner:
        def evaluate() -> float:
            ff.write_forcefield(str(ffield_iter_path))
            sim_results: Dict[str, SimResult] = {}
            for sim_id in needed_sims:
                sim = build_simulation(sim_id, cfg.simulations[sim_id], cfg)
                sim_results[sim_id] = sim.run(
                    ffield_path=ffield_iter_path, work_dir=out_dir, runner=runner,
                )
            ctx = ObjectiveContext(sim_results=sim_results)
            return sum(o.residual(ctx) for o in objectives)

        cost_old = evaluate()
        trace = [cost_old]
        accepted_snap = ParameterSnapshot.capture(ff)
        best_cost = cost_old
        best_snap = accepted_snap.copy()

        T = cfg.optimizer.T
        while T > cfg.optimizer.T_min:
            for _ in range(cfg.optimizer.max_iter):
                _perturb(ff)
                cost_new = evaluate()
                # Boltzmann-factor accept; clip to avoid np.exp overflow on large drops.
                ap_arg = -(cost_new - cost_old) / max(T, 1e-30)
                ap = 1.0 if ap_arg >= 0 else np.exp(ap_arg)
                if ap > random.random():
                    cost_old = cost_new
                    accepted_snap = ParameterSnapshot.capture(ff)
                    if cost_new < best_cost:
                        best_cost = cost_new
                        best_snap = accepted_snap.copy()
                else:
                    accepted_snap.apply(ff)
                trace.append(cost_old)
            T = T * (1 - cfg.optimizer.alpha)

    # Write best forcefield.
    best_path = out_dir / "bestFF.reax"
    best_snap.apply(ff)
    ff.write_forcefield(str(best_path))
    return SAResult(final_cost=best_cost, cost_trace=trace, best_ffield_path=best_path)
