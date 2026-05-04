"""Simulated-annealing driver for the new YAML pipeline.

Phase-2 swaps two hot-path costs the legacy SA paid every iteration:
- Per-step `deepcopy(ff.params)` → `ParameterSnapshot.copy()` (a length-N
  numpy `np.copy`, where N is the number of selected parameters).
- Per-simulation `lammps()` spawn + close → one long-lived
  `LammpsRunner` reused across all simulations and all SA steps; `clear`
  resets state between input files.

`number_of_points` is the number of independent SA walkers. The master
proposes one perturbation per walker each inner step, evaluates all of
them in a single batch (parallel when `optimizer.parallel: true`), then
applies Metropolis per walker. Best-across-walkers is tracked globally.

Reproducibility: with a seeded `optimizer.seed` and any number of walkers
or worker processes, the run is bit-identical — every random number is
drawn on the master before submitting the batch.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from pyfield.config.schema import PyFieldConfig
from pyfield.forcefield.reax import REAX_FF
from pyfield.forcefield.snapshot import ParameterSnapshot
from pyfield.optimizers.parallel import BatchEvaluator, resolve_n_workers


@dataclass
class SAResult:
    final_cost: float
    cost_trace: List[float] = field(default_factory=list)
    best_ffield_path: Optional[Path] = None


def _propose(
    snap: ParameterSnapshot,
    ff: REAX_FF,
    rng: random.Random,
) -> ParameterSnapshot:
    """SA move: kick every selected param by U(-1,1)·delta, clamp to bounds."""
    new_values = snap.values.copy()
    for i, key in enumerate(snap.keys):
        b = ff.param_min_max_delta[key]
        while True:
            v = round(new_values[i] + rng.uniform(-1, 1) * b["delta"], 4)
            if b["min"] <= v <= b["max"]:
                new_values[i] = v
                break
    return ParameterSnapshot(keys=snap.keys, values=new_values)


def _make_progress(total: int, *, enabled: bool, desc: str):
    """Return either a `tqdm.auto` bar or a no-op shim.

    `tqdm.auto` picks the right renderer for the medium (notebook
    widget in Jupyter, text bar in a terminal, plain text otherwise),
    so we don't gate on `sys.stderr.isatty()` — that returned False
    inside Jupyter and silently swallowed the bar.
    """
    if not enabled:
        class _Null:
            def update(self, n=1): pass
            def set_postfix(self, **kw): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Null()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        class _Null:
            def update(self, n=1): pass
            def set_postfix(self, **kw): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Null()
    return tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True)


def run_sa(cfg: PyFieldConfig) -> SAResult:
    """Run SA on a validated config; returns best cost + per-iter trace."""
    o = cfg.optimizer
    rng = random.Random(o.seed)
    if o.seed is not None:
        np.random.seed(o.seed)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()
    n_workers = resolve_n_workers(o.parallel, o.processors)
    n_walkers = max(1, o.number_of_points)

    # Total number of inner iterations the SA loop will run, used to size
    # the progress bar accurately. The closed-form ceil(log/log) version
    # is off-by-one in floating point (e.g. T=1, T_min=0.01, alpha=0.99
    # mathematically gives one cooling step but FP drift on the
    # multiplicative update lands one extra iteration). Simulating the
    # loop is O(n_cooling_steps) — typically a few hundred — and exact.
    n_outer = 0
    if 0 < o.alpha < 1 and o.T > o.T_min:
        T_sim = float(o.T)
        while T_sim > o.T_min:
            n_outer += 1
            T_sim *= (1 - o.alpha)
            if n_outer > 10_000_000:                # safety against alpha≈0 typos
                break
    total_iters = n_outer * o.max_iter

    with BatchEvaluator(
        cfg,
        ffield_in=cfg.forcefield.path,
        params_in=cfg.forcefield.params,
        out_dir=out_dir,
        n_workers=n_workers,
    ) as evaluator:
        # Initial cost — every walker starts at the same (initial) FF.
        initial_snap = ParameterSnapshot.capture(ff)
        initial_cost = evaluator.evaluate_batch([initial_snap.values])[0]

        walker_snaps = [initial_snap.copy() for _ in range(n_walkers)]
        walker_costs = [initial_cost for _ in range(n_walkers)]

        trace = [initial_cost]
        best_cost = initial_cost
        best_snap = initial_snap.copy()

        bar = _make_progress(
            total_iters, enabled=o.show_progress,
            desc=f"SA ({n_walkers}w × {n_workers}p)",
        )
        try:
            T = o.T
            while T > o.T_min:
                for _ in range(o.max_iter):
                    # 1. Master proposes: one perturbation per walker.
                    proposals = [_propose(snap, ff, rng) for snap in walker_snaps]

                    # 2. Workers evaluate the batch in parallel.
                    new_costs = evaluator.evaluate_batch([p.values for p in proposals])

                    # 3. Master decides accept/reject independently per walker.
                    for w in range(n_walkers):
                        delta = new_costs[w] - walker_costs[w]
                        ap = 1.0 if delta <= 0 else float(np.exp(-delta / max(T, 1e-30)))
                        if rng.random() < ap:
                            walker_snaps[w] = proposals[w]
                            walker_costs[w] = new_costs[w]
                            if new_costs[w] < best_cost:
                                best_cost = new_costs[w]
                                best_snap = proposals[w].copy()

                    # 4. Trace = best cost across walkers at this step.
                    trace.append(min(walker_costs))
                    bar.update(1)
                    bar.set_postfix(T=f"{T:.3g}", best=f"{best_cost:.4g}")
                T = T * (1 - o.alpha)
        finally:
            bar.close()

    # Write best forcefield from the master's FF (the workers each have
    # their own; the master's is what we serialise so the user sees a
    # canonical artifact).
    best_snap.apply(ff)
    best_path = out_dir / "bestFF.reax"
    ff.write_forcefield(str(best_path))
    return SAResult(final_cost=best_cost, cost_trace=trace, best_ffield_path=best_path)
