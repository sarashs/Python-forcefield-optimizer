"""CMA-ES driver — a third optimizer alongside SA and GA.

CMA-ES (Hansen et al.) maintains a multivariate Gaussian over the
parameter space and adapts its mean + covariance from the best samples
each generation. Strengths over SA/GA on this problem:

- Adapts step size per direction → handles ill-conditioned cost
  surfaces (where some parameters are 1000× more sensitive than
  others, common in ReaxFF) much better than SA's single delta or
  GA's per-gene Gaussian σ.
- Population-based, so each generation maps cleanly onto our existing
  `BatchEvaluator` — drop-in parallel speedup.
- Global-ish: rebuilds its covariance from scratch each generation,
  so early it's broad like SA/GA, late it tightens like a local
  search. No need for the SA cooling schedule or the GA mutation/
  crossover trade-offs.

Bounds: the existing `param_min_max_delta` per-parameter bounds are
passed to `cma` as a hard `[lower, upper]` constraint — proposals
outside the box get repaired before evaluation.

Reproducibility: with `optimizer.seed` set, cma's RNG is seeded; every
candidate the master proposes is deterministic, every cost we hand
back is deterministic, so seeded runs are bit-identical regardless of
worker count.
"""
from __future__ import annotations

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
class CMAResult:
    final_cost: float
    cost_trace: List[float] = field(default_factory=list)   # best-so-far per generation
    best_ffield_path: Optional[Path] = None


def _make_progress(total: int, *, enabled: bool, desc: str):
    if not enabled or not sys.stderr.isatty():
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


def run_cma(cfg: PyFieldConfig) -> CMAResult:
    """Run CMA-ES on a validated config. Returns best cost + per-gen trace.

    `optimizer.generations` caps the number of CMA generations; the loop
    also stops early if `cma`'s built-in convergence criteria fire (flat
    fitness, tiny step-size, etc.).
    """
    try:
        import cma
    except ImportError as e:
        raise ImportError(
            "method: cma requires the `cma` package. Install with `pip install cma` "
            "or `pip install -e .[cma]` (an extra defined in pyproject.toml)."
        ) from e

    o = cfg.optimizer
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()

    keys = tuple(ff.param_min_max_delta.keys())
    bounds = [ff.param_min_max_delta[k] for k in keys]
    lower = np.array([b["min"] for b in bounds], dtype=float)
    upper = np.array([b["max"] for b in bounds], dtype=float)
    spans = upper - lower
    if np.any(spans <= 0):
        raise ValueError("CMA-ES needs every trainable parameter to have max > min.")

    # x0 = current FF values; sigma0 = sigma fraction of the *median* span
    # so every coordinate gets a reasonable initial spread. We pass the
    # full per-parameter bound list to cma so its boundary handler clips
    # proposals back into the box before they reach the evaluator.
    x0 = ParameterSnapshot.capture(ff).values.astype(float)
    sigma0 = max(1e-6, float(o.cma_sigma0) * float(np.median(spans)))

    cma_opts = {
        "bounds": [list(lower), list(upper)],
        "verbose": -9,                # suppress cma's own stdout chatter
        "maxiter": int(o.generations),
    }
    if o.seed is not None:
        cma_opts["seed"] = int(o.seed) + 1   # cma rejects seed=0
    if o.cma_popsize and o.cma_popsize > 0:
        cma_opts["popsize"] = int(o.cma_popsize)

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
    n_workers = resolve_n_workers(o.parallel, o.processors)
    pop = es.popsize

    bar = _make_progress(
        int(o.generations), enabled=o.show_progress,
        desc=f"CMA (pop={pop} × {n_workers}p)",
    )

    best_cost = float("inf")
    best_values = x0.copy()
    trace: List[float] = []

    with BatchEvaluator(
        cfg,
        ffield_in=cfg.forcefield.path,
        params_in=cfg.forcefield.params,
        out_dir=out_dir,
        n_workers=n_workers,
    ) as evaluator:
        try:
            for gen in range(o.generations):
                if es.stop():
                    break
                candidates = es.ask()                # list of length popsize
                # cma's default BoundPenalty *allows* out-of-bounds proposals
                # (it adds a penalty in tell), but we still call the cost
                # function on them — and a value outside the param's
                # `min/max` can break the ReaxFF text writer (column-width
                # overflow). Hard-clip before evaluating; cma is tolerant
                # of receiving costs for clipped variants.
                # Round to 4dp for the same reason as SA: the ReaxFF text
                # writer pads each column to 10 chars and a 16-digit
                # double like 1.2345678901234567 silently overflows.
                clipped = [np.round(np.clip(c, lower, upper), 4) for c in candidates]
                costs = list(evaluator.evaluate_batch(clipped))
                es.tell(candidates, costs)
                gen_best_idx = int(np.argmin(costs))
                if costs[gen_best_idx] < best_cost:
                    best_cost = float(costs[gen_best_idx])
                    # Save the *clipped* values — those are what produced
                    # the cost we're tracking and what we want to write
                    # back as bestFF.reax.
                    best_values = np.asarray(clipped[gen_best_idx], dtype=float).copy()
                trace.append(best_cost)
                bar.update(1)
                bar.set_postfix(gen=gen + 1, best=f"{best_cost:.4g}",
                                sigma=f"{float(es.sigma):.3g}")
        finally:
            bar.close()

    # Apply best snapshot back to the master FF and write it out.
    best_snap = ParameterSnapshot(keys=keys, values=best_values)
    best_snap.apply(ff)
    best_path = out_dir / "bestFF.reax"
    ff.write_forcefield(str(best_path))
    return CMAResult(final_cost=best_cost, cost_trace=trace, best_ffield_path=best_path)
