"""Genetic-algorithm driver for the new YAML pipeline.

The GA evolves a population of `ParameterSnapshot`s (the same flat
length-N value array the SA optimizer uses), evaluating each member by
running every required simulation and summing the registered objectives.

Operators:

- **Selection** — tournament of size K: pick K members at random, keep
  the lowest-cost one. Avoids the cost-normalisation pitfalls of
  roulette-wheel selection that bit the legacy GA.
- **Crossover** — single-point on the flat array (children are
  parent_a[:k] || parent_b[k:] and the symmetric swap). Triggered with
  probability `crossover_rate`; otherwise the children are clones of
  the parents.
- **Mutation** — per-gene Gaussian kick with probability
  `mutation_rate` and σ = `mutation_sigma * (max - min)` for that gene.
  Clamped back into `[min, max]`.
- **Elitism** — the top `elitism` members are copied unchanged into
  the next generation.

When `method = "sa+ga"`, each newly bred child is also pushed through
`sa_refine_steps` Metropolis steps at temperature T before its fitness
is recorded — a cheap local-search refinement between the global GA
jumps. With `optimizer.parallel: true` every child's refinement runs
on its own worker, so a 16-core box gets a ~16× speedup.

Reproducibility: every random number is drawn on the master (or, for
sa+ga refinement, with a per-child seed derived from the master RNG)
so seeded runs are bit-identical regardless of worker count.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pyfield.config.schema import PyFieldConfig
from pyfield.forcefield.reax import REAX_FF
from pyfield.forcefield.snapshot import ParameterSnapshot
from pyfield.optimizers.parallel import BatchEvaluator, resolve_n_workers


@dataclass
class GAResult:
    final_cost: float
    cost_trace: List[float] = field(default_factory=list)   # best cost per generation
    best_ffield_path: Optional[Path] = None


def _random_within_bounds(ff: REAX_FF, rng: random.Random) -> ParameterSnapshot:
    """Uniform-random snapshot within `param_min_max_delta` for each gene."""
    keys = tuple(ff.param_min_max_delta.keys())
    values = np.empty(len(keys), dtype=float)
    for i, key in enumerate(keys):
        b = ff.param_min_max_delta[key]
        values[i] = round(rng.uniform(b["min"], b["max"]), 4)
    return ParameterSnapshot(keys=keys, values=values)


def _crossover(
    a: ParameterSnapshot,
    b: ParameterSnapshot,
    rate: float,
    rng: random.Random,
) -> Tuple[ParameterSnapshot, ParameterSnapshot]:
    if rng.random() >= rate or len(a) < 2:
        return a.copy(), b.copy()
    k = rng.randint(1, len(a) - 1)
    c1 = ParameterSnapshot(keys=a.keys, values=np.concatenate([a.values[:k], b.values[k:]]))
    c2 = ParameterSnapshot(keys=a.keys, values=np.concatenate([b.values[:k], a.values[k:]]))
    return c1, c2


def _mutate(
    snap: ParameterSnapshot,
    ff: REAX_FF,
    rate: float,
    sigma_frac: float,
    rng: random.Random,
) -> ParameterSnapshot:
    """In-place Gaussian kick with per-gene clamp. Returns the same snapshot."""
    for i, key in enumerate(snap.keys):
        if rng.random() >= rate:
            continue
        b = ff.param_min_max_delta[key]
        sigma = sigma_frac * (b["max"] - b["min"])
        new = snap.values[i] + rng.gauss(0.0, sigma)
        new = min(b["max"], max(b["min"], new))
        snap.values[i] = round(new, 4)
    return snap


def _tournament_pick(
    pop: List[ParameterSnapshot],
    costs: List[float],
    k: int,
    rng: random.Random,
) -> ParameterSnapshot:
    contenders = rng.sample(range(len(pop)), min(k, len(pop)))
    best = min(contenders, key=lambda i: costs[i])
    return pop[best].copy()


def _make_progress(total: int, *, enabled: bool, desc: str):
    """`tqdm.auto` picks notebook widget / terminal text / plain auto;
    we only gate on `show_progress` itself."""
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


def run_ga(cfg: PyFieldConfig) -> GAResult:
    """Run GA (or sa+ga) on a validated config. Returns best cost + per-gen trace."""
    o = cfg.optimizer
    rng = random.Random(o.seed)
    if o.seed is not None:
        np.random.seed(o.seed)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()

    n_workers = resolve_n_workers(o.parallel, o.processors)
    is_sa_ga = o.method == "sa+ga"

    with BatchEvaluator(
        cfg,
        ffield_in=cfg.forcefield.path,
        params_in=cfg.forcefield.params,
        out_dir=out_dir,
        n_workers=n_workers,
    ) as evaluator:
        # Initial population: one snapshot of the seed FF + (N-1) uniform-random.
        seed_snap = ParameterSnapshot.capture(ff)
        population: List[ParameterSnapshot] = [seed_snap]
        for _ in range(o.population_size - 1):
            population.append(_random_within_bounds(ff, rng))

        # Initial costs — single batch.
        costs: List[float] = list(evaluator.evaluate_batch([s.values for s in population]))

        best_idx = int(np.argmin(costs))
        best_snap = population[best_idx].copy()
        best_cost = costs[best_idx]
        trace = [best_cost]

        bar = _make_progress(
            o.generations, enabled=o.show_progress,
            desc=f"GA ({o.method}, pop={o.population_size} × {n_workers}p)",
        )
        try:
            for gen in range(o.generations):
                # Elitism — sort by cost, copy the top E unchanged into the new pop.
                order = np.argsort(costs)
                new_pop: List[ParameterSnapshot] = [population[i].copy() for i in order[:o.elitism]]
                new_costs: List[float] = [costs[i] for i in order[:o.elitism]]

                # Build the rest of the new population on the master (cheap).
                pending: List[ParameterSnapshot] = []
                while len(new_pop) + len(pending) < o.population_size:
                    p1 = _tournament_pick(population, costs, o.tournament_size, rng)
                    p2 = _tournament_pick(population, costs, o.tournament_size, rng)
                    c1, c2 = _crossover(p1, p2, o.crossover_rate, rng)
                    _mutate(c1, ff, o.mutation_rate, o.mutation_sigma, rng)
                    _mutate(c2, ff, o.mutation_rate, o.mutation_sigma, rng)
                    for child in (c1, c2):
                        if len(new_pop) + len(pending) >= o.population_size:
                            break
                        pending.append(child)

                # Evaluate (and optionally SA-refine) the pending children
                # in a single parallel batch.
                if is_sa_ga and o.sa_refine_steps > 0:
                    jobs = [
                        (child.values, o.T, o.sa_refine_steps,
                         rng.randint(0, 2**31 - 1))
                        for child in pending
                    ]
                    refined = evaluator.refine_batch(jobs)
                    for child, (refined_values, cost) in zip(pending, refined):
                        child.values = np.asarray(refined_values, dtype=float)
                        new_pop.append(child)
                        new_costs.append(cost)
                else:
                    eval_costs = evaluator.evaluate_batch([c.values for c in pending])
                    for child, cost in zip(pending, eval_costs):
                        new_pop.append(child)
                        new_costs.append(cost)

                population, costs = new_pop, new_costs
                gen_best_idx = int(np.argmin(costs))
                if costs[gen_best_idx] < best_cost:
                    best_cost = costs[gen_best_idx]
                    best_snap = population[gen_best_idx].copy()
                trace.append(best_cost)
                bar.update(1)
                bar.set_postfix(gen=gen + 1, best=f"{best_cost:.4g}")
        finally:
            bar.close()

    best_path = out_dir / "bestFF.reax"
    best_snap.apply(ff)
    ff.write_forcefield(str(best_path))
    return GAResult(final_cost=best_cost, cost_trace=trace, best_ffield_path=best_path)
