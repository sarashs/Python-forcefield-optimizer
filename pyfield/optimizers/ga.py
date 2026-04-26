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
`sa_refine_steps` Metropolis steps at temperature T before the
generation's fitness is recorded — a cheap local-search refinement
between the global GA jumps.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
class GAResult:
    final_cost: float
    cost_trace: List[float] = field(default_factory=list)   # best cost per generation
    best_ffield_path: Optional[Path] = None


def _random_within_bounds(ff: REAX_FF) -> ParameterSnapshot:
    """Uniform-random snapshot within `param_min_max_delta` for each gene."""
    keys = tuple(ff.param_min_max_delta.keys())
    values = np.empty(len(keys), dtype=float)
    for i, key in enumerate(keys):
        b = ff.param_min_max_delta[key]
        values[i] = round(random.uniform(b["min"], b["max"]), 4)
    return ParameterSnapshot(keys=keys, values=values)


def _crossover(a: ParameterSnapshot, b: ParameterSnapshot, rate: float) -> Tuple[ParameterSnapshot, ParameterSnapshot]:
    if random.random() >= rate or len(a) < 2:
        return a.copy(), b.copy()
    k = random.randint(1, len(a) - 1)
    c1 = ParameterSnapshot(
        keys=a.keys,
        values=np.concatenate([a.values[:k], b.values[k:]]),
    )
    c2 = ParameterSnapshot(
        keys=a.keys,
        values=np.concatenate([b.values[:k], a.values[k:]]),
    )
    return c1, c2


def _mutate(snap: ParameterSnapshot, ff: REAX_FF, rate: float, sigma_frac: float) -> ParameterSnapshot:
    """In-place Gaussian kick with per-gene clamp. Returns the same snapshot."""
    for i, key in enumerate(snap.keys):
        if random.random() >= rate:
            continue
        b = ff.param_min_max_delta[key]
        sigma = sigma_frac * (b["max"] - b["min"])
        new = snap.values[i] + random.gauss(0.0, sigma)
        new = min(b["max"], max(b["min"], new))
        snap.values[i] = round(new, 4)
    return snap


def _tournament_pick(pop: List[ParameterSnapshot], costs: List[float], k: int) -> ParameterSnapshot:
    contenders = random.sample(range(len(pop)), min(k, len(pop)))
    best = min(contenders, key=lambda i: costs[i])
    return pop[best].copy()


def _sa_refine(
    snap: ParameterSnapshot,
    cost: float,
    ff: REAX_FF,
    evaluate: Callable[[], float],
    *,
    T: float,
    steps: int,
) -> Tuple[ParameterSnapshot, float]:
    """Cheap Metropolis local-search around `snap`. Mutates the FF and snap."""
    if steps <= 0:
        return snap, cost
    snap.apply(ff)
    best_snap = snap.copy()
    best_cost = cost
    cur = snap.copy()
    cur_cost = cost
    for _ in range(steps):
        # Tiny perturbation: one random gene, drawn from its `delta` like SA does.
        i = random.randrange(len(cur))
        sec, entry, item = cur.keys[i]
        b = ff.param_min_max_delta[(sec, entry, item)]
        new = cur.values[i] + random.uniform(-1, 1) * b["delta"]
        new = min(b["max"], max(b["min"], round(new, 4)))
        ff.params[sec][entry][item] = float(new)
        new_cost = evaluate()
        accept = (new_cost < cur_cost) or (
            T > 0 and random.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-30))
        )
        if accept:
            cur.values[i] = new
            cur_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_snap = cur.copy()
        else:
            ff.params[sec][entry][item] = float(cur.values[i])
    best_snap.apply(ff)
    return best_snap, best_cost


def run_ga(cfg: PyFieldConfig) -> GAResult:
    """Run GA (or sa+ga) on a validated config. Returns best cost + per-gen trace."""
    o = cfg.optimizer
    if o.seed is not None:
        random.seed(o.seed)
        np.random.seed(o.seed)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = REAX_FF(str(cfg.forcefield.path), str(cfg.forcefield.params))
    ff.parseParamSelectionFile()

    ffield_iter_path = out_dir / "ga_iter.reax"

    objectives: List[Objective] = [build_objective(t) for t in cfg.targets]
    needed_sims: List[str] = sorted({s for ob in objectives for s in ob.required_simulations()})

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

        # Initial population: one snapshot of the seed FF + (N-1) uniform-random.
        seed_snap = ParameterSnapshot.capture(ff)
        population: List[ParameterSnapshot] = [seed_snap]
        for _ in range(o.population_size - 1):
            population.append(_random_within_bounds(ff))

        # Initial costs.
        costs: List[float] = []
        for snap in population:
            snap.apply(ff)
            costs.append(evaluate())

        best_idx = int(np.argmin(costs))
        best_snap = population[best_idx].copy()
        best_cost = costs[best_idx]
        trace = [best_cost]

        for gen in range(o.generations):
            # Elitism — sort by cost, copy the top E unchanged.
            order = np.argsort(costs)
            new_pop: List[ParameterSnapshot] = [population[i].copy() for i in order[:o.elitism]]
            new_costs: List[float] = [costs[i] for i in order[:o.elitism]]

            while len(new_pop) < o.population_size:
                p1 = _tournament_pick(population, costs, o.tournament_size)
                p2 = _tournament_pick(population, costs, o.tournament_size)
                c1, c2 = _crossover(p1, p2, o.crossover_rate)
                _mutate(c1, ff, o.mutation_rate, o.mutation_sigma)
                _mutate(c2, ff, o.mutation_rate, o.mutation_sigma)
                for child in (c1, c2):
                    if len(new_pop) >= o.population_size:
                        break
                    child.apply(ff)
                    cost = evaluate()
                    if o.method == "sa+ga" and o.sa_refine_steps > 0:
                        child, cost = _sa_refine(
                            child, cost, ff, evaluate, T=o.T, steps=o.sa_refine_steps,
                        )
                    new_pop.append(child)
                    new_costs.append(cost)

            population, costs = new_pop, new_costs
            gen_best_idx = int(np.argmin(costs))
            if costs[gen_best_idx] < best_cost:
                best_cost = costs[gen_best_idx]
                best_snap = population[gen_best_idx].copy()
            trace.append(best_cost)

    best_path = out_dir / "bestFF.reax"
    best_snap.apply(ff)
    ff.write_forcefield(str(best_path))
    return GAResult(final_cost=best_cost, cost_trace=trace, best_ffield_path=best_path)
