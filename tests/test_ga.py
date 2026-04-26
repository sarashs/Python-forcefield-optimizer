"""GA operators + a synthetic-objective convergence check. No LAMMPS."""
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pyfield.forcefield.reax import REAX_FF
from pyfield.forcefield.snapshot import ParameterSnapshot
from pyfield.optimizers.ga import (
    _crossover,
    _mutate,
    _random_within_bounds,
    _tournament_pick,
)


@pytest.fixture
def ff():
    f = REAX_FF(os.path.join(ROOT, "tests/ffieldoriginal.txt"),
                os.path.join(ROOT, "tests/params"))
    f.parseParamSelectionFile()
    return f


def test_random_within_bounds_respects_min_max(ff):
    random.seed(0)
    snap = _random_within_bounds(ff)
    for i, key in enumerate(snap.keys):
        b = ff.param_min_max_delta[key]
        assert b["min"] <= snap.values[i] <= b["max"]


def test_crossover_swaps_at_a_single_point(ff):
    random.seed(1)
    a = _random_within_bounds(ff)
    b = _random_within_bounds(ff)
    # Force crossover by setting rate=1.0
    c1, c2 = _crossover(a, b, rate=1.0)
    # Children must be within parents elementwise (each element comes from one parent).
    for i in range(len(a)):
        assert c1.values[i] in (a.values[i], b.values[i])
        assert c2.values[i] in (a.values[i], b.values[i])
    # And at least one position differs from the matching parent — i.e. real crossover.
    assert any(c1.values != a.values) or any(c2.values != a.values)


def test_crossover_clones_when_rate_zero(ff):
    a = _random_within_bounds(ff)
    b = _random_within_bounds(ff)
    c1, c2 = _crossover(a, b, rate=0.0)
    np.testing.assert_array_equal(c1.values, a.values)
    np.testing.assert_array_equal(c2.values, b.values)


def test_mutate_stays_in_bounds(ff):
    random.seed(2)
    snap = _random_within_bounds(ff)
    _mutate(snap, ff, rate=1.0, sigma_frac=2.0)   # extreme σ — anything still clamps
    for i, key in enumerate(snap.keys):
        b = ff.param_min_max_delta[key]
        assert b["min"] <= snap.values[i] <= b["max"]


def test_tournament_picks_lowest_cost(ff):
    pop = [_random_within_bounds(ff) for _ in range(5)]
    costs = [10.0, 1.0, 20.0, 5.0, 50.0]
    random.seed(0)
    # With k=5 we always sample everyone; the pick must be the index-1 minimum.
    pick = _tournament_pick(pop, costs, k=5)
    np.testing.assert_array_equal(pick.values, pop[1].values)


# ---------------------------------------------------------------------------
# End-to-end-ish convergence on a synthetic objective with no LAMMPS.
# We bypass `run_ga` (which calls real simulations) and exercise the GA loop
# logic directly against a quadratic in the flat parameter values.
# ---------------------------------------------------------------------------

def test_ga_loop_drives_synthetic_objective_to_minimum(ff):
    """Plain GA loop on a quadratic objective with a known minimum.

    Uses the same operators run_ga uses, against an in-memory cost
    function so the test stays LAMMPS-free.
    """
    from pyfield.optimizers.ga import _crossover, _mutate, _random_within_bounds, _tournament_pick

    random.seed(123)
    np.random.seed(123)

    # Target: middle of each gene's range. Cost = sum((x - target)^2).
    targets = np.array([
        0.5 * (ff.param_min_max_delta[k]["min"] + ff.param_min_max_delta[k]["max"])
        for k in tuple(ff.param_min_max_delta.keys())
    ])

    def cost(snap):
        return float(np.sum((snap.values - targets) ** 2))

    pop_size = 20
    population = [_random_within_bounds(ff) for _ in range(pop_size)]
    costs = [cost(s) for s in population]
    initial_best = min(costs)

    for _ in range(40):
        order = np.argsort(costs)
        new_pop = [population[i].copy() for i in order[:2]]   # elitism=2
        new_costs = [costs[i] for i in order[:2]]
        while len(new_pop) < pop_size:
            p1 = _tournament_pick(population, costs, k=3)
            p2 = _tournament_pick(population, costs, k=3)
            c1, c2 = _crossover(p1, p2, rate=0.7)
            _mutate(c1, ff, rate=0.3, sigma_frac=0.1)
            _mutate(c2, ff, rate=0.3, sigma_frac=0.1)
            for c in (c1, c2):
                if len(new_pop) >= pop_size:
                    break
                new_pop.append(c)
                new_costs.append(cost(c))
        population, costs = new_pop, new_costs

    final_best = min(costs)
    assert final_best < initial_best, (initial_best, final_best)
    # GA should at least halve the cost over 40 generations on this convex problem.
    assert final_best < 0.5 * initial_best, (initial_best, final_best)
