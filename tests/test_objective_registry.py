"""Objective registry tests. No LAMMPS."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyfield.config.schema import TargetCfg
from pyfield.objectives import (
    Objective,
    ObjectiveContext,
    build_objective,
    register_objective,
    registered_kinds,
)
from pyfield.simulations.base import SimResult


def test_builtins_registered():
    kinds = registered_kinds()
    assert "energy_combination" in kinds
    assert "charges" in kinds


def test_unknown_kind_rejected():
    bad = TargetCfg.model_validate({"kind": "totally_made_up", "weight": 1.0})
    with pytest.raises(KeyError):
        build_objective(bad)


def test_energy_combination_residual():
    tgt = TargetCfg.model_validate({
        "kind": "energy_combination",
        "weight": 2.0,
        "terms": {"a": 1, "b": -1},
        "target": 5.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "a": SimResult(sim_id="a", energy=10.0),
        "b": SimResult(sim_id="b", energy=3.0),
    })
    # value = 10 - 3 = 7; residual = 2*(7-5)^2 = 8
    assert obj.compute(ctx) == pytest.approx(7.0)
    assert obj.residual(ctx) == pytest.approx(8.0)


def test_charges_residual():
    tgt = TargetCfg.model_validate({
        "kind": "charges",
        "weight": 4.0,
        "simulation": "sim1",
        "atoms": {1: 0.5, 2: -0.5},
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, charges=[0.6, -0.4]),
    })
    # mse = ((0.6-0.5)^2 + (-0.4-(-0.5))^2)/2 = 0.01
    # residual = 4 * 0.01 = 0.04
    assert obj.compute(ctx) == pytest.approx(0.01)
    assert obj.residual(ctx) == pytest.approx(0.04)


def test_register_then_resolve():
    @register_objective("__test_only_kind__")
    class Dummy(Objective):
        def required_simulations(self):
            return []
        def compute(self, ctx):
            return 42.0
    tgt = TargetCfg.model_validate({"kind": "__test_only_kind__", "weight": 0.5})
    obj = build_objective(tgt)
    assert isinstance(obj, Dummy)


def test_double_register_rejected():
    with pytest.raises(ValueError):
        @register_objective("energy_combination")
        class Conflict(Objective):
            pass
