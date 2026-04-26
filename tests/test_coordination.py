"""Coordination objective: residual against synthetic frames. No LAMMPS."""
from pathlib import Path

import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.simulations.base import SimResult


_FRAME = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
5
ITEM: BOX BOUNDS pp pp pp
0 100
0 100
0 100
ITEM: ATOMS id type x y z
1 1 50 50 50
2 2 51.0 50 50
3 2 50 51.0 50
4 2 50 50 51.0
5 2 50 50 90
"""
# 1 Si, 4 Os; three Os are at distance 1.0, one is at distance 40.
# With cutoff=2.5 the central Si (type 1) sees 3 type-2 neighbours.


def test_coordination_known_value(tmp_path: Path):
    dump = tmp_path / "c.lammpstrj"
    dump.write_text(_FRAME)
    tgt = TargetCfg.model_validate({
        "kind": "coordination",
        "weight": 4.0,
        "simulation": "sim1",
        "central": "Si",
        "neighbor": "O",
        "cutoff": 2.5,
        "target": 4.0,    # we want 4-coordinated; computed will be 3.0
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(
            sim_id="sim1",
            energy=0.0,
            extras={
                "dump_file": str(dump),
                "type_to_element": {1: "Si", 2: "O"},
            },
        )
    })
    assert obj.compute(ctx) == pytest.approx(3.0)
    # weighted (3-4)^2 * 4 = 4
    assert obj.residual(ctx) == pytest.approx(4.0)


def test_coordination_pbc_minimum_image(tmp_path: Path):
    """Atoms across a periodic boundary should still count as neighbours."""
    dump = tmp_path / "pbc.lammpstrj"
    dump.write_text("""\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0.5 5.0 5.0
2 2 9.5 5.0 5.0
""")
    tgt = TargetCfg.model_validate({
        "kind": "coordination", "weight": 1.0, "simulation": "sim1",
        "central": "A", "neighbor": "B", "cutoff": 2.0, "target": 1.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(
            sim_id="sim1", energy=0.0,
            extras={"dump_file": str(dump), "type_to_element": {1: "A", 2: "B"}},
        )
    })
    # Across PBC the distance is 1.0, well within cutoff 2.0.
    assert obj.compute(ctx) == pytest.approx(1.0)


def test_coordination_no_dump_raises(tmp_path: Path):
    tgt = TargetCfg.model_validate({
        "kind": "coordination", "weight": 1.0, "simulation": "sim1",
        "central": "A", "neighbor": "B", "cutoff": 2.0, "target": 1.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, extras={}),
    })
    with pytest.raises(RuntimeError, match="dump_file"):
        obj.compute(ctx)


def test_coordination_unknown_element_rejected(tmp_path: Path):
    dump = tmp_path / "x.lammpstrj"
    dump.write_text(_FRAME)
    tgt = TargetCfg.model_validate({
        "kind": "coordination", "weight": 1.0, "simulation": "sim1",
        "central": "Mo", "neighbor": "O", "cutoff": 2.5, "target": 1.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(
            sim_id="sim1", energy=0.0,
            extras={"dump_file": str(dump), "type_to_element": {1: "Si", 2: "O"}},
        )
    })
    with pytest.raises(ValueError, match="Mo"):
        obj.compute(ctx)
