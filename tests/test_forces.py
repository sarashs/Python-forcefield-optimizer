"""forces objective. No LAMMPS — synthetic dumps with fx fy fz columns."""
from pathlib import Path

import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.simulations.base import SimResult


def _force_dump(path: Path, forces) -> None:
    n = len(forces)
    rows = "\n".join(
        f"{i+1} 1 0 0 0 0 {f[0]} {f[1]} {f[2]}"
        for i, f in enumerate(forces)
    )
    path.write_text(
        "ITEM: TIMESTEP\n0\n"
        f"ITEM: NUMBER OF ATOMS\n{n}\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z q fx fy fz\n" + rows + "\n"
    )


def test_forces_zero_when_matching(tmp_path):
    forces = [(1.0, -2.0, 0.5), (0.0, 0.5, -1.0)]
    dump = tmp_path / "f.lammpstrj"
    _force_dump(dump, forces)
    tgt = TargetCfg.model_validate({
        "kind": "forces", "weight": 4.0, "simulation": "sp",
        "reference": {1: list(forces[0]), 2: list(forces[1])},
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sp": SimResult(sim_id="sp", energy=0.0, extras={"dump_file": str(dump)})
    })
    assert obj.compute(ctx) == pytest.approx(0.0)
    assert obj.residual(ctx) == pytest.approx(0.0)


def test_forces_known_mse(tmp_path):
    observed = [(1.0, 0.0, 0.0)]
    reference = [(0.0, 0.0, 0.0)]
    dump = tmp_path / "f.lammpstrj"
    _force_dump(dump, observed)
    tgt = TargetCfg.model_validate({
        "kind": "forces", "weight": 3.0, "simulation": "sp",
        "reference": {1: list(reference[0])},
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sp": SimResult(sim_id="sp", energy=0.0, extras={"dump_file": str(dump)})
    })
    # |Δf|² = 1; mean over 1 atom = 1; residual = 3 * 1 = 3
    assert obj.compute(ctx) == pytest.approx(1.0)
    assert obj.residual(ctx) == pytest.approx(3.0)


def test_forces_missing_dump_force_columns_rejected(tmp_path):
    """A dump without fx/fy/fz must produce a clear error."""
    dump = tmp_path / "noforce.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z q\n1 1 0 0 0 0\n"
    )
    tgt = TargetCfg.model_validate({
        "kind": "forces", "weight": 1.0, "simulation": "sp",
        "reference": {1: [0.0, 0.0, 0.0]},
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sp": SimResult(sim_id="sp", energy=0.0, extras={"dump_file": str(dump)})
    })
    with pytest.raises(RuntimeError, match="force columns"):
        obj.compute(ctx)
