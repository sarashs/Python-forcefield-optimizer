"""rdf_peak objective. No LAMMPS — synthetic dumps."""
from pathlib import Path

import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.simulations.base import SimResult


def _dump_with_pair(tmp_path: Path, separation: float, box: float = 100.0) -> Path:
    """Two atoms (different types) at fixed separation in a big box.
    With a large box, all of g(r) is below 1 except a delta-spike at the
    pair separation, so argmax is at that bin."""
    p = tmp_path / "rdf.lammpstrj"
    p.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        f"ITEM: BOX BOUNDS pp pp pp\n0 {box}\n0 {box}\n0 {box}\n"
        "ITEM: ATOMS id type x y z\n"
        f"1 1 50 50 50\n"
        f"2 2 {50 + separation} 50 50\n"
    )
    return p


def test_rdf_peak_finds_pair_separation(tmp_path):
    dump = _dump_with_pair(tmp_path, separation=2.5)
    tgt = TargetCfg.model_validate({
        "kind": "rdf_peak", "weight": 1.0, "simulation": "sim1",
        "central": "Si", "neighbor": "O",
        "r_max": 5.0, "r_min": 0.5, "bins": 100,
        "target": 2.5,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(
            sim_id="sim1", energy=0.0,
            extras={"dump_file": str(dump), "type_to_element": {1: "Si", 2: "O"}},
        )
    })
    # bin width = 5.0 / 100 = 0.05; the peak should be in the bin centred at ~2.5
    assert obj.compute(ctx) == pytest.approx(2.5, abs=0.05)
    assert obj.residual(ctx) == pytest.approx(0.0, abs=0.005)


def test_rdf_peak_residual_grows_with_distance(tmp_path):
    dump = _dump_with_pair(tmp_path, separation=2.0)
    tgt = TargetCfg.model_validate({
        "kind": "rdf_peak", "weight": 1.0, "simulation": "sim1",
        "central": "Si", "neighbor": "O",
        "r_max": 5.0, "target": 3.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(
            sim_id="sim1", energy=0.0,
            extras={"dump_file": str(dump), "type_to_element": {1: "Si", 2: "O"}},
        )
    })
    # peak ≈ 2.0, target = 3.0 → residual ≈ (1.0)^2 = 1.0 (within bin width)
    assert obj.compute(ctx) == pytest.approx(2.0, abs=0.1)
    assert obj.residual(ctx) == pytest.approx(1.0, abs=0.2)
