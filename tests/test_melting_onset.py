"""melting_onset objective. No LAMMPS — synthetic NVT dumps."""
from pathlib import Path

import numpy as np
import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.simulations.base import SimResult


def _msd_growing_dump(path: Path, n_frames: int, drift_per_frame: float):
    """One atom that drifts in +x by `drift_per_frame` each frame.
    MSD vs t is a parabola; the linear-fit slope is drift²·t for last vs 0.
    """
    chunks = []
    for i in range(n_frames):
        chunks.append(
            "ITEM: TIMESTEP\n"
            f"{i}\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 100\n0 100\n0 100\n"
            "ITEM: ATOMS id type x y z\n"
            f"1 1 {50 + drift_per_frame * i} 50 50\n"
        )
    path.write_text("".join(chunks))


def _result(dump_path: str, T: float) -> SimResult:
    return SimResult(
        sim_id="x", energy=0.0,
        extras={"dump_file": dump_path, "temperature": T},
    )


def test_melting_onset_finds_first_high_slope_T(tmp_path):
    # 4 simulations: at 300/500 K very low slope; at 1000/1500 K both
    # at the same high drift, so the (min+max)/2 threshold lands well
    # below their slope and the *first* hot one (1000 K) is reported.
    sims = {}
    for i, (T, drift) in enumerate([(300, 0.001), (500, 0.001), (1000, 1.0), (1500, 1.0)]):
        d = tmp_path / f"sim_{i}.lammpstrj"
        _msd_growing_dump(d, n_frames=10, drift_per_frame=drift)
        sims[f"sim_{i}_T{T}"] = _result(str(d), T)

    tgt = TargetCfg.model_validate({
        "kind": "melting_onset", "weight": 1.0,
        "simulations": list(sims.keys()),
        "target": 1200.0,
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results=sims)
    onset = obj.compute(ctx)
    # The half-way slope threshold is crossed first at T=1000 K, not earlier.
    assert onset == pytest.approx(1000.0)
    # residual = (1000 - 1200)^2 = 40000
    assert obj.residual(ctx) == pytest.approx(40000.0)


def test_melting_onset_all_solid_returns_highest_T(tmp_path):
    sims = {}
    for i, T in enumerate([300, 500, 1000]):
        d = tmp_path / f"sim_{i}.lammpstrj"
        _msd_growing_dump(d, n_frames=10, drift_per_frame=0.001)
        sims[f"s_{T}"] = _result(str(d), T)
    # All slopes are essentially equal so the threshold is the same as the
    # data; everything passes the threshold including the lowest T. The
    # crossing happens at the lowest T.
    tgt = TargetCfg.model_validate({
        "kind": "melting_onset", "weight": 1.0,
        "simulations": list(sims.keys()), "target": 0.0,
    })
    obj = build_objective(tgt)
    onset = obj.compute(ObjectiveContext(sim_results=sims))
    # In a degenerate (all-equal) input the first T crosses immediately.
    assert onset == pytest.approx(300.0)


def test_melting_onset_requires_temperature(tmp_path):
    d = tmp_path / "x.lammpstrj"
    _msd_growing_dump(d, 5, 0.1)
    sims = {"a": SimResult(sim_id="a", energy=0.0, extras={"dump_file": str(d)})}
    tgt = TargetCfg.model_validate({
        "kind": "melting_onset", "weight": 1.0,
        "simulations": ["a"], "target": 100,
    })
    obj = build_objective(tgt)
    with pytest.raises(RuntimeError, match="temperature"):
        obj.compute(ObjectiveContext(sim_results=sims))
