"""NVT backend: render-only golden test (no LAMMPS) + a tiny end-to-end
LAMMPS run that confirms the dump file exists and the dump reader can
parse it.
"""
import os
import sys
from pathlib import Path

import pytest

from pyfield.config.schema import SimulationCfg, StructureCfg
from pyfield.simulations.nvt import NvtSimulation


def _struct():
    return StructureCfg(
        box=(10, 10, 10),
        atoms=[{"element": "Cl", "x": 0, "y": 0, "z": -1.0},
               {"element": "Cl", "x": 0, "y": 0, "z": 1.0}],
    )


def test_render_includes_required_lammps_directives():
    cfg = SimulationCfg.model_validate({
        "structure": "x",
        "type": "nvt",
        "temperature": 300,
        "steps": 100,
        "sample_every": 10,
        "timestep_fs": 0.25,
        "tdamp": 100.0,
        "seed": 7,
    })
    sim = NvtSimulation("sim1", cfg, _struct())
    text = sim.render(
        ffield_path=Path("ffield.reax"),
        elements=["Cl"],
        data_file=Path("sim1.data"),
        log_file=Path("sim1.log"),
        dump_file=Path("sim1.lammpstrj"),
    )
    # Basic shape: every required directive is present.
    for needle in [
        "pair_style reaxff",
        "fix qeq all qeq/reaxff",
        "velocity all create 300.0 7",
        "fix nvt_run all nvt temp 300.0 300.0 100.0",
        "dump traj all custom 10 sim1.lammpstrj",
        "run 100",
        "ffield.reax Cl",   # pair_coeff line with element appended
    ]:
        assert needle in text, needle


def test_render_requires_temperature():
    cfg = SimulationCfg.model_validate({"structure": "x", "type": "nvt"})
    sim = NvtSimulation("sim1", cfg, _struct())
    with pytest.raises(ValueError, match="temperature"):
        sim.render(
            ffield_path=Path("ff.reax"),
            elements=["Cl"],
            data_file=Path("d"),
            log_file=Path("l"),
            dump_file=Path("u"),
        )


lammps = pytest.importorskip("lammps", reason="LAMMPS not installed/loadable")


@pytest.mark.lammps
def test_nvt_runs_end_to_end_and_writes_dump(tmp_path):
    """One short NVT run must produce a parseable dump."""
    from pyfield.io.dump import read_dump
    from pyfield.io.lammps import LammpsRunner

    cfg = SimulationCfg.model_validate({
        "structure": "x",
        "type": "nvt",
        "temperature": 300,
        "steps": 50,
        "sample_every": 10,
        "tdamp": 25.0,
    })
    sim = NvtSimulation("sim1", cfg, _struct())
    with LammpsRunner() as runner:
        result = sim.run(
            ffield_path=Path(os.path.dirname(__file__)) / "ffieldoriginal.txt",
            work_dir=tmp_path,
            runner=runner,
        )
    dump_path = result.extras["dump_file"]
    assert Path(dump_path).exists()
    frames = list(read_dump(dump_path))
    # 50 steps / sample_every=10 => 6 frames (timesteps 0,10,20,30,40,50).
    assert len(frames) >= 5
    assert frames[0].columns == ["id", "type", "x", "y", "z", "q"]
