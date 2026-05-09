"""LammpsRunner: instance reuse + clear semantics. LAMMPS-marked."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


lammps = pytest.importorskip("lammps", reason="LAMMPS not installed/loadable")


@pytest.mark.lammps
def test_runner_reuses_one_lammps_instance(tmp_path):
    from pyfield.io.lammps import LammpsRunner

    in1 = tmp_path / "a.in"
    in2 = tmp_path / "b.in"
    # Trivial single-atom systems.
    in1.write_text(
        "units real\nboundary p p p\nregion box block 0 10 0 10 0 10\n"
        "create_box 1 box\nmass 1 1.0\ncreate_atoms 1 single 5 5 5\n"
        "thermo_style custom step etotal\nrun 0\n"
    )
    in2.write_text(
        "units real\nboundary p p p\nregion box block 0 20 0 20 0 20\n"
        "create_box 1 box\nmass 1 1.0\ncreate_atoms 1 single 10 10 10\n"
        "thermo_style custom step etotal\nrun 0\n"
    )

    with LammpsRunner() as runner:
        runner.run_input_file(in1)
        first_id = id(runner._instance())
        runner.run_input_file(in2)
        second_id = id(runner._instance())

    assert first_id == second_id, "runner must keep the same lammps() instance across calls"


@pytest.mark.lammps
def test_clear_resets_state_between_runs(tmp_path):
    """The second run must not inherit the atoms / box of the first."""
    from pyfield.io.lammps import LammpsRunner

    a = tmp_path / "a.in"
    b = tmp_path / "b.in"
    a.write_text(
        "units real\nboundary p p p\nregion box block 0 10 0 10 0 10\n"
        "create_box 1 box\nmass 1 1.0\n"
        "create_atoms 1 single 1 1 1\ncreate_atoms 1 single 2 2 2\ncreate_atoms 1 single 3 3 3\n"
        "thermo_style custom step etotal\nrun 0\n"
    )
    b.write_text(
        "units real\nboundary p p p\nregion box block 0 10 0 10 0 10\n"
        "create_box 1 box\nmass 1 1.0\ncreate_atoms 1 single 5 5 5\n"
        "thermo_style custom step etotal\nrun 0\n"
    )

    with LammpsRunner() as runner:
        runner.run_input_file(a)
        n_after_a = runner._instance().get_natoms()
        runner.run_input_file(b)
        n_after_b = runner._instance().get_natoms()

    assert n_after_a == 3
    assert n_after_b == 1, "clear should drop atoms from previous run"


@pytest.mark.lammps
def test_yaml_smoke_runs_with_long_lived_runner():
    """Phase 2 acceptance: YAML smoke uses a single lammps() across all sims
    and produces the same final cost as before."""
    from pyfield.config.loader import load_yaml
    from pyfield.optimizers.sa import run_sa

    cfg = load_yaml(os.path.join(ROOT, "tests/cl2.yaml"))
    cfg.output.dir = os.path.join(ROOT, "tests/runs/_phase2_runner_check")
    result = run_sa(cfg)
    assert 0 < result.final_cost < 1e9
    # Phase 2 must not change the answer (same seed, same inputs, same SA logic).
    # 2026-05-08: bumped from 32907.21505210572 after adding `min_modify dmax 0.05`
    # to minimize.in.j2 (caps per-step atomic motion at 50 mÅ — protects
    # unfit seed FFs from blowing the cell up on iteration 1, see GST drift
    # study). The dmax cap forces slightly different per-step kinematics,
    # producing 0.002 kcal/mol drift on the Cl2 minimize. Same converged
    # geometry, same physics — just a numerically-different convergence path.
    assert abs(result.final_cost - 32907.21744068185) < 1e-6
