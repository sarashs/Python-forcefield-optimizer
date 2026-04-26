"""structural_match objective + Kabsch RMSD. No LAMMPS."""
import os
from pathlib import Path

import numpy as np
import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.objectives.structural_match import kabsch_rmsd
from pyfield.simulations.base import SimResult


def _write_dump(path: Path, coords: np.ndarray, types=None):
    if types is None:
        types = [1] * len(coords)
    rows = "\n".join(
        f"{i+1} {types[i]} {c[0]} {c[1]} {c[2]} 0.0"
        for i, c in enumerate(coords)
    )
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n"
        f"{len(coords)}\nITEM: BOX BOUNDS pp pp pp\n"
        "0 100\n0 100\n0 100\n"
        "ITEM: ATOMS id type x y z q\n" + rows + "\n"
    )


def _write_xyz(path: Path, elements, coords):
    lines = [str(len(coords)), ""]
    for el, c in zip(elements, coords):
        lines.append(f"{el} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")


def test_kabsch_rmsd_translation_invariant():
    a = np.random.RandomState(0).rand(5, 3)
    b = a + np.array([10, -3, 7])
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)


def test_kabsch_rmsd_rotation_invariant():
    a = np.random.RandomState(1).rand(8, 3)
    theta = 0.7
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    b = a @ R.T
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)


def test_kabsch_rmsd_known_distance():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    b = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    # Centred, unit-vector difference is 0.5; RMSD = 0.5
    assert kabsch_rmsd(a, b) == pytest.approx(0.5)


def test_structural_match_rmsd_zero_when_identical(tmp_path):
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    dump = tmp_path / "sim1.lammpstrj"
    ref = tmp_path / "ref.xyz"
    _write_dump(dump, coords)
    _write_xyz(ref, ["H", "H", "H"], coords)

    tgt = TargetCfg.model_validate({
        "kind": "structural_match", "weight": 2.0, "simulation": "sim1",
        "reference": str(ref),
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, extras={"dump_file": str(dump)})
    })
    assert obj.compute(ctx) == pytest.approx(0.0, abs=1e-10)
    assert obj.residual(ctx) == pytest.approx(0.0, abs=1e-10)


def test_structural_match_bond_lengths(tmp_path):
    sim_coords = np.array([[0, 0, 0], [1.0, 0, 0]], dtype=float)
    ref_coords = np.array([[0, 0, 0], [2.0, 0, 0]], dtype=float)
    dump = tmp_path / "s.lammpstrj"; ref = tmp_path / "r.xyz"
    _write_dump(dump, sim_coords)
    _write_xyz(ref, ["H", "H"], ref_coords)

    tgt = TargetCfg.model_validate({
        "kind": "structural_match", "weight": 1.0, "simulation": "sim1",
        "reference": str(ref), "metric": "bond_lengths",
        "pairs": [[1, 2]],
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, extras={"dump_file": str(dump)})
    })
    # |1-2|^2 = 1.0
    assert obj.compute(ctx) == pytest.approx(1.0)


def test_structural_match_angles(tmp_path):
    sim_coords = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=float)  # 90°
    ref_coords = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]], dtype=float)  # 180°
    dump = tmp_path / "s.lammpstrj"; ref = tmp_path / "r.xyz"
    _write_dump(dump, sim_coords)
    _write_xyz(ref, ["H", "H", "H"], ref_coords)

    tgt = TargetCfg.model_validate({
        "kind": "structural_match", "weight": 1.0, "simulation": "sim1",
        "reference": str(ref), "metric": "angles",
        "triples": [[1, 2, 3]],
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, extras={"dump_file": str(dump)})
    })
    # (π/2 - π)^2 = (π/2)^2
    assert obj.compute(ctx) == pytest.approx((np.pi / 2) ** 2)


def test_structural_match_atom_count_mismatch(tmp_path):
    sim_coords = np.zeros((3, 3))
    ref_coords = np.zeros((4, 3))
    dump = tmp_path / "s.lammpstrj"; ref = tmp_path / "r.xyz"
    _write_dump(dump, sim_coords)
    _write_xyz(ref, ["H", "H", "H", "H"], ref_coords)
    tgt = TargetCfg.model_validate({
        "kind": "structural_match", "weight": 1.0, "simulation": "sim1",
        "reference": str(ref),
    })
    obj = build_objective(tgt)
    ctx = ObjectiveContext(sim_results={
        "sim1": SimResult(sim_id="sim1", energy=0.0, extras={"dump_file": str(dump)})
    })
    with pytest.raises(ValueError, match="atoms"):
        obj.compute(ctx)
