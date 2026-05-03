"""Smoke test for `pyfield.viz.animate_xyz_dir` — runs headless."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")    # headless

from pyfield.viz import _parse_xyz, animate_xyz_dir


def _write_xyz(path: Path, coords) -> None:
    lines = [str(len(coords)), "test"]
    for el, (x, y, z) in coords:
        lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_parse_xyz_round_trip(tmp_path):
    p = tmp_path / "f.xyz"
    _write_xyz(p, [("Cl", (0, 0, -1.0)), ("Cl", (0, 0, 1.0))])
    elements, coords, comment = _parse_xyz(p)
    assert elements == ["Cl", "Cl"]
    assert comment == "test"
    assert coords.shape == (2, 3)


def test_animate_handles_three_frame_directory(tmp_path):
    for i, d in enumerate([1.0, 1.5, 2.0]):
        _write_xyz(tmp_path / f"Cl2_{i}.xyz",
                   [("Cl", (0, 0, -d / 2)), ("Cl", (0, 0, d / 2))])
    out = animate_xyz_dir(tmp_path, interval_ms=10)
    # Either a Jupyter HTML or a raw FuncAnimation — both have the
    # animation reachable.
    anim = getattr(out, "anim", out)
    assert anim is not None
    # 3 frames, sorted by trailing index.
    assert anim._save_count == 3 or len(list(anim.new_frame_seq())) == 3
