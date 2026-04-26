"""LAMMPS dump frame reader. No LAMMPS."""
import io
import os
from pathlib import Path

import numpy as np
import pytest

from pyfield.io.dump import read_dump


_TWO_FRAMES = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z q
1 1 1.0 1.0 1.0 0.5
2 1 9.0 9.0 9.0 -0.5
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z q
1 1 1.1 1.0 1.0 0.4
2 1 8.9 9.0 9.0 -0.4
"""

_TRICLINIC = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 10 0
0 10 0
0 10 0
ITEM: ATOMS id type x y z
1 1 0 0 0
"""


def test_read_two_frames(tmp_path: Path):
    p = tmp_path / "two.lammpstrj"
    p.write_text(_TWO_FRAMES)
    frames = list(read_dump(p))
    assert len(frames) == 2
    f0, f1 = frames
    assert f0.timestep == 0
    assert f1.timestep == 100
    assert f0.n_atoms == f1.n_atoms == 2
    assert f0.columns == ["id", "type", "x", "y", "z", "q"]
    assert f0.box == (10, 10, 10)
    np.testing.assert_array_equal(f0.col("id"), [1, 2])
    np.testing.assert_allclose(f0.col("x"), [1.0, 9.0])
    np.testing.assert_allclose(f1.col("x"), [1.1, 8.9])


def test_streaming_does_not_load_everything(tmp_path: Path):
    """Generator: pulling one frame must not require having read the rest."""
    p = tmp_path / "two.lammpstrj"
    p.write_text(_TWO_FRAMES)
    it = read_dump(p)
    first = next(it)
    assert first.timestep == 0
    # Closing the iterator without exhausting it should not raise.
    it.close()


def test_triclinic_rejected(tmp_path: Path):
    p = tmp_path / "tri.lammpstrj"
    p.write_text(_TRICLINIC)
    with pytest.raises(NotImplementedError):
        list(read_dump(p))


def test_truncation_rejected(tmp_path: Path):
    """Header-only / mid-frame truncation is reported, not silently skipped."""
    p = tmp_path / "trunc.lammpstrj"
    p.write_text(_TWO_FRAMES.split("ITEM: ATOMS id type x y z q\n", 1)[0]
                 + "ITEM: ATOMS id type x y z q\n1 1 1.0 1.0 1.0 0.5\n")
    with pytest.raises(ValueError):
        list(read_dump(p))
