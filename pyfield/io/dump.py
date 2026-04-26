"""Streaming reader for LAMMPS native dump files.

The format we parse (orthogonal box, the most common case):

    ITEM: TIMESTEP
    <int>
    ITEM: NUMBER OF ATOMS
    <int>
    ITEM: BOX BOUNDS pp pp pp
    <xlo> <xhi>
    <ylo> <yhi>
    <zlo> <zhi>
    ITEM: ATOMS <col1> <col2> …
    <row1>
    <row2>
    …

Triclinic boxes (with `xy xz yz` on the `BOX BOUNDS` lines) are not
supported yet — they raise `NotImplementedError`. Phase 4+ adds them
when the first non-orthogonal-box objective lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


@dataclass
class DumpFrame:
    """One frame of a LAMMPS native dump."""
    timestep: int
    n_atoms: int
    box: Tuple[float, float, float]   # (lx, ly, lz)
    box_origin: Tuple[float, float, float]   # (xlo, ylo, zlo)
    columns: List[str]
    data: np.ndarray   # shape (n_atoms, len(columns)), dtype=float64

    def col(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]


def read_dump(path: str | Path) -> Iterator[DumpFrame]:
    """Yield each frame in `path`.

    Streaming: only one frame is held in memory at a time, so multi-GB
    trajectories work without exploding memory.
    """
    p = Path(path)
    with p.open("r") as f:
        while True:
            header = f.readline()
            if not header:
                return
            if not header.startswith("ITEM: TIMESTEP"):
                raise ValueError(f"{p}: expected 'ITEM: TIMESTEP', got {header!r}")
            timestep = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"{p}: expected 'ITEM: NUMBER OF ATOMS' line")
            n_atoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{p}: expected 'ITEM: BOX BOUNDS' line")
            # Distinguish triclinic ("ITEM: BOX BOUNDS xy xz yz pp pp pp") from
            # orthogonal ("ITEM: BOX BOUNDS pp pp pp"). Triclinic lines have 6
            # fields after "BOX BOUNDS"; orthogonal has 3.
            tokens = line.split()[2:]
            if "xy" in tokens or "xz" in tokens or "yz" in tokens:
                raise NotImplementedError(
                    "Triclinic dump boxes not supported yet — extend pyfield.io.dump "
                    "when the first non-orthogonal-box objective needs them."
                )
            xlo, xhi = (float(x) for x in f.readline().split())
            ylo, yhi = (float(x) for x in f.readline().split())
            zlo, zhi = (float(x) for x in f.readline().split())

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(f"{p}: expected 'ITEM: ATOMS' line")
            columns = line.split()[2:]

            data = np.empty((n_atoms, len(columns)), dtype=np.float64)
            for i in range(n_atoms):
                row = f.readline()
                if not row:
                    raise ValueError(f"{p}: dump truncated mid-frame at atom {i}/{n_atoms}")
                data[i] = [float(x) for x in row.split()]

            yield DumpFrame(
                timestep=timestep,
                n_atoms=n_atoms,
                box=(xhi - xlo, yhi - ylo, zhi - zlo),
                box_origin=(xlo, ylo, zlo),
                columns=columns,
                data=data,
            )
