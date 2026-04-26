"""Minimal XYZ reader. Used by reference-structure objectives.

Format (standard XYZ):
    <N>
    <comment>
    <El> <x> <y> <z>
    ...

Returns (elements, coords) where coords is shape (N, 3) float64.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_xyz(path: str | Path) -> Tuple[List[str], np.ndarray]:
    p = Path(path)
    lines = p.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{p}: not a valid xyz (too short)")
    n = int(lines[0].strip())
    if len(lines) < 2 + n:
        raise ValueError(f"{p}: header says {n} atoms but file has {len(lines) - 2} rows")
    elements: List[str] = []
    coords = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        tok = lines[2 + i].split()
        if len(tok) < 4:
            raise ValueError(f"{p}: atom row {i} has too few columns: {tok!r}")
        elements.append(tok[0])
        coords[i] = [float(tok[1]), float(tok[2]), float(tok[3])]
    return elements, coords
