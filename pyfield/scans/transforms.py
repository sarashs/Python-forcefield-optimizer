"""Pure geometric transformations used by `pyfield make-scan`.

Each function takes a `StructureCfg` plus the scan parameters and returns
a new `StructureCfg` with perturbed atoms. The originals are not mutated.

Atom indices in the public API are **1-based** (matches LAMMPS / xyz
conventions and the rest of the YAML). Internally we convert to 0-based
indices once at the top of each function.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import numpy as np

from pyfield.config.schema import AtomCfg, StructureCfg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coords(structure: StructureCfg) -> np.ndarray:
    """Return an (N, 3) float array of atom positions."""
    return np.array([[a.x, a.y, a.z] for a in structure.atoms], dtype=float)


def _replace_coords(structure: StructureCfg, coords: np.ndarray) -> StructureCfg:
    new_atoms = [
        a.model_copy(update={"x": float(coords[i, 0]),
                             "y": float(coords[i, 1]),
                             "z": float(coords[i, 2])})
        for i, a in enumerate(structure.atoms)
    ]
    return structure.model_copy(update={"atoms": new_atoms})


def _idx(i: int, n_atoms: int, name: str) -> int:
    if not (1 <= i <= n_atoms):
        raise ValueError(f"{name}: 1-based atom index {i} out of range [1, {n_atoms}]")
    return i - 1


def _resolve_direction(d: Union[str, Sequence[float]]) -> np.ndarray:
    if isinstance(d, str):
        if d.lower() == "x": return np.array([1.0, 0.0, 0.0])
        if d.lower() == "y": return np.array([0.0, 1.0, 0.0])
        if d.lower() == "z": return np.array([0.0, 0.0, 1.0])
        raise ValueError(f"direction string must be 'x'/'y'/'z' or a 3-vector, got {d!r}")
    v = np.asarray(d, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"direction vector has zero length: {d!r}")
    return v / n


def _rotate_about_axis(point: np.ndarray, axis: np.ndarray, origin: np.ndarray,
                       angle_rad: float) -> np.ndarray:
    """Rodrigues rotation: rotate `point` by `angle_rad` around the axis
    `axis` (unit) passing through `origin`."""
    p = point - origin
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rotated = p * c + np.cross(axis, p) * s + axis * np.dot(axis, p) * (1 - c)
    return rotated + origin


# ---------------------------------------------------------------------------
# bond_stretch — atoms: [i, j], value = target |r_j − r_i| in Å.
# Midpoint of (i, j) is preserved; both atoms move along the bond axis.
# ---------------------------------------------------------------------------

def bond_stretch(structure: StructureCfg, atoms: Sequence[int], value: float) -> StructureCfg:
    if len(atoms) != 2:
        raise ValueError(f"bond_stretch: atoms must have length 2, got {atoms!r}")
    n = len(structure.atoms)
    i, j = _idx(atoms[0], n, "bond_stretch.atoms[0]"), _idx(atoms[1], n, "bond_stretch.atoms[1]")
    coords = _coords(structure)
    ri, rj = coords[i], coords[j]
    midpoint = 0.5 * (ri + rj)
    axis = rj - ri
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError(f"bond_stretch: atoms {atoms!r} are coincident — can't define bond axis")
    unit = axis / norm
    coords = coords.copy()
    coords[i] = midpoint - 0.5 * value * unit
    coords[j] = midpoint + 0.5 * value * unit
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# angle_bend — atoms: [i, j, k] (j is the vertex), value in degrees.
# i and j are kept fixed; k is rotated around the axis perpendicular to
# the (i, j, k) plane passing through j until the angle equals `value`.
# ---------------------------------------------------------------------------

def angle_bend(structure: StructureCfg, atoms: Sequence[int], value_deg: float) -> StructureCfg:
    if len(atoms) != 3:
        raise ValueError(f"angle_bend: atoms must have length 3, got {atoms!r}")
    n = len(structure.atoms)
    i, j, k = (_idx(atoms[0], n, "angle_bend.atoms[0]"),
               _idx(atoms[1], n, "angle_bend.atoms[1]"),
               _idx(atoms[2], n, "angle_bend.atoms[2]"))
    coords = _coords(structure)
    ri, rj, rk = coords[i], coords[j], coords[k]
    v_ji = ri - rj
    v_jk = rk - rj
    n_ji = np.linalg.norm(v_ji); n_jk = np.linalg.norm(v_jk)
    if n_ji < 1e-12 or n_jk < 1e-12:
        raise ValueError(f"angle_bend: degenerate angle — vertex coincides with neighbour")
    cos_curr = float(np.clip(np.dot(v_ji, v_jk) / (n_ji * n_jk), -1.0, 1.0))
    angle_curr = math.acos(cos_curr)
    angle_target = math.radians(value_deg)
    delta = angle_target - angle_curr
    # Rotation axis = v_ji × v_jk (perpendicular to the plane). For a
    # collinear triple we fall back to any perpendicular to v_jk.
    axis = np.cross(v_ji, v_jk)
    if np.linalg.norm(axis) < 1e-9:
        # pick any axis perpendicular to v_jk
        ref = np.array([1.0, 0.0, 0.0]) if abs(v_jk[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_jk, ref)
    axis = axis / np.linalg.norm(axis)
    coords = coords.copy()
    coords[k] = _rotate_about_axis(rk, axis, rj, delta)
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# dihedral — atoms: [i, j, k, l], value = target dihedral angle (deg).
# Convention: dihedral is the angle between the (i, j, k) and (j, k, l)
# planes, measured looking along j → k. i, j, k are fixed; l is rotated
# around the j–k axis.
# ---------------------------------------------------------------------------

def _dihedral_angle(p1, p2, p3, p4) -> float:
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return math.atan2(y, x)


def dihedral(structure: StructureCfg, atoms: Sequence[int], value_deg: float) -> StructureCfg:
    if len(atoms) != 4:
        raise ValueError(f"dihedral: atoms must have length 4, got {atoms!r}")
    n = len(structure.atoms)
    i, j, k, l = (_idx(atoms[0], n, "dihedral.atoms[0]"),
                  _idx(atoms[1], n, "dihedral.atoms[1]"),
                  _idx(atoms[2], n, "dihedral.atoms[2]"),
                  _idx(atoms[3], n, "dihedral.atoms[3]"))
    coords = _coords(structure)
    p1, p2, p3, p4 = coords[i], coords[j], coords[k], coords[l]
    curr = _dihedral_angle(p1, p2, p3, p4)
    target = math.radians(value_deg)
    delta = target - curr
    axis = p3 - p2
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("dihedral: middle atoms are coincident")
    axis = axis / norm
    # Rodrigues with right-hand rule about (p3 − p2) rotates the IUPAC
    # dihedral by −delta, hence the sign flip below.
    coords = coords.copy()
    coords[l] = _rotate_about_axis(p4, axis, p3, -delta)
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# atom_displacement — single atom shifted along `direction` by `value` Å.
# All other atoms stay put. Useful for vacancy / surface displacement scans.
# ---------------------------------------------------------------------------

def atom_displacement(structure: StructureCfg, atom: int,
                      direction: Union[str, Sequence[float]], value: float) -> StructureCfg:
    n = len(structure.atoms)
    a = _idx(atom, n, "atom_displacement.atom")
    unit = _resolve_direction(direction)
    coords = _coords(structure).copy()
    coords[a] = coords[a] + value * unit
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# dimer_separation — fragments: [[…], […]] of 1-based indices, value =
# target COM-COM distance along `direction` ('auto' or 3-vec). Fragment 1
# is held fixed; fragment 2 is translated along `direction` so the new
# COM-COM distance equals `value`.
# ---------------------------------------------------------------------------

def dimer_separation(structure: StructureCfg, fragments: Sequence[Sequence[int]],
                     direction: Union[str, Sequence[float]], value: float) -> StructureCfg:
    if len(fragments) != 2:
        raise ValueError(f"dimer_separation: fragments must have length 2, got {len(fragments)}")
    n = len(structure.atoms)
    f1 = [_idx(x, n, "dimer_separation.fragments[0]") for x in fragments[0]]
    f2 = [_idx(x, n, "dimer_separation.fragments[1]") for x in fragments[1]]
    if not f1 or not f2:
        raise ValueError("dimer_separation: each fragment must list at least one atom")
    coords = _coords(structure)
    com1 = coords[f1].mean(axis=0)
    com2 = coords[f2].mean(axis=0)
    if isinstance(direction, str) and direction.lower() == "auto":
        sep = com2 - com1
        if np.linalg.norm(sep) < 1e-9:
            raise ValueError("dimer_separation: fragments coincide; pass an explicit direction")
        unit = sep / np.linalg.norm(sep)
    else:
        unit = _resolve_direction(direction)
    new_com2 = com1 + value * unit
    delta = new_com2 - com2
    coords = coords.copy()
    coords[f2] = coords[f2] + delta
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# isotropic_scale — multiply every coordinate (and the box) by `value`.
# Standard EOS / cell-volume scan. Atom positions scale relative to the
# box origin (not COM) so the fractional coordinates of each atom are
# preserved.
# ---------------------------------------------------------------------------

def isotropic_scale(structure: StructureCfg, value: float) -> StructureCfg:
    if value <= 0:
        raise ValueError(f"isotropic_scale: scale factor must be > 0, got {value}")
    coords = _coords(structure) * value
    new_box = tuple(b * value for b in structure.box)
    new_atoms = [
        a.model_copy(update={"x": float(coords[i, 0]),
                             "y": float(coords[i, 1]),
                             "z": float(coords[i, 2])})
        for i, a in enumerate(structure.atoms)
    ]
    return structure.model_copy(update={"atoms": new_atoms, "box": new_box})
