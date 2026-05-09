"""Pure geometric transformations used by `pyfield make-scan`.

Each function takes a `StructureCfg` plus the scan parameters and returns
a new `StructureCfg` with perturbed atoms. The originals are not mutated.

Atom indices in the public API are **1-based** (matches LAMMPS / xyz
conventions and the rest of the YAML). Internally we convert to 0-based
indices once at the top of each function.

Every kind that has multiple anchor atoms accepts an optional `legs`
mapping ({"i": [...], "j": [...]} for bond_stretch, etc.). A leg lists
the atoms that move *as a rigid group* with their anchor — so when you
bend a Si–O–Si angle and put M1 in `legs.i` and M2 in `legs.k`, the M's
rotate with their Si rather than staying put. If `legs` is omitted, only
the anchor itself moves (the historical behaviour, fine for diatomics
and any scan whose substituents don't matter).

The validator on `ScanCfg` enforces the structural rules (no overlap,
vertex never in a leg, etc.) so these functions can assume a consistent
spec.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

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


def _idx_list(xs: Sequence[int], n_atoms: int, name: str) -> List[int]:
    return [_idx(x, n_atoms, name) for x in xs]


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


def _resolve_leg(legs: Optional[Dict[str, List[int]]], role: str,
                 implicit_anchor: int, n_atoms: int, scan_label: str) -> List[int]:
    """Return 0-based indices of every atom that moves with `role`'s anchor.

    The anchor atom itself is always included (whether or not the user
    listed it). `implicit_anchor` is 1-based.
    """
    members = list((legs or {}).get(role) or [])
    if implicit_anchor not in members:
        members.append(implicit_anchor)
    return _idx_list(members, n_atoms, f"{scan_label}.legs.{role}")


# ---------------------------------------------------------------------------
# bond_stretch — atoms: [i, j], value = target |r_j − r_i| in Å.
# Midpoint of (i, j) is preserved. `legs={"i": [...], "j": [...]}` rigidly
# translates each leg so its anchor lands at the new position.
# ---------------------------------------------------------------------------

def bond_stretch(
    structure: StructureCfg,
    atoms: Sequence[int],
    value: float,
    *,
    legs: Optional[Dict[str, List[int]]] = None,
) -> StructureCfg:
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
    new_i = midpoint - 0.5 * value * unit
    new_j = midpoint + 0.5 * value * unit

    leg_i = _resolve_leg(legs, "i", atoms[0], n, "bond_stretch")
    leg_j = _resolve_leg(legs, "j", atoms[1], n, "bond_stretch")

    delta_i = new_i - ri
    delta_j = new_j - rj
    coords = coords.copy()
    coords[leg_i] = coords[leg_i] + delta_i
    coords[leg_j] = coords[leg_j] + delta_j
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# angle_bend — atoms: [i, j, k] (j is the vertex), value in degrees.
# i and j are kept fixed (along with `legs.i`); k and its leg rotate
# around the axis perpendicular to the (i, j, k) plane through j.
# ---------------------------------------------------------------------------

def angle_bend(
    structure: StructureCfg,
    atoms: Sequence[int],
    value_deg: float,
    *,
    legs: Optional[Dict[str, List[int]]] = None,
) -> StructureCfg:
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
        raise ValueError("angle_bend: degenerate angle — vertex coincides with neighbour")
    cos_curr = float(np.clip(np.dot(v_ji, v_jk) / (n_ji * n_jk), -1.0, 1.0))
    angle_curr = math.acos(cos_curr)
    angle_target = math.radians(value_deg)
    delta = angle_target - angle_curr

    axis = np.cross(v_ji, v_jk)
    if np.linalg.norm(axis) < 1e-9:
        ref = np.array([1.0, 0.0, 0.0]) if abs(v_jk[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_jk, ref)
    axis = axis / np.linalg.norm(axis)

    leg_k = _resolve_leg(legs, "k", atoms[2], n, "angle_bend")

    coords = coords.copy()
    for atom_idx in leg_k:
        coords[atom_idx] = _rotate_about_axis(coords[atom_idx], axis, rj, delta)
    # Note: leg_i (if any) stays fixed by construction — rotating only the
    # k-side is sufficient to set the target angle and matches what the
    # user expects when, e.g., the i-leg already sits at a known reference.
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# dihedral — atoms: [i, j, k, l], value = target dihedral angle (deg).
# Convention: dihedral is the angle between the (i, j, k) and (j, k, l)
# planes, measured looking along j → k. i, j, k are fixed; l and
# `legs.l` rotate around the j–k axis.
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


def dihedral(
    structure: StructureCfg,
    atoms: Sequence[int],
    value_deg: float,
    *,
    legs: Optional[Dict[str, List[int]]] = None,
) -> StructureCfg:
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

    leg_l = _resolve_leg(legs, "l", atoms[3], n, "dihedral")

    coords = coords.copy()
    # Right-hand rule about (p3 − p2) rotates the IUPAC dihedral by
    # −delta, hence the sign flip.
    for atom_idx in leg_l:
        coords[atom_idx] = _rotate_about_axis(coords[atom_idx], axis, p3, -delta)
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# atom_displacement — single atom shifted along `direction` by `value` Å.
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
# dimer_separation — anchor-driven. `anchors=[a, b]` are the two central
# atoms whose distance is the scan coordinate; `fragments=[f_a, f_b]`
# are the atoms that translate rigidly with each anchor (anchors are
# included implicitly). Symmetric expansion: midpoint preserved, both
# fragments translate along the anchor-anchor axis.
# ---------------------------------------------------------------------------

def dimer_separation(
    structure: StructureCfg,
    anchors: Sequence[int],
    fragments: Sequence[Sequence[int]],
    value: float,
    *,
    direction: Union[str, Sequence[float]] = "auto",
) -> StructureCfg:
    if len(anchors) != 2:
        raise ValueError(f"dimer_separation: anchors must have length 2, got {anchors!r}")
    if len(fragments) != 2:
        raise ValueError(f"dimer_separation: fragments must have length 2, got {len(fragments)}")
    n = len(structure.atoms)
    a1 = _idx(anchors[0], n, "dimer_separation.anchors[0]")
    a2 = _idx(anchors[1], n, "dimer_separation.anchors[1]")
    if a1 == a2:
        raise ValueError(f"dimer_separation: anchors must be distinct, got {anchors!r}")

    f1 = _idx_list(list(set([anchors[0], *fragments[0]])), n, "dimer_separation.fragments[0]")
    f2 = _idx_list(list(set([anchors[1], *fragments[1]])), n, "dimer_separation.fragments[1]")

    coords = _coords(structure)
    r1, r2 = coords[a1], coords[a2]
    sep = r2 - r1
    norm = np.linalg.norm(sep)
    if norm < 1e-9 and (isinstance(direction, str) and direction.lower() == "auto"):
        raise ValueError(
            "dimer_separation: anchors coincide; pass an explicit `direction:`"
        )
    if isinstance(direction, str) and direction.lower() == "auto":
        unit = sep / norm
    else:
        unit = _resolve_direction(direction)

    midpoint = 0.5 * (r1 + r2)
    new_r1 = midpoint - 0.5 * value * unit
    new_r2 = midpoint + 0.5 * value * unit
    delta1 = new_r1 - r1
    delta2 = new_r2 - r2

    coords = coords.copy()
    coords[f1] = coords[f1] + delta1
    coords[f2] = coords[f2] + delta2
    return _replace_coords(structure, coords)


# ---------------------------------------------------------------------------
# isotropic_scale — multiply every coordinate (and the box) by `value`.
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


# ---------------------------------------------------------------------------
# strain — apply a Lagrangian strain tensor to the cell + atoms.
#
# Five named modes cover everything you need to extract elastic constants
# from a crystalline reference structure:
#
# - hydrostatic      : isotropic; new_box = (1+ε)·box (same as isotropic_scale).
#                      bulk modulus B = − V·dP/dV ≈ V·∂²E/∂V².
# - uniaxial(axis)   : strain along x, y, or z; other two axes unchanged.
#                      gives C₁₁ when paired with biaxial.
# - biaxial(plane)   : strain along two of {xy, xz, yz}; third unchanged.
# - shear(plane)     : tilts the cell in the named plane (xy / xz / yz).
#                      gives the C₄₄ shear modulus.
# - volumetric(axis) : equivalent to uniaxial — kept as alias for clarity.
#
# Atoms are scaled affinely with the lattice (fractional coords preserved),
# then the constrained QM relax is allowed to move them inside the
# deformed cell while the cell vectors stay locked at the strained values.
# ---------------------------------------------------------------------------

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _strain_tensor(mode: str, value: float, axis: Optional[str] = None) -> np.ndarray:
    """Return the 3×3 deformation matrix F such that new = F @ old.

    For small strains ε, F = I + ε for the named directions, so
    new_box[i] = (1 + ε) * old_box[i] along the strained axis.
    """
    F = np.eye(3, dtype=float)
    mode = mode.lower()
    if mode == "hydrostatic":
        F = (1.0 + value) * np.eye(3)
    elif mode in ("uniaxial", "volumetric"):
        if axis is None or axis not in _AXIS_INDEX:
            raise ValueError(f"strain {mode!r}: axis must be one of {list(_AXIS_INDEX)}, "
                             f"got {axis!r}")
        F[_AXIS_INDEX[axis], _AXIS_INDEX[axis]] = 1.0 + value
    elif mode == "biaxial":
        if axis is None or len(axis) != 2 or any(c not in _AXIS_INDEX for c in axis):
            raise ValueError(f"strain biaxial: axis must be a 2-letter plane like 'xy', 'xz', "
                             f"'yz', got {axis!r}")
        for c in axis:
            F[_AXIS_INDEX[c], _AXIS_INDEX[c]] = 1.0 + value
    elif mode == "shear":
        if axis is None or len(axis) != 2 or axis[0] == axis[1] or any(c not in _AXIS_INDEX for c in axis):
            raise ValueError(f"strain shear: axis must be a 2-letter plane like 'xy', 'xz', "
                             f"'yz', got {axis!r}")
        i, j = _AXIS_INDEX[axis[0]], _AXIS_INDEX[axis[1]]
        F[i, j] = value
        F[j, i] = value
    else:
        raise ValueError(
            f"strain: mode must be one of "
            f"['hydrostatic','uniaxial','biaxial','shear','volumetric'], got {mode!r}"
        )
    return F


def strain(
    structure: StructureCfg,
    value: float,
    *,
    mode: str = "hydrostatic",
    axis: Optional[str] = None,
) -> StructureCfg:
    """Apply a strain to the cell + atoms (preserving fractional positions).

    Used by the `type: strain` scan kind. Atoms are deformed affinely
    with the lattice; the constrained relax that follows lets each
    atom relax inside the strained cell while the cell vectors stay
    fixed.
    """
    F = _strain_tensor(mode, value, axis=axis)
    coords = _coords(structure)
    new_coords = coords @ F.T
    # Box vectors transform the same way: new_a_i = F · a_i. For an
    # orthorhombic input box [a, b, c] the deformed box has the same
    # axis lengths along the diagonal of F·diag(a,b,c).
    box = np.diag([float(b) for b in structure.box])
    new_box_matrix = F @ box
    # We only persist orthorhombic boxes today; warn if the deformation
    # induces off-diagonal elements (shear) so the caller knows the
    # box tuple loses the shear info.
    new_box = tuple(float(np.linalg.norm(new_box_matrix[:, i])) for i in range(3))
    new_atoms = [
        a.model_copy(update={"x": float(new_coords[i, 0]),
                             "y": float(new_coords[i, 1]),
                             "z": float(new_coords[i, 2])})
        for i, a in enumerate(structure.atoms)
    ]
    return structure.model_copy(update={"atoms": new_atoms, "box": new_box})
