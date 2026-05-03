"""Geometric correctness of the six scan transformations."""
import math

import numpy as np
import pytest

from pyfield.config.schema import AtomCfg, StructureCfg
from pyfield.scans import transforms as T


def _struct(coords, elements=None, box=(10.0, 10.0, 10.0)):
    if elements is None:
        elements = ["X"] * len(coords)
    atoms = [
        AtomCfg(element=e, x=float(x), y=float(y), z=float(z))
        for e, (x, y, z) in zip(elements, coords)
    ]
    return StructureCfg(box=box, atoms=atoms)


def _coords(s):
    return np.array([[a.x, a.y, a.z] for a in s.atoms])


# ---------------------------------------------------------------------------
# bond_stretch
# ---------------------------------------------------------------------------

def test_bond_stretch_preserves_midpoint():
    s = _struct([[0, 0, -1.0], [0, 0, 1.0]], ["Cl", "Cl"])
    s2 = T.bond_stretch(s, [1, 2], 2.5)
    c = _coords(s2)
    assert np.allclose(0.5 * (c[0] + c[1]), [0, 0, 0])
    assert np.isclose(np.linalg.norm(c[1] - c[0]), 2.5)


def test_bond_stretch_other_atoms_untouched():
    s = _struct([[0, 0, -1.0], [0, 0, 1.0], [3.0, 3.0, 3.0]],
                ["Cl", "Cl", "X"])
    s2 = T.bond_stretch(s, [1, 2], 2.0)
    assert s.atoms[2] == s2.atoms[2]


def test_bond_stretch_rejects_coincident():
    s = _struct([[0, 0, 0], [0, 0, 0]], ["X", "X"])
    with pytest.raises(ValueError, match="coincident"):
        T.bond_stretch(s, [1, 2], 1.5)


def test_bond_stretch_rejects_bad_atoms():
    s = _struct([[0, 0, 0], [0, 0, 1]], ["X", "X"])
    with pytest.raises(ValueError, match="length 2"):
        T.bond_stretch(s, [1], 1.5)


def test_bond_stretch_rejects_out_of_range_index():
    s = _struct([[0, 0, 0], [0, 0, 1]], ["X", "X"])
    with pytest.raises(ValueError, match="out of range"):
        T.bond_stretch(s, [1, 99], 1.5)


# ---------------------------------------------------------------------------
# angle_bend
# ---------------------------------------------------------------------------

def test_angle_bend_sets_target_angle():
    s = _struct([[1, 0, 0], [0, 0, 0], [0, 1, 0]], ["H", "O", "H"])
    s2 = T.angle_bend(s, [1, 2, 3], 104.5)
    c = _coords(s2)
    v1, v2 = c[0] - c[1], c[2] - c[1]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert np.isclose(math.degrees(math.acos(np.clip(cos, -1, 1))), 104.5, atol=1e-6)


def test_angle_bend_keeps_vertex_and_first_atom_fixed():
    s = _struct([[1, 0, 0], [0, 0, 0], [0, 1, 0]], ["H", "O", "H"])
    s2 = T.angle_bend(s, [1, 2, 3], 60.0)
    assert s2.atoms[0] == s.atoms[0]
    assert s2.atoms[1] == s.atoms[1]


def test_angle_bend_handles_collinear_start():
    """A collinear i-j-k should still produce a valid rotated geometry."""
    s = _struct([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], ["H", "O", "H"])
    s2 = T.angle_bend(s, [1, 2, 3], 90.0)
    c = _coords(s2)
    v1, v2 = c[0] - c[1], c[2] - c[1]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert np.isclose(math.degrees(math.acos(np.clip(cos, -1, 1))), 90.0, atol=1e-6)


# ---------------------------------------------------------------------------
# dihedral
# ---------------------------------------------------------------------------

def test_dihedral_sets_target_angle():
    # Start at ~30° dihedral and rotate to 90°.
    s = _struct(
        [[0, 1, 0.5], [0, 0, 0], [1, 0, 0], [1, 1, -0.5]],
        ["H", "O", "O", "H"],
    )
    s2 = T.dihedral(s, [1, 2, 3, 4], 90.0)
    c = _coords(s2)
    angle = T._dihedral_angle(c[0], c[1], c[2], c[3])
    assert np.isclose(math.degrees(angle), 90.0, atol=1e-6)


def test_dihedral_only_moves_last_atom():
    s = _struct(
        [[0, 1, 0.5], [0, 0, 0], [1, 0, 0], [1, 1, -0.5]],
        ["H", "O", "O", "H"],
    )
    s2 = T.dihedral(s, [1, 2, 3, 4], -45.0)
    for i in (0, 1, 2):
        assert s2.atoms[i] == s.atoms[i]


# ---------------------------------------------------------------------------
# atom_displacement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dirstr,expected", [
    ("x", [0.5, 0, 0]),
    ("y", [0, 0.5, 0]),
    ("z", [0, 0, 0.5]),
])
def test_atom_displacement_axis_aliases(dirstr, expected):
    s = _struct([[0, 0, 0]])
    s2 = T.atom_displacement(s, 1, dirstr, 0.5)
    c = _coords(s2)
    assert np.allclose(c[0], expected)


def test_atom_displacement_vector_direction_normalises():
    s = _struct([[0, 0, 0]])
    s2 = T.atom_displacement(s, 1, [1.0, 1.0, 0.0], 1.0)
    c = _coords(s2)
    assert np.allclose(c[0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0])


def test_atom_displacement_rejects_zero_vector():
    s = _struct([[0, 0, 0]])
    with pytest.raises(ValueError, match="zero length"):
        T.atom_displacement(s, 1, [0, 0, 0], 1.0)


# ---------------------------------------------------------------------------
# dimer_separation
# ---------------------------------------------------------------------------

def test_dimer_separation_auto_direction():
    s = _struct([[0, 0, 0], [2, 0, 0]], ["Cl", "Cl"])
    s2 = T.dimer_separation(s, anchors=[1, 2], fragments=[[], []], value=5.0)
    c = _coords(s2)
    # Anchor-anchor distance hits target; midpoint preserved.
    assert np.isclose(np.linalg.norm(c[1] - c[0]), 5.0)
    assert np.allclose(0.5 * (c[0] + c[1]), [1.0, 0, 0])


def test_dimer_separation_preserves_internal_geometry():
    """Translating fragments must keep their internal bonds intact."""
    s = _struct(
        [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]],
        ["O", "H", "O", "H"],
    )
    # anchors = the two O's (atoms 1, 3); H of each goes with its O.
    s2 = T.dimer_separation(s, anchors=[1, 3], fragments=[[2], [4]], value=6.0)
    c = _coords(s2)
    # Internal H-O distance preserved within each fragment.
    assert np.isclose(np.linalg.norm(c[1] - c[0]), 1.0)
    assert np.isclose(np.linalg.norm(c[3] - c[2]), 1.0)
    # Anchor-anchor distance hits target.
    assert np.isclose(np.linalg.norm(c[2] - c[0]), 6.0)


def test_dimer_separation_fragments_translate_with_anchors():
    """Si(OH)4-like: Si-Si separation drags every OH along with its Si."""
    # 2 Si's + 4 dummy O's per Si, all on the x-axis around the Si.
    s = _struct(
        [[0, 0, 0], [0.5, 0, 0],     # Si1, O attached to Si1
         [3, 0, 0], [3.5, 0, 0]],    # Si2, O attached to Si2
        ["Si", "O", "Si", "O"],
    )
    s2 = T.dimer_separation(s, anchors=[1, 3], fragments=[[2], [4]], value=5.0)
    c = _coords(s2)
    # Si-Si distance hits target.
    assert np.isclose(np.linalg.norm(c[2] - c[0]), 5.0)
    # Each O stays at +0.5 from its Si on the x-axis.
    assert np.isclose(c[1][0] - c[0][0], 0.5)
    assert np.isclose(c[3][0] - c[2][0], 0.5)


def test_bond_stretch_with_legs_drags_substituents():
    """Methane-like C-C stretch: H's of each C move with their C."""
    # C1, C2, plus a phantom H attached to each.
    s = _struct(
        [[0, 0, 0], [1.5, 0, 0],     # C1, C2
         [-0.3, 0.5, 0], [1.8, 0.5, 0]],   # H on C1, H on C2
        ["C", "C", "H", "H"],
    )
    s2 = T.bond_stretch(s, [1, 2], 2.0, legs={"i": [3], "j": [4]})
    c = _coords(s2)
    # C-C stretched to 2.0; midpoint preserved at (0.75, 0, 0).
    assert np.isclose(np.linalg.norm(c[1] - c[0]), 2.0)
    assert np.allclose(0.5 * (c[0] + c[1]), [0.75, 0, 0])
    # Each H rides with its C.
    assert np.allclose(c[2] - c[0], [-0.3, 0.5, 0])
    assert np.allclose(c[3] - c[1], [0.3, 0.5, 0])


def test_angle_bend_rotates_leg_k():
    """Si–O–Si bend: M2 rotates around O along with Si2 (its anchor)."""
    # Si1 - O - Si2, with M2 attached to Si2.
    s = _struct(
        [[1.6, 0, 0],   # Si1 (i)
         [0, 0, 0],     # O   (j) — vertex
         [-1.6, 0, 0],  # Si2 (k)
         [-2.6, 0, 0]], # M2 attached to Si2 (in legs.k)
        ["Si", "O", "Si", "M"],
    )
    s2 = T.angle_bend(s, [1, 2, 3], 90.0, legs={"k": [4]})
    c = _coords(s2)
    # Si-O-Si is now 90°.
    v1 = c[0] - c[1]; v2 = c[2] - c[1]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert np.isclose(np.degrees(np.arccos(np.clip(cos_a, -1, 1))), 90.0)
    # Si1 (i, no leg) stayed put.
    assert np.allclose(c[0], [1.6, 0, 0])
    # M2's distance from Si2 is preserved (rigid translation under rotation).
    assert np.isclose(np.linalg.norm(c[3] - c[2]), 1.0)
    # M2 is on the far side of Si2 from O — still collinear with Si2-O reflected.
    along = (c[3] - c[2]) / np.linalg.norm(c[3] - c[2])
    assert np.allclose(along, (c[2] - c[1]) / np.linalg.norm(c[2] - c[1]))


# ---------------------------------------------------------------------------
# isotropic_scale
# ---------------------------------------------------------------------------

def test_isotropic_scale_scales_atoms_and_box():
    s = _struct([[1, 2, 3]], box=(10, 10, 10))
    s2 = T.isotropic_scale(s, 1.5)
    c = _coords(s2)
    assert np.allclose(c[0], [1.5, 3.0, 4.5])
    assert s2.box == (15.0, 15.0, 15.0)


def test_isotropic_scale_rejects_nonpositive():
    s = _struct([[0, 0, 0]])
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be > 0"):
            T.isotropic_scale(s, bad)
