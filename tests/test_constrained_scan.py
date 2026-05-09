"""Constrained-scan path: schema validation, expand_scans wiring, end-to-end
populate_qm with a fake backend that captures the constraint specs."""
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pyfield.config.loader import load_yaml
from pyfield.config.schema import PyFieldConfig
from pyfield.qm.base import QmBackend, QmRelaxResult, QmSinglePoint
from pyfield.qm.prep import cfg_to_yaml, populate_qm
from pyfield.scans import expand_scans


# ---------------------------------------------------------------------------
# Schema validation — leg conflicts must raise ValidationError with a
# message that names the offending atoms.
# ---------------------------------------------------------------------------

def _make_cfg(scan):
    """Minimal cfg with 8 atoms so we can test legs of length up to 4."""
    return {
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "X": {
                "box": [10, 10, 10],
                "atoms": [
                    {"element": "C", "x": float(i), "y": 0, "z": 0} for i in range(8)
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [scan],
    }


def test_bond_stretch_overlapping_legs_rejected():
    bad = _make_cfg({
        "type": "bond_stretch", "reference": "X", "name_prefix": "x",
        "atoms": [1, 2], "values": [1.5],
        "legs": {"i": [3, 4], "j": [3, 5]},     # atom 3 in both
    })
    with pytest.raises(ValidationError, match="overlap"):
        PyFieldConfig.model_validate(bad)


def test_angle_bend_vertex_in_leg_rejected():
    bad = _make_cfg({
        "type": "angle_bend", "reference": "X", "name_prefix": "x",
        "atoms": [1, 2, 3], "values": [90.0],
        "legs": {"i": [1, 2]},                  # vertex 2 inside leg.i
    })
    with pytest.raises(ValidationError, match="vertex"):
        PyFieldConfig.model_validate(bad)


def test_dihedral_middle_atom_in_leg_rejected():
    bad = _make_cfg({
        "type": "dihedral", "reference": "X", "name_prefix": "x",
        "atoms": [1, 2, 3, 4], "values": [90.0],
        "legs": {"l": [4, 3]},                  # atom 3 (k) inside leg.l
    })
    with pytest.raises(ValidationError, match=r"atoms\[2\]"):
        PyFieldConfig.model_validate(bad)


def test_dimer_separation_requires_anchors_and_fragments():
    bad = _make_cfg({
        "type": "dimer_separation", "reference": "X", "name_prefix": "x",
        "values": [3.0],
    })
    with pytest.raises(ValidationError, match="anchors"):
        PyFieldConfig.model_validate(bad)


def test_dimer_separation_overlapping_fragments_rejected():
    bad = _make_cfg({
        "type": "dimer_separation", "reference": "X", "name_prefix": "x",
        "anchors": [1, 5], "fragments": [[2, 3], [3, 6]],
        "values": [3.0],
    })
    with pytest.raises(ValidationError, match="overlap"):
        PyFieldConfig.model_validate(bad)


def test_isotropic_scale_cannot_be_relaxed_constrained():
    bad = _make_cfg({
        "type": "isotropic_scale", "reference": "X", "name_prefix": "x",
        "values": [1.0], "relax_method": "relaxed_constrained",
    })
    with pytest.raises(ValidationError, match="isotropic_scale"):
        PyFieldConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# expand_scans — relaxed_constrained scans emit minimize sims with restraints
# + tag the structures with a constraint spec.
# ---------------------------------------------------------------------------

def _basic_dimer_cfg(extra_scan_fields=None, *, cache_dir=None):
    """Two C atoms (would-be Si), each with one phantom O attached.

    `cache_dir` defaults to a per-test path; tests that share data across
    populate_qm invocations should pass the same directory."""
    qm = {"code": "pyscf"}
    if cache_dir is not None:
        qm["cache_dir"] = str(cache_dir)
    return PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": qm,
        "structures": {
            "Dim_Opt": {
                "box": [50, 50, 50],
                "atoms": [
                    {"element": "C", "x": 0,   "y": 0, "z": 0},
                    {"element": "O", "x": 0.5, "y": 0, "z": 0},
                    {"element": "C", "x": 3,   "y": 0, "z": 0},
                    {"element": "O", "x": 3.5, "y": 0, "z": 0},
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [{
            "type": "dimer_separation",
            "reference": "Dim_Opt",
            "name_prefix": "Dim_d",
            "anchors": [1, 3],
            "fragments": [[2], [4]],
            "values": [3.0, 5.0],
            "relax_method": "relaxed_constrained",
            "restraint_k": 1500.0,
            **(extra_scan_fields or {}),
        }],
    })


def test_relaxed_constrained_dimer_emits_minimize_with_restraints():
    cfg = _basic_dimer_cfg()
    expanded, _ = expand_scans(cfg)
    # Each scan point's sim is a minimize, not a single_point.
    sim0 = expanded.simulations["Dim_d_0_sp"]
    sim1 = expanded.simulations["Dim_d_1_sp"]
    assert sim0.type == "minimize"
    assert sim1.type == "minimize"
    # Restraints are on the bond between anchors at the per-point distance.
    r0 = sim0.__pydantic_extra__["restraints"]
    r1 = sim1.__pydantic_extra__["restraints"]
    assert r0 == ["bond 1 3 1500.0 1500.0 3.0"]
    assert r1 == ["bond 1 3 1500.0 1500.0 5.0"]


def test_relaxed_constrained_dimer_attaches_constraint_to_structure():
    cfg = _basic_dimer_cfg()
    expanded, _ = expand_scans(cfg)
    s0 = expanded.structures["Dim_d_0"]
    s1 = expanded.structures["Dim_d_1"]
    # qm-prep needs the constraint to drive the QM relax.
    assert s0.__pydantic_extra__["constraint"] == {
        "kind": "distance", "atoms": [1, 3], "value": 3.0,
    }
    assert s1.__pydantic_extra__["constraint"]["value"] == 5.0
    # And the structures are flagged for relax (qm-prep will pick them up).
    assert s0.qm_relax is True
    assert s1.qm_relax is True


def test_relaxed_constrained_round_trip_yaml(tmp_path):
    cfg = _basic_dimer_cfg()
    expanded, _ = expand_scans(cfg)
    out = tmp_path / "expanded.yaml"
    out.write_text(cfg_to_yaml(expanded))
    cfg2 = load_yaml(out)
    s = cfg2.structures["Dim_d_0"]
    assert s.qm_relax is True
    assert s.__pydantic_extra__["constraint"]["kind"] == "distance"


def test_angle_bend_relaxed_constrained_emits_angle_restraint():
    cfg = PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "SiOSi": {
                "box": [50, 50, 50],
                "atoms": [
                    {"element": "Si", "x": 1.6, "y": 0, "z": 0},
                    {"element": "O",  "x": 0,   "y": 0, "z": 0},
                    {"element": "Si", "x": -1.6, "y": 0, "z": 0},
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [{
            "type": "angle_bend",
            "reference": "SiOSi",
            "name_prefix": "SiOSi_a",
            "atoms": [1, 2, 3],
            "values": [120.0],
            "relax_method": "relaxed_constrained",
            "restraint_k": 1000.0,
        }],
    })
    expanded, _ = expand_scans(cfg)
    sim = expanded.simulations["SiOSi_a_0_sp"]
    assert sim.__pydantic_extra__["restraints"] == [
        "angle 1 2 3 1000.0 1000.0 120.0"
    ]


# ---------------------------------------------------------------------------
# End-to-end populate_qm: backend.relax must receive the constraint spec.
# ---------------------------------------------------------------------------

class _RecordingBackend(QmBackend):
    name = "recording"
    def __init__(self):
        self.relax_calls = []                    # list of (atom_count, constraint)

    def settings_fingerprint(self):
        return "recording-fp"

    def single_point(self, structure):
        return QmSinglePoint(energy_kcal_mol=0.0)

    def relax(self, structure, constraint=None):
        self.relax_calls.append((len(structure.atoms), constraint))
        return QmRelaxResult(structure=structure.model_copy(update={"qm_relax": False}),
                             energy_kcal_mol=-1.0)


def test_strain_scan_emits_minimize_without_restraints():
    """Strain's constraint is the strained cell itself — no `fix restrain`."""
    cfg = PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "X": {
                "box": [10, 10, 10], "pbc": True,
                "atoms": [
                    {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"element": "C", "x": 5.0, "y": 5.0, "z": 5.0},
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [{
            "type": "strain", "reference": "X", "name_prefix": "X_eos",
            "mode": "hydrostatic",
            "values": [-0.04, 0.0, 0.04],
            "relax_method": "relaxed_constrained",
        }],
    })
    expanded, _ = expand_scans(cfg)
    # Each scan-point structure inherits pbc=True from the reference.
    for i in range(3):
        s = expanded.structures[f"X_eos_{i}"]
        assert s.pbc is True
        assert s.qm_relax is True
        # No `constraint` field — strain has no internal-coord constraint.
        assert "constraint" not in (s.__pydantic_extra__ or {})
        # FF-side sim is `minimize` with no restraints.
        sim = expanded.simulations[f"X_eos_{i}_sp"]
        assert sim.type == "minimize"
        extras = sim.__pydantic_extra__ or {}
        assert not extras.get("restraints")


def test_strain_scan_box_actually_changes():
    cfg = PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "X": {
                "box": [10, 10, 10], "pbc": True,
                "atoms": [
                    {"element": "C", "x": 1.0, "y": 1.0, "z": 1.0},
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [{
            "type": "strain", "reference": "X", "name_prefix": "X_uni",
            "mode": "uniaxial", "axis": "z",
            "values": [-0.05, 0.05],
            "relax_method": "rigid",
        }],
    })
    expanded, _ = expand_scans(cfg)
    s_minus = expanded.structures["X_uni_0"]
    s_plus = expanded.structures["X_uni_1"]
    # Only z should differ from the original box.
    assert s_minus.box == (10.0, 10.0, 9.5)
    assert s_plus.box == (10.0, 10.0, 10.5)
    # The atom's z-coord scales too.
    assert s_minus.atoms[0].z == pytest.approx(0.95)
    assert s_plus.atoms[0].z == pytest.approx(1.05)


def test_populate_qm_forwards_constraint_to_backend(tmp_path):
    cfg = _basic_dimer_cfg(cache_dir=tmp_path / "cache")
    expanded, _ = expand_scans(cfg)
    backend = _RecordingBackend()
    populated, journal = populate_qm(expanded, backend=backend)
    # Two relaxes happened (one per scan point), each with its constraint.
    constraints = [c for _, c in backend.relax_calls]
    kinds = [c["kind"] if c else None for c in constraints]
    values = sorted(c["value"] for c in constraints if c)
    assert kinds == ["distance", "distance"]
    assert values == [3.0, 5.0]
    assert any("relax_constrained" in a for a, _, _ in journal)
