"""End-to-end tests for `expand_scans` (the `make-scan` engine)."""
from pathlib import Path

import pytest

from pyfield.config.loader import load_yaml
from pyfield.config.schema import PyFieldConfig
from pyfield.qm.prep import cfg_to_yaml
from pyfield.scans import expand_scans


def _cl2_with_scan(tmp_path, scan):
    return PyFieldConfig.model_validate({
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "qm": {"code": "pyscf", "functional": "lda", "basis": "sto-3g"},
        "structures": {
            "Cl2_Opt": {
                "box": [50, 50, 50],
                "atoms": [
                    {"element": "Cl", "x": 0, "y": 0, "z": -1.0},
                    {"element": "Cl", "x": 0, "y": 0, "z": +1.0},
                ],
            },
        },
        "simulations": {},
        "targets": [],
        "scans": [scan],
    })


def test_bond_stretch_scan_expansion(tmp_path):
    cfg = _cl2_with_scan(tmp_path, {
        "type": "bond_stretch",
        "reference": "Cl2_Opt",
        "name_prefix": "Cl2_d",
        "atoms": [1, 2],
        "values": [1.5, 2.0, 2.5],
    })
    expanded, summary = expand_scans(cfg, xyz_dir=tmp_path / "scan_xyz")

    # Three new structures + the reference.
    assert set(expanded.structures) == {"Cl2_Opt", "Cl2_d_0", "Cl2_d_1", "Cl2_d_2"}
    # Reference single_point + three scan single_points.
    assert set(expanded.simulations) == {
        "Cl2_Opt_sp", "Cl2_d_0_sp", "Cl2_d_1_sp", "Cl2_d_2_sp",
    }
    # Three energy_combination targets, all `from: dft`.
    assert len(expanded.targets) == 3
    for tgt in expanded.targets:
        extras = tgt.__pydantic_extra__
        assert tgt.kind == "energy_combination"
        assert extras["target"] == {"from": "dft"}
        assert extras["terms"]["Cl2_Opt_sp"] == -1
    # scans block stripped from output.
    assert expanded.scans is None
    # xyz files written.
    written = sorted((tmp_path / "scan_xyz").glob("*.xyz"))
    assert {p.name for p in written} == {
        "Cl2_Opt.xyz", "Cl2_d_0.xyz", "Cl2_d_1.xyz", "Cl2_d_2.xyz",
    }
    # Bond length in the middle frame matches what we asked for.
    middle = (tmp_path / "scan_xyz" / "Cl2_d_1.xyz").read_text().splitlines()
    z = [float(line.split()[3]) for line in middle[2:4]]
    assert abs(abs(z[1] - z[0]) - 2.0) < 1e-9


def test_make_scan_preserves_hand_typed_targets(tmp_path):
    cfg = PyFieldConfig.model_validate({
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "Cl2_Opt": {
                "box": [50, 50, 50],
                "atoms": [
                    {"element": "Cl", "x": 0, "y": 0, "z": -1.0},
                    {"element": "Cl", "x": 0, "y": 0, "z": +1.0},
                ],
            },
        },
        "simulations": {
            "Cl2_Opt_sp": {"structure": "Cl2_Opt", "type": "single_point"},
            "Empirical_sp": {"structure": "Cl2_Opt", "type": "single_point"},
        },
        "targets": [
            {"kind": "energy_combination", "weight": 5.0,
             "terms": {"Empirical_sp": +1, "Cl2_Opt_sp": -1},
             "target": 81.394},
        ],
        "scans": [{
            "type": "bond_stretch",
            "reference": "Cl2_Opt",
            "name_prefix": "Cl2_d",
            "atoms": [1, 2],
            "range": [1.5, 2.5, 3],
        }],
    })
    expanded, _ = expand_scans(cfg)
    # Hand-typed first; scans appended.
    assert expanded.targets[0].__pydantic_extra__["target"] == 81.394
    assert len(expanded.targets) == 4
    # Existing `Cl2_Opt_sp` reused (not duplicated).
    assert "Cl2_Opt_sp" in expanded.simulations


def test_make_scan_yaml_round_trip(tmp_path):
    cfg = _cl2_with_scan(tmp_path, {
        "type": "bond_stretch",
        "reference": "Cl2_Opt",
        "name_prefix": "Cl2_d",
        "atoms": [1, 2],
        "values": [1.5, 2.0],
    })
    expanded, _ = expand_scans(cfg, xyz_dir=tmp_path / "scan_xyz")
    yml_path = tmp_path / "scanned.yaml"
    yml_path.write_text(cfg_to_yaml(expanded))
    cfg2 = load_yaml(yml_path)
    assert set(cfg2.structures) == set(expanded.structures)
    assert set(cfg2.simulations) == set(expanded.simulations)
    assert len(cfg2.targets) == len(expanded.targets)
    assert cfg2.scans is None


def test_make_scan_rejects_path_only_reference(tmp_path):
    """xyz-path structures aren't supported yet — error must be loud."""
    cfg = PyFieldConfig.model_validate({
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "qm": {"code": "pyscf"},
        "structures": {
            "X": {"box": [10, 10, 10], "path": "xyz/x.xyz"},
        },
        "simulations": {},
        "targets": [],
        "scans": [{
            "type": "bond_stretch", "reference": "X",
            "name_prefix": "X_d", "atoms": [1, 2], "values": [1.5],
        }],
    })
    with pytest.raises(ValueError, match="inline `atoms:` is required"):
        expand_scans(cfg)


def test_make_scan_range_grid():
    cfg = _cl2_with_scan(None, {
        "type": "bond_stretch",
        "reference": "Cl2_Opt",
        "name_prefix": "Cl2_d",
        "atoms": [1, 2],
        "range": [1.0, 3.0, 5],     # 1.0, 1.5, 2.0, 2.5, 3.0
    })
    expanded, _ = expand_scans(cfg)
    assert len([k for k in expanded.structures if k.startswith("Cl2_d_")]) == 5


def test_scan_schema_rejects_both_values_and_range():
    """ScanCfg must reject ambiguous grid specs."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="exactly one of"):
        _cl2_with_scan(None, {
            "type": "bond_stretch", "reference": "Cl2_Opt",
            "name_prefix": "Cl2_d", "atoms": [1, 2],
            "values": [1.0], "range": [1.0, 2.0, 3],
        })


def test_scan_schema_rejects_unknown_reference():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="unknown structure"):
        _cl2_with_scan(None, {
            "type": "bond_stretch", "reference": "Nope",
            "name_prefix": "Cl2_d", "atoms": [1, 2], "values": [1.0],
        })
