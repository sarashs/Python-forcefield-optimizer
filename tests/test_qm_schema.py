"""Schema additions for qm-prep: placeholder predicate, qm_relax,
qm-required validation. No LAMMPS, no QM backend."""
import pytest
from pydantic import ValidationError

from pyfield.config.schema import (
    PyFieldConfig,
    QmCfg,
    StructureCfg,
    is_qm_placeholder,
)


def _minimal(**overrides):
    """A minimal PyFieldConfig dict the validator accepts."""
    cfg = {
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "structures": {
            "A": {"box": [10, 10, 10],
                  "atoms": [{"element": "H", "x": 0, "y": 0, "z": 0}]},
        },
        "simulations": {"A_min": {"structure": "A", "type": "minimize"}},
        "targets": [],
    }
    cfg.update(overrides)
    return cfg


def test_is_qm_placeholder_recognises_only_from_dft():
    assert is_qm_placeholder({"from": "dft"})
    assert not is_qm_placeholder({"from": "experiment"})
    assert not is_qm_placeholder({"source": "dft"})
    assert not is_qm_placeholder(81.394)
    assert not is_qm_placeholder("structures/x.xyz")
    assert not is_qm_placeholder(None)


def test_qm_relax_requires_qm_block():
    bad = _minimal()
    bad["structures"]["A"]["qm_relax"] = True
    with pytest.raises(ValidationError, match="qm_relax"):
        PyFieldConfig.model_validate(bad)


def test_from_dft_target_requires_qm_block():
    bad = _minimal()
    bad["targets"] = [
        {"kind": "energy_combination", "weight": 1.0,
         "terms": {"A_min": 1}, "target": {"from": "dft"}},
    ]
    with pytest.raises(ValidationError, match="from: dft"):
        PyFieldConfig.model_validate(bad)


def test_from_dft_passes_when_qm_block_present():
    ok = _minimal(qm={"code": "pyscf"})
    ok["targets"] = [
        {"kind": "energy_combination", "weight": 1.0,
         "terms": {"A_min": 1}, "target": {"from": "dft"}},
    ]
    cfg = PyFieldConfig.model_validate(ok)
    assert cfg.qm.code == "pyscf"
    assert cfg.qm.basis == "sto-3g"          # default


def test_qm_relax_passes_when_qm_block_present():
    ok = _minimal(qm={"code": "pyscf"})
    ok["structures"]["A"]["qm_relax"] = True
    cfg = PyFieldConfig.model_validate(ok)
    assert cfg.structures["A"].qm_relax is True


def test_hand_typed_target_unchanged_by_validator():
    """Plain targets must NOT trigger the qm-required check."""
    cfg = PyFieldConfig.model_validate(_minimal(targets=[
        {"kind": "energy_combination", "weight": 1.0,
         "terms": {"A_min": 1}, "target": 81.394},
    ]))
    assert cfg.qm is None    # no QM needed
    assert cfg.targets[0].__pydantic_extra__["target"] == 81.394


def test_qm_unknown_code_rejected():
    with pytest.raises(ValidationError):
        QmCfg.model_validate({"code": "totally_not_a_code"})
