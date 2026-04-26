"""End-to-end PySCF backend smoke. Skipped if pyscf is not installed."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


pyscf = pytest.importorskip("pyscf", reason="PySCF not installed")
geometric = pytest.importorskip("geometric", reason="geometric (PySCF optimiser) not installed")


@pytest.fixture
def cl2_min_qm_cfg(tmp_path):
    """A tiny PyFieldConfig with one qm_relax + one single-point — smallest
    config that exercises both code paths."""
    from pyfield.config.schema import PyFieldConfig
    return PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf", "functional": "lda", "basis": "sto-3g",
               "cache_dir": str(tmp_path / "cache")},
        "structures": {
            "Cl2_Opt": {"box": [50, 50, 50], "qm_relax": True,
                        "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": -1.0},
                                  {"element": "Cl", "x": 0, "y": 0, "z":  1.0}]},
            "Cl2_long": {"box": [50, 50, 50],
                         "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": -1.5},
                                   {"element": "Cl", "x": 0, "y": 0, "z":  1.5}]},
        },
        "simulations": {
            "Cl2_Opt_min":  {"structure": "Cl2_Opt", "type": "minimize"},
            "Cl2_long_min": {"structure": "Cl2_long", "type": "minimize"},
        },
        "targets": [
            {"kind": "energy_combination", "weight": 1.0,
             "terms": {"Cl2_long_min": +1, "Cl2_Opt_min": -1},
             "target": {"from": "dft"}},
        ],
    })


def test_pyscf_relax_changes_bond_length(cl2_min_qm_cfg):
    """LDA/STO-3G should pull Cl2 to a finite bond length, not collapse."""
    from pyfield.qm.prep import populate_qm

    populated, _ = populate_qm(cl2_min_qm_cfg)
    relaxed = populated.structures["Cl2_Opt"]
    assert relaxed.qm_relax is False
    z1, z2 = relaxed.atoms[0].z, relaxed.atoms[1].z
    bond = abs(z2 - z1)
    # LDA/STO-3G is far from physical (~2.1 Å) but must be in a sane window.
    assert 1.5 < bond < 3.0, bond


def test_pyscf_target_is_positive_for_stretched(cl2_min_qm_cfg):
    """Stretching Cl2 from equilibrium must cost energy — the populated
    target (E_long − E_opt) should come out positive."""
    from pyfield.qm.prep import populate_qm
    populated, _ = populate_qm(cl2_min_qm_cfg)
    target = populated.targets[0].__pydantic_extra__["target"]
    assert isinstance(target, float)
    assert target > 0, target
