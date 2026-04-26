"""ReaxFF parser round-trip: parse → write → re-parse must equal the original.

This is the regression guard for `ForceField.REAX_FF`. It runs without
LAMMPS, so it works in CI on any machine.
"""
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pyfield.forcefield.reax import REAX_FF


FFIELDS = [
    ("tests/ffieldoriginal.txt", "tests/params"),
    ("tests/Zr_Si_forcefield/ffield1.reax", "tests/Zr_Si_forcefield/params"),
]


@pytest.mark.parametrize("ffield_rel, params_rel", FFIELDS)
def test_reax_parser_roundtrip(ffield_rel, params_rel, tmp_path):
    ffield = os.path.join(ROOT, ffield_rel)
    params = os.path.join(ROOT, params_rel)
    assert os.path.exists(ffield), ffield
    assert os.path.exists(params), params

    a = REAX_FF(ffield, params)
    a.parseParamSelectionFile()
    out = tmp_path / "roundtrip.reax"
    a.write_forcefield(str(out))

    b = REAX_FF(str(out), params)
    b.parseParamSelectionFile()

    assert a.Num_Of_GENERAL == b.Num_Of_GENERAL
    assert a.Num_Of_Atoms == b.Num_Of_Atoms
    assert a.Num_Of_BONDS == b.Num_Of_BONDS
    assert a.Num_Of_OFF_DIAG == b.Num_Of_OFF_DIAG
    assert a.Num_Of_ANGLES == b.Num_Of_ANGLES
    assert a.Num_Of_TORSIONS == b.Num_Of_TORSIONS
    assert a.Num_Of_H_BONDS == b.Num_Of_H_BONDS

    # Compare every numeric parameter slot.
    for section in a.params:
        assert section in b.params
        for entry in a.params[section]:
            assert entry in b.params[section], (section, entry)
            for item in a.params[section][entry]:
                assert item in b.params[section][entry], (section, entry, item)
                va = a.params[section][entry][item]
                vb = b.params[section][entry][item]
                assert va == pytest.approx(vb, rel=0, abs=1e-9), (
                    section, entry, item, va, vb
                )

    # Param-selection file produces the same active-tuple list.
    assert a.param_selection == b.param_selection
    assert a.param_min_max_delta == b.param_min_max_delta
