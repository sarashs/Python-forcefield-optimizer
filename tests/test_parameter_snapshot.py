"""ParameterSnapshot regression: snapshot/apply must be equivalent to
deepcopy/restore for the SA accept/reject path. No LAMMPS.
"""
import os
import sys
from copy import deepcopy
import random

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyfield.forcefield.reax import REAX_FF
from pyfield.forcefield.snapshot import ParameterSnapshot


@pytest.fixture
def ff():
    f = REAX_FF(os.path.join(ROOT, "tests/ffieldoriginal.txt"),
                os.path.join(ROOT, "tests/params"))
    f.parseParamSelectionFile()
    return f


def test_capture_then_apply_is_identity(ff):
    snap = ParameterSnapshot.capture(ff)
    # Mutate every selected parameter to something obviously different.
    for sec, entry, item in ff.param_min_max_delta:
        ff.params[sec][entry][item] = 12345.6789
    snap.apply(ff)
    for sec, entry, item in ff.param_min_max_delta:
        assert ff.params[sec][entry][item] == snap.values[snap.keys.index((sec, entry, item))]


def test_snapshot_keys_match_param_min_max_delta(ff):
    snap = ParameterSnapshot.capture(ff)
    assert set(snap.keys) == set(ff.param_min_max_delta.keys())
    assert len(snap) == len(ff.param_min_max_delta)


def test_copy_independence(ff):
    a = ParameterSnapshot.capture(ff)
    b = a.copy()
    b.values[0] = -999.0
    assert a.values[0] != b.values[0]


def test_equivalent_to_deepcopy_under_random_walk(ff):
    """Drive 100 random perturbations through both deepcopy-of-params and
    snapshot. They must give bit-identical results at every step."""
    random.seed(0xC0FFEE)
    snap = ParameterSnapshot.capture(ff)
    deep = deepcopy(ff.params)

    for _ in range(100):
        # Random in-place perturbation in-bounds.
        for sec, entry, item in ff.param_min_max_delta:
            b = ff.param_min_max_delta[(sec, entry, item)]
            ff.params[sec][entry][item] = random.uniform(b["min"], b["max"])

        # Half the time accept (commit both); half reject (restore both).
        if random.random() < 0.5:
            snap = ParameterSnapshot.capture(ff)
            deep = deepcopy(ff.params)
        else:
            snap.apply(ff)
            ff.params = deepcopy(deep)

        # After every step the two restoration mechanisms must agree.
        for sec, entry, item in ff.param_min_max_delta:
            from_snap = snap.values[snap.keys.index((sec, entry, item))]
            from_deep = deep[sec][entry][item]
            assert ff.params[sec][entry][item] == from_snap == from_deep
