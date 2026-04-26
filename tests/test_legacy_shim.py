"""Legacy text-format → PyFieldConfig shim. No LAMMPS."""
import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyfield.config.legacy import from_legacy_files
from pyfield.config.loader import load_yaml


def test_legacy_matches_handwritten_yaml():
    """Cl2 legacy inputs should produce the same structures/sims/targets
    as the hand-written `tests/cl2.yaml` — at least in shape."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = from_legacy_files(
            forcefield=Path(ROOT, "tests/ffieldoriginal.txt"),
            params=Path(ROOT, "tests/params"),
            training=Path(ROOT, "tests/Trainingfile_2.txt"),
            structures=Path(ROOT, "tests/Inputstructurefile.txt"),
        )
    yaml_cfg = load_yaml(os.path.join(ROOT, "tests/cl2.yaml"))

    assert set(legacy.structures) == set(yaml_cfg.structures)
    assert set(legacy.simulations) == set(yaml_cfg.simulations)
    assert len(legacy.targets) == len(yaml_cfg.targets)

    # Every restraint roundtrips.
    for sim_id in yaml_cfg.simulations:
        legacy_r = legacy.simulations[sim_id].__pydantic_extra__.get("restraints") or []
        yaml_r = yaml_cfg.simulations[sim_id].__pydantic_extra__.get("restraints") or []
        assert legacy_r == yaml_r, sim_id


def test_legacy_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        from_legacy_files(
            forcefield=Path(ROOT, "tests/ffieldoriginal.txt"),
            params=Path(ROOT, "tests/params"),
            training=Path(ROOT, "tests/Trainingfile_2.txt"),
            structures=Path(ROOT, "tests/Inputstructurefile.txt"),
        )
