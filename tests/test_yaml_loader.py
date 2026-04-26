"""YAML schema + loader tests. No LAMMPS."""
import os
import sys

import pytest
from pydantic import ValidationError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyfield.config.loader import load_yaml
from pyfield.config.schema import (
    AtomCfg,
    ForceFieldCfg,
    OptimizerCfg,
    OutputCfg,
    PyFieldConfig,
    SimulationCfg,
    StructureCfg,
    TargetCfg,
)


def test_cl2_yaml_loads():
    cfg = load_yaml(os.path.join(ROOT, "tests/cl2.yaml"))
    assert len(cfg.structures) == 20
    assert len(cfg.simulations) == 20
    assert len(cfg.targets) == 19
    assert cfg.optimizer.seed == 0
    assert cfg.optimizer.method == "sa"


def test_simulation_must_pick_type_or_template():
    with pytest.raises(ValidationError):
        SimulationCfg.model_validate({"structure": "X"})  # neither
    with pytest.raises(ValidationError):
        SimulationCfg.model_validate(
            {"structure": "X", "type": "minimize", "template": "x.in.j2"}  # both
        )
    SimulationCfg.model_validate({"structure": "X", "type": "minimize"})  # ok
    SimulationCfg.model_validate({"structure": "X", "template": "x.in.j2"})  # ok


def test_simulation_unknown_structure_rejected(tmp_path):
    bad = {
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "structures": {"A": {"box": [10, 10, 10],
                              "atoms": [{"element": "H", "x": 0, "y": 0, "z": 0}]}},
        "simulations": {"sim1": {"structure": "MISSING", "type": "minimize"}},
        "targets": [],
    }
    with pytest.raises(ValidationError):
        PyFieldConfig.model_validate(bad)


def test_target_terms_must_reference_known_sims():
    bad = {
        "forcefield": {"path": "ffield.reax", "params": "params"},
        "structures": {"A": {"box": [10, 10, 10],
                              "atoms": [{"element": "H", "x": 0, "y": 0, "z": 0}]}},
        "simulations": {"a_min": {"structure": "A", "type": "minimize"}},
        "targets": [
            {"kind": "energy_combination", "weight": 1.0,
             "terms": {"a_min": +1, "ghost_min": -1}, "target": 1.0}
        ],
    }
    with pytest.raises(ValidationError):
        PyFieldConfig.model_validate(bad)


def test_structure_requires_atoms_xor_path():
    with pytest.raises(ValidationError):
        StructureCfg.model_validate({"box": [10, 10, 10]})
    with pytest.raises(ValidationError):
        StructureCfg.model_validate({
            "box": [10, 10, 10], "atoms": [], "path": "x.xyz",
        })
