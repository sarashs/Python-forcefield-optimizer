"""`relax_structures` against the same FakeBackend used in test_qm_prep."""
import numpy as np
import pytest

from pyfield.config.loader import load_yaml
from pyfield.config.schema import PyFieldConfig
from pyfield.qm.base import QmBackend, QmRelaxResult, QmSinglePoint
from pyfield.qm.prep import cfg_to_yaml, relax_structures


class _FakeBackend(QmBackend):
    name = "fake"

    def __init__(self, relax_table=None):
        self.relax_table = relax_table or {}

    def settings_fingerprint(self):
        return "fake-fp"

    def single_point(self, structure):
        return QmSinglePoint(energy_kcal_mol=0.0)

    def relax(self, structure, constraint=None):
        z = self.relax_table.get(structure.atoms[1].z, structure.atoms[1].z)
        new_atoms = [
            structure.atoms[0],
            structure.atoms[1].model_copy(update={"z": z}),
        ]
        return QmRelaxResult(
            structure=structure.model_copy(update={"atoms": new_atoms, "qm_relax": False}),
            energy_kcal_mol=-1.0,
        )


def _cfg(tmp_path):
    return PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf", "cache_dir": str(tmp_path / "cache")},
        "structures": {
            "A": {"box": [10, 10, 10], "qm_relax": True,
                  "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": 0},
                            {"element": "Cl", "x": 0, "y": 0, "z": 1.0}]},
            "B": {"box": [10, 10, 10],
                  "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": 0},
                            {"element": "Cl", "x": 0, "y": 0, "z": 2.0}]},
        },
        "simulations": {},
        "targets": [],
    })


def test_qm_relax_only_runs_flagged_structures(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05})
    populated, journal = relax_structures(cfg, backend=backend)
    actions = [a for a, _, _ in journal]
    assert len(actions) == 1
    assert actions[0].startswith("relax A")            # only A had the flag
    assert populated.structures["A"].qm_relax is False
    assert populated.structures["A"].atoms[1].z == pytest.approx(1.05)
    # B untouched.
    assert populated.structures["B"].atoms[1].z == pytest.approx(2.0)


def test_qm_relax_only_override(tmp_path):
    """`only=[...]` ignores the qm_relax flag entirely."""
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={2.0: 2.07})
    populated, journal = relax_structures(cfg, backend=backend, only=["B"])
    actions = [a for a, _, _ in journal]
    assert len(actions) == 1
    assert actions[0].startswith("relax B")
    assert populated.structures["B"].atoms[1].z == pytest.approx(2.07)


def test_qm_relax_only_rejects_unknown(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="unknown structures"):
        relax_structures(cfg, backend=_FakeBackend(), only=["Nope"])


def test_qm_relax_yaml_round_trip_drops_flag(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05})
    populated, _ = relax_structures(cfg, backend=backend)
    out = tmp_path / "relaxed.yaml"
    out.write_text(cfg_to_yaml(populated))
    cfg2 = load_yaml(out)
    assert cfg2.structures["A"].qm_relax is False
    assert cfg2.structures["A"].atoms[1].z == pytest.approx(1.05)


def test_qm_relax_idempotent_via_cache(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05})
    relax_structures(cfg, backend=backend)
    blind = _FakeBackend()
    pop2, journal2 = relax_structures(cfg, backend=blind)
    assert all(hit for _, hit, _ in journal2)
    assert pop2.structures["A"].atoms[1].z == pytest.approx(1.05)
