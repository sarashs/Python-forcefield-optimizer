"""qm-prep populator against a fake backend. No real QM, no LAMMPS.

Drives the YAML transform end-to-end with a stubbed `QmBackend` that
just returns hand-supplied energies / forces / relaxed coords. This
test exists because the populator is the actual user-visible behaviour
of `pyfield qm-prep` — the PySCF backend is one swap away."""
import numpy as np
import pytest
import yaml

from pyfield.config.loader import load_yaml
from pyfield.config.schema import (
    AtomCfg,
    PyFieldConfig,
    QmCfg,
    StructureCfg,
)
from pyfield.qm.base import QmBackend, QmRelaxResult, QmSinglePoint
from pyfield.qm.prep import cfg_to_yaml, populate_qm


class _FakeBackend(QmBackend):
    name = "fake"

    def __init__(self, sp_table=None, relax_table=None):
        # `sp_table` maps atom-z (the only varying coord in our fixtures)
        # to a single-point energy, so the test case can spec different
        # energies per geometry without inspecting atom hashes.
        self.sp_table = sp_table or {}
        self.relax_table = relax_table or {}

    def settings_fingerprint(self):
        return "fake-fp"

    def single_point(self, structure):
        z = structure.atoms[1].z
        return QmSinglePoint(
            energy_kcal_mol=self.sp_table.get(z, 0.0),
            forces_kcal_mol_per_A=np.zeros((len(structure.atoms), 3)),
        )

    def relax(self, structure, constraint=None):
        z = self.relax_table.get(structure.atoms[1].z, structure.atoms[1].z)
        new_atoms = [
            structure.atoms[0],
            structure.atoms[1].model_copy(update={"z": z}),
        ]
        return QmRelaxResult(
            structure=structure.model_copy(update={"atoms": new_atoms, "qm_relax": False}),
            energy_kcal_mol=self.sp_table.get(z, 0.0),
        )


def _cfg(tmp_path):
    """A 3-structure / 2-target fixture using `from: dft` placeholders."""
    return PyFieldConfig.model_validate({
        "forcefield": {"path": "ff.reax", "params": "params"},
        "qm": {"code": "pyscf", "cache_dir": str(tmp_path / "cache")},
        "structures": {
            "Cl2_Opt": {"box": [10, 10, 10], "qm_relax": True,
                        "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": 0},
                                  {"element": "Cl", "x": 0, "y": 0, "z": 1.0}]},
            "Cl2_414": {"box": [10, 10, 10],
                        "atoms": [{"element": "Cl", "x": 0, "y": 0, "z": 0},
                                  {"element": "Cl", "x": 0, "y": 0, "z": 2.07}]},
        },
        "simulations": {
            "Cl2_Opt_min": {"structure": "Cl2_Opt", "type": "minimize"},
            "Cl2_414_min": {"structure": "Cl2_414", "type": "minimize"},
        },
        "targets": [
            {"kind": "energy_combination", "weight": 1.0,
             "terms": {"Cl2_414_min": +1, "Cl2_Opt_min": -1},
             "target": {"from": "dft"}},
        ],
    })


def test_populator_writes_relaxed_atoms(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(
        relax_table={1.0: 1.05},   # relax pushes the bond from 1.0 → 1.05 Å
        sp_table={1.05: -100.0, 2.07: -10.0},   # combination is -10 - (-100) = 90
    )
    populated, journal = populate_qm(cfg, backend=backend)

    # Relaxed coords landed in atoms[1].z; qm_relax flag is dropped.
    assert populated.structures["Cl2_Opt"].atoms[1].z == pytest.approx(1.05)
    assert populated.structures["Cl2_Opt"].qm_relax is False
    # Target value is the signed sum of the populated single-points.
    assert populated.targets[0].__pydantic_extra__["target"] == pytest.approx(90.0)


def test_populator_journal_reports_actions(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05},
                           sp_table={1.05: 0.0, 2.07: 0.0})
    _, journal = populate_qm(cfg, backend=backend)
    actions = [a for a, _, _ in journal]
    assert "relax Cl2_Opt" in actions
    # Cl2_Opt was just relaxed, so its energy is reused (no redundant SP).
    assert "reuse_relax_energy Cl2_Opt_min" in actions
    # Cl2_414 wasn't relaxed; it gets a single_point.
    assert "single_point Cl2_414_min" in actions


def test_populator_idempotent_via_cache(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05},
                           sp_table={1.05: -100.0, 2.07: -10.0})
    populate_qm(cfg, backend=backend)
    # Replace the backend so the table is gone — cache must still serve.
    blind_backend = _FakeBackend()
    pop2, journal2 = populate_qm(cfg, backend=blind_backend)
    # Every action must be a hit.
    assert all(hit for _, hit, _ in journal2)
    assert pop2.targets[0].__pydantic_extra__["target"] == pytest.approx(90.0)


def test_populated_yaml_round_trips_through_loader(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(relax_table={1.0: 1.05},
                           sp_table={1.05: -1.0, 2.07: 0.5})
    populated, _ = populate_qm(cfg, backend=backend)
    yaml_text = cfg_to_yaml(populated)
    out = tmp_path / "p.yaml"
    out.write_text(yaml_text)
    reloaded = load_yaml(out)
    # Round-trip is a fixed point: every placeholder is gone.
    assert reloaded.structures["Cl2_Opt"].qm_relax is False
    assert isinstance(reloaded.targets[0].__pydantic_extra__["target"], float)


def test_run_refuses_unpopulated_placeholders(tmp_path):
    """`pyfield run` (via _check_no_qm_placeholders) must refuse a config
    that still has `from: dft` slots — one error listing every slot."""
    from pyfield.runner import _check_no_qm_placeholders

    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError, match="qm-prep"):
        _check_no_qm_placeholders(cfg)
