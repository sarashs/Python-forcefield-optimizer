"""QE backend unit tests with a mocked Espresso calculator.

The real `pw.x` lives behind ASE's calculator interface; we substitute
a fake calculator that returns canned energies / forces so the tests
run fast and don't require a QE install. The actual QE integration is
covered by the (slow) `tests/test_qm_pyscf.py`-style smoke test that
gets skipped without `pw.x` on PATH.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from pyfield.config.schema import AtomCfg, QmCfg, StructureCfg
from pyfield.qm.base import ConstraintSpec, make_backend, structure_code


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def pseudo_dir(tmp_path) -> Path:
    """Empty directory — the backend just needs *a* path; the fake
    calculator never touches the files."""
    d = tmp_path / "pseudos"
    d.mkdir()
    return d


@pytest.fixture
def qm_cfg(pseudo_dir):
    return QmCfg.model_validate({
        "code": "qe",
        "functional": "pbe",
        "basis": "pw",                       # informational; QE uses plane-waves
        "cache_dir": str(pseudo_dir.parent / "cache"),
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {
            "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
            "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
        },
        "ecutwfc": 40.0,
        "ecutrho": 320.0,
        "kpts": [2, 2, 2],
    })


def _periodic_struct():
    return StructureCfg(
        box=(5.43, 5.43, 5.43),
        pbc=True,
        atoms=[
            AtomCfg(element="Si", x=0, y=0, z=0),
            AtomCfg(element="Si", x=1.36, y=1.36, z=1.36),
        ],
    )


# ---------------------------------------------------------------------------
# Construction + ASE atom round-trip
# ---------------------------------------------------------------------------

def test_backend_requires_pseudo_dir():
    cfg = QmCfg.model_validate({"code": "qe", "functional": "pbe", "basis": "pw"})
    # No ESPRESSO_PSEUDO env, no pseudo_dir extra → must complain.
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ESPRESSO_PSEUDO", None)
        from pyfield.qm.qe_backend import QEBackend
        with pytest.raises(ValueError, match="pseudo_dir"):
            QEBackend(cfg)


def test_to_ase_round_trip(qm_cfg):
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    s = _periodic_struct()
    atoms = backend._to_ase(s)
    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    np.testing.assert_allclose(atoms.get_positions()[1], [1.36, 1.36, 1.36])
    np.testing.assert_allclose(np.diag(atoms.cell), [5.43, 5.43, 5.43])
    assert all(atoms.pbc)


def test_to_ase_rejects_missing_atoms(qm_cfg):
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    s = StructureCfg.model_construct(box=(5, 5, 5), pbc=True, atoms=None,
                                      path="x.xyz", qm_relax=False)
    with pytest.raises(ValueError, match="inline `atoms:`"):
        backend._to_ase(s)


def test_missing_pseudopotential_raises(qm_cfg):
    """If the structure contains an element not in the pseudo map, fail
    loudly *before* invoking pw.x."""
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    s = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="Ge", x=0, y=0, z=0),
    ])
    atoms = backend._to_ase(s)
    with pytest.raises(ValueError, match="pseudopotential mapping"):
        backend._make_calculator(atoms)


# ---------------------------------------------------------------------------
# input_data construction
# ---------------------------------------------------------------------------

def test_input_data_carries_qe_keywords(qm_cfg):
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    d = backend._build_input_data()
    assert d["control"]["calculation"] == "scf"
    assert d["control"]["tprnfor"] is True
    assert d["control"]["tstress"] is True
    assert d["system"]["ibrav"] == 0
    assert d["system"]["ecutwfc"] == 40.0
    assert d["system"]["ecutrho"] == 320.0
    assert d["system"]["input_dft"] == "PBE"
    assert d["electrons"]["conv_thr"] == 1e-7


def test_qe_input_dft_translation():
    from pyfield.qm.qe_backend import _qe_input_dft
    assert _qe_input_dft("pbe") == "PBE"
    assert _qe_input_dft("PBE") == "PBE"
    assert _qe_input_dft("b3lyp") == "B3LYP"
    assert _qe_input_dft("lda") == "PZ"
    assert _qe_input_dft("pz") == "PZ"


def test_input_data_vc_relax_keywords(qm_cfg):
    """With relax_cell=True the input dict must switch to QE's
    `vc-relax` calculation and add the &IONS / &CELL namelists plus
    cell_factor — that combination is what makes the variable-cell
    relax robust against Pulay stress."""
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    d = backend._build_input_data(relax_cell=True)
    assert d["control"]["calculation"] == "vc-relax"
    assert "forc_conv_thr" in d["control"]
    assert "etot_conv_thr" in d["control"]
    # cell_factor must live in &CELL, not &SYSTEM — QE 6.4.x rejects
    # it from &SYSTEM with "bad line in namelist &system".
    assert "cell_factor" not in d["system"]
    assert d["cell"]["cell_factor"] == 2.0
    assert d["ions"]["ion_dynamics"] == "bfgs"
    assert d["cell"]["cell_dynamics"] == "bfgs"
    assert "press_conv_thr" in d["cell"]
    # cell_dofree must keep the relaxed cell orthorhombic — the
    # StructureCfg.box schema only carries [a, b, c] and would lose
    # any shear from a `cell_dofree: 'all'` run.
    assert d["cell"]["cell_dofree"] == "xyz"
    # nstep generously over QE's 50-default so a vc-relax that needs
    # 6%+ volume change doesn't hit the cap and exit non-zero with a
    # half-converged geometry.
    assert d["control"]["nstep"] >= 100


def test_input_data_default_is_scf(qm_cfg):
    """Sanity check: with relax_cell unset, the calculation type
    stays at `scf` (atoms-only / single-point path)."""
    from pyfield.qm.qe_backend import QEBackend
    d = QEBackend(qm_cfg)._build_input_data()
    assert d["control"]["calculation"] == "scf"
    assert "ions" not in d
    assert "cell" not in d


def test_input_data_spin_polarized(pseudo_dir):
    cfg = QmCfg.model_validate({
        "code": "qe", "functional": "pbe", "basis": "pw",
        "cache_dir": str(pseudo_dir.parent / "cache"),
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {"Fe": "Fe.upf"},
        "spin": 1,
    })
    from pyfield.qm.qe_backend import QEBackend
    d = QEBackend(cfg)._build_input_data()
    assert d["system"]["nspin"] == 2


# ---------------------------------------------------------------------------
# Constraint translation
# ---------------------------------------------------------------------------

def test_distance_constraint_repositions_and_fixes(qm_cfg):
    """For a distance constraint, we move atom j to the target separation
    along the current bond axis, then attach FixBondLengths."""
    from ase import Atoms
    from ase.constraints import FixBondLengths
    from pyfield.qm.qe_backend import QEBackend

    atoms = Atoms(symbols=["Si", "Si"],
                  positions=[(0, 0, 0), (2.0, 0, 0)],
                  cell=np.eye(3) * 6, pbc=True)
    QEBackend._apply_constraint(atoms, ConstraintSpec(
        kind="distance", atoms=[1, 2], value=3.0,
    ))
    np.testing.assert_allclose(atoms.get_positions()[1], [3.0, 0, 0])
    assert any(isinstance(c, FixBondLengths) for c in atoms.constraints)


def test_angle_constraint_uses_fixinternals(qm_cfg):
    from ase import Atoms
    from ase.constraints import FixInternals
    from pyfield.qm.qe_backend import QEBackend
    atoms = Atoms(symbols=["O", "Si", "O"],
                  positions=[(1, 0, 0), (0, 0, 0), (-1, 0, 0)],
                  cell=np.eye(3) * 6, pbc=True)
    QEBackend._apply_constraint(atoms, ConstraintSpec(
        kind="angle", atoms=[1, 2, 3], value=120.0,
    ))
    assert any(isinstance(c, FixInternals) for c in atoms.constraints)


def test_dihedral_constraint_uses_fixinternals(qm_cfg):
    from ase import Atoms
    from ase.constraints import FixInternals
    from pyfield.qm.qe_backend import QEBackend
    atoms = Atoms(symbols=["O", "Si", "O", "Si"],
                  positions=[(1, 0, 0), (0, 0, 0), (-1, 0, 0), (-1, 1, 0)],
                  cell=np.eye(3) * 6, pbc=True)
    QEBackend._apply_constraint(atoms, ConstraintSpec(
        kind="dihedral", atoms=[1, 2, 3, 4], value=90.0,
    ))
    assert any(isinstance(c, FixInternals) for c in atoms.constraints)


# ---------------------------------------------------------------------------
# Mocked single_point / relax via fake ASE calculator
# ---------------------------------------------------------------------------

class _FakeCalc:
    """Stand-in for the ASE Espresso calculator. Returns canned energies
    + forces so the test doesn't need pw.x."""

    def __init__(self, energy_ev=-100.0, forces_ev_a=None):
        self.energy_ev = energy_ev
        self.forces_ev_a = forces_ev_a if forces_ev_a is not None else None
        self.atoms = None

    def calculate(self, atoms=None, **kwargs):
        self.atoms = atoms
        self.results = {"energy": self.energy_ev,
                        "forces": self.forces_ev_a if self.forces_ev_a is not None
                                  else np.zeros((len(atoms), 3))}

    def get_potential_energy(self, atoms=None, **kwargs):
        if atoms is not None:
            self.atoms = atoms
        return self.energy_ev

    def get_forces(self, atoms=None, **kwargs):
        if atoms is not None:
            self.atoms = atoms
        return (self.forces_ev_a if self.forces_ev_a is not None
                else np.zeros((len(self.atoms or [None, None]), 3)))


def test_single_point_converts_units(qm_cfg, monkeypatch):
    from pyfield.qm import qe_backend
    backend = qe_backend.QEBackend(qm_cfg)

    fake = _FakeCalc(energy_ev=-2.5)
    monkeypatch.setattr(backend, "_make_calculator",
                        lambda atoms, functional=None, directory=None: fake)

    s = _periodic_struct()
    sp = backend.single_point(s)
    # eV → kcal/mol: -2.5 eV * 23.0605 = -57.65 kcal/mol
    assert sp.energy_kcal_mol == pytest.approx(-2.5 * 23.06054783, rel=1e-6)


def test_per_structure_functional_override(qm_cfg):
    """Per-structure `qm_functional` must override the global qm.functional
    when QE constructs the input_data dict — otherwise the wrong
    functional silently leaks into pw.x runs."""
    from pyfield.qm.qe_backend import QEBackend
    backend = QEBackend(qm_cfg)
    s = StructureCfg(box=(5, 5, 5), pbc=True, qm_functional="pbe", atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])
    assert backend._effective_functional(s) == "pbe"
    s_default = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])
    assert backend._effective_functional(s_default) == qm_cfg.functional


def test_settings_fingerprint_changes_with_conv_thr(pseudo_dir):
    """conv_thr affects forces/stress at converged SCF — bumping
    1e-7 → 1e-9 must invalidate cached relaxes (otherwise tightening
    the threshold silently hits stale, looser results)."""
    from pyfield.qm.qe_backend import QEBackend

    base = QmCfg.model_validate({
        "code": "qe", "functional": "pbe", "basis": "pw",
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {"Si": "x.UPF"},
        "conv_thr": 1.0e-7,
    })
    other = QmCfg.model_validate({
        "code": "qe", "functional": "pbe", "basis": "pw",
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {"Si": "x.UPF"},
        "conv_thr": 1.0e-9,
    })
    fp1 = QEBackend(base).settings_fingerprint()
    fp2 = QEBackend(other).settings_fingerprint()
    assert fp1 != fp2


def test_settings_fingerprint_changes_with_ecut(pseudo_dir):
    from pyfield.qm.qe_backend import QEBackend

    base = QmCfg.model_validate({
        "code": "qe", "functional": "pbe", "basis": "pw",
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {"Si": "x.UPF"},
        "ecutwfc": 40.0, "ecutrho": 320.0, "kpts": [1, 1, 1],
    })
    other = QmCfg.model_validate({
        "code": "qe", "functional": "pbe", "basis": "pw",
        "pseudo_dir": str(pseudo_dir),
        "pseudopotentials": {"Si": "x.UPF"},
        "ecutwfc": 50.0, "ecutrho": 400.0, "kpts": [1, 1, 1],
    })
    fp1 = QEBackend(base).settings_fingerprint()
    fp2 = QEBackend(other).settings_fingerprint()
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# Factory dispatch + per-structure override
# ---------------------------------------------------------------------------

def test_factory_dispatches_to_qe(qm_cfg):
    backend = make_backend(qm_cfg)
    assert backend.name == "qe"


def test_structure_code_override():
    """A per-structure `qm_code: qe` overrides the global default."""
    s_default = StructureCfg(box=(5, 5, 5), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
    ])
    s_qe = StructureCfg(box=(5, 5, 5), qm_code="qe", atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
    ])
    assert structure_code(s_default, fallback="pyscf") == "pyscf"
    assert structure_code(s_qe, fallback="pyscf") == "qe"


def test_relax_dispatch_on_qm_relax_cell():
    """The relax() top-level dispatcher must route to the vc path
    only when both qm_relax_cell=True AND constraint is None.
    Constrained scan points keep the strained cell fixed and must
    fall through to atoms-only relax even if the flag is set."""
    from pyfield.qm.qe_backend import QEBackend
    sentinel_atoms = object()
    sentinel_vc = object()
    s_atoms_only = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])
    s_vc = StructureCfg(box=(5, 5, 5), pbc=True, qm_relax_cell=True, atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])

    class _Stub(QEBackend):
        def __init__(self): pass
        def _relax_atoms_only(self, structure, constraint=None):
            return sentinel_atoms
        def _relax_vc(self, structure):
            return sentinel_vc

    stub = _Stub()
    assert stub.relax(s_atoms_only) is sentinel_atoms
    assert stub.relax(s_vc) is sentinel_vc
    # qm_relax_cell + constraint → atoms-only branch (constraint wins)
    assert stub.relax(s_vc, constraint=ConstraintSpec(
        kind="distance", atoms=[1, 2], value=2.0,
    )) is sentinel_atoms


def test_cache_key_includes_qm_code():
    """Same atoms with `qm_code: qe` vs without must have different
    cache keys — same SCF run on PySCF and QE are different results."""
    from pyfield.qm.cache import _key

    s_pyscf = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])
    s_qe = StructureCfg(box=(5, 5, 5), pbc=True, qm_code="qe", atoms=[
        AtomCfg(element="Si", x=0, y=0, z=0),
    ])
    assert _key(s_pyscf, "fp", "single_point") != _key(s_qe, "fp", "single_point")
