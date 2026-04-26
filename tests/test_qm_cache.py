"""QM cache: hit/miss/forced miss + content-keyed independence. No QM backend."""
import numpy as np
import pytest

from pyfield.config.schema import AtomCfg, StructureCfg
from pyfield.qm.base import QmRelaxResult, QmSinglePoint
from pyfield.qm.cache import QmCache


def _struct(z=1.0):
    return StructureCfg(box=(10, 10, 10), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=z),
    ])


def test_first_call_misses_then_hits(tmp_path):
    cache = QmCache(tmp_path)
    calls = {"n": 0}
    def compute():
        calls["n"] += 1
        return QmSinglePoint(energy_kcal_mol=-1.234)

    s = _struct()
    r1, key1, hit1 = cache.memoise_single_point(s, "fp1", "single_point", compute)
    assert calls["n"] == 1 and hit1 is False
    r2, key2, hit2 = cache.memoise_single_point(s, "fp1", "single_point", compute)
    assert calls["n"] == 1 and hit2 is True              # served from cache
    assert key1 == key2
    assert r1.energy_kcal_mol == r2.energy_kcal_mol


def test_force_invalidates_cache(tmp_path):
    cache = QmCache(tmp_path)
    calls = {"n": 0}
    def compute():
        calls["n"] += 1
        return QmSinglePoint(energy_kcal_mol=-1.234)
    s = _struct()
    cache.memoise_single_point(s, "fp1", "single_point", compute)
    cache.memoise_single_point(s, "fp1", "single_point", compute, force=True)
    assert calls["n"] == 2


def test_different_structure_different_key(tmp_path):
    cache = QmCache(tmp_path)
    a = _struct(z=1.0)
    b = _struct(z=1.5)
    _, key_a, _ = cache.memoise_single_point(a, "fp", "sp", lambda: QmSinglePoint(0.0))
    _, key_b, _ = cache.memoise_single_point(b, "fp", "sp", lambda: QmSinglePoint(0.0))
    assert key_a != key_b


def test_different_settings_different_key(tmp_path):
    cache = QmCache(tmp_path)
    s = _struct()
    _, key_pbe,  _ = cache.memoise_single_point(s, "pbe-fp", "sp", lambda: QmSinglePoint(0.0))
    _, key_pbe0, _ = cache.memoise_single_point(s, "pbe0-fp", "sp", lambda: QmSinglePoint(0.0))
    assert key_pbe != key_pbe0


def test_relax_round_trip(tmp_path):
    cache = QmCache(tmp_path)
    s = _struct()
    relaxed = StructureCfg(box=(10, 10, 10), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    def compute():
        return QmRelaxResult(structure=relaxed, energy_kcal_mol=-100.0)
    r1, _, _ = cache.memoise_relax(s, "fp", "relax", compute)
    r2, _, hit = cache.memoise_relax(s, "fp", "relax", compute)
    assert hit is True
    assert r1.energy_kcal_mol == r2.energy_kcal_mol
    assert r1.structure.atoms[1].z == r2.structure.atoms[1].z == 0.74


def test_forces_round_trip(tmp_path):
    cache = QmCache(tmp_path)
    s = _struct()
    f = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    def compute():
        return QmSinglePoint(energy_kcal_mol=-1.0, forces_kcal_mol_per_A=f)
    r1, _, _ = cache.memoise_single_point(s, "fp", "sp", compute)
    r2, _, hit = cache.memoise_single_point(s, "fp", "sp", compute)
    assert hit is True
    np.testing.assert_array_equal(r1.forces_kcal_mol_per_A, r2.forces_kcal_mol_per_A)
