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


def test_relax_cache_preserves_pbc_and_extras(tmp_path):
    """Cache hits on a relax must return a structure with the SAME
    extras the input had — `pbc`, `qm_code`, `qm_functional`,
    `constraint`, etc. Earlier the cache only stored `box / atoms /
    qm_relax`, which silently routed PBC structures through PySCF on
    cache hits because `pbc` and `qm_code` got dropped."""
    from pyfield.qm.base import structure_code

    cache = QmCache(tmp_path)
    # Periodic structure with the full set of per-structure overrides.
    s = StructureCfg.model_validate({
        "box": [6.0, 6.0, 6.0],
        "pbc": True,
        "qm_relax": True,
        "qm_code": "qe",
        "qm_functional": "pbe",
        "qm_basis": "def2-svp",
        "atoms": [
            {"element": "Si", "x": 0, "y": 0, "z": 0},
            {"element": "Si", "x": 1.36, "y": 1.36, "z": 1.36},
        ],
    })
    relaxed = s.model_copy(update={
        "atoms": [
            AtomCfg(element="Si", x=0.01, y=0.02, z=0.03),
            AtomCfg(element="Si", x=1.40, y=1.40, z=1.40),
        ],
        "qm_relax": False,
    })

    def compute():
        return QmRelaxResult(structure=relaxed, energy_kcal_mol=-100.0)

    # First call stores; second call hits the cache.
    r1, _, hit1 = cache.memoise_relax(s, "fp", "relax", compute)
    r2, _, hit2 = cache.memoise_relax(s, "fp", "relax", compute)
    assert hit1 is False and hit2 is True

    # The cache-hit structure must have the same dispatch profile as the
    # cache-miss one — otherwise scan-point structures derived from the
    # cached reference get the wrong backend on subsequent runs.
    assert r2.structure.pbc is True
    extras = r2.structure.__pydantic_extra__ or {}
    assert extras.get("qm_code") == "qe"
    assert extras.get("qm_functional") == "pbe"
    assert extras.get("qm_basis") == "def2-svp"
    assert structure_code(r2.structure, fallback="pyscf") == "qe"


def test_relax_memoise_merges_input_extras_on_legacy_entries(tmp_path):
    """A legacy cache entry (no `structure` key in result.json) doesn't
    carry pbc/qm_code/qm_functional. `memoise_relax` must re-attach
    them from the input structure on every cache hit, so dispatch still
    works correctly across runs against an old cache."""
    import json
    from pyfield.qm.base import structure_code

    cache = QmCache(tmp_path)
    # Pre-populate a legacy cache entry by hand.
    s_input = StructureCfg.model_validate({
        "box": [6.0, 6.0, 6.0],
        "pbc": True,
        "qm_relax": True,
        "qm_code": "qe",
        "qm_functional": "pbe",
        "atoms": [
            {"element": "Si", "x": 0, "y": 0, "z": 0},
            {"element": "Si", "x": 1.36, "y": 1.36, "z": 1.36},
        ],
    })
    key = "fakekeyfakekey00"
    entry = tmp_path / key
    entry.mkdir()
    legacy = {
        "kind": "relax",
        "energy_kcal_mol": -42.0,
        "atoms": [
            {"element": "Si", "x": 0.01, "y": 0.0, "z": 0.0, "charge": 0.0},
            {"element": "Si", "x": 1.40, "y": 1.40, "z": 1.40, "charge": 0.0},
        ],
        "box": [6.0, 6.0, 6.0],
    }
    (entry / "result.json").write_text(json.dumps(legacy))

    # Force memoise_relax to find this key by passing matching content.
    # We can't construct the right input deterministically, so call
    # _load_relax + _merge_relax_with_input directly to test the merge.
    from pyfield.qm.cache import _merge_relax_with_input
    cached = cache._load_relax(key)
    merged = _merge_relax_with_input(cached.structure, s_input)
    assert merged.pbc is True                 # restored from input
    extras = merged.__pydantic_extra__ or {}
    assert extras.get("qm_code") == "qe"
    assert extras.get("qm_functional") == "pbe"
    assert structure_code(merged, fallback="pyscf") == "qe"
    # Cached atoms still survive (not overwritten by input).
    assert merged.atoms[0].x == 0.01


def test_relax_cache_legacy_entries_still_load(tmp_path):
    """Older cache entries only stored `box / atoms / qm_relax`. The
    loader must still handle that shape so users don't have to wipe
    their cache after the schema upgrade."""
    import json

    cache = QmCache(tmp_path)
    key = "abcdef0123456789"
    entry = tmp_path / key
    entry.mkdir()
    legacy = {
        "kind": "relax",
        "energy_kcal_mol": -42.0,
        "atoms": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0, "charge": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74, "charge": 0.0},
        ],
        "box": [10.0, 10.0, 10.0],
    }
    (entry / "result.json").write_text(json.dumps(legacy))
    result = cache._load_relax(key)
    assert result.energy_kcal_mol == -42.0
    assert result.structure.box == (10.0, 10.0, 10.0)
    assert result.structure.atoms[1].z == 0.74
    # Legacy entries didn't store pbc, so it falls back to the schema default.
    assert result.structure.pbc is False


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
