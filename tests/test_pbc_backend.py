"""PBC dispatch in the PySCF backend.

The `pbc: true` flag should route through `pyscf.pbc.gto.Cell` instead
of the molecular `pyscf.gto.M`. We verify by inspecting the SCF object
the backend builds — actual SCF convergence on a real periodic cell
is left to the slow `test_qm_pyscf.py` integration.
"""
from pathlib import Path

import pytest

from pyfield.config.schema import (
    AtomCfg,
    PyFieldConfig,
    QmCfg,
    StructureCfg,
)


pyscf = pytest.importorskip("pyscf")


def _backend(tmp_path):
    from pyfield.qm.pyscf_backend import PySCFBackend
    return PySCFBackend(QmCfg(code="pyscf", functional="lda", basis="sto-3g",
                              cache_dir=tmp_path / "cache"))


def test_cluster_mode_builds_molecular_scf(tmp_path):
    backend = _backend(tmp_path)
    s = StructureCfg(box=(10, 10, 10), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    mf = backend._build_mf(s)
    # Molecular SCF carries a `mol` attribute; PBC SCF carries `cell`.
    assert hasattr(mf, "mol")
    assert mf.mol.natm == 2


def test_pbc_mode_builds_periodic_scf(tmp_path):
    backend = _backend(tmp_path)
    s = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    mf = backend._build_pbc_mf(s)
    # Periodic SCF object has a `cell` (a pyscf.pbc.gto.Cell).
    assert hasattr(mf, "cell")
    cell = mf.cell
    assert cell.natm == 2
    # Lattice vectors round-trip — `cell.a` is in Bohr after build, so
    # we just check shape and that it's not zero.
    import numpy as np
    a = np.asarray(cell.a)
    assert a.shape == (3, 3)
    assert np.linalg.det(a) > 0


def test_dispatch_routes_on_pbc_flag(tmp_path):
    backend = _backend(tmp_path)
    cluster = StructureCfg(box=(10, 10, 10), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    periodic = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    assert backend._is_pbc(cluster) is False
    assert backend._is_pbc(periodic) is True


def test_cache_key_separates_pbc_and_cluster(tmp_path):
    """Same atoms + same box but different `pbc:` give different keys."""
    from pyfield.qm.cache import _key

    cluster = StructureCfg(box=(5, 5, 5), atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    periodic = StructureCfg(box=(5, 5, 5), pbc=True, atoms=[
        AtomCfg(element="H", x=0, y=0, z=0),
        AtomCfg(element="H", x=0, y=0, z=0.74),
    ])
    k_cluster = _key(cluster, "fp", "single_point")
    k_periodic = _key(periodic, "fp", "single_point")
    assert k_cluster != k_periodic
