"""CMA-ES end-to-end tests against a real LAMMPS Cl₂ scan.

Mirrors the parallel-optimizer tests for SA / GA: serial vs parallel
must give bit-identical costs at a fixed seed; CMA must reduce the
initial cost meaningfully on a small problem.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
POPULATED = ROOT / "tests" / "cl2_scan.populated.yaml"

cma_module = pytest.importorskip("cma")


pytestmark = pytest.mark.skipif(
    not POPULATED.exists(),
    reason="needs tests/cl2_scan.populated.yaml (run `pyfield qm-prep "
           "tests/cl2_scan.scanned.yaml` first)",
)


@pytest.fixture(autouse=True)
def _chdir_to_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _load(processors=0, parallel=False, generations=4, popsize=4):
    from pyfield.config.loader import load_yaml
    from pyfield.io.lammps import preload_libmpi
    preload_libmpi()
    cfg = load_yaml(POPULATED)
    cfg.optimizer.method = "cma"
    cfg.optimizer.parallel = parallel
    cfg.optimizer.processors = processors
    cfg.optimizer.generations = generations
    cfg.optimizer.cma_sigma0 = 0.3
    cfg.optimizer.cma_popsize = popsize
    cfg.optimizer.seed = 42
    cfg.optimizer.show_progress = False
    return cfg


def test_cma_parallel_matches_serial_for_same_seed():
    """CMA cost is deterministic across worker counts when seeded."""
    from pyfield.optimizers.cma import run_cma

    serial = run_cma(_load(parallel=False))
    parallel = run_cma(_load(parallel=True, processors=4))
    assert serial.final_cost == parallel.final_cost, (
        serial.final_cost, parallel.final_cost
    )
    assert serial.cost_trace == parallel.cost_trace


def test_cma_is_deterministic_across_runs():
    """Re-running the same parallel CMA must give the same cost."""
    from pyfield.optimizers.cma import run_cma
    cfg = _load(parallel=True, processors=4)
    a = run_cma(cfg)
    b = run_cma(cfg)
    assert a.final_cost == b.final_cost
    assert a.cost_trace == b.cost_trace


def test_cma_reduces_initial_cost():
    """4 generations × popsize 4 = 16 evals; cost should beat the seed FF."""
    from pyfield.diagnostics import cost_breakdown
    from pyfield.optimizers.cma import run_cma

    cfg = _load(parallel=True, processors=4, generations=4, popsize=4)
    initial = cost_breakdown(cfg).total_cost
    result = run_cma(cfg)
    assert result.final_cost < initial, (initial, result.final_cost)


def test_cma_writes_best_ffield():
    """The best parameter set is written to bestFF.reax under output.dir."""
    from pyfield.optimizers.cma import run_cma
    cfg = _load(parallel=False, generations=2, popsize=4)
    result = run_cma(cfg)
    assert result.best_ffield_path is not None
    assert result.best_ffield_path.exists()
    assert result.best_ffield_path.read_text(encoding="utf-8")
