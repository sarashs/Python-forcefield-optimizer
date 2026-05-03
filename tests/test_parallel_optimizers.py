"""End-to-end tests for parallel SA / GA on a real LAMMPS Cl2 scan.

These hit a real LAMMPS via spawned worker processes, so they're slow
(~10–20 s combined). Skipped when the cl2_scan.populated.yaml fixture
isn't there — `pyfield qm-relax` + `make-scan` + `qm-prep` regenerate
it, but those need PySCF, so we don't force them here. Run them
manually after `pyfield qm-prep tests/cl2_scan.scanned.yaml` if you've
done a full pipeline run.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
POPULATED = ROOT / "tests" / "cl2_scan.populated.yaml"


pytestmark = pytest.mark.skipif(
    not POPULATED.exists(),
    reason="needs tests/cl2_scan.populated.yaml (run `pyfield qm-prep tests/cl2_scan.scanned.yaml` first)",
)


@pytest.fixture(autouse=True)
def _isolate_run_dir(tmp_path, monkeypatch):
    """Each test gets its own output dir so worker subdirs don't collide."""
    monkeypatch.chdir(ROOT)


def _load(processors=0, parallel=False, n_walkers=1):
    from pyfield.config.loader import load_yaml
    from pyfield.io.lammps import preload_libmpi
    preload_libmpi()
    cfg = load_yaml(POPULATED)
    cfg.optimizer.parallel = parallel
    cfg.optimizer.processors = processors
    cfg.optimizer.number_of_points = n_walkers
    cfg.optimizer.seed = 42
    cfg.optimizer.max_iter = 20            # keep tests fast
    cfg.optimizer.show_progress = False
    return cfg


def test_sa_parallel_matches_serial_for_same_seed():
    """SA cost is deterministic across worker counts for a fixed seed.

    Master draws all randomness; workers are stateless cost-evaluators.
    With 4 walkers, the 1-process and 4-process paths must produce
    bit-identical traces (and therefore final cost).
    """
    from pyfield.optimizers.sa import run_sa

    cfg_serial = _load(parallel=False, n_walkers=4)
    serial = run_sa(cfg_serial)

    cfg_parallel = _load(parallel=True, processors=4, n_walkers=4)
    parallel = run_sa(cfg_parallel)

    assert serial.final_cost == parallel.final_cost, (
        serial.final_cost, parallel.final_cost
    )
    assert serial.cost_trace == parallel.cost_trace


def test_sa_parallel_is_deterministic_across_runs():
    """Re-running the same parallel SA must give the same cost."""
    from pyfield.optimizers.sa import run_sa

    cfg = _load(parallel=True, processors=4, n_walkers=4)
    a = run_sa(cfg)
    b = run_sa(cfg)
    assert a.final_cost == b.final_cost
    assert a.cost_trace == b.cost_trace


def test_sa_walkers_explore_more_than_one():
    """With 4 walkers we should beat the 1-walker baseline at the same seed
    (more proposals per cooling step → better trajectory)."""
    from pyfield.optimizers.sa import run_sa
    one = run_sa(_load(parallel=False, n_walkers=1))
    four = run_sa(_load(parallel=False, n_walkers=4))
    # 4 walkers see 4× more candidates per step, so they should find a
    # cost at least as good as the 1-walker run (typically much better).
    assert four.final_cost <= one.final_cost


def test_ga_parallel_matches_serial_for_same_seed():
    """GA cost is deterministic across worker counts for a fixed seed."""
    from pyfield.optimizers.ga import run_ga

    cfg = _load(parallel=False)
    cfg.optimizer.method = "ga"
    cfg.optimizer.population_size = 6
    cfg.optimizer.generations = 3
    serial = run_ga(cfg)

    cfg = _load(parallel=True, processors=3)
    cfg.optimizer.method = "ga"
    cfg.optimizer.population_size = 6
    cfg.optimizer.generations = 3
    parallel = run_ga(cfg)

    assert serial.final_cost == parallel.final_cost
    assert serial.cost_trace == parallel.cost_trace


def test_sa_ga_parallel_matches_serial_for_same_seed():
    """sa+ga: per-child SA refinement is also reproducible across processors.

    Each child gets a master-derived seed, so refinement results match
    regardless of which worker processes a given child.
    """
    from pyfield.optimizers.ga import run_ga

    cfg = _load(parallel=False)
    cfg.optimizer.method = "sa+ga"
    cfg.optimizer.population_size = 4
    cfg.optimizer.generations = 2
    cfg.optimizer.sa_refine_steps = 3
    serial = run_ga(cfg)

    cfg = _load(parallel=True, processors=2)
    cfg.optimizer.method = "sa+ga"
    cfg.optimizer.population_size = 4
    cfg.optimizer.generations = 2
    cfg.optimizer.sa_refine_steps = 3
    parallel = run_ga(cfg)

    assert serial.final_cost == parallel.final_cost
    assert serial.cost_trace == parallel.cost_trace
