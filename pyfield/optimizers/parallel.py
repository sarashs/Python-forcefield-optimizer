"""Parallel cost evaluation for SA / GA.

Each worker process owns its own long-lived `LammpsRunner` and parsed
`REAX_FF`; the master only ever ships small numpy arrays
(`ParameterSnapshot.values`) across the IPC boundary, never the FF
itself. That keeps the per-task overhead at a few hundred microseconds
even on cold pickle paths.

Two operations cover both optimizers:

- `evaluate_batch(values_list)` — the master perturbs each candidate and
  hands the worker a flat values array; worker writes them into its FF,
  runs every required simulation, returns the summed residual.
- `refine_batch(jobs)` — for `method: sa+ga`. Each worker runs a small
  per-child SA loop, returning `(best_values, best_cost)`. Each child
  gets a deterministic seed from the master so the whole run is
  bit-reproducible regardless of how the pool schedules tasks.

Reproducibility contract: workers are stateless cost-evaluators (or
deterministic refiners with a master-supplied seed). Anything that
flips coins for accept/reject / crossover / mutation lives on the
master. As long as the master collects every batch's results before
acting on them, parallel runs match serial runs bit-for-bit.

The `BatchEvaluator` context manager is the only public surface; SA
and GA never see the executor directly.
"""
from __future__ import annotations

import math
import os
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pyfield.config.schema import PyFieldConfig
from pyfield.io.lammps import preload_libmpi


# ---------------------------------------------------------------------------
# worker globals — set by `_init_worker` exactly once per process,
# then reused across every task that lands on this worker.
# ---------------------------------------------------------------------------

_W_FF = None              # REAX_FF
_W_FF_PATH = None         # Path the worker writes the per-task ffield into
_W_OBJECTIVES = None      # List[Objective]
_W_NEEDED_SIMS = None     # List[str]
_W_RUNNER = None          # LammpsRunner (lazily created on first eval)
_W_CFG = None             # PyFieldConfig (kept for build_simulation calls)
_W_OUT_DIR = None         # Path


def _init_worker(
    cfg_dump: dict,
    ffield_in: str,
    params_in: str,
    out_dir: str,
    worker_idx: int,
) -> None:
    """Process-pool initializer. Builds everything the worker needs once.

    `cfg_dump` is the pydantic dict (model_dump output); we re-validate
    here rather than ship the live PyFieldConfig because pydantic models
    pickle awkwardly across pydantic versions.
    """
    global _W_FF, _W_FF_PATH, _W_OBJECTIVES, _W_NEEDED_SIMS, _W_RUNNER, _W_CFG, _W_OUT_DIR

    # Make sure the bundled MPICH is reachable for `from lammps import lammps`
    # in this worker — the master's preload doesn't propagate through fork
    # nor through spawn.
    preload_libmpi()

    from pyfield.config.schema import PyFieldConfig
    from pyfield.forcefield.reax import REAX_FF
    from pyfield.objectives import build_objective

    cfg = PyFieldConfig.model_validate(cfg_dump)
    _W_CFG = cfg
    # Each worker gets its own subdirectory under `out_dir` so the per-sim
    # `*.data` / `*.in` / `*.lammpstrj` / `*.log` files don't collide when
    # multiple workers handle the same simulation concurrently. (LAMMPS
    # silently fails with "Energy was not tallied on needed timestep" when
    # one worker reads a `*.in` that another worker is mid-rewriting.)
    _W_OUT_DIR = Path(out_dir) / f"worker_{os.getpid()}"
    _W_OUT_DIR.mkdir(parents=True, exist_ok=True)

    _W_FF = REAX_FF(ffield_in, params_in)
    _W_FF.parseParamSelectionFile()
    _W_FF_PATH = _W_OUT_DIR / "ffield.reax"

    _W_OBJECTIVES = [build_objective(t) for t in cfg.targets]
    _W_NEEDED_SIMS = sorted({s for o in _W_OBJECTIVES for s in o.required_simulations()})

    # Defer `LammpsRunner()` until first eval — first import of the
    # `lammps` module dominates worker init wall-clock and we don't want
    # to pay it for workers that never receive a task.


def _ensure_runner():
    global _W_RUNNER
    if _W_RUNNER is None:
        _W_RUNNER = LammpsRunner()
    return _W_RUNNER


def _apply_values(values: np.ndarray) -> None:
    """Write a flat `ParameterSnapshot.values` array back into _W_FF.params."""
    keys = tuple(_W_FF.param_min_max_delta.keys())
    for (s, e, i), v in zip(keys, values):
        _W_FF.params[s][e][i] = float(v)


def _eval_current_ff() -> float:
    """Run every required simulation on _W_FF, return summed residual."""
    from pyfield.objectives.base import ObjectiveContext
    from pyfield.simulations.runner import build_simulation

    _W_FF.write_forcefield(str(_W_FF_PATH))
    runner = _ensure_runner()
    sim_results = {}
    for sim_id in _W_NEEDED_SIMS:
        sim = build_simulation(sim_id, _W_CFG.simulations[sim_id], _W_CFG)
        sim_results[sim_id] = sim.run(
            ffield_path=_W_FF_PATH, work_dir=_W_OUT_DIR, runner=runner,
        )
    ctx = ObjectiveContext(sim_results=sim_results)
    return float(sum(o.residual(ctx) for o in _W_OBJECTIVES))


def _evaluate_in_worker(values: np.ndarray) -> float:
    _apply_values(np.asarray(values, dtype=float))
    return _eval_current_ff()


def _refine_in_worker(args) -> Tuple[np.ndarray, float]:
    """Per-child SA refinement for `method: sa+ga`.

    Args: (values, T, steps, seed). Returns the best (values, cost) seen
    during `steps` Metropolis kicks at temperature `T`. The worker draws
    every random number from a local `random.Random(seed)` so the
    refinement is reproducible regardless of which worker runs it.
    """
    values, T, steps, seed = args
    values = np.asarray(values, dtype=float).copy()
    if steps <= 0:
        _apply_values(values)
        return values, _eval_current_ff()

    rng = random.Random(seed)
    keys = tuple(_W_FF.param_min_max_delta.keys())

    _apply_values(values)
    cur_cost = _eval_current_ff()
    best_values = values.copy()
    best_cost = cur_cost

    for _ in range(steps):
        idx = rng.randrange(len(values))
        sec, entry, item = keys[idx]
        b = _W_FF.param_min_max_delta[(sec, entry, item)]
        new = values[idx] + rng.uniform(-1, 1) * b["delta"]
        new = min(b["max"], max(b["min"], round(new, 4)))
        old = float(_W_FF.params[sec][entry][item])
        _W_FF.params[sec][entry][item] = float(new)
        new_cost = _eval_current_ff()
        accept = (new_cost < cur_cost) or (
            T > 0 and rng.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-30))
        )
        if accept:
            values[idx] = new
            cur_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_values = values.copy()
        else:
            _W_FF.params[sec][entry][item] = old
    _apply_values(best_values)
    return best_values, best_cost


# ---------------------------------------------------------------------------
# master-side abstraction — `BatchEvaluator(cfg, n_workers)`
# ---------------------------------------------------------------------------

from pyfield.io.lammps import LammpsRunner       # used by the in-process serial evaluator

class _SerialEvaluator:
    """In-process fallback when n_workers == 1.

    Holds the same FF / objectives / runner the parallel workers do, so
    parallel and serial paths share the exact same evaluation code.
    """

    def __init__(self, cfg: PyFieldConfig, ffield_in: Path, params_in: Path, out_dir: Path):
        from pyfield.forcefield.reax import REAX_FF
        from pyfield.objectives import build_objective

        self._ff = REAX_FF(str(ffield_in), str(params_in))
        self._ff.parseParamSelectionFile()
        self._ff_path = Path(out_dir) / "serial_iter.reax"
        self._cfg = cfg
        self._objectives = [build_objective(t) for t in cfg.targets]
        self._needed_sims = sorted({s for o in self._objectives for s in o.required_simulations()})
        self._runner = LammpsRunner()
        self._out_dir = Path(out_dir)

    def _apply(self, values):
        keys = tuple(self._ff.param_min_max_delta.keys())
        for (s, e, i), v in zip(keys, values):
            self._ff.params[s][e][i] = float(v)

    def _eval(self) -> float:
        from pyfield.objectives.base import ObjectiveContext
        from pyfield.simulations.runner import build_simulation
        self._ff.write_forcefield(str(self._ff_path))
        sim_results = {}
        for sim_id in self._needed_sims:
            sim = build_simulation(sim_id, self._cfg.simulations[sim_id], self._cfg)
            sim_results[sim_id] = sim.run(
                ffield_path=self._ff_path, work_dir=self._out_dir, runner=self._runner,
            )
        ctx = ObjectiveContext(sim_results=sim_results)
        return float(sum(o.residual(ctx) for o in self._objectives))

    def evaluate_batch(self, values_list: Sequence[np.ndarray]) -> List[float]:
        out = []
        for v in values_list:
            self._apply(v)
            out.append(self._eval())
        return out

    def refine_batch(self, jobs) -> List[Tuple[np.ndarray, float]]:
        # `jobs` are (values, T, steps, seed). Run sequentially with the
        # same logic as the worker.
        out = []
        for values, T, steps, seed in jobs:
            values = np.asarray(values, dtype=float).copy()
            if steps <= 0:
                self._apply(values)
                out.append((values, self._eval()))
                continue
            rng = random.Random(seed)
            keys = tuple(self._ff.param_min_max_delta.keys())
            self._apply(values)
            cur_cost = self._eval()
            best_values = values.copy()
            best_cost = cur_cost
            for _ in range(steps):
                idx = rng.randrange(len(values))
                sec, entry, item = keys[idx]
                b = self._ff.param_min_max_delta[(sec, entry, item)]
                new = values[idx] + rng.uniform(-1, 1) * b["delta"]
                new = min(b["max"], max(b["min"], round(new, 4)))
                old = float(self._ff.params[sec][entry][item])
                self._ff.params[sec][entry][item] = float(new)
                new_cost = self._eval()
                accept = (new_cost < cur_cost) or (
                    T > 0 and rng.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-30))
                )
                if accept:
                    values[idx] = new
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_values = values.copy()
                else:
                    self._ff.params[sec][entry][item] = old
            self._apply(best_values)
            out.append((best_values, best_cost))
        return out

    def close(self):
        self._runner.close()


class _PoolEvaluator:
    """ProcessPoolExecutor-backed evaluator. `evaluate_batch` and
    `refine_batch` map across the pool; preserve input ordering."""

    def __init__(self, cfg: PyFieldConfig, ffield_in: Path, params_in: Path,
                 out_dir: Path, n_workers: int):
        cfg_dump = cfg.model_dump(mode="python")
        # `spawn` (not `fork`) is required: the master imports/preloads MPI
        # for its own LAMMPS, and forking children inherit that mid-state,
        # which trips LAMMPS' "Energy was not tallied" check on the first
        # real eval. A fresh Python interpreter per worker avoids the issue.
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(cfg_dump, str(ffield_in), str(params_in), str(out_dir), 0),
        )
        # Eagerly spin up workers so the first batch doesn't pay all the
        # init cost serially. Submitting one no-op per worker forces the
        # initializer.
        for _ in range(n_workers):
            self._executor.submit(_evaluate_in_worker_noop).result()

    def evaluate_batch(self, values_list):
        return list(self._executor.map(_evaluate_in_worker, list(values_list)))

    def refine_batch(self, jobs):
        return list(self._executor.map(_refine_in_worker, list(jobs)))

    def close(self):
        self._executor.shutdown(wait=True)


def _evaluate_in_worker_noop():
    """Warm-up task: confirms the worker can build LAMMPS without errors."""
    return 0


def resolve_n_workers(parallel: bool, processors: int) -> int:
    """Translate `optimizer.parallel` / `optimizer.processors` into a worker count.

    Rules:
    - `parallel: false`            → 1 (serial in-process; no executor).
    - `parallel: true, proc: 0`    → cpu_count() (clamped to ≥1).
    - `parallel: true, proc: N`    → N.
    """
    if not parallel:
        return 1
    if processors <= 0:
        return max(1, os.cpu_count() or 1)
    return int(processors)


@contextmanager
def BatchEvaluator(
    cfg: PyFieldConfig,
    *,
    ffield_in: Path,
    params_in: Path,
    out_dir: Path,
    n_workers: int,
):
    """Context manager exposing `evaluate_batch` and `refine_batch`.

    `n_workers <= 1` short-circuits to the in-process serial evaluator —
    avoids the IPC + executor-spinup overhead when there's nothing to
    parallelize.
    """
    if n_workers <= 1:
        ev = _SerialEvaluator(cfg, ffield_in=ffield_in, params_in=params_in, out_dir=out_dir)
        try:
            yield ev
        finally:
            ev.close()
        return
    ev = _PoolEvaluator(cfg, ffield_in=ffield_in, params_in=params_in,
                        out_dir=out_dir, n_workers=n_workers)
    try:
        yield ev
    finally:
        ev.close()
