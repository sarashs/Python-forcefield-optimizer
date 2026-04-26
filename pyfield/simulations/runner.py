"""Picks the right backend per `SimulationCfg` and dispatches.

Phase 1 only knows about `type: minimize`. Adding a backend in Phase 3+
is a single registry entry plus a Jinja template under `templates/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyfield.config.schema import PyFieldConfig, SimulationCfg
from pyfield.io.lammps import LammpsRunner
from pyfield.simulations.base import SimResult, Simulation
from pyfield.simulations.minimize import MinimizeSimulation
from pyfield.simulations.npt import NptSimulation
from pyfield.simulations.nvt import NvtSimulation
from pyfield.simulations.single_point import SinglePointSimulation
from pyfield.simulations.templated import TemplatedSimulation


_BUILTIN_BACKENDS = {
    "minimize": MinimizeSimulation,
    "nvt": NvtSimulation,
    "npt": NptSimulation,
    "single_point": SinglePointSimulation,
}


def build_simulation(sim_id: str, sim_cfg: SimulationCfg, cfg: PyFieldConfig) -> Simulation:
    """Resolve a SimulationCfg → ready-to-run `Simulation` instance."""
    structure = cfg.structures[sim_cfg.structure]
    if sim_cfg.template is not None:
        return TemplatedSimulation(sim_id, sim_cfg, structure)
    backend = _BUILTIN_BACKENDS.get(sim_cfg.type)
    if backend is None:
        raise NotImplementedError(
            f"simulation type {sim_cfg.type!r} is reserved but not yet implemented. "
            f"Built-ins available: {sorted(_BUILTIN_BACKENDS)}."
        )
    return backend(sim_id, sim_cfg, structure)


def run_simulation(
    sim_id: str,
    cfg: PyFieldConfig,
    *,
    ffield_path: Path,
    work_dir: Path,
    runner: Optional[LammpsRunner] = None,
) -> SimResult:
    sim = build_simulation(sim_id, cfg.simulations[sim_id], cfg)
    return sim.run(ffield_path=ffield_path, work_dir=work_dir, runner=runner)
