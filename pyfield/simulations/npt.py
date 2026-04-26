"""Built-in `type: npt` simulation backend.

Same shape as NVT but adds box dynamics via `fix npt … iso`. SimResult
extras carry `temperature`, `pressure`, and the equilibrated mean
volume (read off the dump's BOX BOUNDS) for downstream EOS objectives.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pyfield.config.schema import SimulationCfg, StructureCfg
from pyfield.io.dump import read_dump
from pyfield.io.lammps import LammpsRunner, energy_charge
from pyfield.io.structures import write_lammps_data
from pyfield.simulations.base import SimResult, Simulation


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
)


_DEFAULTS = {
    "timestep_fs": 0.25,
    "steps": 1000,
    "sample_every": 100,
    "tdamp": 100.0,
    "pdamp": 1000.0,
    "seed": 12345,
    "average_window": None,
}


def _field(sim_cfg: SimulationCfg, key: str):
    extras = sim_cfg.__pydantic_extra__ or {}
    return extras.get(key, _DEFAULTS[key])


def _mean_volume_from_dump(dump_path: str, window) -> float:
    """Average the box volume over the post-equilibration frames."""
    vols = []
    for frame in read_dump(dump_path):
        if window is not None:
            lo, hi = window
            if frame.timestep < lo or frame.timestep > hi:
                continue
        vols.append(frame.box[0] * frame.box[1] * frame.box[2])
    if not vols:
        return 0.0
    return float(sum(vols) / len(vols))


class NptSimulation(Simulation):
    name = "npt"

    def __init__(self, sim_id: str, sim_cfg: SimulationCfg, structure: StructureCfg):
        self.sim_id = sim_id
        self.cfg = sim_cfg
        self.structure = structure

    def render(
        self,
        *,
        ffield_path: Path,
        elements: Sequence[str],
        data_file: Path,
        log_file: Path,
        dump_file: Path,
    ) -> str:
        extras = self.cfg.__pydantic_extra__ or {}
        for required in ("temperature", "pressure"):
            if required not in extras:
                raise ValueError(
                    f"npt simulation {self.sim_id!r} requires a `{required}` field"
                )
        return _env.get_template("npt.in.j2").render(
            FFIELD_PATH=str(ffield_path),
            ELEMENTS=list(elements),
            DATA_FILE=str(data_file),
            LOG_FILE=str(log_file),
            DUMP_FILE=str(dump_file),
            TEMPERATURE=float(extras["temperature"]),
            PRESSURE=float(extras["pressure"]),
            STEPS=int(_field(self.cfg, "steps")),
            TIMESTEP_FS=float(_field(self.cfg, "timestep_fs")),
            SAMPLE_EVERY=int(_field(self.cfg, "sample_every")),
            TDAMP=float(_field(self.cfg, "tdamp")),
            PDAMP=float(_field(self.cfg, "pdamp")),
            SEED=int(_field(self.cfg, "seed")),
        )

    def run(
        self,
        *,
        ffield_path: Path,
        work_dir: Path,
        runner: Optional[LammpsRunner] = None,
    ) -> SimResult:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        data_file = work_dir / f"{self.sim_id}.data"
        in_file = work_dir / f"{self.sim_id}.in"
        log_file = work_dir / f"{self.sim_id}.log"
        dump_file = work_dir / f"{self.sim_id}.lammpstrj"

        elements = write_lammps_data(self.structure, data_file)
        in_file.write_text(self.render(
            ffield_path=ffield_path, elements=elements,
            data_file=data_file, log_file=log_file, dump_file=dump_file,
        ))
        if runner is not None:
            energy, charges = runner.run_input_file(in_file)
        else:
            energy, charges = energy_charge(str(in_file))

        extras = self.cfg.__pydantic_extra__ or {}
        window = _field(self.cfg, "average_window")
        mean_volume = _mean_volume_from_dump(str(dump_file), window) if dump_file.exists() else 0.0

        return SimResult(
            sim_id=self.sim_id, energy=energy, charges=charges,
            extras={
                "dump_file": str(dump_file),
                "type_to_element": {i + 1: el for i, el in enumerate(elements)},
                "average_window": window,
                "sample_every": int(_field(self.cfg, "sample_every")),
                "temperature": float(extras["temperature"]),
                "pressure": float(extras["pressure"]),
                "mean_volume": mean_volume,
            },
        )
