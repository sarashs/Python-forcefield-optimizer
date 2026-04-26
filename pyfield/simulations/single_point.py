"""`type: single_point` — energy + per-atom forces at the input geometry.

A `run 0` LAMMPS evaluation. The dump file carries forces (`fx fy fz`)
so the `forces` objective can compute residuals against QM reference
forces.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pyfield.config.schema import SimulationCfg, StructureCfg
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


class SinglePointSimulation(Simulation):
    name = "single_point"

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
        return _env.get_template("single_point.in.j2").render(
            FFIELD_PATH=str(ffield_path),
            ELEMENTS=list(elements),
            DATA_FILE=str(data_file),
            LOG_FILE=str(log_file),
            DUMP_FILE=str(dump_file),
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
        return SimResult(
            sim_id=self.sim_id, energy=energy, charges=charges,
            extras={
                "dump_file": str(dump_file) if dump_file.exists() else None,
                "type_to_element": {i + 1: el for i, el in enumerate(elements)},
            },
        )
