"""Built-in `type: nvt` simulation backend.

Renders `templates/nvt.in.j2` against (structure, ffield) and runs an
NVT trajectory through LAMMPS. The trajectory dump is the *primary*
output — energy and final-frame charges still come back on `SimResult`
for objectives that care, and `extras['dump_file']` carries the path
to the LAMMPS native dump for trajectory-consuming objectives
(`coordination`, `rdf_peak`, MSD, …).
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


# Reasonable defaults so a minimal `type: nvt` block runs. Override by
# adding keys to the SimulationCfg in the YAML.
_DEFAULTS = {
    "timestep_fs": 0.25,
    "steps": 1000,
    "sample_every": 100,
    "tdamp": 100.0,           # NVT thermostat damping in time units
    "seed": 12345,
    # average_window is consumed by trajectory-objectives, not by the
    # simulation; we just ignore it here. Stored on the SimResult.extras
    # so objectives don't need a second config lookup.
    "average_window": None,
}


def _field(sim_cfg: SimulationCfg, key: str):
    extras = sim_cfg.__pydantic_extra__ or {}
    if key in extras:
        return extras[key]
    return _DEFAULTS[key]


class NvtSimulation(Simulation):
    name = "nvt"

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
        if "temperature" not in extras:
            raise ValueError(f"nvt simulation {self.sim_id!r} requires a `temperature` field")
        template = _env.get_template("nvt.in.j2")
        return template.render(
            FFIELD_PATH=str(ffield_path),
            ELEMENTS=list(elements),
            DATA_FILE=str(data_file),
            LOG_FILE=str(log_file),
            DUMP_FILE=str(dump_file),
            TEMPERATURE=float(extras["temperature"]),
            STEPS=int(_field(self.cfg, "steps")),
            TIMESTEP_FS=float(_field(self.cfg, "timestep_fs")),
            SAMPLE_EVERY=int(_field(self.cfg, "sample_every")),
            TDAMP=float(_field(self.cfg, "tdamp")),
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
            ffield_path=ffield_path,
            elements=elements,
            data_file=data_file,
            log_file=log_file,
            dump_file=dump_file,
        ))
        if runner is not None:
            energy, charges = runner.run_input_file(in_file)
        else:
            energy, charges = energy_charge(str(in_file))
        return SimResult(
            sim_id=self.sim_id,
            energy=energy,
            charges=charges,
            extras={
                "dump_file": str(dump_file),
                # type → element mapping so trajectory objectives can resolve
                # `central_type: Cl` against the integer types LAMMPS dumps.
                "type_to_element": {i + 1: el for i, el in enumerate(elements)},
                "average_window": _field(self.cfg, "average_window"),
                "sample_every": int(_field(self.cfg, "sample_every")),
                # `melting_onset` and other multi-NVT objectives need to
                # know what temperature this run was performed at.
                "temperature": float((self.cfg.__pydantic_extra__ or {}).get("temperature", 0.0)),
            },
        )
