"""Built-in `type: minimize` simulation backend.

Renders `templates/minimize.in.j2` against (structure, ffield) and runs it
through LAMMPS. Returns a `SimResult(energy=…, charges=[…])` — the same
shape every other backend produces.
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
    undefined=StrictUndefined,    # missing variable → render error, never silent
    trim_blocks=False,
    lstrip_blocks=False,
)


class MinimizeSimulation(Simulation):
    name = "minimize"

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
        template = _env.get_template("minimize.in.j2")
        return template.render(
            FFIELD_PATH=str(ffield_path),
            ELEMENTS=list(elements),
            DATA_FILE=str(data_file),
            LOG_FILE=str(log_file),
            DUMP_FILE=str(dump_file),
            # `hftn` (Hessian-free truncated Newton) is the default minimizer
            # because `cg` deadlocks LAMMPS' setup phase on PBC ReaxFF cells
            # whose seed FF is far from physical (verified empirically on the
            # GST_rocksalt cell with the placeholder-fitted GST seed: cg hangs
            # indefinitely in `Min::setup`'s pre-iteration force eval, hftn
            # converges in 0.26 s, fire in 4.6 s, sd hangs). Per-sim override
            # is still possible via `min_style:` in the simulation's YAML.
            MIN_STYLE=self.cfg.__pydantic_extra__.get("min_style", "hftn"),
            RESTRAINTS=self.cfg.__pydantic_extra__.get("restraints") or [],
        )

    def run(
        self,
        *,
        ffield_path: Path,
        work_dir: Path,
        runner: Optional[LammpsRunner] = None,
    ) -> SimResult:
        """Run this simulation.

        If `runner` is given, it's reused (Phase-2 long-lived LAMMPS); else
        a one-shot runner is created and closed for backwards compatibility.
        """
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
                # Single-frame dump containing the post-minimization
                # geometry — used by `structural_match` and similar.
                "dump_file": str(dump_file) if dump_file.exists() else None,
                "type_to_element": {i + 1: el for i, el in enumerate(elements)},
            },
        )
