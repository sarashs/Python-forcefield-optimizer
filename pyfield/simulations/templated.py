"""User-supplied Jinja templates — the escape hatch for non-standard runs.

The variable-leakage contract from DEV.md §7:

- `{{ FFIELD_PATH }}` must appear **exactly once** in the template (the
  one `pair_coeff` line). Zero means the iterating force-field is being
  ignored; more than one means something is being redefined.
- The runner injects `FFIELD_PATH`, `DATA_FILE`, `ELEMENTS` (alongside
  whatever the user puts under `variables:`). Any optimizer-tuned value
  must come through `variables:` — never hard-coded into the template.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pyfield.config.schema import SimulationCfg, StructureCfg
from pyfield.io.lammps import LammpsRunner, energy_charge
from pyfield.io.structures import write_lammps_data
from pyfield.simulations.base import SimResult, Simulation


_FFIELD_TOKEN = re.compile(r"\{\{\s*FFIELD_PATH\s*\}\}")


class VariableLeakageError(ValueError):
    """Raised when a user template doesn't reference {{ FFIELD_PATH }} exactly once."""


def _validate_template(text: str, *, source: Path) -> None:
    n = len(_FFIELD_TOKEN.findall(text))
    if n != 1:
        raise VariableLeakageError(
            f"{source}: a templated simulation must reference "
            f"{{{{ FFIELD_PATH }}}} exactly once (found {n}). "
            "See DEV.md §7 — variable-leakage contract."
        )


class TemplatedSimulation(Simulation):
    name = "templated"

    def __init__(self, sim_id: str, sim_cfg: SimulationCfg, structure: StructureCfg):
        if sim_cfg.template is None:
            raise ValueError("TemplatedSimulation requires sim_cfg.template")
        self.sim_id = sim_id
        self.cfg = sim_cfg
        self.structure = structure
        self.template_path = Path(sim_cfg.template)
        # Load and validate up-front so a broken template fails the cost
        # evaluation immediately, not deep in a LAMMPS error message.
        text = self.template_path.read_text()
        _validate_template(text, source=self.template_path)
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_path.parent)),
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        self._template = self._env.get_template(self.template_path.name)

    def render(
        self,
        *,
        ffield_path: Path,
        elements: Sequence[str],
        data_file: Path,
        log_file: Path,
        dump_file: Path,
    ) -> str:
        ctx = dict(self.cfg.variables)  # user-declared
        # Runner-injected. User may not override these via `variables:`.
        for k, v in {
            "FFIELD_PATH": str(ffield_path),
            "ELEMENTS": list(elements),
            "DATA_FILE": str(data_file),
            "LOG_FILE": str(log_file),
            "DUMP_FILE": str(dump_file),
        }.items():
            ctx[k] = v
        return self._template.render(**ctx)

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
                # We can't know whether the user template wrote a dump,
                # but if it did it'll be at this conventional path.
                "dump_file": str(dump_file) if dump_file.exists() else None,
                "type_to_element": {i + 1: el for i, el in enumerate(elements)},
            },
        )
