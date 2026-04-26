"""User-supplied template (escape hatch): variable-leakage validator.
No LAMMPS — we exercise loading + rendering only.
"""
from pathlib import Path

import pytest

from pyfield.config.schema import SimulationCfg, StructureCfg
from pyfield.simulations.templated import TemplatedSimulation, VariableLeakageError


def _struct():
    return StructureCfg(
        box=(10, 10, 10),
        atoms=[{"element": "Cl", "x": 0, "y": 0, "z": -1.0},
               {"element": "Cl", "x": 0, "y": 0, "z": 1.0}],
    )


_OK_TEMPLATE = """\
units real
read_data {{ DATA_FILE }}
pair_style reaxff NULL
pair_coeff * * {{ FFIELD_PATH }}{% for el in ELEMENTS %} {{ el }}{% endfor %}
fix nvt all nvt temp {{ TEMP }} {{ TEMP }} 100
run {{ STEPS }}
"""


def test_template_loads_and_renders(tmp_path: Path):
    tmpl = tmp_path / "ok.in.j2"
    tmpl.write_text(_OK_TEMPLATE)
    cfg = SimulationCfg.model_validate({
        "structure": "x",
        "template": str(tmpl),
        "variables": {"TEMP": 300, "STEPS": 1000},
    })
    sim = TemplatedSimulation("sim1", cfg, _struct())
    rendered = sim.render(
        ffield_path=Path("ffield.reax"),
        elements=["Cl"],
        data_file=Path("/tmp/sim1.data"),
        log_file=Path("/tmp/sim1.log"),
        dump_file=Path("/tmp/sim1.lammpstrj"),
    )
    assert "ffield.reax" in rendered
    assert "Cl" in rendered
    assert "temp 300 300 100" in rendered
    assert "run 1000" in rendered


@pytest.mark.parametrize("body, n_expected", [
    ("read_data x.data\nrun 0\n", 0),
    ("pair_coeff * * {{ FFIELD_PATH }}\npair_coeff * * {{ FFIELD_PATH }}\nrun 0\n", 2),
])
def test_missing_or_double_ffield_path_rejected(tmp_path: Path, body, n_expected):
    tmpl = tmp_path / "bad.in.j2"
    tmpl.write_text(body)
    cfg = SimulationCfg.model_validate({
        "structure": "x",
        "template": str(tmpl),
    })
    with pytest.raises(VariableLeakageError, match=f"found {n_expected}"):
        TemplatedSimulation("sim1", cfg, _struct())
