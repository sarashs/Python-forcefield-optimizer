"""Seeded smoke through the new YAML pipeline. LAMMPS-marked."""
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


lammps = pytest.importorskip("lammps", reason="LAMMPS not installed/loadable")


@pytest.mark.lammps
def test_yaml_smoke_is_reproducible(tmp_path):
    """Two seeded YAML runs must produce the same final cost."""
    from pyfield.config.loader import load_yaml
    from pyfield.optimizers.sa import run_sa

    cfg = load_yaml(os.path.join(ROOT, "tests/cl2.yaml"))

    def one_run(out_dir: Path) -> float:
        cfg.output.dir = out_dir
        return run_sa(cfg).final_cost

    c1 = one_run(tmp_path / "run1")
    c2 = one_run(tmp_path / "run2")
    assert c1 == pytest.approx(c2, rel=0, abs=1e-9), (c1, c2)
    assert 0.0 < c1 < 1e9   # finite, non-degenerate
