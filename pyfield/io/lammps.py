"""LAMMPS-side I/O.

`preload_libmpi` makes the bundled MPICH wheel work without an external
`LD_LIBRARY_PATH`. `LammpsRunner` keeps one long-lived `lammps()`
instance and resets state with `clear` between input files — the legacy
code spawned a fresh `lammps()` per call, which was the dominant cost
in the inner SA loop (LAMMPS init dominates over the actual minimization
on 2-atom Cl₂).
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def preload_libmpi() -> bool:
    """Load `<sys.prefix>/lib/libmpi.so.12` with RTLD_GLOBAL if present.

    glibc caches `LD_LIBRARY_PATH` at process start; setting it from
    inside Python is too late. Loading the lib explicitly puts the symbols
    in the global namespace before liblammps's DT_NEEDED resolves.
    """
    libmpi = os.path.join(sys.prefix, "lib", "libmpi.so.12")
    if not os.path.exists(libmpi):
        return False
    try:
        ctypes.CDLL(libmpi, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


class LammpsRunner:
    """Holds one `lammps()` instance and reuses it across input files.

    Use either as a context manager or call `.close()` explicitly. `clear`
    between runs deletes atoms / fixes / computes / dumps, so each input
    file gets a clean LAMMPS environment without paying the spawn cost.
    """

    def __init__(self, quiet: bool = True) -> None:
        self._lmp = None
        self._quiet = quiet

    def _instance(self):
        if self._lmp is None:
            from lammps import lammps  # imported lazily — not every test path needs it
            cmdargs = ["-screen", "none", "-log", "none"] if self._quiet else []
            self._lmp = lammps(cmdargs=cmdargs)
        return self._lmp

    def run_input_file(self, path: Path | str) -> Tuple[float, List[float]]:
        """Run a LAMMPS input file and return (etotal, [per-atom charges]).

        Charges come back as `[]` if the atom style has no `q` property
        (`atomic`, `bond`, ...). Energy is always returned.
        """
        lmp = self._instance()
        lmp.command("clear")
        lmp.file(str(path))
        energy = round(lmp.get_thermo("etotal"), 5)
        try:
            q = lmp.gather_atoms("q", 1, 1)
            charges = [q[i] for i in range(len(q))]
        except Exception:
            charges = []
        return energy, charges

    def close(self) -> None:
        if self._lmp is not None:
            try:
                self._lmp.close()
            finally:
                self._lmp = None

    def __enter__(self) -> "LammpsRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Backwards-compat: legacy `energy_charge(path)` users get a per-call instance.
# Kept so the pre-Phase-1 LAMMPS_Utils shim and any straggler imports keep
# working until the legacy modules are removed.
# ---------------------------------------------------------------------------

def energy_charge(input_file_path):
    """Legacy single-shot: run an input file, return [energy, [charges]]."""
    runner = LammpsRunner()
    try:
        energy, charges = runner.run_input_file(input_file_path)
    finally:
        runner.close()
    return [energy, charges]
