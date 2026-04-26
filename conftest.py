"""Repo-root conftest. Preloads `libmpi.so.12` once for the whole pytest
session so any test that imports `lammps` works without an external
`LD_LIBRARY_PATH`. The legacy sys.path hack was removed once the legacy
top-level modules were deleted — `pyfield` is now pip-installed and
imports normally.
"""
import ctypes
import os
import sys

_libmpi = os.path.join(sys.prefix, "lib", "libmpi.so.12")
if os.path.exists(_libmpi):
    try:
        ctypes.CDLL(_libmpi, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # the @pytest.mark.lammps tests will skip if lammps isn't usable
