"""Rescue: stash a preserved-on-failure QE vc-relax run into the QM cache.

When ASE's subprocess wrapper raised on QE's non-zero exit, the QE
output (including a valid final image) was preserved at
`/tmp/qe_vcrelax_failed_<...>/`. After patching the backend to tolerate
that exit code, the natural fix would be to re-run — but a single
`vc-relax` on the GST cell takes ~2 hours of compute. This script reads
the preserved `espresso.pwo`, builds a `QmRelaxResult`, and writes it
into the QM cache directly so the next `relax_structures` call short-
circuits as a cache hit.

Usage:
    python scripts/rescue_qe_vcrelax.py \
        --cfg studies/gst_drift/gst_drift.yaml \
        --structure GST_rocksalt \
        --pwo /tmp/qe_vcrelax_failed_qe_vcrelax_gdo13q8p/espresso.pwo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from pyfield.config.loader import load_yaml
from pyfield.qm.base import make_backend, structure_code
from pyfield.qm.cache import QmCache, _key
from pyfield.qm.base import QmRelaxResult
from pyfield.qm.prep import _relax_op


_EV_TO_KCAL = 23.06054783


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", required=True, help="path to the source YAML (e.g. gst_drift.yaml)")
    p.add_argument("--structure", required=True, help="structure name in the YAML")
    p.add_argument("--pwo", required=True, help="path to preserved espresso.pwo")
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    if cfg.qm is None:
        raise SystemExit("config has no `qm:` block")
    if args.structure not in cfg.structures:
        raise SystemExit(f"unknown structure {args.structure!r}")

    struct = cfg.structures[args.structure]

    # Read the relaxed cell + atoms + final SCF energy from the QE output.
    final = ase_read(args.pwo, index=-1)
    e_ev = float(final.get_potential_energy())
    cell = np.asarray(final.get_cell().array, dtype=float)
    # Mirror the tolerance in pyfield.qm.qe_backend._relax_vc:
    # off-diagonals < 5 % of axis length → project to diagonal with
    # a warning; ≥ 5 % → refuse (the diagonal would mis-describe the
    # cell). 5 % is the regime where "essentially orthorhombic plus
    # noise from a `cell_dofree: 'all'` run on a low-symmetry input"
    # is faithfully summarised by [a, b, c].
    off = cell - np.diag(np.diag(cell))
    diag = np.abs(np.diag(cell))
    rel_shear = float(np.max(np.abs(off) / diag.max()))
    if rel_shear >= 0.05:
        raise SystemExit(
            f"non-orthorhombic relaxed cell (max shear {rel_shear:.1%}) — "
            f"schema box: [a, b, c] won't carry it faithfully:\n{cell}"
        )
    if rel_shear > 1e-4:
        print(f"warning: relaxed cell has {rel_shear:.1%} shear; "
              "projecting to box diagonal.")
    new_box = (float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2]))
    coords = final.get_positions()

    new_atoms = [
        orig.model_copy(update={
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
        })
        for i, orig in enumerate(struct.atoms)
    ]
    new_structure = struct.model_copy(update={
        "atoms": new_atoms,
        "box": new_box,
        "qm_relax": False,
        "qm_relax_cell": False,
    })

    # Compute cache key the same way the populator would.
    code = structure_code(struct, fallback=cfg.qm.code)
    backend = make_backend(cfg.qm, code_override=code)
    fp = backend.settings_fingerprint()
    op = _relax_op(struct, constraint=None)   # = "vc-relax" because struct.qm_relax_cell is True
    if op != "vc-relax":
        raise SystemExit(
            f"expected op=vc-relax (qm_relax_cell must be True on {args.structure!r}); got {op!r}"
        )
    key = _key(struct, fp, op)

    cache = QmCache(cfg.qm.cache_dir)
    result = QmRelaxResult(
        structure=new_structure,
        energy_kcal_mol=e_ev * _EV_TO_KCAL,
    )
    cache._store_relax(key, result)
    print(f"OK — cached vc-relax of {args.structure!r}")
    print(f"  cache dir : {cache.cache_dir / key}")
    print(f"  box       : {new_box}")
    print(f"  energy    : {e_ev * _EV_TO_KCAL:.3f} kcal/mol  ({e_ev:.6f} eV)")
    print(f"  atoms     : {len(new_atoms)}")
    print()
    print("Next: re-run `relax_structures(cfg)` in the notebook — should be a cache hit.")


if __name__ == "__main__":
    main()
