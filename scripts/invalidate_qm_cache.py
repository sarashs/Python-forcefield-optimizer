"""Invalidate stale QM-cache entries for one or more structures.

Use this when an entry's cached *energy* was produced under different
settings than its *key* currently implies. The canonical instance is
the salvaged `GST_rocksalt` vc-relax:

  - The cache key was computed under post-bump production settings
    (ecutwfc=50, kpts=2×2×2, conv_thr=1e-9 — see EXPERIMENT.md §10).
  - The energy stored in the entry was actually produced by the
    13-hour pre-bump run (ecutwfc=40, kpts=Γ-only, conv_thr=1e-7),
    captured by `rescue_qe_vcrelax.py` from a `/tmp/qe_vcrelax_failed_*`
    directory.

The mismatch is invisible to `cache.has()` — it returns True — but the
energy is wrong by ~45 kcal/mol on an 18-atom GST cell, and the
relaxed geometry sits at a Γ-only-minimum rather than the 2×2×2
minimum. The downstream symptom is a strain-scan ΔE ladder that goes
*monotonically negative* with expansion (the reference cell is too
compressed for the current settings — see EXPERIMENT.md §10.5).

What this script does:

1. Computes the current cache key for the named structure(s) under
   each relax op (`relax`, `vc-relax`, `relax_constrained`) and for
   `single_point`. Whichever entries exist get deleted.
2. Leaves dependent entries (strain-scan-point relaxes) alone — they
   key on the input structure's *content* hash. When the reference is
   re-relaxed and `make-scan` regenerates strain points off the new
   geometry, the keys change and the old entries simply become
   orphaned (harmless, just disk).

Usage:
    python scripts/invalidate_qm_cache.py \\
        --cfg studies/gst_drift/gst_drift.yaml \\
        --structure GST_rocksalt

    # Dry-run mode prints what *would* be deleted without touching disk:
    python scripts/invalidate_qm_cache.py \\
        --cfg studies/gst_drift/gst_drift.yaml \\
        --structure GST_rocksalt --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pyfield.config.loader import load_yaml
from pyfield.qm.base import make_backend, structure_code
from pyfield.qm.cache import QmCache, _key


_OPS = ("relax", "vc-relax", "relax_constrained")


def _invalidate_one(
    cache: QmCache, structure, fingerprint: str, *, dry_run: bool
) -> list[tuple[str, Path]]:
    """Return [(op_or_sp, entry_dir), ...] for everything that was (or
    would be) deleted for this structure."""
    deleted: list[tuple[str, Path]] = []
    # 1) Relax entries keyed on the *input* structure. We don't know
    #    which op was used to produce the stale entry, so check all
    #    three; misses are harmless. Before deleting a relax entry,
    #    read its post-relax geometry so we can also find the orphan
    #    single_point that the populator stored at that geometry.
    post_relax_structs = []
    for op in _OPS:
        key = _key(structure, fingerprint, op)
        d = cache._entry_dir(key)
        if not d.exists():
            continue
        try:
            res = cache._load_relax(key)
            post_relax_structs.append(res.structure)
        except Exception:
            pass        # corrupt entry — still delete the dir
        deleted.append((op, d))
        if not dry_run:
            shutil.rmtree(d)

    # 2) Single_point on the *input* geometry. Rare, but possible if
    #    the populator's reuse_relax_energy path was bypassed.
    key_sp_input = _key(structure, fingerprint, "single_point")
    d_sp_input = cache._entry_dir(key_sp_input)
    if d_sp_input.exists():
        deleted.append(("single_point[input geom]", d_sp_input))
        if not dry_run:
            shutil.rmtree(d_sp_input)

    # 3) Single_point on each *post-relax* geometry — this is the entry
    #    the populator actually stores for reference SCFs (the cached
    #    relax already gave the populator the relaxed structure, and
    #    populate_qm then keys its reference-single_point off that).
    for prs in post_relax_structs:
        key_sp = _key(prs, fingerprint, "single_point")
        d_sp = cache._entry_dir(key_sp)
        if d_sp.exists() and d_sp not in {d for _, d in deleted}:
            deleted.append(("single_point[post-relax geom]", d_sp))
            if not dry_run:
                shutil.rmtree(d_sp)
    return deleted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", required=True, help="path to the source YAML")
    p.add_argument("--structure", required=True, action="append",
                   help="structure name to invalidate (repeat for multiple)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be deleted, don't touch disk")
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    if cfg.qm is None:
        print("error: config has no `qm:` block", file=sys.stderr)
        return 2

    cache = QmCache(cfg.qm.cache_dir)
    unknown = [n for n in args.structure if n not in cfg.structures]
    if unknown:
        print(f"error: unknown structures {unknown!r}", file=sys.stderr)
        return 2

    grand_total = 0
    for name in args.structure:
        struct = cfg.structures[name]
        code = structure_code(struct, fallback=cfg.qm.code)
        be = make_backend(cfg.qm, code_override=code)
        fp = be.settings_fingerprint()
        deleted = _invalidate_one(cache, struct, fp, dry_run=args.dry_run)
        verb = "would delete" if args.dry_run else "deleted"
        if not deleted:
            print(f"{name}: no cached entries at the current fingerprint")
        else:
            for op, d in deleted:
                print(f"{name}: {verb} [{op}]  {d}")
        grand_total += len(deleted)

    print()
    print(f"total entries {'matched' if args.dry_run else 'deleted'}: {grand_total}")
    if args.dry_run:
        print("(dry run — nothing was touched. Re-run without --dry-run to apply.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
