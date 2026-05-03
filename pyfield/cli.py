"""PyField command-line interface."""
import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pyfield", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a YAML config")
    run.add_argument("config", help="Path to a pyfield YAML config")

    run_legacy = sub.add_parser(
        "run-legacy",
        help="Run with the pre-Phase-1 text formats (deprecated)",
    )
    run_legacy.add_argument("training", help="Trainingfile.txt")
    run_legacy.add_argument("structures", help="Inputstructurefile.txt")
    run_legacy.add_argument("--ff", required=True, help="Path to ffield.reax")
    run_legacy.add_argument("--params", required=True, help="Path to params file")
    run_legacy.add_argument("--out", default=".", help="Output directory")

    qm_prep = sub.add_parser(
        "qm-prep",
        help="Populate `from: dft` placeholders in a YAML config via QM",
    )
    qm_prep.add_argument("config", help="Path to a pyfield YAML config")
    qm_prep.add_argument("-o", "--output", help="Path to write the populated YAML "
                                                "(default: <config>.populated.yaml)")
    qm_prep.add_argument("--in-place", action="store_true",
                         help="Mutate `config` in place (writes a .bak first)")
    qm_prep.add_argument("--force", action="store_true",
                         help="Ignore the QM cache and re-run every job")

    qm_relax = sub.add_parser(
        "qm-relax",
        help="QM geom-opt every `qm_relax: true` structure; write back the relaxed coords",
    )
    qm_relax.add_argument("config", help="Path to a pyfield YAML config")
    qm_relax.add_argument("-o", "--output", help="Path to write the relaxed YAML "
                                                 "(default: <config>.relaxed.yaml)")
    qm_relax.add_argument("--structures", nargs="+", default=None,
                          help="Override: only relax these structures (ignores `qm_relax:` flags)")
    qm_relax.add_argument("--force", action="store_true",
                          help="Ignore the QM cache and re-run every relax")

    make_scan = sub.add_parser(
        "make-scan",
        help="Expand a `scans:` block into structures + single_point sims + targets",
    )
    make_scan.add_argument("config", help="Path to a pyfield YAML config")
    make_scan.add_argument("-o", "--output", help="Path to write the expanded YAML "
                                                  "(default: <config>.scanned.yaml)")
    make_scan.add_argument("--xyz-dir", help="Directory to dump generated structures as .xyz "
                                             "(default: <output_yaml_dir>/scan_structures/)")

    args = parser.parse_args(argv)

    # Import lazily so `pyfield --help` doesn't pay for pydantic / lammps imports.
    if args.cmd == "run":
        from pyfield.runner import run_from_yaml
        return run_from_yaml(args.config)
    if args.cmd == "run-legacy":
        from pyfield.runner import run_from_legacy
        return run_from_legacy(
            training=args.training,
            structures=args.structures,
            ff=args.ff,
            params=args.params,
            out=args.out,
        )
    if args.cmd == "qm-prep":
        from pyfield.runner import run_qm_prep
        return run_qm_prep(
            args.config,
            output=args.output,
            in_place=args.in_place,
            force=args.force,
        )
    if args.cmd == "qm-relax":
        from pyfield.runner import run_qm_relax
        return run_qm_relax(
            args.config,
            output=args.output,
            only=args.structures,
            force=args.force,
        )
    if args.cmd == "make-scan":
        from pyfield.runner import run_make_scan
        return run_make_scan(
            args.config,
            output=args.output,
            xyz_dir=args.xyz_dir,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
