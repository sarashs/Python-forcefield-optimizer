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
    return 2


if __name__ == "__main__":
    sys.exit(main())
