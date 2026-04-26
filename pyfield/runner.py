"""High-level `run` entry points used by the CLI."""
from pathlib import Path

from pyfield.io.lammps import preload_libmpi


def _dispatch(cfg) -> int:
    """Pick SA or GA (or sa+ga) based on cfg.optimizer.method."""
    method = cfg.optimizer.method
    if method == "sa":
        from pyfield.optimizers.sa import run_sa
        result = run_sa(cfg)
    elif method in ("ga", "sa+ga"):
        from pyfield.optimizers.ga import run_ga
        result = run_ga(cfg)
    else:
        raise NotImplementedError(f"unknown optimizer.method={method!r}")
    print(f"FINAL cost: {result.final_cost}")
    print(f"trace length: {len(result.cost_trace)}")
    print(f"best ffield written to: {result.best_ffield_path}")
    return 0


def run_from_yaml(config_path: str) -> int:
    preload_libmpi()
    from pyfield.config.loader import load_yaml

    cfg = load_yaml(config_path)
    return _dispatch(cfg)


def run_from_legacy(*, training: str, structures: str, ff: str, params: str, out: str) -> int:
    preload_libmpi()
    from pyfield.config.legacy import from_legacy_files
    from pyfield.config.schema import OptimizerCfg, OutputCfg

    cfg = from_legacy_files(
        forcefield=Path(ff),
        params=Path(params),
        training=Path(training),
        structures=Path(structures),
        output_dir=Path(out),
        optimizer=OptimizerCfg(),
    )
    return _dispatch(cfg)
