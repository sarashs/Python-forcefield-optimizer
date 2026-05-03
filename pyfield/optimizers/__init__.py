"""PyField optimizers."""
from pyfield.optimizers.cma import run_cma
from pyfield.optimizers.ga import run_ga
from pyfield.optimizers.sa import run_sa


def run_optimizer(cfg):
    """Dispatch by `cfg.optimizer.method`.

    Public wrapper around the SA / GA / CMA drivers. Useful from
    notebooks and scripts that want one entry point matching whatever
    the YAML's `optimizer.method:` says.
    """
    method = cfg.optimizer.method
    if method == "sa":
        return run_sa(cfg)
    if method in ("ga", "sa+ga"):
        return run_ga(cfg)
    if method == "cma":
        return run_cma(cfg)
    raise NotImplementedError(f"unknown optimizer.method={method!r}")


__all__ = ["run_sa", "run_ga", "run_cma", "run_optimizer"]
