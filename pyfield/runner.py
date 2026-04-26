"""High-level `run` entry points used by the CLI."""
from pathlib import Path
from typing import List

from pyfield.io.lammps import preload_libmpi


def _check_no_qm_placeholders(cfg) -> None:
    """Pre-flight: refuse to run an optimiser on an unpopulated config.

    Lists every offending slot (qm_relax structures, `from:`-tagged
    target / reference fields) so the user knows exactly what to fix.
    """
    from pyfield.config.schema import is_qm_placeholder

    issues: List[str] = []
    for name, s in cfg.structures.items():
        if s.qm_relax:
            issues.append(f"structure {name!r} still has qm_relax: true")
    for i, t in enumerate(cfg.targets):
        extras = getattr(t, "__pydantic_extra__", {}) or {}
        for slot in ("target", "reference"):
            v = extras.get(slot)
            if isinstance(v, dict) and "from" in v:
                issues.append(f"target #{i} (kind={t.kind!r}): {slot}: {v!r}")
    if issues:
        raise RuntimeError(
            "Config contains unpopulated placeholders — run `pyfield qm-prep` "
            "first or supply concrete values:\n  " + "\n  ".join(issues)
        )


def _dispatch(cfg) -> int:
    """Pick SA or GA (or sa+ga) based on cfg.optimizer.method."""
    _check_no_qm_placeholders(cfg)
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


def run_qm_prep(
    config_path: str,
    *,
    output: str = None,
    in_place: bool = False,
    force: bool = False,
) -> int:
    """`pyfield qm-prep` entry point. Populates `from: dft` slots in a YAML.

    Doesn't preload libmpi — qm-prep doesn't touch LAMMPS.
    """
    import shutil
    from pyfield.config.loader import load_yaml
    from pyfield.qm.prep import cfg_to_yaml, populate_qm

    in_path = Path(config_path)
    if in_place:
        out_path = in_path
        shutil.copy(in_path, in_path.with_suffix(in_path.suffix + ".bak"))
    elif output:
        out_path = Path(output)
    else:
        out_path = in_path.with_suffix(".populated.yaml")

    cfg = load_yaml(in_path)
    if cfg.qm is None:
        print(f"{in_path}: no `qm:` block configured — nothing to populate.")
        return 0

    populated, journal = populate_qm(cfg, force=force)

    n_total = len(journal)
    n_hits = sum(1 for _, hit, _ in journal if hit)
    for action, hit, key in journal:
        tag = "[cache hit ]" if hit else "[running   ]"
        print(f"{tag} {action}   ({cfg.qm.cache_dir}/{key})")
    out_path.write_text(cfg_to_yaml(populated))
    print(f"populated {in_path} → {out_path}  ({n_total} jobs, {n_hits} cached, "
          f"{n_total - n_hits} ran)")
    return 0


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
