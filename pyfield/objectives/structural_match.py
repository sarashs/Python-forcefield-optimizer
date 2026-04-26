"""`kind: structural_match` — geometry of a minimized structure vs reference.

Reads the final-frame positions from the simulation's dump (the
`minimize` backend's `write_dump` after minimization) and compares to a
reference xyz. Three metrics are supported:

- `rmsd` (default): centred + Kabsch-aligned root-mean-square deviation.
- `bond_lengths`: MSE of |r_i - r_j| for an explicit list of pairs.
- `angles`: MSE of bond angles for an explicit list of triples.

Schema example:

    - kind: structural_match
      weight: 5.0
      simulation: SiOH4_min
      reference: structures/SiOH4_dft_optimized.xyz
      metric: rmsd                                 # default
      pairs:    [[1, 2], [1, 3]]                   # only for bond_lengths
      triples:  [[1, 2, 3], [1, 2, 4]]             # only for angles
      # `target` defaults to 0.0 (perfect match) for rmsd / bond_lengths
      # / angles since each metric is already a scalar deviation.

Atom ordering: the dump and the reference xyz must agree on the order.
The `read_data` writer in `pyfield.io.structures` writes atoms in the
order they appear in `StructureCfg.atoms`, so a YAML structure block
ordered the same way as the reference xyz is enough.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from pyfield.io.dump import read_dump
from pyfield.io.xyz import read_xyz
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Return the RMSD between (N, 3) point clouds after centring and the
    Kabsch optimal rotation of `b` onto `a`."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = b.T @ a
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    b_aligned = b @ rot.T
    diff = a - b_aligned
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))


def _bond_length_msd(coords: np.ndarray, ref: np.ndarray, pairs: Iterable[Iterable[int]]) -> float:
    """MSE over |Δr| for the listed 1-based atom pairs."""
    diffs = []
    for i, j in pairs:
        i_, j_ = int(i) - 1, int(j) - 1
        di = float(np.linalg.norm(coords[i_] - coords[j_]))
        dj = float(np.linalg.norm(ref[i_] - ref[j_]))
        diffs.append((di - dj) ** 2)
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _angle_msd(coords: np.ndarray, ref: np.ndarray, triples: Iterable[Iterable[int]]) -> float:
    """MSE over bond-angle differences (radians) for 1-based (i, j, k) triples."""
    diffs = []
    for i, j, k in triples:
        ai = _angle(coords, int(i) - 1, int(j) - 1, int(k) - 1)
        bj = _angle(ref, int(i) - 1, int(j) - 1, int(k) - 1)
        diffs.append((ai - bj) ** 2)
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    """Bond angle at atom j between i—j—k, in radians."""
    a = coords[i] - coords[j]
    b = coords[k] - coords[j]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _last_frame_xyz(dump_path: str) -> np.ndarray:
    """Read the *last* frame of a dump and return its (N, 3) coords sorted by id."""
    last = None
    for f in read_dump(dump_path):
        last = f
    if last is None:
        raise RuntimeError(f"dump {dump_path!r} contained no frames")
    ids = last.col("id").astype(int)
    order = np.argsort(ids)
    return np.column_stack([last.col("x"), last.col("y"), last.col("z")])[order]


@register_objective("structural_match")
class StructuralMatch(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulation: str,
        reference: str,
        metric: str = "rmsd",
        target: float = 0.0,
        pairs: Optional[List[List[int]]] = None,
        triples: Optional[List[List[int]]] = None,
        **_ignored,
    ):
        super().__init__(
            weight,
            simulation=simulation, reference=reference, metric=metric,
            target=target, pairs=pairs, triples=triples,
        )
        self._sim = simulation
        self._reference = Path(reference)
        self._metric = metric
        self._target = float(target)
        self._pairs = pairs
        self._triples = triples
        if metric not in ("rmsd", "bond_lengths", "angles"):
            raise ValueError(f"unknown structural_match metric {metric!r}")
        if metric == "bond_lengths" and not pairs:
            raise ValueError("metric=bond_lengths requires `pairs:`")
        if metric == "angles" and not triples:
            raise ValueError("metric=angles requires `triples:`")

    def required_simulations(self) -> List[str]:
        return [self._sim]

    def compute(self, ctx: ObjectiveContext) -> float:
        sim_result: SimResult = ctx.sim_results[self._sim]
        dump_path = sim_result.extras.get("dump_file")
        if not dump_path:
            raise RuntimeError(
                f"structural_match: simulation {self._sim!r} produced no dump_file"
            )
        coords = _last_frame_xyz(str(dump_path))
        _, ref = read_xyz(self._reference)
        if coords.shape != ref.shape:
            raise ValueError(
                f"structural_match: simulation has {coords.shape[0]} atoms but "
                f"reference {self._reference!r} has {ref.shape[0]}"
            )
        if self._metric == "rmsd":
            return kabsch_rmsd(coords, ref)
        if self._metric == "bond_lengths":
            return _bond_length_msd(coords, ref, self._pairs or [])
        return _angle_msd(coords, ref, self._triples or [])

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
