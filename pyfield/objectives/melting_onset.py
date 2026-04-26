"""`kind: melting_onset` — onset temperature inferred from MSD-vs-T.

Consumes a list of NVT trajectories run at increasing temperatures.
Computes mean-squared displacement vs frame for each, fits its slope
(linear regression of MSD against time), and reports the lowest
temperature whose slope crosses (min + max) / 2 — a crude but
dependency-free step-finder.

Schema:

    - kind: melting_onset
      weight: 1.0
      simulations: [SiOH4_300K, SiOH4_500K, SiOH4_1000K, SiOH4_1500K]
      target: 1200            # K

Each referenced simulation must:
- be `type: nvt` (or a templated NVT) so its dump exists,
- carry its `temperature` in `SimResult.extras["temperature"]`.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from pyfield.io.dump import DumpFrame, read_dump
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


def _msd_slope_for_dump(dump_path: str) -> float:
    """Linear-regression slope of <|r(t) - r(0)|^2> vs frame index."""
    times: List[int] = []
    msd: List[float] = []
    initial = None
    initial_ids = None
    for frame in read_dump(dump_path):
        ids = frame.col("id").astype(int)
        order = np.argsort(ids)
        coords = np.column_stack([frame.col("x"), frame.col("y"), frame.col("z")])[order]
        if initial is None:
            initial = coords.copy()
            initial_ids = ids[order]
            times.append(frame.timestep)
            msd.append(0.0)
            continue
        # Atom-index order assumed stable across frames in a single LAMMPS run.
        d = coords - initial
        msd.append(float(np.mean(np.einsum("ij,ij->i", d, d))))
        times.append(frame.timestep)
    if len(times) < 2:
        return 0.0
    t = np.asarray(times, dtype=np.float64)
    s = np.asarray(msd, dtype=np.float64)
    # Linear fit slope only (intercept ignored). np.polyfit uses least squares.
    slope = float(np.polyfit(t, s, 1)[0])
    return slope


@register_objective("melting_onset")
class MeltingOnset(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulations: List[str],
        target: float,
        **_ignored,
    ):
        super().__init__(weight, simulations=simulations, target=target)
        if not simulations:
            raise ValueError("melting_onset requires a non-empty `simulations:` list")
        self._sims = list(simulations)
        self._target = float(target)

    def required_simulations(self) -> List[str]:
        return list(self._sims)

    def compute(self, ctx: ObjectiveContext) -> float:
        temperatures: List[float] = []
        slopes: List[float] = []
        for sim_id in self._sims:
            res = ctx.sim_results[sim_id]
            dump_path = res.extras.get("dump_file")
            if not dump_path:
                raise RuntimeError(f"melting_onset: {sim_id!r} has no dump_file")
            t = res.extras.get("temperature")
            if t is None:
                raise RuntimeError(f"melting_onset: {sim_id!r} has no extras.temperature")
            temperatures.append(float(t))
            slopes.append(_msd_slope_for_dump(dump_path))
        order = np.argsort(temperatures)
        T = np.asarray(temperatures, dtype=np.float64)[order]
        S = np.asarray(slopes, dtype=np.float64)[order]
        if S.size < 2:
            return float(T[0])
        threshold = 0.5 * (S.min() + S.max())
        # Lowest T whose slope clears the half-way mark; if none do
        # (everything stayed solid), return the highest T as the
        # under-estimate of onset.
        crossing = np.where(S >= threshold)[0]
        if crossing.size == 0:
            return float(T[-1])
        return float(T[int(crossing[0])])

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
