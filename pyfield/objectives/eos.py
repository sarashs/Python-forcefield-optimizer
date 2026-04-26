"""`kind: eos` — bulk modulus from a series of NPT runs.

Consumes a list of NPT simulations at different pressures (same
structure, same temperature) and fits the linear bulk modulus
B = -V (dP/dV) at the lowest-pressure volume:

    B ≈ -V₀ * Δp / Δv     (centered finite difference around p=0)

The simple linear fit avoids a scipy dependency and gives a sane
starting target. Phase 5+ can swap in Birch-Murnaghan when needed.

Schema:

    - kind: eos
      weight: 1.0
      simulations: [SiOH4_p0, SiOH4_p1, SiOH4_p10]
      target: 38.0          # GPa
"""
from __future__ import annotations

from typing import List

import numpy as np

from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


# Convert atm → GPa : 1 atm = 1.01325e-4 GPa.
_ATM_TO_GPA = 1.01325e-4


@register_objective("eos")
class EosObjective(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulations: List[str],
        target: float,
        **_ignored,
    ):
        super().__init__(weight, simulations=simulations, target=target)
        if len(simulations) < 2:
            raise ValueError("eos requires at least 2 NPT simulations at different pressures")
        self._sims = list(simulations)
        self._target = float(target)

    def required_simulations(self) -> List[str]:
        return list(self._sims)

    def compute(self, ctx: ObjectiveContext) -> float:
        pressures: List[float] = []   # atm
        volumes: List[float] = []     # Å³
        for sim_id in self._sims:
            res: SimResult = ctx.sim_results[sim_id]
            p = res.extras.get("pressure")
            v = res.extras.get("mean_volume")
            if p is None or v is None:
                raise RuntimeError(
                    f"eos: simulation {sim_id!r} is missing extras.pressure / "
                    "extras.mean_volume — was it a `type: npt` run?"
                )
            pressures.append(float(p))
            volumes.append(float(v))

        order = np.argsort(pressures)
        P = np.asarray(pressures, dtype=np.float64)[order]
        V = np.asarray(volumes, dtype=np.float64)[order]

        # Linear least-squares slope: dV/dP (Å³ / atm)
        slope = float(np.polyfit(P, V, 1)[0])
        if slope == 0:
            return 0.0
        # Use the volume at the lowest pressure as V₀ (closest to ambient).
        V0 = V[0]
        # B = -V₀ / (dV/dP)
        B_atm = -V0 / slope
        return float(B_atm * _ATM_TO_GPA)

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
