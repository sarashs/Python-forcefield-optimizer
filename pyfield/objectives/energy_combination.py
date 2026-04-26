"""`kind: energy_combination` — signed sum of simulation energies vs target.

Generalizes the legacy ENERGY block: each entry was
    weight  c_a*structA  c_b*structB   ΔE_target
and we matched (c_a*E_A + c_b*E_B) to ΔE_target. The YAML form lets
arbitrarily many simulations contribute to one combination.
"""
from __future__ import annotations

from typing import Dict, List

from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective


@register_objective("energy_combination")
class EnergyCombination(Objective):
    def __init__(self, weight: float, *, terms: Dict[str, float], target: float, **_ignored):
        super().__init__(weight, terms=terms, target=target)
        self._terms: Dict[str, float] = dict(terms)
        self._target: float = float(target)

    def required_simulations(self) -> List[str]:
        return list(self._terms.keys())

    def compute(self, ctx: ObjectiveContext) -> float:
        return sum(coeff * ctx.sim_results[sim].energy for sim, coeff in self._terms.items())

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
