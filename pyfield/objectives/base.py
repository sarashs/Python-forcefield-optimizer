"""Objective interface used by the optimizer.

Every objective answers three questions:
1. Which simulations do you need run?  (`required_simulations`)
2. Reduce those sim outputs to a scalar.  (`compute`)
3. Translate the scalar to a residual.    (`residual`)

The optimizer never touches anything domain-specific — it only sees
residuals and sums them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pyfield.simulations.base import SimResult


@dataclass(frozen=True)
class ObjectiveContext:
    """Read-only view of the per-iteration simulation results.

    Made a dataclass (rather than a bare dict) so future additions like a
    weight summary, parameter snapshot, or trajectory accessors can be
    bolted on without rewriting every objective signature.
    """
    sim_results: Dict[str, SimResult]


class Objective:
    kind: str = "abstract"

    def __init__(self, weight: float, **fields):
        self.weight = weight
        self.fields = fields

    def required_simulations(self) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def compute(self, ctx: ObjectiveContext) -> float:  # pragma: no cover
        raise NotImplementedError

    def residual(self, ctx: ObjectiveContext) -> float:
        """Default residual: weighted squared error against `target`."""
        value = self.compute(ctx)
        target = float(self.fields.get("target", 0.0))
        return self.weight * (value - target) ** 2
