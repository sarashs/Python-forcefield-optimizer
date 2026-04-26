"""`kind: charges` — match per-atom QEq charges against a target dict.

Replaces the legacy CHARGE block. `atoms:` is a mapping `{atom_id: q}`
where `atom_id` is 1-based (matching how LAMMPS / the legacy structure
file numbered atoms).
"""
from __future__ import annotations

from typing import Dict, List

from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective


@register_objective("charges")
class ChargesTarget(Objective):
    def __init__(self, weight: float, *, simulation: str, atoms: Dict[int, float], **_ignored):
        super().__init__(weight, simulation=simulation, atoms=atoms)
        self._sim: str = simulation
        # YAML keys may arrive as strings; normalise to int.
        self._atoms: Dict[int, float] = {int(k): float(v) for k, v in atoms.items()}

    def required_simulations(self) -> List[str]:
        return [self._sim]

    def compute(self, ctx: ObjectiveContext) -> float:
        # Per-atom MSE against target; returned value is the "score" the
        # objective wants to drive to zero, so the residual just is the
        # weighted MSE.
        result = ctx.sim_results[self._sim]
        per_atom = []
        for atom_id, q_target in self._atoms.items():
            idx = atom_id - 1
            if idx < 0 or idx >= len(result.charges):
                raise IndexError(
                    f"charges target references atom #{atom_id} but simulation "
                    f"{self._sim!r} produced only {len(result.charges)} charges"
                )
            per_atom.append((result.charges[idx] - q_target) ** 2)
        return sum(per_atom) / max(len(per_atom), 1)

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * self.compute(ctx)
