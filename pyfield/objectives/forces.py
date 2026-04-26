"""`kind: forces` — per-atom force matching against a reference.

Reads the `single_point` (or any other) simulation's dump and pulls out
the `fx fy fz` columns. Compares against an inline reference dictionary
of per-atom force vectors. Residual = weighted MSE over all components.

Schema:

    - kind: forces
      weight: 1.0
      simulation: my_single_point
      reference:
        1: [-0.123, 0.456, 0.000]   # force on atom 1 (kcal/mol/Å)
        2: [ 0.123,-0.456, 0.000]
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from pyfield.io.dump import read_dump
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


def _last_frame_forces(dump_path: str) -> Dict[int, np.ndarray]:
    """Read the last frame; return {atom_id: [fx, fy, fz]}."""
    last = None
    for f in read_dump(dump_path):
        last = f
    if last is None:
        raise RuntimeError(f"forces: dump {dump_path!r} has no frames")
    for col in ("fx", "fy", "fz"):
        if col not in last.columns:
            raise RuntimeError(
                f"forces: dump {dump_path!r} does not include force columns "
                f"(missing {col!r}). Use a `type: single_point` simulation."
            )
    ids = last.col("id").astype(int)
    f = np.column_stack([last.col("fx"), last.col("fy"), last.col("fz")])
    return {int(ids[i]): f[i] for i in range(len(ids))}


@register_objective("forces")
class ForcesObjective(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulation: str,
        reference: Dict[int, Sequence[float]],
        **_ignored,
    ):
        super().__init__(weight, simulation=simulation, reference=reference)
        self._sim = simulation
        self._reference = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in reference.items()
        }
        for atom_id, vec in self._reference.items():
            if vec.shape != (3,):
                raise ValueError(
                    f"forces: reference[{atom_id}] must be a 3-vector, got shape {vec.shape}"
                )

    def required_simulations(self) -> List[str]:
        return [self._sim]

    def compute(self, ctx: ObjectiveContext) -> float:
        sim_result: SimResult = ctx.sim_results[self._sim]
        dump_path = sim_result.extras.get("dump_file")
        if not dump_path:
            raise RuntimeError(f"forces: simulation {self._sim!r} produced no dump_file")
        observed = _last_frame_forces(str(dump_path))
        sq = []
        for atom_id, ref_vec in self._reference.items():
            if atom_id not in observed:
                raise KeyError(
                    f"forces: reference atom #{atom_id} not present in "
                    f"simulation {self._sim!r} (observed atoms: {sorted(observed)[:10]}…)"
                )
            d = observed[atom_id] - ref_vec
            sq.append(float(np.dot(d, d)))
        return float(np.mean(sq))

    def residual(self, ctx: ObjectiveContext) -> float:
        # Already an MSE; just weight it.
        return self.weight * self.compute(ctx)
