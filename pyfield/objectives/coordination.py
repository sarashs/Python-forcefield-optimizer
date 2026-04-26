"""`kind: coordination` — average coordination number from an MD trajectory.

Reads the LAMMPS dump produced by an `nvt` (or templated) simulation,
counts neighbour-of-target atoms within `cutoff` of each central atom in
each frame, and averages the per-frame mean over the configured
`average_window`. Returns the residual against `target` (squared error,
weighted).

YAML schema (consumed via the registry's TargetCfg.__pydantic_extra__):

    - kind: coordination
      weight: 2.0
      simulation: SiOH4_300K
      central: Si           # element symbol or 1-based LAMMPS type
      neighbor: O
      cutoff: 2.5           # Å
      target: 4.0
      average_window: [100000, 200000]   # optional; defaults to whole traj
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from pyfield.io.dump import DumpFrame, read_dump
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


_TypeRef = Union[str, int]


def _resolve_type(ref: _TypeRef, type_to_element: Dict[int, str]) -> int:
    """Element symbol → LAMMPS type integer, or pass an integer through."""
    if isinstance(ref, int):
        return ref
    el_to_type = {el: t for t, el in type_to_element.items()}
    if ref not in el_to_type:
        raise ValueError(
            f"coordination references element {ref!r}, but the simulation only "
            f"has {sorted(el_to_type)}"
        )
    return el_to_type[ref]


def _frames_in_window(
    dump_path: str,
    average_window: Optional[Tuple[int, int]],
) -> List[DumpFrame]:
    frames = []
    for frame in read_dump(dump_path):
        if average_window is not None:
            lo, hi = average_window
            if frame.timestep < lo or frame.timestep > hi:
                continue
        frames.append(frame)
    return frames


def _coordination_one_frame(
    frame: DumpFrame,
    central_type: int,
    neighbor_type: int,
    cutoff: float,
) -> float:
    """Mean number of neighbour-type atoms within `cutoff` of each central atom.

    Uses minimum-image distances under periodic boundary conditions
    (orthogonal box only). Self-pairs (i==j when central==neighbor) are
    excluded.
    """
    types = frame.col("type").astype(int)
    xyz = np.column_stack([frame.col("x"), frame.col("y"), frame.col("z")])
    central_idx = np.where(types == central_type)[0]
    neighbor_idx = np.where(types == neighbor_type)[0]
    if central_idx.size == 0 or neighbor_idx.size == 0:
        return 0.0

    box = np.asarray(frame.box, dtype=np.float64)
    cutoff2 = cutoff * cutoff

    n_for_central: List[int] = []
    for i in central_idx:
        # Vector from atom i to all neighbour candidates, with minimum-image PBC.
        d = xyz[neighbor_idx] - xyz[i]
        d -= np.round(d / box) * box
        r2 = np.einsum("ij,ij->i", d, d)
        # Exclude self when the central and neighbour types coincide.
        if central_type == neighbor_type:
            r2[neighbor_idx == i] = np.inf
        n_for_central.append(int(np.count_nonzero(r2 < cutoff2)))
    return float(np.mean(n_for_central))


@register_objective("coordination")
class CoordinationObjective(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulation: str,
        central: _TypeRef,
        neighbor: _TypeRef,
        cutoff: float,
        target: float,
        average_window: Optional[List[int]] = None,
        **_ignored,
    ):
        super().__init__(
            weight,
            simulation=simulation,
            central=central,
            neighbor=neighbor,
            cutoff=cutoff,
            target=target,
            average_window=average_window,
        )
        self._sim = simulation
        self._central = central
        self._neighbor = neighbor
        self._cutoff = float(cutoff)
        self._target = float(target)
        self._window = tuple(average_window) if average_window else None

    def required_simulations(self) -> List[str]:
        return [self._sim]

    def compute(self, ctx: ObjectiveContext) -> float:
        sim_result: SimResult = ctx.sim_results[self._sim]
        dump_path = sim_result.extras.get("dump_file")
        if not dump_path:
            raise RuntimeError(
                f"coordination objective requires a trajectory dump from "
                f"simulation {self._sim!r}, but its SimResult.extras has no "
                "'dump_file' entry. Use a `type: nvt` simulation or a templated "
                "one whose .in.j2 writes a dump to {{ DUMP_FILE }}."
            )
        type_to_element = sim_result.extras.get("type_to_element") or {}
        ctype = _resolve_type(self._central, type_to_element)
        ntype = _resolve_type(self._neighbor, type_to_element)
        window = self._window if self._window else sim_result.extras.get("average_window")

        frames = _frames_in_window(str(dump_path), window)
        if not frames:
            raise RuntimeError(
                f"coordination: dump {dump_path!r} contains no frames in the "
                f"requested window {window!r}"
            )
        per_frame = [
            _coordination_one_frame(f, ctype, ntype, self._cutoff) for f in frames
        ]
        return float(np.mean(per_frame))

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
