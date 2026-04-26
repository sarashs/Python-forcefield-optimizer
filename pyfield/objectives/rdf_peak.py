"""`kind: rdf_peak` — position of the peak in g(r) for a pair, vs target.

Computes the radial distribution function for the requested
`(central, neighbor)` pair from an NVT dump (PBC minimum-image) and
reports the r-value where g(r) is maximal in `[r_min, r_max]`.

Schema:

    - kind: rdf_peak
      weight: 1.0
      simulation: SiOH4_300K
      central: Si           # element symbol or 1-based LAMMPS type
      neighbor: O
      r_max: 5.0            # Å
      r_min: 0.5            # Å (avoid the noise spike near zero)
      bins: 100
      target: 1.62          # Å
      average_window: [t_lo, t_hi]   # optional
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from pyfield.io.dump import DumpFrame, read_dump
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.coordination import _resolve_type
from pyfield.objectives.registry import register_objective
from pyfield.simulations.base import SimResult


_TypeRef = Union[str, int]


def _accumulate_pair_distances(
    frame: DumpFrame,
    central_type: int,
    neighbor_type: int,
    r_max: float,
) -> np.ndarray:
    types = frame.col("type").astype(int)
    xyz = np.column_stack([frame.col("x"), frame.col("y"), frame.col("z")])
    central_idx = np.where(types == central_type)[0]
    neighbor_idx = np.where(types == neighbor_type)[0]
    if central_idx.size == 0 or neighbor_idx.size == 0:
        return np.empty(0)
    box = np.asarray(frame.box, dtype=np.float64)
    out = []
    for i in central_idx:
        d = xyz[neighbor_idx] - xyz[i]
        d -= np.round(d / box) * box
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        if central_type == neighbor_type:
            r = r[neighbor_idx != i]
        r = r[(r > 0) & (r < r_max)]
        out.append(r)
    if not out:
        return np.empty(0)
    return np.concatenate(out)


def _gr(
    distances: np.ndarray,
    n_central: int,
    n_neighbor: int,
    box_volume: float,
    r_min: float,
    r_max: float,
    bins: int,
    same_type: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute g(r). Returns (r_centres, g(r)). Properly normalised by
    the ideal-gas pair density in the box."""
    edges = np.linspace(0.0, r_max, bins + 1)
    counts, _ = np.histogram(distances, bins=edges)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # Number of distinct pairs in an ideal gas:
    #   same-type:    n*(n-1) / 2 pairs, each counted once
    #   cross-type:   n_central * n_neighbor
    if same_type:
        n_pairs = n_central * (n_central - 1)
    else:
        n_pairs = n_central * n_neighbor
    if n_pairs == 0 or box_volume == 0:
        return r, np.zeros_like(r)
    rho = n_pairs / box_volume
    g = counts / (shell_vol * rho)
    # Mask out anything below r_min so the divergent low-r noise doesn't
    # win the peak search.
    g = np.where(r >= r_min, g, 0.0)
    return r, g


@register_objective("rdf_peak")
class RdfPeakObjective(Objective):
    def __init__(
        self,
        weight: float,
        *,
        simulation: str,
        central: _TypeRef,
        neighbor: _TypeRef,
        target: float,
        r_max: float = 5.0,
        r_min: float = 0.5,
        bins: int = 100,
        average_window: Optional[List[int]] = None,
        **_ignored,
    ):
        super().__init__(
            weight,
            simulation=simulation, central=central, neighbor=neighbor,
            target=target, r_max=r_max, r_min=r_min, bins=bins,
            average_window=average_window,
        )
        self._sim = simulation
        self._central = central
        self._neighbor = neighbor
        self._target = float(target)
        self._r_max = float(r_max)
        self._r_min = float(r_min)
        self._bins = int(bins)
        self._window = tuple(average_window) if average_window else None

    def required_simulations(self) -> List[str]:
        return [self._sim]

    def compute(self, ctx: ObjectiveContext) -> float:
        sim_result: SimResult = ctx.sim_results[self._sim]
        dump_path = sim_result.extras.get("dump_file")
        if not dump_path:
            raise RuntimeError(
                f"rdf_peak: simulation {self._sim!r} produced no dump_file"
            )
        type_to_element = sim_result.extras.get("type_to_element") or {}
        ctype = _resolve_type(self._central, type_to_element)
        ntype = _resolve_type(self._neighbor, type_to_element)
        same_type = ctype == ntype
        window = self._window if self._window else sim_result.extras.get("average_window")

        all_distances: List[np.ndarray] = []
        n_central_total = 0
        n_neighbor_total = 0
        n_frames = 0
        box_volume = None
        for frame in read_dump(str(dump_path)):
            if window is not None:
                lo, hi = window
                if frame.timestep < lo or frame.timestep > hi:
                    continue
            n_frames += 1
            types = frame.col("type").astype(int)
            n_central_total += int(np.count_nonzero(types == ctype))
            n_neighbor_total += int(np.count_nonzero(types == ntype))
            box_volume = float(frame.box[0] * frame.box[1] * frame.box[2])
            all_distances.append(_accumulate_pair_distances(frame, ctype, ntype, self._r_max))
        if n_frames == 0:
            raise RuntimeError(
                f"rdf_peak: dump {dump_path!r} has no frames in window {window!r}"
            )
        distances = np.concatenate(all_distances) if all_distances else np.empty(0)
        # Average atom counts per frame; used in the ideal-gas normaliser.
        n_central_avg = n_central_total / n_frames
        n_neighbor_avg = n_neighbor_total / n_frames
        # The histogram aggregates `n_frames` frames, so divide once.
        if distances.size == 0:
            return 0.0
        r, g = _gr(
            distances=distances / 1.0,   # numpy passthrough; explicit for clarity
            n_central=n_central_avg,
            n_neighbor=n_neighbor_avg,
            box_volume=box_volume * n_frames,
            r_min=self._r_min,
            r_max=self._r_max,
            bins=self._bins,
            same_type=same_type,
        )
        if not np.any(g > 0):
            return 0.0
        return float(r[int(np.argmax(g))])

    def residual(self, ctx: ObjectiveContext) -> float:
        return self.weight * (self.compute(ctx) - self._target) ** 2
