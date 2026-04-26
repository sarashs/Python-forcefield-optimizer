"""Simulation interface + result type used by every backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimResult:
    """What every simulation produces.

    Phase-1 minimize-only fills `energy` and `charges`. Phase 3 trajectory
    backends extend this with a `frames` payload (lazy iterator) and let
    objectives consume it through the same SimResult shape.
    """
    sim_id: str
    energy: float
    charges: List[float] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)


class Simulation:
    """Interface every simulation backend implements.

    Concrete backends live next to a Jinja template that renders to a real
    LAMMPS input file. The runner picks the backend by `cfg.type` (built-in)
    or `cfg.template` (user-supplied) — see DEV.md §7.
    """

    name: str = "abstract"

    def render(self, *args, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError

    def run(self, *args, **kwargs) -> SimResult:  # pragma: no cover
        raise NotImplementedError
