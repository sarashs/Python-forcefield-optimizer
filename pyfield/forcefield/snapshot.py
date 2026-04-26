"""Flat-array parameter snapshot for SA-style optimizers.

Replaces `deepcopy(ff.params)` per accept/reject. The legacy SA copied the
entire parsed force-field dict (~100s of nested entries) every iteration
even though only a handful of `param_min_max_delta` entries actually move.
A `ParameterSnapshot` is a length-N `np.ndarray` plus an ordered key list
that matches `ff.param_min_max_delta`. capture / apply are O(N).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pyfield.forcefield.reax import REAX_FF


_Key = Tuple[int, int, int]


@dataclass
class ParameterSnapshot:
    keys: Tuple[_Key, ...]
    values: np.ndarray

    @classmethod
    def capture(cls, ff: REAX_FF) -> "ParameterSnapshot":
        keys = tuple(ff.param_min_max_delta.keys())
        values = np.array(
            [ff.params[s][e][i] for (s, e, i) in keys],
            dtype=float,
        )
        return cls(keys=keys, values=values)

    def apply(self, ff: REAX_FF) -> None:
        """Write this snapshot's values back into `ff.params` in-place."""
        for (s, e, i), v in zip(self.keys, self.values):
            ff.params[s][e][i] = float(v)

    def copy(self) -> "ParameterSnapshot":
        return ParameterSnapshot(keys=self.keys, values=self.values.copy())

    def __len__(self) -> int:
        return len(self.values)
