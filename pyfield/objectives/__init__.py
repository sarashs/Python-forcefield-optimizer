"""Objective registry + bundled objectives.

Importing this package registers every built-in objective. Adding a new
objective is just a new file under `pyfield/objectives/` decorated with
`@register_objective("kind")`.
"""
from pyfield.objectives.base import Objective, ObjectiveContext
from pyfield.objectives.registry import (
    build_objective,
    register_objective,
    registered_kinds,
)

# Importing for side-effect: each module registers its kind on import.
from pyfield.objectives import charges as _charges  # noqa: F401
from pyfield.objectives import coordination as _coordination  # noqa: F401
from pyfield.objectives import energy_combination as _energy_combination  # noqa: F401
from pyfield.objectives import eos as _eos  # noqa: F401
from pyfield.objectives import forces as _forces  # noqa: F401
from pyfield.objectives import melting_onset as _melting_onset  # noqa: F401
from pyfield.objectives import rdf_peak as _rdf_peak  # noqa: F401
from pyfield.objectives import structural_match as _structural_match  # noqa: F401

__all__ = [
    "Objective",
    "ObjectiveContext",
    "build_objective",
    "register_objective",
    "registered_kinds",
]
