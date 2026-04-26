"""Decorator-based registry mapping target.kind → Objective subclass."""
from __future__ import annotations

from typing import Dict, List, Type

from pyfield.config.schema import TargetCfg
from pyfield.objectives.base import Objective


_REGISTRY: Dict[str, Type[Objective]] = {}


def register_objective(kind: str):
    """Decorator: bind a kind string to an Objective subclass."""
    def _decorator(cls: Type[Objective]) -> Type[Objective]:
        if kind in _REGISTRY:
            raise ValueError(f"objective kind {kind!r} is already registered")
        cls.kind = kind
        _REGISTRY[kind] = cls
        return cls
    return _decorator


def registered_kinds() -> List[str]:
    return sorted(_REGISTRY)


def build_objective(target: TargetCfg) -> Objective:
    """Instantiate an objective from a validated TargetCfg."""
    cls = _REGISTRY.get(target.kind)
    if cls is None:
        raise KeyError(
            f"unknown target kind {target.kind!r}; registered: {registered_kinds()}"
        )
    extras = dict(getattr(target, "__pydantic_extra__", {}) or {})
    return cls(weight=target.weight, **extras)
