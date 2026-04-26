"""Load a PyField YAML config file → validated PyFieldConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from pyfield.config.schema import PyFieldConfig


def load_yaml(path: Union[str, Path]) -> PyFieldConfig:
    """Read a YAML file and return a validated PyFieldConfig.

    Pydantic validation errors from this function are the primary user-
    facing error surface for malformed configs.
    """
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a YAML mapping at the top level")
    return PyFieldConfig.model_validate(raw)
