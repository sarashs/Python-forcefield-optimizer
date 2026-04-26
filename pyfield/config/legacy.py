"""Convert the pre-Phase-1 text formats into a `PyFieldConfig`.

Reads `Trainingfile.txt` + `Inputstructurefile.txt` and produces an
in-memory `PyFieldConfig`. The `params` file path is passed through
unchanged (its format is owned by ReaxFF/LAMMPS, not by us).

This is a compatibility shim. It emits a DeprecationWarning so callers
remember to migrate. Drop after one release.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyfield.config.schema import (
    AtomCfg,
    ForceFieldCfg,
    OptimizerCfg,
    OutputCfg,
    PyFieldConfig,
    SimulationCfg,
    StructureCfg,
    TargetCfg,
)


def _parse_structure_blocks(structure_file: Path) -> Dict[str, dict]:
    """Return {structure_name: {atoms, box, restraints}}."""
    text = structure_file.read_text().splitlines()

    blocks: Dict[str, dict] = {}
    i = 0
    while i < len(text):
        line = text[i]
        if line.startswith("#structure "):
            name = line.replace("#structure ", "").strip()
            n_atoms = int(text[i + 1])
            # walk past #weights into the atom-type masses (until #dimensions)
            j = i + 3
            while j < len(text) and not text[j].startswith("#dimensions"):
                j += 1
            # box: line after #dimensions
            dims = re.findall(r"[-+]?\d*\.\d+|\d+", text[j + 1])
            box = (float(dims[0]), float(dims[1]), float(dims[2]))
            # atoms: n_atoms lines starting at j+2
            atoms: List[AtomCfg] = []
            for k in range(n_atoms):
                row = text[j + 2 + k].split()
                # legacy row: "1 Cl  0  0  0  -1.02802"
                # = idx, element, charge, x, y, z
                _, element, q, x, y, z = (
                    row[0], row[1], row[2], row[3], row[4], row[5]
                )
                atoms.append(AtomCfg(
                    element=element, charge=float(q),
                    x=float(x), y=float(y), z=float(z),
                ))
            # restraints: optional, after the atom block
            restraints: List[str] = []
            after_atoms = j + 2 + n_atoms
            if after_atoms < len(text) and text[after_atoms].startswith("#restrain"):
                m = after_atoms + 1
                while m < len(text) and not text[m].startswith("#structure"):
                    if text[m].strip():
                        restraints.append(text[m].strip())
                    m += 1
                i = m
            else:
                i = after_atoms
            blocks[name] = {"atoms": atoms, "box": box, "restraints": restraints}
        else:
            i += 1
    return blocks


def _parse_training_file(training_file: Path) -> List[TargetCfg]:
    """Convert ENERGY/CHARGE blocks into a list of TargetCfg."""
    targets: List[TargetCfg] = []
    text = training_file.read_text().splitlines()

    section: Optional[str] = None
    for line in text:
        if line.startswith("ENERGY"):
            section = "ENERGY"
            continue
        if line.startswith("CHARGE"):
            section = "CHARGE"
            continue
        if not line.strip():
            continue

        if section == "ENERGY":
            # "1.0  1*Cl2_414 -1*Cl2_Opt 81.394"
            m = re.match(
                r"\s*([-+]?\d+\.?\d*)\s+"
                r"([-+]?\d+\.?\d*)\*([A-Za-z0-9_]+)\s+"
                r"([-+]?\d+\.?\d*)\*([A-Za-z0-9_]+)\s+"
                r"([-+]?\d+\.?\d*)",
                line,
            )
            if not m:
                continue
            weight, c_a, name_a, c_b, name_b, dE = m.groups()
            terms = {f"{name_a}_min": float(c_a), f"{name_b}_min": float(c_b)}
            targets.append(TargetCfg.model_validate({
                "kind": "energy_combination",
                "weight": float(weight),
                "terms": terms,
                "target": float(dE),
            }))
        elif section == "CHARGE":
            # "5.01 SiO4 1 +4 2 -1 3 -1 4 -1 5 -1"
            tokens = line.split()
            if len(tokens) < 4:
                continue
            weight = float(tokens[0])
            sim_name = tokens[1] + "_min"
            atoms: Dict[int, float] = {}
            for k in range(2, len(tokens) - 1, 2):
                atoms[int(tokens[k])] = float(tokens[k + 1])
            targets.append(TargetCfg.model_validate({
                "kind": "charges",
                "weight": weight,
                "simulation": sim_name,
                "atoms": atoms,
            }))
    return targets


def from_legacy_files(
    *,
    forcefield: Path,
    params: Path,
    training: Path,
    structures: Path,
    output_dir: Path = Path("runs/legacy"),
    optimizer: Optional[OptimizerCfg] = None,
) -> PyFieldConfig:
    """Build a PyFieldConfig from the four pre-Phase-1 inputs."""
    warnings.warn(
        "Loading from Trainingfile.txt + Inputstructurefile.txt is deprecated. "
        "Migrate to a YAML config (see DEV.md §7).",
        DeprecationWarning,
        stacklevel=2,
    )

    raw = _parse_structure_blocks(Path(structures))
    structures_map: Dict[str, StructureCfg] = {
        name: StructureCfg(box=blk["box"], atoms=blk["atoms"])
        for name, blk in raw.items()
    }
    simulations_map: Dict[str, SimulationCfg] = {}
    for name, blk in raw.items():
        simulations_map[f"{name}_min"] = SimulationCfg.model_validate({
            "structure": name,
            "type": "minimize",
            "min_style": "cg",
            "restraints": blk["restraints"],
        })

    targets = _parse_training_file(Path(training))

    return PyFieldConfig(
        forcefield=ForceFieldCfg(path=Path(forcefield), params=Path(params)),
        structures=structures_map,
        simulations=simulations_map,
        targets=targets,
        optimizer=optimizer or OptimizerCfg(),
        output=OutputCfg(dir=Path(output_dir)),
    )
