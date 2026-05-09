"""Quantum ESPRESSO backend via ASE.

PySCF's molecular DFT is excellent for cluster training, but its PBC
gradient infrastructure is fragile for moderately sized supercells —
SCF convergence under the multigrid integrator stalls for ~3-atom
test cells regardless of DIIS / cycle settings, and `nuc_grad_method`
on density-fit hybrid functionals is partial. QE is the de-facto
plane-wave PBC code with bulletproof SCF + analytic forces + stresses
and battle-tested cell / atom relaxation.

We drive QE through ASE's `Espresso` calculator. ASE handles input
file generation, subprocess management, and output parsing; we just
build the right input dict and invoke ASE's `BFGS` optimiser for
relaxation.

Setup the user supplies (outside the YAML):
- `pw.x` on PATH (or `ESPRESSO_COMMAND` env var with mpirun wrapper).
- A directory of UPF pseudopotentials (`qm.pseudo_dir` in YAML or
  `ESPRESSO_PSEUDO` env). The Standard Solid-State Pseudopotentials
  library (SSSP, Materials Cloud) is the recommended source.

Constraint mapping:

    PyField ConstraintSpec       ASE constraint
    distance i j r0          →   FixBondLengths([(i-1, j-1)])
    angle    i j k θ         →   FixInternals(angles_deg=[[θ, [i-1, j-1, k-1]]])
    dihedral i j k l φ       →   FixInternals(dihedrals_deg=[[φ, [i-1, j-1, k-1, l-1]]])

For a distance constraint we also pre-position the atoms at the
target separation before running BFGS — `FixBondLengths` keeps a bond
*at its current value*, not at a target value.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyfield.config.schema import StructureCfg
from pyfield.qm.base import (
    ConstraintSpec,
    QmBackend,
    QmRelaxResult,
    QmSinglePoint,
)


# Energy: eV (ASE's native) → kcal/mol; Force: eV/Å → kcal/mol/Å.
_EV_TO_KCAL = 23.06054783    # NIST 2019 conversion
_EV_PER_A_TO_KCAL_PER_A = _EV_TO_KCAL


def _qe_input_dft(name: str) -> str:
    """Map a friendly functional name to QE's `input_dft` keyword."""
    return {
        "lda": "PZ", "pz": "PZ",
        "pbe": "PBE",
        "pbesol": "PBESOL",
        "blyp": "BLYP",
        "b3lyp": "B3LYP",
        "scan": "SCAN",
        "hse": "HSE", "hse06": "HSE",
    }.get(name.lower(), name.upper())


class QEBackend(QmBackend):
    name = "qe"

    def __init__(self, qm_cfg):
        # Lazy import — ASE doesn't have to be installed for users not using QE.
        import ase  # noqa: F401
        self.cfg = qm_cfg
        # Default functional — actual functional used per-call honors any
        # per-structure `qm_functional:` override (see _effective_functional).
        self.functional = qm_cfg.functional
        self.basis = qm_cfg.basis  # not used; QE is plane-wave. Kept for symmetry with PySCF.
        extras = qm_cfg.__pydantic_extra__ or {}

        # Pseudopotential directory + per-element filename map.
        self.pseudo_dir = extras.get("pseudo_dir") or os.environ.get("ESPRESSO_PSEUDO")
        if not self.pseudo_dir:
            raise ValueError(
                "QE backend requires `qm.pseudo_dir:` in the YAML or "
                "`ESPRESSO_PSEUDO` env var pointing at a directory of UPF "
                "files. Recommended: download SSSP from materialscloud.org."
            )
        self.pseudopotentials: Dict[str, str] = dict(extras.get("pseudopotentials", {}))

        # Plane-wave parameters.
        self.ecutwfc = float(extras.get("ecutwfc", 50.0))    # Ry — orbital cutoff
        self.ecutrho = float(extras.get("ecutrho", 400.0))   # Ry — density cutoff (≈ 8×ecutwfc)
        self.kpts: Tuple[int, int, int] = tuple(extras.get("kpts", (1, 1, 1)))

        # Spin / charge — same convention as PySCF backend.
        self.spin = int(extras.get("spin", 0))
        self.charge = int(extras.get("charge", 0))

        # SCF / electronic settings (overridable).
        self.conv_thr = float(extras.get("conv_thr", 1e-7))
        self.mixing_beta = float(extras.get("mixing_beta", 0.5))
        self.degauss = float(extras.get("degauss", 0.01))    # Ry — Methfessel-Paxton smearing

        # ASE Espresso calculator: optional command override, otherwise
        # ASE looks for `pw.x` via `ESPRESSO_COMMAND`. We pass the
        # parameters explicitly so behaviour is reproducible regardless
        # of environment.
        self.command = extras.get("command", os.environ.get("ESPRESSO_COMMAND"))

    # ------------------------------------------------------------------
    # ASE plumbing
    # ------------------------------------------------------------------

    def _effective_functional(self, structure: StructureCfg) -> str:
        """Honor per-structure `qm_functional` override (matches PySCF backend)."""
        extras = getattr(structure, "__pydantic_extra__", {}) or {}
        return extras.get("qm_functional", self.functional)

    def _to_ase(self, structure: StructureCfg):
        from ase import Atoms
        if structure.atoms is None:
            raise ValueError("QE backend requires inline `atoms:` in the StructureCfg.")
        symbols = [a.element for a in structure.atoms]
        positions = [(float(a.x), float(a.y), float(a.z)) for a in structure.atoms]
        a, b, c = structure.box
        cell = np.diag([float(a), float(b), float(c)])
        # PBC even for cluster mode: QE is intrinsically periodic. For
        # cluster-like systems users should provide a sufficiently large
        # box (~10–15 Å of vacuum padding around the atoms).
        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    def _missing_pseudos(self, atoms) -> List[str]:
        return sorted({sym for sym in atoms.get_chemical_symbols()
                       if sym not in self.pseudopotentials})

    def _build_input_data(self, functional: Optional[str] = None) -> dict:
        xc = _qe_input_dft(functional or self.functional)
        return {
            "control": {
                "calculation": "scf",
                "verbosity": "low",
                "tprnfor": True,
                "tstress": True,
                "outdir": "./tmp",
                "prefix": "pwscf",
                "disk_io": "low",
            },
            "system": {
                "ibrav": 0,                 # explicit cell in CELL_PARAMETERS
                "ecutwfc": self.ecutwfc,
                "ecutrho": self.ecutrho,
                "input_dft": xc,
                "occupations": "smearing",
                "smearing": "mp",
                "degauss": self.degauss,
                "tot_charge": self.charge,
                **({"nspin": 2} if self.spin > 0 else {}),
            },
            "electrons": {
                "conv_thr": self.conv_thr,
                "mixing_beta": self.mixing_beta,
                "electron_maxstep": 200,
            },
        }

    def _make_calculator(self, atoms, functional: Optional[str] = None):
        from ase.calculators.espresso import Espresso, EspressoProfile
        missing = self._missing_pseudos(atoms)
        if missing:
            raise ValueError(
                f"QE backend: no pseudopotential mapping for elements {missing}. "
                "Add them under qm.pseudopotentials in the YAML."
            )
        # ASE 3.23+ uses an EspressoProfile to bundle command + pseudo_dir.
        cmd = self.command or "pw.x"
        profile = EspressoProfile(command=cmd, pseudo_dir=self.pseudo_dir)
        return Espresso(
            profile=profile,
            pseudopotentials=self.pseudopotentials,
            input_data=self._build_input_data(functional=functional),
            kpts=tuple(self.kpts),
        )

    # ------------------------------------------------------------------
    # constraint translation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_constraint(atoms, constraint: ConstraintSpec) -> None:
        """Apply a PyField ConstraintSpec to an ASE Atoms object.

        For a distance constraint we also pre-position the second atom
        at the target separation along the current bond axis — ASE's
        `FixBondLengths` keeps the bond at *its current value*, so the
        starting geometry must already match the target.
        """
        from ase.constraints import FixBondLengths, FixInternals

        kind = constraint["kind"]
        ids = [int(i) - 1 for i in constraint["atoms"]]
        target = float(constraint["value"])

        if kind == "distance":
            i, j = ids
            pos = atoms.get_positions()
            sep = pos[j] - pos[i]
            cur = float(np.linalg.norm(sep))
            if cur < 1e-9:
                raise ValueError(
                    f"QE distance constraint: atoms {ids} are coincident; "
                    "can't define a bond axis."
                )
            unit = sep / cur
            pos[j] = pos[i] + target * unit
            atoms.set_positions(pos)
            atoms.set_constraint(FixBondLengths([(i, j)]))
        elif kind == "angle":
            i, j, k = ids
            atoms.set_constraint(FixInternals(angles_deg=[[target, [i, j, k]]]))
        elif kind == "dihedral":
            i, j, k, l = ids
            atoms.set_constraint(FixInternals(dihedrals_deg=[[target, [i, j, k, l]]]))
        else:
            raise NotImplementedError(f"QE backend: unknown constraint kind {kind!r}")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def single_point(self, structure: StructureCfg) -> QmSinglePoint:
        atoms = self._to_ase(structure)
        atoms.calc = self._make_calculator(atoms, functional=self._effective_functional(structure))
        e_ev = float(atoms.get_potential_energy())
        try:
            f_ev = np.asarray(atoms.get_forces(), dtype=float)
            forces = f_ev * _EV_PER_A_TO_KCAL_PER_A
        except Exception:
            forces = None
        return QmSinglePoint(
            energy_kcal_mol=e_ev * _EV_TO_KCAL,
            forces_kcal_mol_per_A=forces,
        )

    def relax(
        self,
        structure: StructureCfg,
        constraint: Optional[ConstraintSpec] = None,
    ) -> QmRelaxResult:
        from ase.optimize import BFGS

        atoms = self._to_ase(structure)
        atoms.calc = self._make_calculator(atoms, functional=self._effective_functional(structure))
        if constraint is not None:
            self._apply_constraint(atoms, constraint)
        BFGS(atoms, logfile=None).run(fmax=0.025)   # 0.025 eV/Å ≈ 0.5 kcal/mol/Å
        e_ev = float(atoms.get_potential_energy())

        # Pull relaxed coords back into a StructureCfg.
        coords = atoms.get_positions()
        new_atoms = [
            orig.model_copy(update={
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            })
            for i, orig in enumerate(structure.atoms)
        ]
        new_structure = structure.model_copy(update={"atoms": new_atoms, "qm_relax": False})
        return QmRelaxResult(
            structure=new_structure,
            energy_kcal_mol=e_ev * _EV_TO_KCAL,
        )

    def settings_fingerprint(self) -> str:
        return json.dumps({
            "code": "qe",
            "functional": self.functional,
            "ecutwfc": self.ecutwfc,
            "ecutrho": self.ecutrho,
            "kpts": list(self.kpts),
            "spin": self.spin,
            "charge": self.charge,
            "degauss": self.degauss,
        }, sort_keys=True)
