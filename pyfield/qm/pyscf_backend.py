"""PySCF backend for `pyfield qm-prep`.

Why PySCF as MVP:
- pure-Python, ships pre-built pip wheels, no LIBXC / FFTW / pseudopotential
  hassle on a fresh CI runner;
- supports HF / DFT / post-HF on Gaussian basis sets — enough for any
  refit that doesn't need plane-waves;
- forces and Hartree-units energies fall out of the standard interface
  (`Gradients.kernel()` and `mf.kernel()`).

For periodic systems / plane waves, swap to `pyfield.qm.qe_backend`
when it lands. The `QmBackend` interface is identical.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Optional

import numpy as np

from pyfield.config.schema import StructureCfg
from pyfield.qm.base import QmBackend, QmRelaxResult, QmSinglePoint


# Hartree → kcal/mol; Ångström unchanged.
_HA_TO_KCAL = 627.5095


def _render_geometric_constraint(constraint) -> str:
    """Render a `ConstraintSpec` as a geomeTRIC `$set` block (1-based atoms).

    geomeTRIC syntax:
        $set
        distance i j r0
        angle i j k theta0
        dihedral i j k l phi0
        $end
    """
    kind = constraint["kind"]
    atoms = " ".join(str(a) for a in constraint["atoms"])
    value = constraint["value"]
    return f"$set\n{kind} {atoms} {value}\n$end\n"
# PySCF derivatives are in Hartree/Bohr → convert to kcal/mol/Å.
_BOHR_TO_A = 0.529177210903
_GRAD_HA_BOHR_TO_KCAL_A = _HA_TO_KCAL / _BOHR_TO_A


def _atoms_to_pyscf_geom(structure: StructureCfg) -> str:
    """`StructureCfg` → PySCF's '<El> <x> <y> <z>; …' geometry string."""
    if structure.atoms is None:
        raise NotImplementedError(
            "PySCF backend currently requires inline `atoms:` in the StructureCfg "
            "(xyz `path:` loading lands when the same xyz reader is shared with "
            "the FF-side simulations)."
        )
    parts = [f"{a.element} {a.x} {a.y} {a.z}" for a in structure.atoms]
    return "; ".join(parts)


class PySCFBackend(QmBackend):
    name = "pyscf"

    def __init__(self, qm_cfg):
        # Lazy import — the rest of pyfield works without pyscf installed.
        from pyscf import gto, dft, scf  # noqa: F401  (probes import)
        self.cfg = qm_cfg
        self.basis = qm_cfg.basis
        self.functional = qm_cfg.functional
        # Per-code knobs (e.g. spin, charge) live in __pydantic_extra__.
        extras = qm_cfg.__pydantic_extra__ or {}
        self.spin = int(extras.get("spin", 0))
        self.charge = int(extras.get("charge", 0))

    # ---------- internals ------------------------------------------------

    def _build_mf(self, structure: StructureCfg):
        from pyscf import gto, dft, scf
        mol = gto.M(
            atom=_atoms_to_pyscf_geom(structure),
            basis=self.basis,
            spin=self.spin,
            charge=self.charge,
            verbose=0,
            unit="Angstrom",
        )
        if self.functional.lower() in ("hf",):
            return scf.RHF(mol) if self.spin == 0 else scf.UHF(mol)
        rks_cls = dft.RKS if self.spin == 0 else dft.UKS
        mf = rks_cls(mol)
        mf.xc = self.functional
        return mf

    # ---------- public API ----------------------------------------------

    def single_point(self, structure: StructureCfg) -> QmSinglePoint:
        mf = self._build_mf(structure)
        e_ha = mf.kernel()
        try:
            grad_ha_bohr = mf.Gradients().kernel()
            forces = -np.asarray(grad_ha_bohr) * _GRAD_HA_BOHR_TO_KCAL_A
        except Exception:
            forces = None
        return QmSinglePoint(
            energy_kcal_mol=float(e_ha) * _HA_TO_KCAL,
            forces_kcal_mol_per_A=forces,
        )

    def relax(
        self,
        structure: StructureCfg,
        constraint=None,
    ) -> QmRelaxResult:
        mf = self._build_mf(structure)
        # geometric_solver is the standard ASE-free PySCF optimiser.
        from pyscf.geomopt.geometric_solver import optimize as geom_optimize
        kwargs = {}
        if constraint is not None:
            # geomeTRIC reads constraints from a $set / $freeze block in
            # a small text file. We render an in-memory string and write
            # it to a temp file because the API takes a path.
            import tempfile
            spec = _render_geometric_constraint(constraint)
            tf = tempfile.NamedTemporaryFile(
                "w", suffix=".geometric.txt", delete=False
            )
            tf.write(spec)
            tf.close()
            kwargs["constraints"] = tf.name
        new_mol = geom_optimize(mf, **kwargs)
        # Recompute the energy at the relaxed geometry to be sure.
        from pyscf import dft, scf
        if self.functional.lower() in ("hf",):
            new_mf = scf.RHF(new_mol) if self.spin == 0 else scf.UHF(new_mol)
        else:
            rks_cls = dft.RKS if self.spin == 0 else dft.UKS
            new_mf = rks_cls(new_mol)
            new_mf.xc = self.functional
        e_ha = new_mf.kernel()
        # Pull Cartesian coordinates back out and patch the StructureCfg.
        coords_a = new_mol.atom_coords(unit="Angstrom")   # numpy (N, 3)
        new_atoms = [
            orig.model_copy(update={
                "x": float(coords_a[i, 0]),
                "y": float(coords_a[i, 1]),
                "z": float(coords_a[i, 2]),
            })
            for i, orig in enumerate(structure.atoms)
        ]
        new_structure = structure.model_copy(update={"atoms": new_atoms, "qm_relax": False})
        return QmRelaxResult(
            structure=new_structure,
            energy_kcal_mol=float(e_ha) * _HA_TO_KCAL,
        )

    # ---------- cache fingerprint ---------------------------------------

    def settings_fingerprint(self) -> str:
        return json.dumps({
            "code": "pyscf",
            "basis": self.basis,
            "functional": self.functional,
            "spin": self.spin,
            "charge": self.charge,
        }, sort_keys=True)
