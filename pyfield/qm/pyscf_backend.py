"""PySCF backend for `pyfield qm-prep`.

Why PySCF as MVP:

- pure-Python, ships pre-built pip wheels, no LIBXC / FFTW / pseudopotential
  hassle on a fresh CI runner;
- supports HF / DFT / post-HF on Gaussian basis sets — enough for any
  refit that doesn't need plane-waves;
- has a parallel cluster (`pyscf.gto`) and PBC (`pyscf.pbc`) interface
  that share the same SCF code, so we can dispatch on
  `StructureCfg.pbc` from the same backend without an extra module;
- forces and Hartree-units energies fall out of the standard interface
  (`Gradients.kernel()` and `mf.kernel()`).

The cluster path uses `pyscf.gto.M` + `pyscf.dft.RKS`. The periodic
path (when `structure.pbc` is true) uses `pyscf.pbc.gto.Cell` +
`pyscf.pbc.dft.RKS` at Γ-only k-sampling — accurate for moderately
sized supercells where the BZ is already well-sampled by the
supercell folding (≥ 32 atoms in metallic / narrow-gap systems is the
rule of thumb).
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
# PySCF derivatives are in Hartree/Bohr → convert to kcal/mol/Å.
_BOHR_TO_A = 0.529177210903
_GRAD_HA_BOHR_TO_KCAL_A = _HA_TO_KCAL / _BOHR_TO_A


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


def _orthorhombic_lattice(structure: StructureCfg) -> np.ndarray:
    """Return the 3×3 lattice matrix (in Å) for a periodic StructureCfg.

    For v1, `box: [a, b, c]` is interpreted as orthorhombic — the only
    triclinic angles are 90°. Triclinic / monoclinic cells (a future
    `lattice:` field) plug in here without changing call sites.
    """
    a, b, c = structure.box
    return np.diag([float(a), float(b), float(c)])


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

    def _effective_settings(self, structure: StructureCfg):
        """Per-structure overrides win over the global qm: block.

        Lets a single training set mix functionals — typically B3LYP for
        cluster training (where charge transfer matters) and PBE for
        PBC training (where hybrid-functional gradients are still
        unreliable in PySCF). Per-structure overrides go in
        `__pydantic_extra__` (StructureCfg has `extra="allow"`) under
        the keys `qm_functional` and `qm_basis`.
        """
        extras = getattr(structure, "__pydantic_extra__", {}) or {}
        functional = extras.get("qm_functional", self.functional)
        basis = extras.get("qm_basis", self.basis)
        return functional, basis

    def _build_mf(self, structure: StructureCfg):
        """Cluster (molecular) SCF object.

        For Gaussian basis sets like `def2-svp`, heavy elements (Z > 36 —
        Ge, Sb, Te, …) need an effective core potential (ECP). PySCF's
        `dft.RKS.Gradients()` raises `NotImplementedError` for SCFs
        built without ECPs on those elements, so we always pass
        `ecp=basis` to `gto.M`. PySCF emits a "ECP X not found for H"
        warning per light element and silently skips them — the warning
        is noise, not an error.
        """
        from pyscf import gto, dft, scf
        functional, basis = self._effective_settings(structure)
        mol = gto.M(
            atom=_atoms_to_pyscf_geom(structure),
            basis=basis,
            ecp=basis,
            spin=self.spin,
            charge=self.charge,
            verbose=0,
            unit="Angstrom",
        )
        if functional.lower() in ("hf",):
            mf = scf.RHF(mol) if self.spin == 0 else scf.UHF(mol)
        else:
            rks_cls = dft.RKS if self.spin == 0 else dft.UKS
            mf = rks_cls(mol)
            mf.xc = functional
        # Density fitting on by default: cuts the cost of hybrid DFT
        # exact-exchange evaluation by ~10× on medium-sized
        # heavy-element clusters (37-atom GeTe6 with B3LYP/def2-svp
        # is otherwise prohibitively slow). Density-fitting error on
        # energies / gradients is ~µHa per atom — well below DFT
        # accuracy. PySCF auto-picks `weigend` auxbasis to match
        # def2-svp / def2-tzvp.
        mf = mf.density_fit()
        # Heavy-element clusters (Te, Sb, Ge with ECP) sometimes need
        # more than PySCF's default 50 SCF cycles inside the gradient
        # scanner, especially on the first geom-opt step. Bump the cap
        # so geometric_solver doesn't fail with "Nuclear gradients of
        # ... not converged" after burning 50 iters.
        mf.max_cycle = 200
        return mf

    def _build_pbc_mf(self, structure: StructureCfg, *, with_gradients: bool = False):
        """Periodic SCF object — Γ-only k-sampling, Gaussian density fitting.

        PySCF's default `FFTDF` integrator builds a plane-wave mesh
        whose density is set by the basis set's stiffness; for
        molecular Gaussian bases like def2-svp on a ~6 Å cell that
        mesh balloons to ≥10⁹ points and the calculation OOMs.
        `GDF` (Gaussian density fitting) sidesteps this entirely by
        re-expanding the density in an auxiliary Gaussian basis, which
        is the standard PBC approach for molecular-style basis sets.

        When `with_gradients=True`, install `MultiGridNumInt2` as the
        numerical integrator: `pbc.dft.RKS.Gradients` raises
        `NotImplementedError` unless the SCF was built with that
        integrator. PBC gradients with hybrid functionals are still
        partial in PySCF — PBE works, B3LYP / PBE0 may not. Use
        `qm_relax: true` on PBC structures with caution.
        """
        from pyscf.pbc import gto as pgto, dft as pdft, scf as pscf
        from pyscf.pbc import df as pdf
        functional, basis = self._effective_settings(structure)
        cell = pgto.Cell()
        cell.atom = _atoms_to_pyscf_geom(structure)
        cell.a = _orthorhombic_lattice(structure)
        cell.basis = basis
        cell.unit = "Angstrom"
        cell.spin = self.spin
        cell.charge = self.charge
        cell.verbose = 0
        # Cap the kinetic-energy cutoff. PySCF auto-derives `ke_cutoff`
        # from the most diffuse basis function; def2-svp on a small cell
        # gives a cutoff that produces a 10⁴³-element FFT mesh. 100 Ha
        # (~2.7 keV) is plenty for typical Gaussian-basis PBC DFT and
        # caps the mesh at sane sizes.
        cell.ke_cutoff = float(
            (self.cfg.__pydantic_extra__ or {}).get("pbc_ke_cutoff", 100.0)
        )
        cell.build()
        if functional.lower() in ("hf",):
            mf = pscf.RHF(cell) if self.spin == 0 else pscf.UHF(cell)
            mf.with_df = pdf.GDF(cell)
        else:
            mf = pdft.RKS(cell) if self.spin == 0 else pdft.UKS(cell)
            mf.xc = functional
            if with_gradients:
                # Multigrid path: handles XC + J via the same grid;
                # supports analytic gradients for *pure* functionals
                # (PBE, LDA). Don't combine with GDF — they're
                # alternative approaches and PySCF's gradient code
                # gets confused. Hybrid functionals (B3LYP) don't
                # currently have a working gradient path here; the
                # `qm_functional: pbe` per-structure override on PBC
                # cells is the recommended workaround.
                from pyscf.pbc.dft.multigrid import MultiGridNumInt2
                mf._numint = MultiGridNumInt2(cell)
            else:
                # Energy-only path: GDF keeps the Coulomb integrals
                # tractable for molecular Gaussian bases like
                # def2-svp without the multigrid setup cost.
                mf.with_df = pdf.GDF(cell)
        return mf

    def _is_pbc(self, structure: StructureCfg) -> bool:
        return bool(getattr(structure, "pbc", False))

    # ---------- public API ----------------------------------------------

    def single_point(self, structure: StructureCfg) -> QmSinglePoint:
        if self._is_pbc(structure):
            return self._single_point_pbc(structure)
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

    def _single_point_pbc(self, structure: StructureCfg) -> QmSinglePoint:
        # Forces aren't needed for single-points unless we're targeting a
        # `forces` objective. Build without multigrid to skip the heavier
        # integrator setup; if forces are requested we can opt in below.
        mf = self._build_pbc_mf(structure, with_gradients=False)
        e_ha = mf.kernel()
        return QmSinglePoint(
            energy_kcal_mol=float(e_ha) * _HA_TO_KCAL,
            forces_kcal_mol_per_A=None,
        )

    def relax(
        self,
        structure: StructureCfg,
        constraint=None,
    ) -> QmRelaxResult:
        if self._is_pbc(structure):
            return self._relax_pbc(structure, constraint=constraint)
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

    def _relax_pbc(self, structure: StructureCfg, constraint=None) -> QmRelaxResult:
        """Periodic geom-opt with the cell vectors *fixed*.

        For a strain scan we want atoms to relax inside a strained cell
        without the lattice itself relaxing back. PySCF's pbc geomopt
        does exactly that — it optimises atomic positions only, never
        the cell. Constraints (distance/angle/dihedral) are passed
        through the same way as the molecular relax.
        """
        # `with_gradients=True` installs MultiGridNumInt2 — required by
        # `pbc.dft.RKS.Gradients`. SCF time is similar to plain numint
        # at this cell size; skipping it would land us in
        # `NotImplementedError('pbc-RKS must be computed with MultiGridNumInt2')`.
        mf = self._build_pbc_mf(structure, with_gradients=True)
        from pyscf.geomopt.geometric_solver import optimize as geom_optimize
        kwargs = {}
        if constraint is not None:
            import tempfile
            spec = _render_geometric_constraint(constraint)
            tf = tempfile.NamedTemporaryFile(
                "w", suffix=".geometric.txt", delete=False
            )
            tf.write(spec)
            tf.close()
            kwargs["constraints"] = tf.name
        new_cell = geom_optimize(mf, **kwargs)
        # Recompute energy at the relaxed geometry. new_cell is a Cell.
        from pyscf.pbc import dft as pdft, scf as pscf
        if self.functional.lower() in ("hf",):
            new_mf = pscf.RHF(new_cell) if self.spin == 0 else pscf.UHF(new_cell)
        else:
            new_mf = pdft.RKS(new_cell) if self.spin == 0 else pdft.UKS(new_cell)
            new_mf.xc = self.functional
        e_ha = new_mf.kernel()
        coords_a = new_cell.atom_coords(unit="Angstrom")
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
