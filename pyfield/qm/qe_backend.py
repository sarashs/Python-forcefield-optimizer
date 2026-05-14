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

import datetime as _dt
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

        # Optional persistent log/work directory. When set (by the
        # populator), every QE call writes its input/output files to a
        # stable, predictable subdirectory of `log_dir` instead of a
        # tempfile, and the directory survives the call regardless of
        # exit status. `run_label` is a short name (typically
        # "<op>_<structure>") that the populator updates before each
        # call so the workdir is recognisable from the outside while a
        # long run is in progress.
        self.log_dir: Optional[Path] = None
        self.run_label: Optional[str] = None

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

    def _build_input_data(
        self,
        functional: Optional[str] = None,
        *,
        relax_cell: bool = False,
    ) -> dict:
        """Construct the QE input-data dict.

        With `relax_cell=True` we switch `calculation` from `scf` to
        `vc-relax` and add the `&IONS` and `&CELL` namelists QE needs
        to drive a variable-cell BFGS internally. `cell_factor` pre-
        sizes the plane-wave basis to tolerate the cell expanding
        during the relax (~2× volume range without the basis going
        stale — the classic Pulay-stress fix).
        """
        xc = _qe_input_dft(functional or self.functional)
        data: dict = {
            "control": {
                "calculation": "vc-relax" if relax_cell else "scf",
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
        if relax_cell:
            data["control"]["forc_conv_thr"] = 1.0e-3   # Ry/bohr (~0.025 eV/Å)
            data["control"]["etot_conv_thr"] = 1.0e-4   # Ry
            # QE's default nstep for vc-relax is 50 — too tight for an
            # underbound seed cell that needs 6%+ volume change. Bump
            # to 200 so the relax can finish cleanly inside the limit.
            data["control"]["nstep"] = 200
            data["ions"] = {"ion_dynamics": "bfgs"}
            data["cell"] = {
                "cell_dynamics": "bfgs",
                "press_conv_thr": 0.5,                  # kbar
                # 'xyz' = independent a, b, c, no shear → orthorhombic
                # relaxed cell, which is what our `box: [a, b, c]`
                # schema can carry. 'all' (full 9 DOFs) would let
                # vc-relax break symmetry into a sheared cell whenever
                # the structure has no enforced point group; we'd then
                # raise in `_relax_vc` because the schema would lose
                # the off-diagonal terms. 'xyz' also has fewer DOFs to
                # optimize (3 cell vs 9), so the relax converges in
                # fewer ionic steps.
                "cell_dofree": "xyz",
                # cell_factor lives in &CELL (not &SYSTEM — older QE
                # docs are inconsistent about this and 6.4.x rejects
                # it from &SYSTEM with "bad line in namelist &system").
                # Pre-allocates the plane-wave basis as if the cell
                # were 2× larger so it doesn't go stale during the
                # internal BFGS — the standard Pulay-stress fix.
                "cell_factor": 2.0,
            }
        return data

    def _make_calculator(
        self,
        atoms,
        functional: Optional[str] = None,
        *,
        relax_cell: bool = False,
        directory: Optional[str] = None,
    ):
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
        kw = dict(
            profile=profile,
            pseudopotentials=self.pseudopotentials,
            input_data=self._build_input_data(functional=functional, relax_cell=relax_cell),
            kpts=tuple(self.kpts),
        )
        if directory is not None:
            kw["directory"] = directory
        return Espresso(**kw)

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
        import shutil
        atoms = self._to_ase(structure)
        workdir, persistent = self._resolve_workdir("sp")
        try:
            atoms.calc = self._make_calculator(
                atoms,
                functional=self._effective_functional(structure),
                directory=str(workdir),
            )
            e_ev = float(atoms.get_potential_energy())
            try:
                f_ev = np.asarray(atoms.get_forces(), dtype=float)
                forces = f_ev * _EV_PER_A_TO_KCAL_PER_A
            except Exception:
                forces = None
        finally:
            if not persistent:
                shutil.rmtree(workdir, ignore_errors=True)
        return QmSinglePoint(
            energy_kcal_mol=e_ev * _EV_TO_KCAL,
            forces_kcal_mol_per_A=forces,
        )

    def relax(
        self,
        structure: StructureCfg,
        constraint: Optional[ConstraintSpec] = None,
    ) -> QmRelaxResult:
        # Variable-cell branch: structure asks for it AND there's no
        # internal-coordinate constraint (constrained scans freeze the
        # strained cell as part of the constraint, atoms-only relax inside).
        if getattr(structure, "qm_relax_cell", False) and constraint is None:
            return self._relax_vc(structure)
        return self._relax_atoms_only(structure, constraint=constraint)

    def _relax_atoms_only(
        self,
        structure: StructureCfg,
        constraint: Optional[ConstraintSpec] = None,
    ) -> QmRelaxResult:
        import shutil
        from ase.optimize import BFGS

        atoms = self._to_ase(structure)
        op = "relax_constrained" if constraint is not None else "relax"
        workdir, persistent = self._resolve_workdir(op)
        try:
            atoms.calc = self._make_calculator(
                atoms,
                functional=self._effective_functional(structure),
                directory=str(workdir),
            )
            if constraint is not None:
                self._apply_constraint(atoms, constraint)
            bfgs_log = str(workdir / "bfgs.log") if persistent else None
            BFGS(atoms, logfile=bfgs_log).run(fmax=0.025)   # 0.025 eV/Å ≈ 0.5 kcal/mol/Å
            e_ev = float(atoms.get_potential_energy())
        finally:
            if not persistent:
                shutil.rmtree(workdir, ignore_errors=True)

        coords = atoms.get_positions()
        new_atoms = [
            orig.model_copy(update={
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            })
            for i, orig in enumerate(structure.atoms)
        ]
        new_structure = structure.model_copy(update={
            "atoms": new_atoms,
            "qm_relax": False,
            "qm_relax_cell": False,
        })
        return QmRelaxResult(
            structure=new_structure,
            energy_kcal_mol=e_ev * _EV_TO_KCAL,
        )

    def _resolve_workdir(self, op: str) -> Tuple[Path, bool]:
        """Return `(workdir, persistent)` for one QE call.

        `persistent=True` means the populator wired a `log_dir` and we
        should keep the directory after the call so the user can `tail
        -f` live runs and post-mortem on failures. `persistent=False` is
        the original tempfile path used for unit tests and ad-hoc
        backend instantiation, where the directory is deleted on
        success.

        The timestamp suffix means re-runs (e.g. `force=True`) don't
        clobber each other — each call gets a fresh subdir and the
        previous one stays around for inspection.
        """
        if self.log_dir is None:
            import tempfile
            return Path(tempfile.mkdtemp(prefix=f"qe_{op}_")), False
        label = self.run_label or "anon"
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        wd = Path(self.log_dir) / f"{label}_{ts}"
        wd.mkdir(parents=True, exist_ok=True)
        return wd, True

    @staticmethod
    def _read_qe_tail(workdir: Path, *, n: int = 80) -> str:
        """Concatenate the last n lines of QE's stdout / stderr files."""
        chunks: List[str] = []
        for fname in ("espresso.pwo", "espresso.err"):
            p = workdir / fname
            if not p.exists() or p.stat().st_size == 0:
                continue
            try:
                lines = p.read_text(errors="replace").splitlines()
            except Exception:
                continue
            chunks.append(f"\n----- {fname} (last {n} lines) -----")
            chunks.extend(lines[-n:])
        return "\n".join(chunks) if chunks else "(no QE output captured)"

    def _relax_vc(self, structure: StructureCfg) -> QmRelaxResult:
        """Variable-cell relax via QE's native `calculation: vc-relax`.

        ASE's `BFGS + ExpCellFilter` would fight Pulay stress (the
        plane-wave basis is fixed at the input cell). QE does it right:
        with `cell_factor: 2.0` the basis tolerates the cell changing
        during the internal BFGS without going stale.

        ASE's calculator returns the final SCF energy but doesn't
        update the atoms object's positions/cell on a vc-relax. We
        read the last image of the QE trajectory file explicitly.

        On QE failure we preserve the work directory and embed the
        tail of `espresso.pwo` / `espresso.err` in the raised
        exception — without this, ASE's `subprocess.CalledProcessError`
        only reports the exit code and the temp dir is gone.
        """
        import shutil
        import subprocess
        import tempfile
        from ase.io import read as ase_read

        atoms = self._to_ase(structure)
        workdir, persistent = self._resolve_workdir("vcrelax")
        try:
            atoms.calc = self._make_calculator(
                atoms,
                functional=self._effective_functional(structure),
                relax_cell=True,
                directory=str(workdir),
            )
            try:
                e_ev = float(atoms.get_potential_energy())   # triggers pw.x; QE drives BFGS internally
            except subprocess.CalledProcessError as e:
                # QE returns non-zero even when it wrote a valid final
                # image. Common case: STOP 3 = hit `nstep` without
                # satisfying both `forc_conv_thr` and `press_conv_thr`,
                # but the last step's geometry + energy + forces are
                # still in espresso.pwo and "JOB DONE" appears at the
                # bottom. Don't throw away an hour-plus of compute over
                # a soft-fail like that — read the last image, warn,
                # and proceed.
                pwo = workdir / "espresso.pwo"
                pwo_text = pwo.read_text(errors="replace") if pwo.exists() else ""
                if "JOB DONE" in pwo_text:
                    import warnings
                    warnings.warn(
                        f"QE vc-relax exited {e.returncode} but completed "
                        "with JOB DONE — likely hit `nstep` without fully "
                        "converging both forc_conv_thr and press_conv_thr. "
                        "Using the last step's geometry; if you need a "
                        "tighter relax, rerun with a larger nstep or "
                        "loosen the thresholds.",
                        RuntimeWarning,
                    )
                    e_ev = float(atoms.calc.results.get("energy", 0.0)) \
                        if atoms.calc.results.get("energy") is not None else None
                    if e_ev is None:
                        # ASE didn't parse the energy off the failed run.
                        # Pull it from the final image we'll read below.
                        final_tmp = ase_read(str(pwo), index=-1)
                        e_ev = float(final_tmp.get_potential_energy())
                else:
                    tail = self._read_qe_tail(workdir, n=80)
                    if persistent:
                        # Already in a stable location — don't move it,
                        # just raise pointing at it.
                        kept = workdir
                    else:
                        kept = Path(tempfile.gettempdir()) / f"qe_vcrelax_failed_{workdir.name}"
                        shutil.move(str(workdir), str(kept))
                        workdir = None
                    raise RuntimeError(
                        f"QE vc-relax failed (exit {e.returncode}). "
                        f"Inputs/outputs preserved at: {kept}\n{tail}"
                    ) from e
            out = workdir / "espresso.pwo"
            final = ase_read(str(out), index=-1)
        finally:
            # Only clean up tempfile-mode workdirs on success. Persistent
            # ones (set by the populator) stay behind so the user can
            # `tail -f espresso.pwo` after the fact and audit the run.
            if workdir is not None and not persistent:
                shutil.rmtree(workdir, ignore_errors=True)

        coords = final.get_positions()
        cell = np.asarray(final.get_cell().array, dtype=float)
        # The StructureCfg schema only carries `box: [a, b, c]` (orthorhombic).
        # Two regimes for what we do with QE's relaxed cell:
        # - off-diagonals < 5 % of axis length: project to the diagonal
        #   with a RuntimeWarning. This is the realistic "noisy near-
        #   orthorhombic" case that pops up with `cell_dofree: 'all'`
        #   on low-symmetry inputs (disordered cation sublattices etc.)
        #   — schema can't carry the shear, but it's small enough that
        #   the diagonal is a faithful summary of the relaxed cell.
        # - off-diagonals ≥ 5 %: raise. The cell is meaningfully sheared
        #   (e.g. Peierls-distorted) and the diagonal would mis-describe
        #   it. User should re-run with `cell_dofree: 'xyz'` (our
        #   current default) or extend the schema to a 3×3 lattice.
        off = cell - np.diag(np.diag(cell))
        diag = np.abs(np.diag(cell))
        rel_shear = np.max(np.abs(off) / diag.max())
        if rel_shear >= 0.05:
            raise RuntimeError(
                "QE vc-relax produced a strongly non-orthorhombic cell "
                f"(max |off-diagonal| / |axis| = {rel_shear:.1%}):\n{cell}\n"
                "The current StructureCfg.box schema is `[a, b, c]` only. "
                "Re-run with `cell_dofree: 'xyz'` in the QE input "
                "(the new default), or extend the schema to a full "
                "3×3 lattice."
            )
        if rel_shear > 1e-4:
            import warnings
            warnings.warn(
                f"QE vc-relax cell has {rel_shear:.1%} shear; projecting "
                "to the diagonal `box: [a, b, c]` (the schema doesn't "
                "carry off-diagonal cell terms). Diagonal is a faithful "
                "summary at this magnitude of shear.",
                RuntimeWarning,
            )
        new_box = (float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2]))

        new_atoms = [
            orig.model_copy(update={
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            })
            for i, orig in enumerate(structure.atoms)
        ]
        new_structure = structure.model_copy(update={
            "atoms": new_atoms,
            "box": new_box,
            "qm_relax": False,
            "qm_relax_cell": False,
        })
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
            # SCF tolerance affects the converged result (forces /
            # stress, in particular) — different conv_thr → different
            # cache entry. mixing_beta is omitted because it only
            # changes the convergence path, not the converged values.
            "conv_thr": self.conv_thr,
        }, sort_keys=True)
