# GST drift mechanism study — experiment tracking

**Question.** What is the molecular mechanism of resistance drift in
amorphous Ge₂Sb₂Te₅ phase-change memory?

**Status (start date 2026-05-03).** Project bootstrapped, force field
seeded, training set scaffolded. No QM has been run yet.

---

## 1 · What we're trying to find out

Phase-change memory (PCM) cells store bits as the resistance of a tiny
chunk of GST — low resistance ≈ "1" (crystalline), high resistance ≈
"0" (amorphous). The amorphous state is created by a 5 ns melt-quench
pulse and is far from thermodynamic equilibrium; it slowly relaxes
over time, and as it does, the electrical resistance **drifts upward**
following an empirical power law

> R(t) ∝ tᵛ,    ν ≈ 0.1 at room temperature

This drift is the single biggest reason multi-level cell (MLC) PCM
isn't commercial: encoded resistance levels overlap as they drift.

There are three published hypotheses for the molecular cause; they
have never been simultaneously discriminated by direct simulation:

| Mechanism | Atomic-scale story | Time-evolving observable |
|-----------|---------------------|--------------------------|
| **A — wrong-bond annealing** | Quench-frozen amorphous GST has excess Te–Te / Ge–Ge / Sb–Sb homopolar bonds; over time these convert to Ge–Te / Sb–Te. | Pair-type populations f(Ge–Te), f(Te–Te), … vs time |
| **B — Peierls re-ordering** | Some Ge sites are frozen in near-symmetric 6-fold coordination; over time they relax to the bulk-favoured asymmetric 3+3 ("Peierls-distorted") arrangement. | Distribution of Ge–Te bond-length asymmetry vs time |
| **C — void coalescence** | Free-volume regions slowly merge and migrate. | Voronoi free-volume distribution and Te slow modes |

We aim to **rule mechanisms in or out from the same simulated
trajectory**, by measuring all three observables on a 5–10 nm
amorphous slab held at 300–425 K for ~100 ns. Existing tools can't do
this:

- AIMD reaches ~200 atoms × ~100 ps. Drift happens in cells of ~10⁵
  atoms over hours — five orders of magnitude short of what's
  needed.
- Empirical Tersoff potentials (Sosso 2012) capture the static
  amorphous structure but cannot reorganise bonds — they're blind to
  drift by construction.

A reactive force field calibrated to the four chemistry questions
that drift asks (see §3) closes the gap. That's what this project
fits.

## 2 · Why the answer matters

- **Direct industrial impact.** If mechanism A dominates, doping with
  elements that suppress homopolar bonds (e.g., N — empirically slows
  drift) follows from a clean design rule. If B dominates, off-Ge
  stoichiometry helps. The recipe-from-physics is genuinely missing
  from the field.
- **MLC enablement.** Quantifying the drift timescale at different
  cell volumes / temperatures predicts where in the design space
  multi-level PCM becomes shippable.
- **Endurance prediction.** The same FF supports cycling-fatigue MD
  (SET/RESET pulses), which addresses a separate but related
  industrial pain point.

## 3 · How we plan to find it out

The plan splits into two phases:

### 3.1 · Force-field training (this folder)

Build a ReaxFF for {Ge, Sb, Te, H} that gets four chemistry questions
right. Each question maps to one or more reference structures in the
training set:

| Question drift asks | Reference structure(s) | QM backend | Section |
|---|---|---|---|
| What's the equilibrium bonding + elastic response of the ordered phase? | `GST_rocksalt` (PBC, plane-wave) | **Quantum ESPRESSO**, PBE | §4.1 |
| Does Ge prefer 3+3 asymmetric or 6-fold symmetric? | `GeTe6_Peierls` (cluster, H-passivated) | PySCF, B3LYP/def2-svp | §4.2 |
| What's the energy gap between octahedral and tetrahedral Ge? | `GeTe4_tet` (cluster, H-passivated) | PySCF, B3LYP/def2-svp | §4.3 |
| What's the cost of a homopolar wrong bond? | `Te2_wrong`, `Ge2_dumbbell`, `Sb2_dumbbell` (clusters, H-passivated) | PySCF, B3LYP/def2-svp | §4.4 |

The two-backend split is deliberate — see §5.6 for the rationale.
PyField dispatches per-structure based on the `qm_code:` override on
each `StructureCfg`.

H atoms appear only as **bond-saturation caps** on the cluster
references. They are not part of the simulated GST cell at production
time. They add H–Ge / H–Sb / H–Te bond, off-diagonal, and angle
parameters to the FF, but those parameters aren't part of the
trainable subset (we leave them at literature CHO-ReaxFF values).

### 3.2 · Production simulations (downstream of this folder)

Once the FF is fitted, plain LAMMPS MD runs:

1. Build a 5–10 nm × 5–10 nm × 5–10 nm amorphous Ge₂Sb₂Te₅ slab by
   melt-quench (1500 K → 300 K @ 10¹² K/s).
2. Anneal at 300, 350, 400, 425 K for 50–100 ns each.
3. Measure all three drift observables simultaneously from the
   trajectory. Compare to:
   - **EXAFS bond populations** — Kolobov & Fons, *Nat. Mater.* **3**, 703 (2004); Kolobov et al., *Phys. Rev. B* **76**, 224107 (2007).
   - **PDF from neutron / X-ray** — Caravati et al., *J. Phys.: Condens. Matter* **21**, 255501 (2009); Sebastian et al., *Adv. Mater.* **30**, 1801268 (2018).
   - **Drift coefficient** ν ≈ 0.1 — Boniardi et al., *J. Appl. Phys.* **110**, 124313 (2011); Salinga et al., *Nat. Commun.* **9**, 2900 (2018).

These production runs are *not* part of this folder. They use the FF
this folder produces.

## 4 · Training reference structures

Sized so each fits its QM backend in the 5–60 minute range per
single-point on a laptop CPU. **Two QM backends in one training set**:
QE (plane-wave PBC) for the bulk cell, PySCF (Gaussian-basis cluster)
for everything else. PyField dispatches per-structure based on the
`qm_code:` override.

### 4.1 · Bulk: rocksalt GST cell (PBC, Quantum ESPRESSO)

```
18 atoms in a cubic supercell, lattice parameter ~6.0 Å, stoichiometry
Ge:Sb:Te = 2:2:5 with structural vacancies on the cation sublattice.
```

**Composition.** A 2 × 2 × 1 supercell of the Ge₂Sb₂Te₅ rocksalt
primitive cell. Te occupies the anion sublattice (8 sites in our
trimmed-down demo cell — the full 12-site cell relaxes with QE too,
just slower), and the cation sublattice is shared between Ge, Sb, and
2 vacancies (the canonical 20 % structural vacancy concentration that
defines the GST cubic phase — Yamada 1991).

**Reference data.**
- Lattice constant a₀ ≈ 6.02 Å (room T), Yamada *Jpn. J. Appl. Phys.*
  **30**, 2606 (1991). The YAML uses this as a *seed* for vc-relax
  (`qm_relax_cell: true`) — QE finds its own equilibrium volume for
  the chosen functional / pseudo combo, which can differ from the
  experimental value by a few percent and lands in `populated.yaml`
  alongside the relaxed atomic coordinates. See §10 (2026-05-09) for
  why anchoring the strain ladder on the literature value rather than
  on QE's `V_eq` was the source of the 1.5 × 10⁶ CMA plateau.
- Bulk modulus B = 39 GPa, Sun et al. *Phys. Rev. B* **76**, 245206
  (2007).

> **⚠️ IMPORTANT — seed cell is unphysically compressed; the QE
> relaxation expanded it ~28 % linearly (~108 % volume).**
>
> The literature 6.02 Å refers to the conventional 8-atom rocksalt
> unit cell. Our training cell packs 18 atoms (8 Te + 5 Ge + 5 Sb,
> with 2 cation vacancies on a 12-site cation sublattice) into the
> same 6.02 Å cube — i.e. ~12 Å³/atom, **double-compressed** relative
> to real GST (~25 Å³/atom from Yamada). PBE/SSSP vc-relax responded
> by expanding the cell to `(7.61, 7.75, 7.69) Å` (≈ 7.65 Å mean
> linear, ~25.2 Å³/atom — physically correct).
>
> Net effect: PBE was **not** over-expanding by 28 %. It was relieving
> the seed's unphysical compression. The salvaged reference is
> physically reasonable; the strain ladder anchored on it (in the
> current `scanned.yaml`) is correct.
>
> If you re-seed this study, start `GST_rocksalt` at `~[7.65, 7.65, 7.65] Å`
> rather than `[6.02, 6.02, 6.02]` so the vc-relax has less work to
> do and the ionic-relax history is cleaner. The current cached
> result is fine for training; this is forward-looking advice for
> v2 of the training set. See §10 (2026-05-10) for the diagnostic
> chain.

**QM backend.** Quantum ESPRESSO via the `qm_code: qe` per-structure
override. Plane-wave SCF + analytic forces + ASE BFGS relaxation —
the path PySCF's PBC infrastructure can't handle reliably for our
cell sizes (see §5.6). PBE/def2-svp-equivalent settings:

- `ecutwfc = 50` Ry (orbital cutoff)
- `ecutrho = 400` Ry (= 8 × ecutwfc, SSSP recommendation)
- `kpts = [2, 2, 2]` (8 k-points on the 18-atom cubic cell)
- `conv_thr = 1e-9` (SCF tolerance — tight enough that stress
  noise sits at ~0.1 kbar, well below the 0.5 kbar `press_conv_thr`
  vc-relax targets)
- Pseudopotentials: SSSP 1.1.2 PBE Efficiency UPFs
  (`ge_pbe_v1.4.uspp.F.UPF`, `sb_pbe_v1.4.uspp.F.UPF`,
  `Te_pbe_v1.uspp.F.UPF`).

These were demo-grade (`ecutwfc: 40, kpts: [1,1,1], conv_thr: 1e-7`)
through the first plateau-diagnosis pass; bumped to the production
values above on 2026-05-09 once it became clear the strain-target
errors from Γ-only sampling were comparable to the residuals CMA was
trying to fit. See §10 (2026-05-09) for the accuracy-vs-cost table
and rationale.

**Scans on this structure (PBC, relaxed_constrained).**
- Hydrostatic strain, range [-0.06, +0.06] in 7 points → **B (bulk
  modulus)**.
- Uniaxial strain along z, range [-0.04, +0.04] in 4 points →
  **C₁₁** (combined with hydrostatic).
- Biaxial strain in xy, range [-0.04, +0.04] in 4 points → **C₁₁ +
  C₁₂**.
- Shear in xy, range [-0.04, +0.04] in 4 points → **C₄₄**.

At each strain magnitude QE locks the lattice at the deformed cell
and relaxes atoms inside via ASE BFGS — that's the **relaxed**
elastic constant the experiments measure. (The clamped-ion
approximation we'd get from rigid single-points typically
underestimates these by 10–25 %.)

**Wall-clock note.** Single-core `pw.x` on the 18-atom cell at the
demo settings takes ~100 s per single-point and ~25–50 min per
constrained relax (BFGS to fmax = 0.025 eV/Å). Running with
`ESPRESSO_COMMAND='mpirun -np 8 pw.x ...'` cuts this 5–7×. Plan ~6
hours for the full bulk training set on 8 cores with cold cache.

### 4.2 · Cluster: Peierls Ge–Te₆ octahedron (H-capped) — *DEFERRED to v2*

The 37-atom (1 Ge + 6 Te + 30 H caps) cluster is **too expensive for
PySCF molecular DFT on a typical workstation**: B3LYP/def2-svp + ECP
+ density fitting still budgets > 30 min per geom-opt step on this
machine, dominated by exact-exchange evaluation over 30 H caps.
PBE on the same atom count is ~3–5× faster but still timed out on
our 10–15 minute test budgets.

**v2 unblockers (any one suffices):**

- **ORCA backend** (`pyfield.qm.orca_backend` — planned). RIJCOSX
  hybrid-DFT acceleration is ~10× faster than PySCF density fitting
  on this size of cluster, and ORCA handles heavy-element symmetry
  more robustly.
- **Closed-shell molecular cluster** instead of H-passivated. E.g.
  (GeTe)₄ cubane (8 atoms, no caps needed), naturally closed-shell,
  captures Peierls-like bond-length asymmetry. Smaller and cleaner.
- **QE-on-cluster mode**: surround with vacuum padding and use the
  plane-wave QE backend.

**The Peierls double-well physics is partially captured by the bulk
strain scans on `GST_rocksalt`**: the rocksalt phase has the
asymmetric 3+3 Ge–Te split built in, and the strain response under
all four deformation modes measures how stiff that asymmetry is.
v1 of the FF therefore still has *some* Peierls signal, just weaker
than a direct cluster scan would give.

### 4.2.skip · (legacy section — kept for reference)

```
1 Ge in the centre of 6 Te atoms (octahedral). Each Te is then bonded
to 5 H atoms on its outward face to passivate the dangling bonds it
would have had in the real bulk environment (Te valence = 2 in
chalcogenides; in rocksalt-GST each Te has ~6 cation neighbours, so
each surface Te needs 5 caps).
```

**Atom count.** 1 Ge + 6 Te + 30 H = 37 atoms.

**Why this cluster.** The Peierls distortion is the *defining*
chemical feature of crystalline GST. Each Ge sits asymmetrically
inside its Te₆ cage — three short bonds (~2.85 Å) on one side and
three long bonds (~3.15 Å) on the other. Symmetric 6-fold coordination
is a saddle point. **Drift mechanism B** is precisely the slow relaxation
of frozen-in symmetric Ge sites toward the asymmetric ground state, so
the energy curve along the Ge displacement axis (a double well with a
small central barrier) is the most discriminating signal we can put
into the FF.

**Reference data.** Equilibrium Ge–Te bond split 2.85 Å / 3.15 Å —
Lencer et al., *Adv. Mater.* **25**, 6710 (2013). Symmetry-breaking
energy ~70 meV per Ge — Caravati 2009 AIMD.

**Scans (cluster, relaxed_constrained, H-frozen).**
- `atom_displacement` on Ge along the C₃ axis, range [-0.4, +0.4] Å
  in 9 points → the Peierls double-well curve.
- `bond_stretch` on one Ge–Te bond with the other 5 Te atoms in
  legs.j, range [2.5, 3.5] Å in 7 points → bond anharmonicity.

### 4.3 · Cluster: tetrahedral Ge–Te₄ (H-capped) — *DEFERRED to v2*

The 9-atom Td cluster has degenerate `t₂` frontier orbitals at the
central Ge that PySCF's DIIS SCF can't navigate cleanly — the geom-opt
runs into near-zero Hessian eigenvalues and either oscillates or burns
> 10 minutes per relax even with broken-symmetry starting geometries
and Newton SCF. We defer this cluster to v2, when a more robust QM
backend (e.g. ORCA via a future `pyfield.qm.orca_backend`) handles
heavy-element Td clusters reliably.

**The physics it covers is partially captured by `GeTe6_Peierls`**:
that cluster's atom-displacement scan probes the Ge coordination
preference (4-fold vs 6-fold) along the C₃ axis, which is what
`GeTe4_tet` would have measured directly. So v1 of the FF still has
a meaningful signal for this drift mechanism, just from a different
angle.

### 4.3.skip · (legacy section — kept for reference)

```
1 Ge + 4 Te in a tetrahedral arrangement; each Te capped by 1 H.
```

**Atom count.** 1 Ge + 4 Te + 4 H = 9 atoms.

**Why this cluster.** Some defective Ge sites in amorphous GST are
4-fold tetrahedrally coordinated — a "wrong" coordination for
crystalline GST but a known motif in the disordered phase. Drift
mechanism B's full picture is the slow Ge migration between 4-fold
tetrahedral sites and 6-fold octahedral sites; getting the relative
energy gap right (~0.3–0.5 eV per site at QM, Caravati 2009) requires
training against this cluster.

**Scans.**
- `bond_stretch` on one Ge–Te bond, range [2.4, 3.0] Å in 7 points.
- `angle_bend` Te–Ge–Te around the tetrahedral 109.5° vertex, range
  [90°, 130°] in 5 points → distortion from tetrahedral toward planar
  / pyramidal.

### 4.4 · Clusters: homopolar bond fragments (H-capped)

Three small clusters, each isolating one homopolar bond:

| Cluster | Atoms | Cap pattern |
|---|---|---|
| `Te2_wrong` | Te–Te dimer | 1 H on each Te (Te is divalent) |
| `Ge2_dumbbell` | Ge–Ge dimer | 3 H on each Ge (Ge is tetravalent) |
| `Sb2_dumbbell` | Sb–Sb dimer | 2 H on each Sb (Sb is trivalent in chalcogenides) |

**Why these clusters.** Drift mechanism A is precisely the slow
conversion of homopolar bonds (Te–Te, Ge–Ge, Sb–Sb) to heteropolar
ones (Ge–Te, Sb–Te). Without an explicit calibration of how
energetically expensive each homopolar bond is (relative to the
heteropolar alternative), the FF can't quantitatively predict
mechanism-A-driven drift.

**Scans (each).**
- `bond_stretch` on the homopolar bond from 2.5 → 4.0 Å in 7 points
  (drives dissociation; energy at large separation = the "no wrong
  bond" baseline).

### Summary of training set (v1, what actually ships)

| Reference | Mode | Backend | Heavy atoms | H caps | Total | Scan points |
|---|---|---|---|---|---|---|
| `GST_rocksalt` | PBC | QE / PBE | 18 | 0 | 18 | 19 (across 4 strain modes) |
| `Te2_wrong` | cluster | PySCF / B3LYP | 2 | 2 | 4 | 7 |
| `Ge2_dumbbell` | cluster | PySCF / B3LYP | 2 | 6 | 8 | 7 |
| `Sb2_dumbbell` | cluster | PySCF / B3LYP | 2 | 4 | 6 | 7 |
| **TOTAL** | | | **24** | **12** | **36** | **40** |

**Deferred to v2** (need a faster QM backend like ORCA):

- `GeTe6_Peierls` (37 atoms — was the headline drift-mechanism-B
  signal; bulk strain scans partially cover it).
- `GeTe4_tet` (9 atoms but Td orbital degeneracy stalls PySCF SCF —
  smaller fix needed).

**Geometry notes (added during v1 debug).** Early runs showed that
PySCF's DIIS SCF can't navigate the orbital degeneracies of perfectly
high-symmetry cluster geometries — collinear H-Te-Te-H, σ_h-symmetric
Sb₂ dumbbell, perfect-Td GeTe₄ all produced
`Nuclear gradients of <RKS_Scanner> not converged` errors inside
`geometric_solver`. Two fixes applied to the v1 starting geometries:
1. **Te₂_wrong** — H caps moved to a bent ~92° H-Te-Te angle (matches
   real H₂Te chemistry; breaks the degenerate π_y / π_z manifold).
2. **Sb₂_dumbbell** — H caps perturbed off the z = 0 plane by ±0.3 Å
   to break σ_h symmetry.
3. **GeTe₄_tet** — deferred to v2 (see §4.3); we tried both small
   off-Ge perturbations and staggered Ge-Te bond lengths, neither
   stabilised the SCF reliably.

These geometry tweaks are *starting points* — the constrained relax
finds the true symmetric minimum from there in 10–30 s per cluster.

**QM cold-cache budget (v1, what actually runs).** Roughly **5–8
hours on 8 MPI cores**:

- PBC strain scans (QE, dominant): 19 constrained relaxes × ~25 min
  each on `mpirun -np 8 pw.x` ≈ **~6 hours**.
- Cluster relaxes (3 references + 21 scan points, PySCF B3LYP/def2-svp
  + DF + ECP): each fresh relax ~1–4 min on the small clusters →
  **~1 hour total**.

Without MPI, the QE side is **5–7× slower** (~30 hours cold cache).
Set `ESPRESSO_COMMAND='mpirun -np N pw.x ...'` before launching the
notebook; ASE picks it up automatically. All steps are content-keyed
in `studies/gst_drift/runs/qm_cache/` so re-running after FF tweaks
is a no-op.

## 5 · Force-field seed and parameter provenance

ReaxFF for {Ge, Sb, Te} is **not** included in any LAMMPS-bundled
potential, and no fully self-consistent published parameter set covers
the whole quartet. We build the seed force field by combining four
sources, with the optimization step cleaning up the seam between
them:

### 5.1 · CHO base (Chenoweth/van Duin/Goddard 2008)

The base file is the LAMMPS-bundled `ffield.reax.cho` — Chenoweth, K.;
van Duin, A. C. T.; Goddard, W. A. *J. Phys. Chem. A* **112**, 1040
(2008). It defines C, H, O atomic parameters; angles; torsions;
hydrogen-bond corrections. We **inherit** this entire file as the
starting point. C is dormant (no carbon in our training set); H caps
the cluster surfaces.

**Citation:** doi:10.1021/jp709896w. The exact `ffield.reax.cho` file
shipped with LAMMPS 22-Jul-2025 is the source.

### 5.2 · Atomic parameters (chi, eta, gamma, R, R_vdW)

Per-element atomic numbers come from physical / spectroscopic
constants:

- Mass: IUPAC 2021 atomic weights.
- Covalent radii (rₛ, rπ, rππ): Pyykkö & Atsumi, *Chem. Eur. J.* **15**, 12770 (2009).
- van der Waals radii: Bondi, *J. Phys. Chem.* **68**, 441 (1964).
- Mulliken electronegativity (χ_EEM): from atomic IP and EA in the
  CRC Handbook of Chemistry and Physics, 102nd ed.
- Chemical hardness (η_EEM): η = (IP − EA)/2 from the same source.

Resulting values used as the seed (see `make_starter_ffield.py` for
the exact numbers):

| Element | r_σ (Å) | mass (u) | R_vdW (Å) | γ | χ_EEM (eV) | η_EEM (eV) |
|---|---|---|---|---|---|---|
| Ge | 1.20 | 72.640 | 2.11 | 0.62 | 4.30 | 4.50 |
| Sb | 1.40 | 121.760 | 2.06 | 0.45 | 4.00 | 4.50 |
| Te | 1.30 | 127.600 | 2.06 | 0.45 | 5.50 | 4.50 |

### 5.3 · Bond / off-diagonal seed values

Bond dissociation energies are seeded from gas-phase diatomic data
(Huber & Herzberg 1979; Luo, *Comprehensive Handbook of Chemical Bond
Energies*, 2007). Off-diagonal Lennard-Jones / Morse parameters are
estimated by Lorentz-Berthelot combination rules from elemental
self-pair values, where missing.

| Bond | D_e (kcal/mol) | r_e (Å) | Source |
|---|---|---|---|
| Ge–Te | 80 | 2.85 | GeTe(g) D₀ = 91 kcal/mol from Pauling-style estimate; r_e from rocksalt |
| Sb–Te | 70 | 2.96 | SbTe gas-phase ~58 kcal/mol; bulk r_e from Sb₂Te₃ |
| Te–Te | 50 | 2.84 | Te₂(g) D₀ = 64 kcal/mol; r_e from Te₈ ring |
| Ge–Ge | 65 | 2.45 | Ge₂(g) D₀ = 64 kcal/mol; r_e diamond cubic Ge |
| Sb–Sb | 70 | 2.90 | Sb₂(g) D₀ = 71 kcal/mol; r_e from rhombohedral Sb |

**These values are deliberately approximate — the optimization fits
them.** The seed only needs to be in the right ballpark for SA / CMA
to start moving in a useful direction.

### 5.4 · Angle parameters

Angles use the standard ReaxFF form. Equilibrium angles seeded from
the structural geometry of the corresponding crystal:

| Angle | θ₀ | Notes |
|---|---|---|
| Te–Ge–Te (octahedral Ge) | 90° | Rocksalt Te–Ge–Te is 90° |
| Te–Ge–Te (tetrahedral Ge) | 109.5° | Tetrahedral defects in amorphous |
| Ge–Te–Ge | 90° | Rocksalt geometry |
| Te–Sb–Te | 90° | Rocksalt geometry |
| Sb–Te–Sb | 90° | Rocksalt geometry |
| Te–Ge–Sb / Sb–Ge–Te | 90° | Mixed cation environment |

p(val1) and p(val2) (force constants) are seeded at 30 / 1.0 (the
typical CHO-family scale) and left for the optimization to refine.

### 5.5 · Trainable subset (~30 parameters)

Listed in `params_GST`. The selection covers the chemistry questions
in §3 directly:

- **Bond De,σ** for Ge–Te, Sb–Te, Te–Te, Ge–Ge, Sb–Sb
  (5 params; bond-energy magnitudes — drives the homopolar/heteropolar
  energy gap).
- **Bond p(be1)** for the same 5 pairs (5 params; bond-curvature shape).
- **Off-diagonal Dij and R_vdW** for Ge–Te, Sb–Te, Te–Te (6 params;
  the long-range tail).
- **Atomic χ_EEM and η_EEM** for Ge, Sb, Te (6 params; QEq partial
  charges, which set the polarisation of Ge–Te bonds).
- **Angle θ₀ and p(val1)** for Te–Ge–Te, Te–Sb–Te, Ge–Te–Ge (6
  params; controls the rocksalt-vs-tetrahedral preference of Ge).
- **Atomic R_vdW** for Ge, Sb, Te (3 params; bounds [1.85, 2.40] Å,
  current values Ge=2.11, Sb=2.06, Te=2.06; covers the Bondi/Pyykkö
  physical range for these elements). Added 2026-05-08 after CMA gen
  ~200 plateaued at residuals ~1.5–4 × 10⁶ on the compressive GST
  strain points (FF=161 vs target=2177 kcal/mol at ε=−0.06): bulk
  modulus is dominated by short-range repulsion, which comes from
  the per-atom R_vdW (item 4 on the atom row), and that knob wasn't
  in the original 28-param subset. The 6 off-diagonal entries
  (4 4 2 / 4 5 2 / 4 6 2) only set the *cross-pair* R_vdW; the
  diagonal R_vdW per element is an independent degree of freedom.

= **31 trainable parameters** (28 original + 3 atomic R_vdW). The
remaining ~250 parameters in the ffield are frozen at their seed
values.

## 5.6 · Mixed backend across the training set (QE for PBC, PySCF for clusters)

The PBC and cluster training pieces use **different QM backends and
functionals** by design:

| Mode | Backend | Functional / Basis | Why |
|---|---|---|---|
| Cluster (Peierls, tetrahedral, homopolar) | **PySCF** | **B3LYP / def2-svp** (with `def2-ecp` for heavy elements) | Molecular Gaussian DFT is exactly what PySCF was built for. Hybrid is required for the Peierls splitting (pure DFT underestimates the asymmetric ground state by ~30 %; Caravati 2009). |
| PBC (rocksalt + strain scans) | **Quantum ESPRESSO** | **PBE / SSSP UPF pseudopotentials** | Plane-wave SCF + analytic forces + ASE BFGS handle production cell sizes cleanly. PySCF's PBC nuclear-gradient implementation stalls inside `geometric_solver` for our 18-atom cells regardless of multigrid / DIIS settings. PBE is the standard choice for bulk GST AIMD anyway (Sosso 2012). |

Mixing backends is **standard ReaxFF parameterization practice** —
the canonical Sosso 2011 / Caravati 2009 GST AIMD work used
plane-wave bulk + Gaussian molecular cluster from different codes,
combined into one FF training set.

**Implementation.** The bulk structure (`GST_rocksalt`) carries a
`qm_code: qe` per-structure override that flips PyField's backend
dispatch; the global `qm.code: pyscf` stays as the default for
clusters. The QM cache key folds the per-structure backend code
into the hash so PySCF and QE results on the same atoms can't
collide.

PyField's `_BackendCache` lazily builds one backend instance per
unique code seen across the structures, so a single `populate_qm`
call can drive both backends without double-init cost.

If you need to unify on a single backend (e.g. for cleaner
methodology):

- **All-PySCF** would require waiting for PySCF's PBC hybrid-gradient
  path to mature (currently fragile).
- **All-QE** is doable today: set the global `qm.code: qe` and add
  `pseudopotentials` for H. The clusters get larger PBC cells with
  vacuum padding (~10–15 Å) to mimic the cluster picture; QE handles
  this fine but costs more per single-point than the molecular path.

## 6 · Charge handling in ReaxFF

ReaxFF computes per-atom partial charges **automatically** at every MD
step via QEq (charge equilibration; Rappé & Goddard, *J. Phys. Chem.*
**95**, 3358 (1991)). The user does not specify charges in the YAML;
LAMMPS' `fix qeq/reaxff` does it during each evaluation, given each
atom's electronegativity (χ_EEM) and hardness (η_EEM) from the
ffield. Total charge is conserved at zero for a neutral cell.

For training, the QEq parameters are tuned indirectly by fitting to QM
energies — if the FF gets the energy of the polar Ge–Te bond right,
the implied partial charges are already roughly correct. We do *not*
need explicit `kind: charges` targets matching QM Mulliken / Hirshfeld
charges for this project, though the framework supports them if we
wanted to add them later for, e.g., a uranyl glass project where
explicit oxidation states matter.

## 7 · Optimization plan

CMA-ES with thousands of generations. The configuration in
`gst_drift.yaml` runs:

```yaml
optimizer:
  method: cma
  generations: 2000
  cma_sigma0: 0.3        # 30 % of median parameter span as initial step
  cma_popsize: 0         # use cma's default popsize (4 + floor(3 ln N))
  parallel: true
  processors: 8
  seed: 0
  show_progress: true
```

Default `cma_popsize` for N = 28 trainable parameters is `4 + ⌊3 ·
ln(28)⌋ = 13`. With 8 worker processes, each generation evaluates 13
candidates in two batches (8 + 5). Wall-clock per generation: ~25–45 s
(each evaluation runs all 68 LAMMPS sims; 19 of those are constrained
`minimize` runs which dominate). 2000 generations ≈ 14–25 h of
compute, depending on system. Cache hits are obvious in the bar.

**Stopping criteria:** `cma` will stop early via its built-in
convergence test (`tolfun`, `tolx`); 2000 is a hard cap.

## 8 · Validation and analysis

Once the FF is fitted, validate against:

1. **Held-out scan points** (not used in training): a subset of strain
   and bond-stretch points are reserved. Their FF energies should
   predict the held-out QM energies within ~5 kcal/mol.
2. **Crystalline rocksalt MD at 300 K** (ReaxFF run, separate from
   PyField): a 200-atom NVT trajectory should preserve the rocksalt
   structure for at least 100 ps (no spurious melting, no Te–Te bond
   formation).
3. **Density of amorphous melt-quench at 300 K**: target 6.27 g/cm³
   (Wełnic 2007). Expect ±5 % from the FF.
4. **Coordination number of Ge**: target 5.5 (mixed octahedral and
   tetrahedral) from EXAFS (Kolobov 2007). Expect ±0.3 from the FF.

Failure modes (how we'd know we're stuck):

- **Crystalline rocksalt melts spuriously** → bond De,σ is too low;
  re-run optimization with larger weight on the rocksalt cohesion
  target.
- **Ge always 6-fold or always 4-fold** → Peierls double-well missing
  from the FF; re-train with Cluster 4.2 weight bumped 5×.
- **CMA stalls** at high cost → seed FF is in the wrong basin; restart
  from random or different seed.

## 9 · Project structure

```
studies/gst_drift/
├── EXPERIMENT.md                  # this file
├── gst_drift.yaml                 # PyField config (structures + scans + optimizer)
├── make_starter_ffield.py         # programmatic ffield builder (see §5)
├── ffield.reax.GST                # the seed ffield (generated by the script)
├── params_GST                     # trainable parameter selection (§5.5)
├── gst_drift_walkthrough.ipynb    # the run-the-pipeline notebook
└── runs/
    └── qm_cache/                  # content-keyed QM cache (PySCF + QE entries)
```

**External requirements** (not in this folder):

- **Quantum ESPRESSO**: `pw.x` on `PATH`. See README §"Setting up
  Quantum ESPRESSO" for install instructions (`apt`, `conda`, or
  source).
- **SSSP pseudopotentials**: ~70 UPF files extracted to
  `~/qe_pseudos/` (or any directory pointed at by `qm.pseudo_dir` in
  the YAML). README has the exact `curl` command for the SSSP 1.1.2
  PBE Efficiency tarball (~37 MB compressed, ~100 MB extracted). The
  three filenames the YAML expects:
  - `ge_pbe_v1.4.uspp.F.UPF`
  - `sb_pbe_v1.4.uspp.F.UPF`
  - `Te_pbe_v1.uspp.F.UPF`
- **MPI for QE** (strongly recommended): set
  `ESPRESSO_COMMAND='mpirun -np N pw.x ...'` before running the
  notebook. Single-core `pw.x` makes the bulk training take ~30 hours
  cold-cache; 8-core MPI cuts that to ~6 hours.

## 10 · Status / progress log

| Date | Milestone |
|------|-----------|
| 2026-05-03 | Project scaffolded. Force field seeded from CHO + literature. Training YAML drafted. QM not yet run. |
| 2026-05-03 | First QE relax attempted on `GST_rocksalt` — caught a per-structure-functional override bug (B3LYP was used instead of PBE) and added a `_effective_functional` helper to the QE backend. |
| 2026-05-04 | QE backend wired in PyField with `qm_code: qe` per-structure override. SSSP 1.1.2 PBE Efficiency pseudopotentials installed at `/home/ubuntu/qe_pseudos`. PBE / Γ-only / `ecutwfc=40` Ry single-point on the 18-atom rocksalt cell: 106 s. Strain scans flipped back to `relaxed_constrained`. |
| 2026-05-04 | Cluster-relax debug pass. PySCF DFT on heavy-element clusters needs four fixes the backend now bakes in: `ecp=basis` for Te/Sb/Ge gradients, `mf.density_fit()` to make hybrid DFT tractable, `mf.max_cycle = 200` to ride out SCF instabilities in geom-opt loops, and bent / off-axis starting geometries to break orbital-degenerate symmetries. Te₂_wrong, Ge₂_dumbbell, Sb₂_dumbbell now relax cleanly in 1–4 min each. **GeTe₆_Peierls (37 atoms) and GeTe₄_tet (9 atoms but perfect Td) still defeat PySCF — deferred to v2 pending an ORCA backend or closed-shell molecular cluster substitute.** v1 of the FF therefore covers bulk strain + homopolar bond costs only; Peierls drift physics is partially captured indirectly through the bulk strain response. |
| 2026-05-08 | **Optimizer wedged: `cg` minimizer hangs on PBC GST cell with seed FF**. Symptom: `cost_breakdown(populated)` and the SA/CMA driver both stuck on the very first simulation (`GST_bi_xy_0_sp` — the unstrained GST rocksalt minimize) for 24+ hours of pegged-CPU wall-clock with **zero log output past the LAMMPS pair-init line**. The kernel was alive (99% CPU) but stuck inside `lmp.file()` → `Min::setup()`, which is *before* the iteration loop, so neither LAMMPS-internal iter caps nor `timer timeout` fired — both only check inside the iteration loop. **Diagnosis (took several rounds)**: (1) Confirmed via `py-spy dump` that the kernel was inside `lammps/core.py:815`. (2) Verified the seed FF can compute single-point energies fine on Te₂ / GeTe / Sb₂ / GeSbTe / GST_rocksalt-PBC (all <100 ms, plausible energies 47–149 kcal/mol) — so the FF parameters themselves are not the problem. (3) Verified `cg` minimize works on cluster cells (Te₂, GeTe converge in 7 iters, <1 s). (4) Tested all four `min_style` options on the PBC GST cell with subprocess-level timeouts (SIGALRM doesn't interrupt LAMMPS' C-extension calls): **`cg` → hangs, `sd` → hangs, `hftn` → 0.26 s, `fire` → 4.59 s**. Switched the project default `min_style` from `cg` to `hftn` in `pyfield/simulations/minimize.py`. Also wired up tqdm + per-sim timing logs in `pyfield.diagnostics.cost_breakdown` (was previously silent), tightened the `minimize.in.j2` template caps from `2e5 / 2e6` to `2000 / 20000` iters/evals, and added `timer timeout 0:05:00` and `min_modify dmax 0.05` (per-step motion cap). The Cl₂ smoke test target was bumped from `32907.21505210572` to `32907.21744068185` (0.002 kcal/mol drift from the dmax cap; same converged geometry, different per-step kinematics). |
| 2026-05-08 | **CMA plateau on compressive GST → expanded `params_GST` with atomic R_vdW**. After the `ff_relax_method: rigid` switch (entry above), CMA-ES kicked off cleanly and dropped total cost from `9.91 × 10⁶` to `1.96 × 10⁶` by gen 200 (~5× drop). Top residuals stayed dominated by the compressive GST_hydro / GST_bi_xy / GST_uni_z scan points (FF underestimates strain energy by ~6× — `FF=161 vs target=2177` at ε=−0.06). Reasoning: compressive bulk modulus is set by short-range vdW repulsion, which depends most directly on **atomic R_vdW** (item 4 on the atom row). The original 28-param subset trained the *off-diagonal* (cross-pair) R_vdW (entries `4 4 2`, `4 5 2`, `4 6 2`) but not the *diagonal* per-element R_vdW. Added 3 entries to `params_GST`: `2 4 4`, `2 5 4`, `2 6 4` (Ge / Sb / Te), bounds `[1.85, 2.40]` Å — covers Bondi/Pyykkö physical range and includes the seed values (Ge=2.11, Sb=2.06, Te=2.06). **31 trainable params total now**; §5.5 updated. To restart CMA from the current best FF (carries forward most of the 5× gain already made), copy `runs/sa/bestFF.reax` over `studies/gst_drift/ffield.reax.GST` (or point `forcefield.path` at it) before re-running — `pyfield.optimizers.cma` reads `x0` from `ParameterSnapshot.capture(ff)` at startup, so whatever FF is on disk becomes the warm-start seed. |
| 2026-05-09 | **CMA plateau diagnosed: bulk strain ladder was monotone, not parabolic — wrong reference cell**. After 1000 CMA generations on the expanded 31-param subset, total cost stalled at `~1.5 × 10⁶` (down from `9.91 × 10⁶` at gen 0, `1.96 × 10⁶` at gen 200). Inspection of the GST_hydro target ladder showed it was a *slope*, not a *bowl*: targets descended monotonically `+2177, +1327, +608, 0, −514, −948, −1316` kcal/mol from ε=−0.06 to ε=+0.06. A physical E(V) curve at equilibrium is parabolic (compression *and* expansion both cost). A monotone ladder means the entire scan window sat to one side of the QE equilibrium volume — the reference cell `box: [6.02, 6.02, 6.02]` (literature lattice from Yamada 1991) is not where PBE actually wants to place GST rocksalt; the seven scan points sample only the descending wall of the parabola. **ReaxFF's E(V) is parabolic around its own minimum no matter how the parameters are tuned — it physically cannot fit an asymmetric ladder.** CMA's best response was slope-matching the midpoint, eating ±1000 kcal residuals at the endpoints — that *is* the 1.5 × 10⁶ floor. **Root cause in the QM backend**: `pyfield/qm/qe_backend.py:relax` ran ASE's plain `BFGS` (atoms only, cell pinned at `box:`) for every `qm_relax: true` structure. QE's stress tensor screamed but the optimizer wasn't listening. **Fix**: added `StructureCfg.qm_relax_cell: bool = False` to the schema (`pyfield/config/schema.py`); when true on a `pbc: true` structure, the QE backend switches to QE's native `calculation: vc-relax` (atoms + cell vectors) with `cell_factor: 2.0` (pre-allocate the plane-wave basis to tolerate cell expansion without Pulay-stress drift), `&IONS{ion_dynamics: bfgs}`, `&CELL{cell_dynamics: bfgs, press_conv_thr: 0.5}`. Added `_relax_vc` — reads the final image of `espresso.pwo` via `ase.io.read(..., index=-1)` because ASE's Espresso calculator doesn't auto-update the atoms object after a vc-relax. Returns a new `StructureCfg` with both `box:` and `atoms:` updated; non-orthorhombic relax results raise (the `box: [a, b, c]` schema doesn't carry shear). Cache-key separation: `pyfield/qm/prep.py:_relax_op` distinguishes `relax` / `vc-relax` / `relax_constrained` so flipping the flag doesn't alias old atoms-only entries. **YAML change**: `gst_drift.yaml` now sets `qm_relax_cell: true` on `GST_rocksalt` only (cluster references stay `pbc: false` and the validator rejects vc-relax there — vacuum-padding boxes would collapse). Strain scans don't change — they re-anchor automatically once the reference's box updates in `populated.yaml`. **Why QE's native vc-relax instead of ASE's `ExpCellFilter`**: ASE's filter trick keeps the basis pinned at the initial cell, so it goes stale as the cell expands and Pulay stress biases the result toward the wrong volume. QE's `cell_factor` solves it cleanly. **Expected effect on cost**: target magnitudes for the bulk strain set drop ~10× (from |target| ~ 2000 → ~200 kcal/mol once the ladder becomes parabolic centred on `V_eq`), residuals² drop ~100×, CMA cost floor drops by ~2 orders of magnitude. **Action required**: re-run `pyfield qm-prep` (cache invalidates automatically — the QM cache is content-keyed on `box`, so old atoms-only entries don't shadow the new vc-relax run). Tests: 174 still pass + 4 new (schema validators for `qm_relax_cell + pbc`, auto-implication of `qm_relax`, default-false; QE-backend `_build_input_data(relax_cell=True)` keyword test; relax dispatch test that vc path is skipped when a constraint is present). Doc updates in DEV.md §9 (change log) and §10 (this entry). |
| 2026-05-10 | **⚠️ IMPORTANT — seed cell `[6.02, 6.02, 6.02]` was unphysically compressed; vc-relax expanded it ~28 % linearly to ~7.65 Å mean** (full diagnostic moved up to §4.1 as a prominent callout — readers shouldn't have to dig through the changelog to learn that the central reference moved this much). The literature 6.02 Å refers to the conventional 8-atom rocksalt unit cell; our 18-atom training cell packs roughly twice the atom count into the same volume (~12 Å³/atom vs the real GST ~25 Å³/atom), so QE wasn't over-expanding — it was relieving compression. Confirmed by inspecting the preserved `/tmp/qe_vcrelax_failed_qe_vcrelax_ncyvq0bo/espresso.pwi`: the run started from `(6.02, 6.02, 6.02)` cubic with `Te (0, 0, 0)` as atom 1 — those are `GST_rocksalt`'s seed values verbatim, *not* a strain-scan point that misfired via the `expand_scans` `qm_relax_cell`-propagation bug (which was a real worry given the bug's existence at the time — see the adjacent entry). The salvaged cache entry under `runs/qm_cache/6732606334e8d965/` is therefore genuine `GST_rocksalt` data, and the on-disk `scanned.yaml` strain ladder anchored on `(7.61, 7.75, 7.69) Å` is correct. Forward-looking note for v2: re-seed the structure at `~[7.65, 7.65, 7.65] Å` so the vc-relax has less work to do (shorter wall-clock, cleaner ionic-relax history, no shear projection). Current cached result fine for v1 training. |
| 2026-05-10 | **Real root cause of the cell-[7] long QE runs: `qm_relax_cell` propagated through `expand_scans`**. After kernel restart, the qm-prep cell (code-cell `[8]` = `populate_qm`, which the user counts as cell 7 since they skip the diagnostic-only cells) was still triggering 13-hour vc-relaxes. Tracing through: `pyfield/scans/__init__.py:expand_scans` stripped `qm_relax` from inherited scan points but never `qm_relax_cell`. So if the relaxed `GST_rocksalt` still carried `qm_relax_cell: true` (older cache entries from before `_merge_relax_with_input` stripped the flag, or any code path that passed the raw cfg into `expand_scans`), every one of the 19 strain scan points inherited it. `populate_qm` then walked them, saw no internal-coord constraint (strain scans use cell-as-constraint), and dispatched each to `_relax_vc` — i.e. 19 × variable-cell relaxes queued up. The single 13h run the user reported was the first of those, not the `GST_rocksalt` reference relax (which had long since been a cache hit). Fix: `expand_scans` now unconditionally strips both `qm_relax` and `qm_relax_cell` from scan points before re-attaching `qm_relax: true` for the constrained-relax variants. Tested via a new regression in `tests/test_constrained_scan.py::test_strain_scan_strips_qm_relax_cell_from_scan_points`. The current on-disk `scanned.yaml` already has `qm_relax_cell: false` on every scan point (the latest run happened to start from a relaxed reference whose flag was stripped), so no re-make-scan needed in the user's current state. |
| 2026-05-10 | **Rescued a 13-hour vc-relax that failed under stale kernel state**. Re-running the qm-prep cell after the YAML setting bumps (kpts=[2,2,2], ecutwfc=50, conv_thr=1e-9) without restarting the jupyter kernel meant: (1) `ESPRESSO_COMMAND` from the new `~/.bashrc` wasn't visible to the kernel — QE ran single-core (`Parallel version (MPI), running on 1 processors`), turning a ~3h MPI run into a 13h serial run; (2) the kernel still held the *old* `qe_backend.py` in memory, so `nstep` was the QE-default 50 (not our 200) and the `JOB DONE`-tolerance exception handler wasn't applied — QE hit `STOP 3` legitimately at step 50 with a valid final image, but the kernel raised through it. (3) `cell_dofree` was also the old `'all'` (not `'xyz'`), producing a 3 % sheared cell at `(7.61, 7.75, 7.69)` Å — close enough to orthorhombic that the diagonal is a faithful summary, but the strict `atol=1e-4` check refused to project. **Salvage**: loosened the orthorhombicity guard in `pyfield/qm/qe_backend.py:_relax_vc` and `scripts/rescue_qe_vcrelax.py` to tolerate up to 5 % shear (project to diagonal with `RuntimeWarning`), then imported the preserved `espresso.pwo` directly into the QM cache via the rescue script — no recompute. **Lesson**: kernel restart isn't optional after touching the QE backend or `~/.bashrc`; the YAML reload happens at cell run-time but Python imports and shell-env snapshotting do not. The next strain-scan run will hit the cached vc-relax for `GST_rocksalt` and only pay for the 19 strain-point single-points (~10 hours on 8 ranks at the bumped settings; was ~30 hours single-core without the MPI fix). **Structure insight**: the relaxed volume is `~452 Å³ / 18 atoms ≈ 25 Å³/atom` — exactly the GST norm. The 6.02 Å seed was at `12 Å³/atom`, double-compressed; PBE expanded the cell ~28 % linearly to relieve this. The seed cell was wrong, not the QE settings. |
| 2026-05-09 | **QM accuracy review — bumped k-points / cutoff / SCF tolerance to production values**. Audit of QM settings prompted by "are we converged?" question, pre-empting another long re-run with under-converged inputs. Findings:<br><br>**Functionals are fine.** PBE (GGA) for bulk QE; B3LYP (hybrid) for cluster PySCF — both appropriate. PBE typically over-expands lattice constants 1–3 % and underestimates B by 5–15 % for chalcogenides, but reproduces E(V) shape correctly, which is what ReaxFF training needs. Hybrid B3LYP is required for the cluster Peierls splitting per §4.2. PBEsol would be a free upgrade (often ~1 % closer lattice constant for solids); deferred — current PBE+SSSP is well-validated.<br><br>**k-points were the real weakness.** Γ-only on an 18-atom cell with ~1 Å⁻¹ Brillouin zone gives ~5–15 % errors in the stress tensor and ~10–25 % in elastic constants — directly the targets the strain scans hit. Also explains why the first vc-relax bounced past `press_conv_thr` and hit `nstep`: the stress signal was below the noise floor.<br><br>**Plane-wave cutoff borderline.** `ecutwfc: 40` Ry is the SSSP-Efficiency minimum — fine for energies, but leaves residual Pulay-like stress of ~1–10 kbar even with `cell_factor: 2.0`. 50 Ry takes that to <1 kbar at ~1.4× SCF cost.<br><br>**SCF threshold was loose for stress.** `conv_thr: 1e-7` is fine for SCF energies; `1e-9` is the right setting for accurate forces/stress and adds <10 % SCF cost. Also folded `conv_thr` into `QEBackend.settings_fingerprint` — without that, tightening the threshold would silently hit cached looser results.<br><br>Cluster basis (def2-svp) left as-is for now — moderate impact on Te₂/Ge₂/Sb₂ stretch targets but not load-bearing for the bulk-strain plateau we're currently fighting. Revisit once bulk fit is in shape.<br><br>**Settings table — current state of the QM stack** (post-bump):<br><br><table><tr><th>Knob</th><th>Old (demo)</th><th>New (production)</th><th>Cost factor</th><th>Why</th></tr><tr><td><code>kpts</code> (bulk)</td><td><code>[1,1,1]</code></td><td><code>[2,2,2]</code></td><td>~8×</td><td>Cuts stress / elastic-constant errors 3–5×; biggest single accuracy lever for the strain scans</td></tr><tr><td><code>ecutwfc</code> (bulk)</td><td>40 Ry</td><td>50 Ry</td><td>~1.4×</td><td>Drops Pulay stress floor below `press_conv_thr`; vc-relax converges instead of wandering</td></tr><tr><td><code>ecutrho</code> (bulk)</td><td>320 Ry</td><td>400 Ry</td><td>(included in ecutwfc cost)</td><td>Maintains 8:1 SSSP-recommended ratio</td></tr><tr><td><code>conv_thr</code> (bulk)</td><td>1e-7</td><td>1e-9</td><td>~1.1×</td><td>Stress noise from ~1 kbar → ~0.1 kbar; nearly free</td></tr><tr><td><code>functional</code> (bulk)</td><td>PBE</td><td>PBE</td><td>—</td><td>Already appropriate for ReaxFF training trends</td></tr><tr><td><code>functional</code> (cluster)</td><td>B3LYP</td><td>B3LYP</td><td>—</td><td>Hybrid required for Peierls splitting</td></tr><tr><td><code>basis</code> (cluster)</td><td>def2-svp</td><td>def2-svp</td><td>—</td><td>Holds for now; def2-tzvp is a candidate upgrade (~3× cost, drops bond-energy error 5–10 → 1–2 kcal/mol). Revisit if cluster targets dominate residuals after bulk fit converges.</td></tr></table><br><br>Total cost factor on the bulk QE stack ≈ 8 × 1.4 × 1.1 ≈ **~12×** wall-clock per QE run, or ~3 hours on 8 MPI ranks for a fresh `GST_rocksalt` vc-relax. Cache fingerprint changes invalidate all bulk QM entries automatically (the schema fingerprint includes ecutwfc / ecutrho / kpts / conv_thr); cluster entries unaffected. |
| 2026-05-09 | **vc-relax follow-up: QE 6.4.1 gotchas**. The first vc-relax run on `GST_rocksalt` failed twice on the way to a working configuration: (1) `cell_factor` was placed in `&SYSTEM` (where some older QE docs put it) — QE 6.4.1 rejects it there with "bad line in namelist", moved to `&CELL` (one-line edit, instant rerun). (2) With `cell_dofree: 'all'` the vc-relax produced a sheared / non-orthorhombic cell `[[7.83, -0.09, -0.23], [-0.15, 7.72, 0.25], [-0.48, 0.66, 5.47]]` after 50 ionic steps — likely the disordered cation sublattice (5 Ge + 5 Sb on 12 sites with 2 vacancies) wants a Peierls-like distortion, which our `box: [a, b, c]` schema can't carry. Switched to `cell_dofree: 'xyz'` (independent `a, b, c`, no shear) — fewer cell DOFs to optimize and produces an orthorhombic answer that fits the schema. Also bumped `nstep: 200` (default 50) so the relax has room to finish without `STOP 3`. Even when the soft-fail does happen, `_relax_vc` now tolerates exit codes >0 if QE wrote "JOB DONE" — uses the last step's geometry with a `RuntimeWarning` rather than discarding ~2h of compute. The previous 1h56m attempt (sheared output preserved at `/tmp/qe_vcrelax_failed_qe_vcrelax_gdo13q8p/`) is unsalvageable for the orthorhombic schema, so the next run starts from scratch — but with `cell_dofree: 'xyz'` it should converge in materially fewer steps. |
| 2026-05-08 | **`hftn` ran but produced NaN atoms — strain scans gave identical E**. After fixing the hang, `cost_breakdown` printed for all 39 sims but every PBC GST minimize returned the same `E = -53.060 kcal/mol` regardless of strain mode/value. Inspection of the dump files showed every atom coord = `nan`: the seed FF's first minimize step (even with `dmax 0.05`) produces forces that collapse the cell into an unphysical configuration; LAMMPS reports the last finite energy and the atoms NaN out silently. Independently, the `Ge₂_d_*` long-bond stretches gave runaway energies (`Ge2_d_4 → −1.9 × 10⁹`, then `−1.2 × 10¹²` after dmax) — the seed Ge–Ge bond parameters have unphysical attraction at long separation. Total cost diverged to `3.7 × 10¹⁸`. **Decision: rather than hand-fit the seed FF (would take days and require GST ReaxFF expertise we don't have on hand), introduce a per-side `ff_relax_method` override on `ScanCfg`** so the QM side keeps its `relaxed_constrained` regimen (preserves the cached QM-relaxed geometry per scan point) while the FF side is evaluated as a `single_point` at that QM geometry. This answers the right question for early-fit CMA: "what does the FF *say* the energy is at the same geometry QM relaxed to?". Once CMA produces an FF physical enough to relax the cells without exploding, drop the override to evaluate both sides consistently. **Implementation**: (1) Schema change — added `ff_relax_method: Optional[RelaxMethod] = None` to `ScanCfg` (`pyfield/config/schema.py:233`), with `None` meaning "inherit from `relax_method`". (2) Engine change — `pyfield/scans/__init__.py:248` now uses `scan.ff_relax_method or scan.relax_method` to decide the FF sim type. The QM dispatch path (`pyfield.qm.prep`) ignores the new field. (3) Source change — `studies/gst_drift/gst_drift.yaml` now sets `ff_relax_method: rigid` on all 7 scan blocks, alongside the existing `relax_method: relaxed_constrained`. Regenerated `gst_drift.scanned.yaml` and `gst_drift.populated.yaml`; QM cache hits make this fast. **Result on `cost_breakdown`**: total cost dropped from `3.7 × 10¹⁸` to ~`9.9 × 10⁶`; GST strain energies span `54.1 → 310.7 kcal/mol` (physical, monotonic with strain); Ge₂ stretches give `16.7 → −4.6` (still off vs QM target but finite, no blow-up); dominant residual is `GST_hydro_0_sp` (FF=161, target=2177) — exactly the "compressed cell is way too soft under the seed" gap CMA needs to close. **The seed `ffield.reax.GST` itself is unchanged**; the underlying parameters are still the provenance documented in §5.1–5.4 below. The change is robust against re-running `qm-prep` because it lives in the source YAML, not as a one-shot patch on the populated artifact. |

Update this section as we go.

## 11 · References

Listed in order of appearance. Full citations:

1. Chenoweth, K.; van Duin, A. C. T.; Goddard, W. A. *J. Phys. Chem.
   A* **112**, 1040–1053 (2008). doi:10.1021/jp709896w. *(CHO ReaxFF base)*
2. Pyykkö, P.; Atsumi, M. *Chem. Eur. J.* **15**, 12770–12779 (2009).
   *(covalent radii)*
3. Bondi, A. *J. Phys. Chem.* **68**, 441–451 (1964). *(vdW radii)*
4. Huber, K. P.; Herzberg, G. *Constants of Diatomic Molecules*, Van
   Nostrand-Reinhold (1979). *(diatomic D_e)*
5. Luo, Y.-R. *Comprehensive Handbook of Chemical Bond Energies*, CRC
   Press (2007). *(bond dissociation energies)*
6. Yamada, N. *Jpn. J. Appl. Phys.* **30**, 2606–2611 (1991).
   *(rocksalt GST lattice)*
7. Sun, Z. et al. *Phys. Rev. B* **76**, 245206 (2007). *(GST bulk
   modulus)*
8. Lencer, D. et al. *Adv. Mater.* **25**, 6710 (2013). *(Peierls
   distortion in GST)*
9. Caravati, S. et al. *J. Phys.: Condens. Matter* **21**, 255501
   (2009). *(amorphous GST AIMD)*
10. Kolobov, A. V.; Fons, P. *Nat. Mater.* **3**, 703–708 (2004); and
    Kolobov, A. V. et al. *Phys. Rev. B* **76**, 224107 (2007).
    *(EXAFS bond populations)*
11. Sebastian, A. et al. *Adv. Mater.* **30**, 1801268 (2018).
    *(synchrotron diffuse scattering)*
12. Boniardi, M. et al. *J. Appl. Phys.* **110**, 124313 (2011).
    *(drift coefficient ν ≈ 0.1)*
13. Salinga, M. et al. *Nat. Commun.* **9**, 2900 (2018). *(positron
    annihilation drift study)*
14. Sosso, G. C. et al. *J. Chem. Phys.* **135**, 014506 (2011).
    *(GST AIMD reference)*
15. Spreadborough, B. et al. *J. Mater. Res.* **28**, 1857 (2013).
    *(GST ReaxFF — partial fit, our literature comparison point)*
16. Wełnic, W. et al. *Nat. Mater.* **6**, 122 (2007). *(amorphous GST
    density / structure)*
17. Rappé, A. K.; Goddard, W. A. *J. Phys. Chem.* **95**, 3358–3363
    (1991). *(QEq method)*
