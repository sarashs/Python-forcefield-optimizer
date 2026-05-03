# PyField

PyField is a Python tool for refitting classical molecular-dynamics force
fields against quantum-chemistry reference data. It optimises ReaxFF
parameters by repeatedly running LAMMPS simulations and minimising a
weighted sum of objectives (energies, charges, structural agreement,
trajectory observables) defined in a single YAML config.

Supported today:

- **Force fields**: ReaxFF.
- **Optimisers**: simulated annealing (`sa`), genetic algorithm (`ga`),
  hybrid (`sa+ga`).
- **Simulation backends**: `minimize`, `nvt`, `npt`, `single_point`,
  plus a user-supplied Jinja template escape hatch.
- **Objectives**: energy combinations, per-atom QEq charges,
  coordination number, structural matching (Kabsch RMSD / bond
  lengths / bond angles), RDF peak position, force matching,
  melting-onset temperature (multi-NVT), bulk modulus (multi-NPT).
- **Platforms**: Linux x86_64/aarch64, macOS, WSL2.

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Examples](#examples)
- [Features](#features)
- [Citation](#citation)
- [License](#license)

## Architecture

PyField is a `pyfield/` Python package. The pieces:

- **`pyfield.forcefield`** — parser/writer for force-field files.
  Currently ReaxFF (`REAX_FF`); the `ForceField` base is shaped for
  COMB / Tersoff / OPLS subclasses to slot in later.
- **`pyfield.config`** — pydantic schema, YAML loader, and a deprecation
  shim that converts the pre-Phase-1 text formats into the new schema.
- **`pyfield.simulations`** — Jinja-templated simulation backends.
  Ships `minimize`, `nvt`, `npt`, and `single_point`, plus a
  user-template escape hatch (`template:`) for non-standard runs
  (validated against the variable-leakage contract — the user template
  must reference `{{ FFIELD_PATH }}` exactly once and let the runner
  inject everything the optimiser tunes).
- **`pyfield.objectives`** — registry of cost-function pieces. Ships
  `energy_combination`, `charges`, `coordination`, `structural_match`
  (Kabsch RMSD / `bond_lengths` / `angles`), `rdf_peak`, `forces`,
  `melting_onset`, and `eos`. Adding a new objective is one new file
  under this directory; the schema is untouched.
- **`pyfield.optimizers`** — drivers that consume the objective
  registry. `sa` (simulated annealing), `ga` (genetic algorithm with
  tournament selection / single-point crossover / Gaussian mutation /
  elitism), and `sa+ga` (GA with per-child SA refinement).
- **`pyfield.io`** — structure-file writer, the long-lived
  `LammpsRunner` (one LAMMPS instance reused across simulations with
  `clear` between), and a streaming `read_dump` + xyz reader for
  trajectory objectives.
- **`pyfield.qm`** — the `pyfield qm-prep` and `pyfield qm-relax`
  subcommands and their pluggable QM backends. `single_point` +
  `relax` cover every registered objective. PySCF backend ships
  today; xTB / QE / GPAW / ORCA / CP2K slot in behind the same
  interface (one new file each).
- **`pyfield.scans`** — the `pyfield make-scan` engine. Six geometric
  perturbations (bond stretch, angle bend, dihedral, atom
  displacement, dimer separation, isotropic scale); each scan kind is
  one small function under `pyfield/scans/transforms.py`.
- **`pyfield.viz`** — `animate_xyz_dir` for previewing scans in a
  notebook. Pure matplotlib (no extra deps).

Adding a new objective or simulation type is a new file under the
relevant subpackage — no schema or optimiser edits.

## Installation

Quickstart (Linux / macOS, Python ≥ 3.8):

```bash
python -m venv .venv && source .venv/bin/activate
pip install 'lammps[mpi]'
pip install -e .
pyfield run tests/cl2.yaml
```

This installs PyField as an editable package, registers the `pyfield`
CLI on your PATH, and runs the Cl₂ ReaxFF smoke optimisation against
`tests/cl2.yaml` — finishes in well under a second with a
`FINAL cost: 32907.21505210572` line (`seed: 0` is set in the YAML so
the run is bit-reproducible).

The pre-Phase-1 text-format inputs still work via a deprecation shim:

```bash
pyfield run-legacy tests/Trainingfile_2.txt tests/Inputstructurefile.txt \
  --ff tests/ffieldoriginal.txt --params tests/params --out tests/runs/legacy
```

### Notes on LAMMPS

The PyPI `lammps[mpi]` wheel bundles a manylinux MPICH library at
`<sys.prefix>/lib/libmpi.so.12`. glibc caches `LD_LIBRARY_PATH` at
process start, so PyField's CLI preloads that library with
`RTLD_GLOBAL` before importing `lammps` (see
`pyfield.io.lammps.preload_libmpi`) — you don't need to set any
environment variables yourself. If the wheel doesn't work for your
platform, fall back to conda-forge (`conda install -c conda-forge
lammps`) or build LAMMPS from source with `PKG_REAXFF=on PKG_QEQ=on
BUILD_SHARED_LIBS=on`.

## Examples

The bundled `tests/cl2.yaml` is a complete, runnable Cl₂ ReaxFF refit
that exercises all the major moving parts. Run it with `pyfield run
tests/cl2.yaml`. A guided walkthrough that loads the same YAML, runs
the optimiser, and plots the cost trace is in
[`examples/cl2_walkthrough.ipynb`](examples/cl2_walkthrough.ipynb).
The notebook is also a regression test — `pytest examples/` re-executes
every cell.

### Generating QM training data

The full pipeline starts from a rough geometry guess plus a perturbation
grid; everything else is filled in for you:

```bash
pip install -e .[qm]                                          # pyscf + geometric

pyfield qm-relax  tests/cl2_scan.yaml          -o cl2.relaxed.yaml
pyfield make-scan cl2.relaxed.yaml             -o cl2.scanned.yaml \
                                               --xyz-dir runs/scan_xyz
pyfield qm-prep   cl2.scanned.yaml             -o cl2.populated.yaml
pyfield run       cl2.populated.yaml
```

Step by step:

- **`qm-relax`** runs PySCF geom-opt for every structure flagged
  `qm_relax: true` and writes the relaxed coordinates back into the
  YAML's `atoms:` block. Skip it if you already know the equilibrium
  geometry.
- **`make-scan`** consumes a top-level `scans:` block and stamps out N
  perturbed structures + N matching `single_point` simulations + N
  `energy_combination` targets (`Scan_i − reference`, with
  `target: { from: dft }` waiting for `qm-prep`). Six scan kinds:
  `bond_stretch`, `angle_bend`, `dihedral`, `atom_displacement`,
  `dimer_separation`, `isotropic_scale`. Each generated structure also
  lands as an `.xyz` under `--xyz-dir` so you can inspect them in
  OVITO — or animate them inline (see below).
- **`qm-prep`** fills every `target: { from: dft }` slot via the
  configured QM backend. Hand-typed targets (empirical or experimental
  numbers) pass through untouched, so DFT-driven and hand-typed targets
  can coexist in one config.
- **`pyfield run`** does the SA / GA refit on the fully populated
  config.

Currently uses **PySCF** (pip-installable, no licence portal); xTB,
Quantum ESPRESSO, GPAW, ORCA, and CP2K backends slot in behind the same
interface (one new file each). Commercial codes — Gaussian, VASP,
Molpro — are intentionally not bundled.

The QM cache is content-keyed (`qm_cache/<sha256>/`), so re-running any
step after only FF-side edits is a no-op.

#### `scans:` schema

```yaml
scans:
  - { type: bond_stretch,      reference: Cl2_Opt,
      atoms: [1, 2], values: [1.6, 1.9, 2.2, 2.5, 3.0],
      name_prefix: Cl2_d }
  - { type: angle_bend,        reference: H2O_Opt,
      atoms: [1, 2, 3],     range: [80, 130, 11],
      name_prefix: H2O_a }                            # degrees
  - { type: dihedral,          reference: H2O2_Opt,
      atoms: [1, 2, 3, 4],  range: [-180, 180, 13],
      name_prefix: H2O2_t }
  - { type: atom_displacement, reference: Slab_Opt,
      atom: 5, direction: [0, 0, 1], range: [-0.5, 0.5, 11],
      name_prefix: Slab_z }
  - { type: dimer_separation,  reference: Dim_Opt,
      fragments: [[1, 2, 3], [4, 5, 6]], direction: auto,
      values: [2.5, 3.0, 3.5, 4.0, 5.0],   name_prefix: Dim_r }
  - { type: isotropic_scale,   reference: Cell_Opt,
      range: [0.95, 1.05, 11], name_prefix: Cell_s }   # multiplier on box+atoms
```

Pick exactly one of `values:` (explicit list) or `range: [start, stop,
num]` (linspace) per scan. Atom indices are 1-based.

#### Visualising scans inline

`pyfield.viz.animate_xyz_dir(path)` reads every `.xyz` in a directory,
builds a 3D-scatter `matplotlib` animation (atoms coloured by element),
and embeds the play/slider widget in the notebook:

```python
from pyfield.viz import animate_xyz_dir
animate_xyz_dir('runs/scan_xyz', pattern='Cl2_d_*.xyz', interval_ms=400)
```

Use it to confirm perturbations look the way you intended *before*
paying for QM. Pure matplotlib — no extra deps.

A minimal user-facing schema looks like:

```yaml
forcefield:
  path: ffield.reax
  type: reaxff
  params: params

structures:
  Cl2_Opt:
    box: [100, 100, 100]
    atoms:
      - { element: Cl, x: 0, y: 0, z: -1.028 }
      - { element: Cl, x: 0, y: 0, z:  1.028 }

simulations:
  Cl2_Opt_min:
    structure: Cl2_Opt
    type: minimize
    restraints: ["bond 1 2 2000 2000 2.056"]

  # An NVT trajectory feeding a coordination objective:
  SiOH4_300K:
    structure: SiOH4
    type: nvt
    temperature: 300
    steps: 200000
    sample_every: 100

  # Power-user escape hatch — any LAMMPS-native run:
  Custom:
    structure: SiOH4
    template: lammps_inputs/explosion.in.j2
    variables: { TEMP: 2500, SEED: 42 }

targets:
  - kind: energy_combination
    weight: 1.0
    terms: { Cl2_414_min: +1, Cl2_Opt_min: -1 }
    target: 81.394           # ΔE in kcal/mol

  - kind: coordination
    weight: 2.0
    simulation: SiOH4_300K
    central: Si
    neighbor: O
    cutoff: 2.5
    target: 4.0

  - kind: structural_match
    weight: 5.0
    simulation: SiOH4_min
    reference: structures/SiOH4_dft.xyz
    metric: rmsd

optimizer:
  method: sa                 # or ga, sa+ga
  T: 0.2
  T_min: 0.1
  alpha: 0.5
  max_iter: 2
  number_of_points: 1
  seed: 0

output:
  dir: runs/my_refit
```

## Features

- One YAML config drives everything (no separate Trainingfile.txt /
  Inputstructurefile.txt).
- Pydantic-validated schema with cross-reference checks (every
  simulation references a known structure; every target references
  known simulations).
- Plug-in objective registry — adding `coordination`, `rdf_peak`, etc.
  is one new file, no schema edit.
- Long-lived LAMMPS instance with `clear` between simulations
  (faster than spawning a fresh `lammps()` per call).
- Bit-reproducible runs via `optimizer.seed`.
- Single-MD-feeds-many-objectives via simulation deduplication in the
  cost-evaluation loop.
- Jinja escape hatch for power users with non-standard LAMMPS recipes,
  validated against a variable-leakage contract.

## Citation

A peer-reviewed publication is in preparation. Until then, please cite
this repository:

```
PyField: a Python force-field optimisation tool.
https://github.com/sarashs/Python-forcefield-optimizer
```

## License

GPL-2.0-or-later (matching the bundled LAMMPS dependencies).
