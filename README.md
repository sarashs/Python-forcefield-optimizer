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
- **`pyfield.qm`** — the `pyfield qm-prep` subcommand and its
  pluggable QM backends. `single_point` + `relax` cover every
  registered objective. PySCF backend ships today; xTB / QE / GPAW /
  ORCA slot in behind the same interface (one new file each).

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

`pyfield qm-prep config.yaml` runs single-points and geometry
optimisations for every `target: { from: dft }` placeholder in the
config and writes a populated copy you can hand to `pyfield run`.
Currently uses **PySCF** (pip-installable, no licence portal); xTB,
Quantum ESPRESSO, GPAW, and ORCA backends slot in behind the same
interface (one new file each). Commercial codes — Gaussian, VASP,
Molpro — are intentionally not bundled.

```bash
pip install -e .[qm]                    # adds pyscf + ase
pyfield qm-prep tests/cl2_qm.yaml       # populates the placeholders
pyfield run     tests/cl2_qm.populated.yaml
```

The QM cache is content-keyed (`qm_cache/<sha256>/`), so re-running
`qm-prep` after only FF-side edits is a no-op.

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
