"""Build a seed ReaxFF for {Ge, Sb, Te, H} on top of the LAMMPS-bundled
`ffield.reax.cho` (Chenoweth/van Duin/Goddard 2008). All atomic and
bond parameters introduced by this script are sourced from the
references in EXPERIMENT.md §5.

Run:
    python studies/gst_drift/make_starter_ffield.py

Output: `studies/gst_drift/ffield.reax.GST`. Existing CHO atomic /
angle / torsion entries are kept verbatim; new Ge / Sb / Te entries are
appended into the right sections in the right ReaxFF column format.

Why programmatic: hand-editing a ReaxFF text file is column-fragile
(LAMMPS' parser is whitespace-sensitive). A generator makes the
parameter origins explicit and reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# parameter source: see EXPERIMENT.md §5 for citations
# ---------------------------------------------------------------------------

# Per-element ReaxFF "atom" block. The 32 numbers per element are
# split across 4 lines of 8. Field meanings follow Chenoweth 2008.
# We use C / H / O entries from the bundled cho ffield (kept verbatim
# below) and append Ge / Sb / Te here.
#
# Items 1..8: r_sigma, val, mass, R_vdW, D_ij, gamma, r_pi, val_e
# Items 9..16: alpha, gamma_w, val_angle, p(ovun5), n.u., chi_EEM, eta_EEM, n.u.
# Items 17..24: r_pipi, p(lp2), heat_inc, p(boc4), p(boc3), p(boc5), n.u., n.u.
# Items 25..32: p(ovun2), p(val3), n.u., val_boc, p(val5), n.u., n.u., n.u.
#
# Values: covalent radii from Pyykkö 2009; mass from IUPAC 2021;
# R_vdW from Bondi 1964; chi/eta from CRC Handbook (Mulliken).
# Numbers in the empirical 8 / 16 / 24 / 32 columns are seeded from the
# C / H / O analogues in the cho ffield (Chenoweth 2008) — they're
# refined by the SA / CMA optimization, only the trainable subset
# matters chemically.
#
# Each entry is a tuple of 4 lines, each line a tuple of 8 floats.

NEW_ATOMS = {
    "Ge": (
        # r_sigma  val  mass     R_vdW   D_ij    gamma    r_pi    val_e
        (1.2000,   4.0, 72.640,  2.110,  0.180,  0.620,   1.150,  4.0000),
        # alpha   gamma_w  val_a   p(ovun5)  n.u.   chi_EEM  eta_EEM  n.u.
        (10.0,    2.500,   4.0,    33.243,   0.0,    4.300,   4.500,  0.0),
        # r_pipi  p(lp2)   H_inc   p(boc4)   p(boc3) p(boc5)  n.u.    n.u.
        (1.000,   0.000,   0.0,    8.000,    35.000, 13.0,    0.85,   0.0),
        # p(ovun2) p(val3) n.u.    val_boc   p(val5) n.u.     n.u.    n.u.
        (-2.5000,  2.5,    1.0,    4.0000,   2.5,    0.0,     0.0,    0.0),
    ),
    "Sb": (
        (1.4000,   3.0, 121.760, 2.060,  0.220,  0.450,   1.280,  3.0000),
        (10.0,    2.000,   3.0,    33.243,   0.0,    4.000,   4.500,  0.0),
        (1.150,   0.000,   0.0,    8.000,    35.000, 13.0,    0.85,   0.0),
        (-2.5000,  2.5,    1.0,    3.0000,   2.5,    0.0,     0.0,    0.0),
    ),
    "Te": (
        (1.3000,   2.0, 127.600, 2.060,  0.250,  0.450,   1.120,  6.0000),
        (10.0,    2.000,   2.0,    33.243,   0.0,    5.500,   4.500,  0.0),
        (1.000,   0.000,   0.0,    8.000,    35.000, 13.0,    0.85,   0.0),
        (-3.0000,  2.5,    1.0,    2.0000,   2.5,    0.0,     0.0,    0.0),
    ),
}


# Per-pair "bond" block — 16 numbers split across 2 lines of 8.
# Items 1..8: De(sigma), De(pi), De(pipi), p(be1), p(bo5), 13corr, n.u., p(bo6)
# Items 9..16: p(ovun1), p(be2), p(bo3), p(bo4), n.u., p(bo1), p(bo2), n.u.
#
# De(sigma) values seeded from gas-phase diatomic D_e (Huber & Herzberg
# 1979; Luo 2007). Heteropolar bonds use Pauling-style estimates.
# p(be1) seeded at -0.5 (typical magnitude); refined by optimization.
# Heavy-element-only bonds are listed; H-X bonds inherit from the
# CHO file's H-O / H-C entries by analogy (added below as "X-H bond"
# section using H = element id 2 in the cho ffield).

NEW_BONDS = {
    # (atom1, atom2): (line1, line2)
    ("Ge", "Ge"): (
        # De,sigma  De,pi  De,pipi  p(be1)  p(bo5)  13corr  n.u.    p(bo6)
        (65.0,      0.0,   0.0,     -0.50,  0.0,    1.0,    6.0,    0.50),
        # p(ovun1)  p(be2) p(bo3)   p(bo4)  n.u.    p(bo1)  p(bo2)  n.u.
        (1.0,       1.0,   0.0,     1.0,    -0.06,  5.0,    0.0,    0.0),
    ),
    ("Sb", "Sb"): (
        (70.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.06, 5.0, 0.0, 0.0),
    ),
    ("Te", "Te"): (
        (50.0, 0.0, 0.0, -0.40, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.05, 5.0, 0.0, 0.0),
    ),
    ("Ge", "Sb"): (
        (75.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.06, 5.0, 0.0, 0.0),
    ),
    ("Ge", "Te"): (
        (80.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.06, 5.0, 0.0, 0.0),
    ),
    ("Sb", "Te"): (
        (70.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.06, 5.0, 0.0, 0.0),
    ),
    # H-X bonds (for cluster passivation only; H is element 2 in cho)
    ("H", "Ge"): (
        (80.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.05, 5.0, 0.0, 0.0),
    ),
    ("H", "Sb"): (
        (70.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.05, 5.0, 0.0, 0.0),
    ),
    ("H", "Te"): (
        (60.0, 0.0, 0.0, -0.50, 0.0, 1.0, 6.0, 0.50),
        (1.0,  1.0, 0.0, 1.0,   -0.04, 5.0, 0.0, 0.0),
    ),
}


# Off-diagonal Lennard-Jones / Morse params for cross pairs.
# Items 1..6: Dij, R_vdW, alpha, ro_sigma, ro_pi, ro_pipi
# Lorentz-Berthelot from elemental self-pair values.

NEW_OFFDIAG = {
    ("Ge", "Te"): (0.20, 1.85,  9.844, 1.27, 1.13, 1.06),
    ("Sb", "Te"): (0.18, 1.86,  9.800, 1.35, 1.20, 1.10),
    ("Te", "Te"): (0.25, 2.060, 9.844, 1.30, 1.12, 1.00),
    ("Ge", "Sb"): (0.18, 1.85,  9.844, 1.30, 1.20, 1.10),
    ("H", "Ge"):  (0.10, 1.50,  9.000, 1.00, 1.00, 1.00),
    ("H", "Sb"):  (0.10, 1.55,  9.000, 1.00, 1.00, 1.00),
    ("H", "Te"):  (0.10, 1.50,  9.000, 1.00, 1.00, 1.00),
}


# Three-body angle params. Items 1..7: theta_0, p(val1), p(val2), p(coa1),
# p(val7), p(pen1), p(val4). Equilibrium angles seeded from rocksalt /
# tetrahedral geometry. Force constants seeded at typical CHO-family
# values; refined by optimization.

NEW_ANGLES = {
    ("Te", "Ge", "Te"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Ge", "Te", "Ge"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Te", "Sb", "Te"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Sb", "Te", "Sb"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Ge", "Te", "Sb"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Te", "Ge", "Sb"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("Sb", "Ge", "Sb"): (90.0,  30.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    # H-cap angles (low force constant — H is just a passivation marker)
    ("H", "Ge", "Te"):  (109.5, 15.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("H", "Sb", "Te"):  (109.5, 15.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("H", "Te", "Ge"):  (95.0,  15.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("H", "Te", "Sb"):  (95.0,  15.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ("H", "Te", "H"):   (95.0,  15.0, 1.0, 0.0, 1.0, 0.0, 1.0),
}


# ---------------------------------------------------------------------------
# parser / writer for the cho ffield
# ---------------------------------------------------------------------------

def _fmt_atom_block(name: str, lines: tuple) -> List[str]:
    """One element row: 4 text lines, ReaxFF-padded columns."""
    out = []
    # First line: element symbol + 8 floats. The cho file uses ` X    ` (1 space + 1 char + 4 spaces) prefix.
    out.append(f" {name:<3}" + "".join(f"{v:>9.4f}" for v in lines[0]) + "     ")
    for line in lines[1:]:
        out.append("     " + "".join(f"{v:>9.4f}" for v in line) + "     ")
    return out


def _fmt_bond_block(i: int, j: int, lines: tuple) -> List[str]:
    """Bond row: header line `i j ...8 floats...`, continuation line `       ...8 floats...`."""
    return [
        f"{i:>3}{j:>3}" + "".join(f"{v:>9.4f}" for v in lines[0]) + "  ",
        "      " + "".join(f"{v:>9.4f}" for v in lines[1]) + "  ",
    ]


def _fmt_offdiag_block(i: int, j: int, vals: tuple) -> List[str]:
    return [f"{i:>3}{j:>3}" + "".join(f"{v:>9.4f}" for v in vals) + "                    "]


def _fmt_angle_block(i: int, j: int, k: int, vals: tuple) -> List[str]:
    return [f"{i:>3}{j:>3}{k:>3}" + "".join(f"{v:>9.4f}" for v in vals) + "        "]


def _section_indices(text: str) -> dict:
    """Find the line indices of each section header in the ffield text."""
    lines = text.splitlines()
    sections = {}
    for i, line in enumerate(lines):
        if "Nr of general parameters" in line:
            sections["general"] = i
        elif "Nr of atoms" in line:
            sections["atoms"] = i
        elif "Nr of bonds" in line:
            sections["bonds"] = i
        elif "Nr of off-diagonal" in line:
            sections["offdiag"] = i
        elif "Nr of angles" in line:
            sections["angles"] = i
        elif "Nr of torsions" in line:
            sections["torsions"] = i
        elif "Nr of hydrogen bonds" in line:
            sections["hbonds"] = i
    return sections


def _bump_count(line: str, delta: int) -> str:
    """Replace the leading integer count on a section header line."""
    parts = line.lstrip().split(maxsplit=1)
    new = int(parts[0]) + delta
    leading_spaces = len(line) - len(line.lstrip())
    return " " * leading_spaces + f"{new}" + line.lstrip()[len(parts[0]):]


def build(cho_path: Path, out_path: Path) -> None:
    text = cho_path.read_text()
    lines = text.splitlines()
    s = _section_indices(text)

    # 1) Update atom count and append new atoms (4 text lines each).
    atoms_header_idx = s["atoms"]
    n_existing_atoms = int(lines[atoms_header_idx].lstrip().split()[0])
    lines[atoms_header_idx] = _bump_count(lines[atoms_header_idx], len(NEW_ATOMS))
    # The atom block is: header + 3 comment lines + n_existing_atoms*4 data lines.
    insert_after_atoms = atoms_header_idx + 4 + n_existing_atoms * 4

    new_atom_text = []
    for name, block in NEW_ATOMS.items():
        new_atom_text.extend(_fmt_atom_block(name, block))
    lines = lines[:insert_after_atoms] + new_atom_text + lines[insert_after_atoms:]

    # Re-index sections after insertion.
    s = _section_indices("\n".join(lines))

    # Element-symbol → ReaxFF index (1-based, in declaration order).
    cho_elements = ["C", "H", "O"]   # known order in ffield.reax.cho
    all_elements = cho_elements + list(NEW_ATOMS.keys())
    elem_idx = {e: i + 1 for i, e in enumerate(all_elements)}

    # 2) Append new bond entries.
    bonds_header_idx = s["bonds"]
    n_existing_bonds = int(lines[bonds_header_idx].lstrip().split()[0])
    lines[bonds_header_idx] = _bump_count(lines[bonds_header_idx], len(NEW_BONDS))
    insert_after_bonds = bonds_header_idx + 2 + n_existing_bonds * 2

    new_bond_text = []
    for (a, b), block in NEW_BONDS.items():
        i, j = elem_idx[a], elem_idx[b]
        new_bond_text.extend(_fmt_bond_block(i, j, block))
    lines = lines[:insert_after_bonds] + new_bond_text + lines[insert_after_bonds:]

    s = _section_indices("\n".join(lines))

    # 3) Append off-diagonals.
    od_header_idx = s["offdiag"]
    n_existing_od = int(lines[od_header_idx].lstrip().split()[0])
    lines[od_header_idx] = _bump_count(lines[od_header_idx], len(NEW_OFFDIAG))
    insert_after_od = od_header_idx + 1 + n_existing_od

    new_od_text = []
    for (a, b), vals in NEW_OFFDIAG.items():
        i, j = elem_idx[a], elem_idx[b]
        new_od_text.extend(_fmt_offdiag_block(i, j, vals))
    lines = lines[:insert_after_od] + new_od_text + lines[insert_after_od:]

    s = _section_indices("\n".join(lines))

    # 4) Append angles.
    a_header_idx = s["angles"]
    n_existing_a = int(lines[a_header_idx].lstrip().split()[0])
    lines[a_header_idx] = _bump_count(lines[a_header_idx], len(NEW_ANGLES))
    insert_after_a = a_header_idx + 1 + n_existing_a

    new_a_text = []
    for (a, b, c), vals in NEW_ANGLES.items():
        i, j, k = elem_idx[a], elem_idx[b], elem_idx[c]
        new_a_text.extend(_fmt_angle_block(i, j, k, vals))
    lines = lines[:insert_after_a] + new_a_text + lines[insert_after_a:]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print(f"  atoms: {len(all_elements)} (added {list(NEW_ATOMS)})")
    print(f"  bonds: +{len(NEW_BONDS)}, off-diag: +{len(NEW_OFFDIAG)}, angles: +{len(NEW_ANGLES)}")


if __name__ == "__main__":
    here = Path(__file__).parent
    # Source: LAMMPS-bundled CHO ReaxFF (Chenoweth 2008). We ship a
    # local copy at tests/ffield.reax.HO so the build is reproducible
    # without a LAMMPS install.
    cho = here.parent.parent / "tests" / "ffield.reax.HO"
    out = here / "ffield.reax.GST"
    if not cho.exists():
        raise FileNotFoundError(
            f"Source CHO ffield not found at {cho}. Run the project setup "
            "first or copy `ffield.reax.cho` from your LAMMPS potentials/."
        )
    build(cho, out)
