"""Structure → LAMMPS data-file writers.

`geofilecreator` is the legacy text-format reader (kept for back-compat).
`write_lammps_data` is the new entry point used by Phase-1 simulations,
which works directly from a validated `StructureCfg` so we don't need the
intermediate text format any more.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from pyfield.config.schema import StructureCfg


# Periodic-table masses for the elements we care about. Mirrors the
# atomic_weight_dict in the legacy LAMMPS_Utils.
ATOMIC_WEIGHT = {
    "H": 1.0079, "He": 4.0026, "Li": 6.941, "Be": 9.0122, "B": 10.811,
    "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984, "Ne": 20.1797,
    "Na": 22.9897, "Mg": 24.305, "Al": 26.9815, "Si": 28.0855, "P": 30.9738,
    "S": 32.065, "Cl": 35.453, "K": 39.0983, "Ar": 39.948, "Ca": 40.078,
    "Sc": 44.9559, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938,
    "Fe": 55.845, "Ni": 58.6934, "Co": 58.9332, "Cu": 63.546, "Zn": 65.39,
    "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96, "Br": 79.904,
    "Kr": 83.8, "Rb": 85.4678, "Sr": 87.62, "Y": 88.9059, "Zr": 91.224,
    "Nb": 92.9064, "Mo": 95.94, "Tc": 98.00, "Ru": 101.07, "Rh": 102.9055,
    "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818, "Sn": 118.71,
    "Sb": 121.76, "I": 126.9045, "Te": 127.6, "Xe": 131.293, "Cs": 132.9055,
    "Ba": 137.327, "La": 138.9055, "Hf": 178.49, "Ta": 180.9479, "W": 183.84,
    "Pt": 195.078, "Au": 196.9665, "Hg": 200.59, "Pb": 207.2, "Bi": 208.9804,
    "U": 238.0289,
}


def _ordered_elements(atoms) -> List[str]:
    """Return unique element symbols in first-appearance order."""
    seen, out = set(), []
    for a in atoms:
        if a.element not in seen:
            seen.add(a.element)
            out.append(a.element)
    return out


def write_lammps_data(structure: StructureCfg, out_path: Path | str) -> List[str]:
    """Write a LAMMPS data file from a `StructureCfg`.

    Returns the ordered list of element symbols (so callers can write a
    matching `pair_coeff * * <ffield> El1 El2 …` line).
    """
    if structure.atoms is None:
        raise NotImplementedError("StructureCfg.path (xyz) loading lands in Phase 2")
    elements = _ordered_elements(structure.atoms)
    type_of = {el: i + 1 for i, el in enumerate(elements)}

    lines = []
    lines.append("# System description #######################")
    lines.append("#")
    lines.append("")
    lines.append(f"{len(structure.atoms)}  atoms")
    lines.append(f"{len(elements)} atom types")
    bx, by, bz = structure.box
    lines.append(f"0 {bx:.6f} xlo xhi")
    lines.append(f"0 {by:.6f} ylo yhi")
    lines.append(f"0 {bz:.6f} zlo zhi")
    lines.append("#")
    lines.append("# for a crystal:")
    lines.append("# lx=a;  ly2+xy2=b2;  lz2+xz2+yz2=c2")
    lines.append("# xz=c*cos(beta);  xy=b*cos(gamma)")
    lines.append("# xy*xz+ly*yz=b*c*cos(alpha)")
    lines.append("#")
    lines.append("")
    lines.append("# Elements #################################")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for el in elements:
        lines.append(f"{type_of[el]} {ATOMIC_WEIGHT[el]}")
    lines.append("")
    lines.append("Atoms")
    lines.append("")
    for i, atom in enumerate(structure.atoms, start=1):
        lines.append(
            f"{i:<4d} {type_of[atom.element]:>2d} {atom.charge:.5f}    "
            f"{atom.x:.6f}    {atom.y:.6f}    {atom.z:.6f}"
        )

    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n")
    return elements


def geofilecreator(Input_structure_file="Inputstructurefile.txt", file_path=""):
    """Legacy `Inputstructurefile.txt` → LAMMPS `*.data` writer.

    Kept verbatim from the pre-Phase-1 code so the legacy smoke and the
    deprecation shim keep producing identical output.
    """
    f = open(Input_structure_file, 'r')
    l = f.readlines()
    for item in l:
        atom_type = 0
        if '#structure ' in item:
            LAMMPS_Data_file = file_path + l[l.index(item)].replace('#structure ', '').replace('\n', '').replace(' ', '') + ".data"
            s = open(LAMMPS_Data_file, 'w')
            s.close()
            s = open(LAMMPS_Data_file, 'a')
            s.write('# System description #######################\n')
            s.write('#\n')
            s.write('\n')
            s.write(l[l.index(item) + 1].replace('\n', '  atoms\n'))
            number_of_atoms = int(l[l.index(item) + 1])
            for item2 in l[(l.index(item) + 3):]:
                if not ('#dimensions' in item2):
                    atom_type = atom_type + 1
                else:
                    break
            s.write('%d atom types\n' % atom_type)
            dimensions = re.findall(r"[-+]?\d*\.\d+|\d+", l[(l.index(item) + 4 + atom_type)])
            s.write('0 %f xlo xhi\n' % float(dimensions[0]))
            s.write('0 %f ylo yhi\n' % float(dimensions[1]))
            s.write('0 %f zlo zhi\n' % float(dimensions[2]))
            s.write('#\n')
            s.write('# for a crystal:\n')
            s.write('# lx=a;  ly2+xy2=b2;  lz2+xz2+yz2=c2\n')
            s.write('# xz=c*cos(beta);  xy=b*cos(gamma)\n')
            s.write('# xy*xz+ly*yz=b*c*cos(alpha)\n')
            s.write('#\n\n')
            s.write('# Elements #################################\n\n')
            s.write('Masses\n\n')
            for i in range(1, atom_type + 1):
                s.write(l[l.index(item) + 2 + i].replace(l[l.index(item) + 2 + i][0:2], '%d ' % i))
            s.write('\nAtoms\n')
            for item2 in l[(l.index(item) + 5 + atom_type):(l.index(item) + 5 + atom_type + number_of_atoms)]:
                for i in range(1, atom_type + 1):
                    item2 = item2.replace(l[l.index(item) + 2 + i][0:2], '%d ' % i)
                s.write('\n' + item2.replace('\n', ''))
            s.close()
    f.close()
