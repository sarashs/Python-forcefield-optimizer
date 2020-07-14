#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:33:21 2020

@author: sarashs
LAMMPS utils containing:
    
    Parameters:
    atomic_weight_dict
    
    Functions:
        
    lammps_input_creator : Which prepares the lammps inputfile.dat 
    geofilecreator : Prepares lammps atom files [also known as geometry file] atom.data 
    append_2structure_file : Avogadro xyz to input structure file
    gaussian_energy_extractor : extracts gaussian log file energies and converts them to kj/mol and eV
"""
import re

atomic_weight_dict = {"H" : 1.0079, "He" : 4.0026,"Li" : 6.941,"Be" : 9.0122,\
                      "B" : 10.811, "C" : 12.0107,"N" : 14.0067,"O" : 15.9994,\
                      "F" : 18.9984, "Ne" : 20.1797, "Na" : 22.9897, "Mg" : 24.305,\
                      "Al" : 26.9815, "Si" : 28.0855, "P" : 30.9738, "S" : 32.065,\
                      "Cl" : 35.453, "K" : 39.0983, "Ar" : 39.948, "Ca" : 40.078,\
                      "Sc" : 44.9559, "Ti" : 47.867, "V" : 50.9415, "Cr" : 51.9961,\
                      "Mn" : 54.938, "Fe" : 55.845, "Ni" : 58.6934, "Co" : 58.9332,\
                      "Cu" : 63.546, "Zn" : 65.39, "Ga" : 69.723, "Ge" : 72.64,\
                      "As" : 74.9216, "Se" : 78.96, "Br" : 79.904, "Kr" : 83.8,\
                      "Rb" : 85.4678, "Sr" : 87.62, "Y" : 88.9059, "Zr" : 91.224,\
                      "Nb" : 92.9064, "Mo" : 95.94, "Tc" : 98.00, "Ru" : 101.07,\
                      "Rh" : 102.9055, "Pd" : 106.42, "Ag" : 107.8682, "Cd" : 112.411,\
                      "In" : 114.818, "Sn" : 118.71, "Sb" : 121.76, "I" : 126.9045,\
                      "Te" : 127.6, "Xe" : 131.293, "Cs" : 132.9055, "Ba" : 137.327,\
                      "La" : 138.9055, "Hf" : 178.49, "Ta" : 180.9479, "W" : 183.84,\
                      "Pt" : 195.078, "Au" : 196.9665, "Hg" : 200.59, "Pb" : 207.2,\
                      "Bi" : 208.9804, "U" : 238.0289}

def lammps_input_creator(Input_structure_file="Inputstructurefile.txt",Input_forcefield='ffield.reax',Forcefield_type = 'reax', file_path = ""):
    """
    This function creates the lammps input file
    :param Input_structure_file:
    :param Input_forcefield:
    :return: Input_data_file_list
    """
    Input_data_file_list=[]
    try:
        f=open(Input_structure_file,'U')
        l=f.readlines()
        for item in l:
            if '#structure ' in item:
                Input_data_file_list.append(item.replace('#structure ','').replace('\n','').replace(' ','')+".dat")
        f.close()
    except IOError:
        print('An error occured trying to read the training data file.')
    f=open(Input_structure_file,'r')
    l=f.readlines()
    for item in l:
           atom_type=0
           if '#structure ' in item:
              LAMMPS_Data_file = file_path + l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".data"
              LAMMPS_Input_file = file_path + l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+Input_forcefield.replace('.reax','')+".dat"
              s=open(LAMMPS_Input_file,'w')
              s.close()
              s=open(LAMMPS_Input_file,'a')
              #for n in lists:
              s.write('# 1.- Inizialization #######################\n')
              s.write('units real\n')
              s.write('  #mass = grams/mole\n')
              s.write('  #distance = Angstroms\n')
              s.write('  #time = femtoseconds\n')
              s.write('  #energy = kcal/mol\n')
              s.write('  #velocity = Angstroms/femtosecond\n')
              s.write('  #force = kcal/mol.Angstrom\n')
              s.write('  #torque = kcal/mole\n')
              s.write('  #temperature = degrees K\n')
              s.write('  #pressure = atmospheres (0.1013 GPa)\n')
              s.write('  #dynamic viscosity = Poise\n')
              s.write('  #charge = multiple of electron charge (+1.0 is a proton)\n')
              s.write('  #dipole = charge*Angstroms\n')
              s.write('  #electric field = volts/Angstrom\n')
              s.write('dimension 3\n')
              s.write('processors * * *\n')
              s.write('##\n')
              s.write('boundary p p p\n')
              s.write('atom_style charge\n\n# 2.- Atom definition ######################\n\n')
              s.write('atom_modify map hash\n')
              s.write('read_data   '+LAMMPS_Data_file+'\n')
              s.write('\n# 3.- Force-Field ##########################\n\n')
              number_of_atoms=int(l[l.index(item)+1])
              #Forcefield params
              if(Forcefield_type in 'reax'):
                  s.write('pair_style reax/c NULL\n')
                  s.write('pair_coeff * * ' + file_path + Input_forcefield)
              #calculate number of atom types
              for item2 in l[(l.index(item)+3):]:
                 if not ('#dimensions' in item2):
                    atom_type=atom_type+1
                 else:
                    break
              #Add elements to the forcefield line
              for i in range(1,atom_type+1):
                 s.write(' '+l[l.index(item)+2+i][0:2])
              s.write('\n'+'fix 99 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n')
              s.write('neighbor        2.0 bin\n')
              s.write('neigh_modify    every 10 check yes\n\n')
              s.write('## 4.- MD & relax parameters ################\n\n')
              ######
              s.write('dump DUMP2 all custom 1000000 '+LAMMPS_Data_file.replace('.data','')+Input_forcefield.replace('.reax','')+'.lammpstrj'+' id type x y z q #this size \n')
              s.write('thermo_style custom step etotal ke pe temp press pxx pyy pzz \n')
              s.write('thermo 1000000\n')
              #####fix restraints
              i=l.index(item)+atom_type+number_of_atoms+6
              if '#restrain' in l[i-1]:
                  s.write('fix holdem all restrain')
                  while i<len(l) and ('#structure ' not in l[i]):
                             s.write(' '+l[i].replace('\n',''))
                             i=i+1
                  s.write('\n')
              #This line decides whether or not the fix is going to be in the calculated energy
                  s.write('fix_modify holdem energy yes\n') 
                  s.write('min_style cg\n')
                  s.write('minimize 1.0e-12 1.0e-12 20000 200000\n')
                  s.write('undump DUMP2\n')
                  s.write('unfix holdem\n')
              else:
                  s.write('min_style cg\n')
                  s.write('minimize 1.0e-12 1.0e-12 20000 200000\n')                  
                  s.write('undump DUMP2\n')
              s.close()
    f.close()
    return Input_data_file_list

def geofilecreator(Input_structure_file="Inputstructurefile.txt", file_path = ""):
    """
    This function creates lammps data files
    """
    f=open(Input_structure_file,'r')
    l=f.readlines()
    for item in l:
           atom_type=0
           if '#structure ' in item:
              LAMMPS_Data_file = file_path + l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".data"
              s=open(LAMMPS_Data_file,'w')
              s.close()
              s=open(LAMMPS_Data_file,'a')
              #for n in lists:
              s.write('# System description #######################\n')
              s.write('#\n')
              s.write('\n')
              #line for number of atoms
              s.write(l[l.index(item)+1].replace('\n','  atoms\n')) 
              number_of_atoms=int(l[l.index(item)+1])
              #line for atom types
              for item2 in l[(l.index(item)+3):]:
                 if not ('#dimensions' in item2):
                    atom_type=atom_type+1
                 else:
                    break   
              s.write('%d atom types\n' % atom_type)
              #line for box dimentions
              dimensions=re.findall(r"[-+]?\d*\.\d+|\d+",l[(l.index(item)+4+atom_type)])
              #s.write(l[(l.index(item)+3+atom_type)])
              s.write('0 %f xlo xhi\n' % float(dimensions[0]) )
              s.write('0 %f ylo yhi\n' % float(dimensions[1]) )
              s.write('0 %f zlo zhi\n' % float(dimensions[2]) )
              s.write('#\n')
              s.write('# for a crystal:\n')
              s.write('# lx=a;  ly2+xy2=b2;  lz2+xz2+yz2=c2\n')
              s.write('# xz=c*cos(beta);  xy=b*cos(gamma)\n')
              s.write('# xy*xz+ly*yz=b*c*cos(alpha)\n')
              s.write('#\n\n')
              s.write('# Elements #################################\n\n')
              s.write('Masses\n\n')
              # line for atomic masses
              for i in range(1,atom_type+1):
                 s.write(l[l.index(item)+2+i].replace(l[l.index(item)+2+i][0:2],'%d ' % i))
              s.write('\nAtoms\n')
              for item2 in l[(l.index(item)+5+atom_type):(l.index(item)+5+atom_type+number_of_atoms)]:
                 for i in range(1,atom_type+1):
                        item2=item2.replace(l[l.index(item)+2+i][0:2],'%d ' % i)
                 s.write('\n'+item2.replace('\n',''))
           s.close()
    f.close()

def append_2structure_file(input_files_path, output_files_path, input_xyz_name, structure_file_name, box_dim = [100, 100, 100]):
    """This function reformats and appends a given XYZ file to the input structure file. 
    This function does not add the bond retraints etc.
    :param input_files_path:
    :param output_file_path:
    :param xyz input file name:
    :param structure file name:
    :param box_dim:[x y z]
    :return: NULL"""
    if ".xyz" in input_xyz_name[-4:]:
        input_xyz_name = input_xyz_name[:-4]
    if ".txt" in structure_file_name[-4:]:
        structure_file_name = structure_file_name[:-4]
    try:
        f=open(input_files_path + input_xyz_name + ".xyz",'U')
        l=f.readlines()
        f.close()
    except IOError:
        print('An error occured trying to read the xyz file.')
    num_atoms = int(l[0].replace('\n','').replace(' ',''))
    set_of_atoms = set([])
    for item in l[2:(2+num_atoms)]:
        set_of_atoms.add(item[:3].replace(' ',''))
        
    try:
        S=open(output_files_path + structure_file_name + ".txt",'a')
    except IOError:
        print('The structure (output) file cannot be opened.')
    S.write('#structure ' + input_xyz_name + "\n")
    S.write(l[0])
    S.write('#weights' + "\n")
    for item in set_of_atoms:
        S.write(item + " " + str(atomic_weight_dict[item]) + "\n")
    S.write("#dimensions\n")
    S.write(str(box_dim[0]) + " " + str(box_dim[1]) + " " + str(box_dim[2]) + "\n")
    S.writelines(l[2:(2+num_atoms)])
    S.close()

def gaussian_energy_extractor(input_files_path, output_files_path, input_gaussian_file_name, energy_file_name, flag = "HF="):
    """ This function extracts the structure energies from a gaussian output (.log) file"""
    if ".log" in input_gaussian_file_name[-4:]:
        input_gaussian_file_name = input_gaussian_file_name[:-4]
    if ".txt" in energy_file_name[-4:]:
        energy_file_name = energy_file_name[:-4]
    try:
        f=open(input_files_path + input_gaussian_file_name + ".log",'U')
        l=f.readlines()
        f.close()
    except IOError:
        print('An error occured trying to read the log file.')
    for item in l:
        index = item.find(flag)
        if index > -1 :
            item2 = item[index:] 
            item2 = item2.replace(flag, '')
            index2 = item2.find("\\")
            energy = float(item2[:index2])
            break
    try:
        S=open(output_files_path + energy_file_name + ".txt",'a')
    except IOError:
        print('The structure (output) file cannot be opened.')
    S.write(input_gaussian_file_name + "   " + str(energy * 627.5094740631) + "  kcal/mol  " + str(energy * 27.211386245) + "  eV" + "\n")
    S.close()      