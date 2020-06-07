#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:33:21 2020

@author: sarashs
LAMMPS utils containing:
    lammps_input_creator : Which prepares the lammps inputfile.dat 
    geofilecreator : Prepares lammps atom files [also known as geometry file] atom.data 

"""
import re
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
              LAMMPS_Data_file=l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".data"
              LAMMPS_Input_file=file_path + l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".dat"
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
              s.write('read_data   '+'scripts/'+LAMMPS_Data_file+'\n')
              s.write('\n# 3.- Force-Field ##########################\n\n')
              number_of_atoms=int(l[l.index(item)+1])
              #Forcefield params
              if(Forcefield_type in 'reax'):
                  s.write('pair_style reax/c NULL\n')
                  s.write('pair_coeff * * '+'scripts/'+Input_forcefield)
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
              s.write('dump DUMP2 all custom 1000000 '+ 'logs/lmp_logs/'+LAMMPS_Data_file.replace('.data','.out')+'.lammpstrj'+'id type x y z q #this size \n')
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
              s.write('fix_modify holdem energy yes\n')
              s.write('min_style cg\n')
              s.write('minimize 1.0e-7 1.0e-9 20000 20000\n')
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