 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import sys
#sys.path.append('../')
###Testing the forcefield
#from REAX_FF import REAX_FF
from SA import SA_REAX_FF 
from GA import GA_REAX_FF
from LAMMPS_Utils import lammps_input_creator, geofilecreator, append_2structure_file, gaussian_energy_extractor, gaussian_xyz_extractor
from datetime import datetime
from scipy import optimize

#from lammps import lammps
import pylab
#from mpi4py import MPI

"""Training data is now + path"""
#For ZrOSi, Genetic algorithm
ff_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/ffield1.reax'
ParamSelect_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/params'
Training_file = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/Trainingfile.txt'
Input_structure_file = '/home/sarashs/Python-forcefield-optimizer/tests/4SiOH3_O_Zr_center_structure.txt'#2SiOH4_H_bond_structure.txt' #4SiOH3_O_Zr_center_structure.txt'#OH3_Si_O_Zr_OH3_structure.txt'  #Zr_Si_forcefield/Zr_O_Si_structure.txt'#OH3_Si_O_Zr_OH3_structure.txt' #Zr_Si_forcefield/Zr_O_Si_structure.txt'  #2SiOH4_H_bond_structure.txt' #Zr_Si_forcefield/Zr_O_Si_structure.txt' # # #2SiOH4_H_bond_structure.txt'
output_path = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/'
geofilecreator(Input_structure_file, output_path)

a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=0.4, T_min=0.01, Temperature_decreasing_factor=0.1, max_iter=10, number_of_points=2, min_style = 'cg', processors = 12)
startTime = datetime.now()
#a.init_search('annealer_0.reax', 20, parallel = 'YES')
#a.anneal(record_costs = "YES", repelling_weight = 0.01, parallel = 'YES')
###
##g = GA_REAX_FF.from_forcefield_list(a.sol_, a.cost_, output_path)
##for i in range(5):
#i = 0
#while min(list(a.cost_.values())) > 0.5 and i < 30:
#    i += 1
#    a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=0.1, T_min=0.001, Temperature_decreasing_factor=0.1, max_iter=5, number_of_points=1, min_style = 'cg', processors = 12)
#    a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'YES')
###    g.next_generation(4, Keep_the_best = "YES", mode = "average")
print(datetime.now() - startTime)
#
a.Individual_Energy(parallel = "NO")
a.clean_the_mess(lammpstrj = "NO")
#pylab.figure(figsize=(12,8))
#REAX = [a.structure_energies['annealer_0.reax'][item] for item in a.structure_energies['annealer_0.reax'].keys() if 'eq' not in item]
#minimum = [a.structure_energies['annealer_0.reax'][item] for item in a.structure_energies['annealer_0.reax'].keys() if 'eq' in item]
#REAX = [item - minimum[0] for item in REAX]
#pylab.plot([3.3, 3.6, 3.8, 3.9, 3.95, 4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 6], [i[5] for i in a.Training_info.training_energy],'-ob', markersize=10.0,linewidth=2.0, label='B3LYP') #[3.3, 3.6, 3.8, 3.9, 3.95, 4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 6]
#pylab.plot([3.3, 3.6, 3.8, 3.9, 3.95, 4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 6], REAX,'^r', alpha=0.6, markersize=15.0, linewidth=4.0, label='REAX') #[148, 152, 155, 158, 160, 162, 164.3, 165, 166, 168, 170, 172, 175, 178]
#                                               
##pylab.plot([i-164.3 for i in [148, 152, 155, 158, 160, 162, 164.3, 165, 166, 168, 170, 172, 175, 178]], [i[5] for i in a.Training_info.training_energy],'-ob', markersize=10.0,linewidth=2.0, label='B3LYP') #[3.3, 3.6, 3.8, 3.9, 3.95, 4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 6]
##pylab.plot([i-164.3 for i in [148, 152, 155, 158, 160, 162, 164.3, 165, 166, 168, 170, 172, 175, 178]], REAX,'^r', alpha=0.6, markersize=15.0, linewidth=4.0, label='REAX') 
##pylab.plot([106, 108, 109, 110, 111.7, 113, 115, 118, 123, 128], [i[5] for i in a.Training_info.training_energy],'-ob', markersize=10.0,linewidth=1.0, label='B3LYP') #[3.3, 3.6, 3.8, 3.9, 3.95, 4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 6]
##pylab.plot([106, 108, 109, 110, 111.7, 113, 115, 118, 123, 128], REAX,'-.^r', alpha=0.6, markersize=15.0, linewidth=4.0, label='REAX') #
#pylab.xlabel('$Si-Si$ distance ($\AA$)', fontsize=14) #'$Si-O-Zr$ angle from equilibrium (Deg)'
#pylab.ylabel('$\Delta E$ (Kcal/Mol)', fontsize=14)
#pylab.xticks(fontsize=14)
#pylab.yticks(fontsize=14)
#pylab.legend(fontsize=14)
#pylab.grid()
#pylab.title('Hydrogen Bond Optimization', fontsize=14) #Hydrogen Bond
#pylab.savefig(output_path + 'Hydrogen_Bond.png') #
#pylab.show()

##For CL
#ff_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/ffieldoriginal.txt'
#ParamSelect_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/params'
#Training_file = 'Trainingfile_2.txt'
#Input_structure_file = '/home/sarashs/Python-forcefield-optimizer/tests/Inputstructurefile.txt'
#output_path = '/home/sarashs/Python-forcefield-optimizer/tests/'
#geofilecreator(Input_structure_file, output_path)
#
#a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=1, T_min=0.1, Temperature_decreasing_factor=0.1, max_iter=3, number_of_points=1, min_style = 'cg', processors = 6)
#startTime = datetime.now()
#a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'YES')
##for i in range(5):
##    a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'NO')
####g = GA_REAX_FF.from_forcefield_list(a.sol_, a.cost_, output_path)
####g.next_generation(5, Keep_the_best = "YES", mode = "average")
##    a.T = 0.2
###a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'NO')
#a.clean_the_mess(lammpstrj = "YES")
#print(datetime.now() - startTime)
#listaa = [item['annealer_0.reax'] for item in a.costs]
#pylab.plot(listaa)
#a.Individual_Energy(parallel = "NO")
#a.clean_the_mess(lammpstrj = "NO")
#pylab.plot([item-min(list(a.structure_energies['annealer_0.reax'].values())) for item in a.structure_energies['annealer_0.reax'].values()])

#create the xyz files
#import os
#path = "/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/2SiOH4_H_bond_B3LYP_disperssion/"
#for file_name in os.listdir(path):
#    if file_name.endswith(".log"):
#        gaussian_xyz_extractor(path, file_name)
#
#
###create the input_structure_file
#for file_name in os.listdir(path):
#    if file_name.endswith(".xyz"):
#        append_2structure_file(path, "/home/sarashs/Python-forcefield-optimizer/tests/", file_name, "2SiOH4_H_bond_structure", box_dim = [100, 100, 100],  restrain = ["bond 1 5 2000 2000 x\n"])
#  
###extract the energies     
#for file_name in os.listdir(path):
#    if file_name.endswith(".log"):
#        gaussian_energy_extractor(path, "/home/sarashs/Python-forcefield-optimizer/tests/", file_name, "2SiOH4_H_bond_energies")
