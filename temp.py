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
#from lammps import lammps
import pylab
#from mpi4py import MPI

"""Training data is now + path"""

#ff_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/ffield2.reax'
#ParamSelect_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/params'
#Training_file = 'Trainingfile.txt'
#Input_structure_file = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/Zr_O_Si_structure.txt'
#output_path = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/'
#geofilecreator(Input_structure_file, output_path)
#a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=0.3, T_min=0.01, Temperature_decreasing_factor=0.1, max_iter=30, number_of_points=5, min_style = 'cg')

#For ZrOSi, Genetic algorithm
#ff_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/ffield1.reax'
#ParamSelect_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/params'
#Training_file = 'Trainingfile.txt'
#Input_structure_file = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/Zr_O_Si_structure.txt'
#output_path = '/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/'
#geofilecreator(Input_structure_file, output_path)
#a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=1, T_min=0.01, Temperature_decreasing_factor=0.1, max_iter=1, number_of_points=1, min_style = 'cg')
#a.anneal(record_costs = "YES", repelling_weight = 0)
##temp_list = [0.5]*5 + [0.2]*5 + [0.1]*10 + [0.05]*5
##for initial_temp in temp_list: #range(30): 
##    a.T = initial_temp
##    g = GA_REAX_FF.from_forcefield_list(a.sol_, a.cost_, output_path)
##    g.next_generation(5, Keep_the_best = "YES", mode = "average")
##    a.anneal(record_costs = "YES", repelling_weight = 0)
#listaa = [item['annealer_0.reax'] for item in a.costs]
#pylab.plot(listaa)
#a.Individual_Energy(parallel = "NO")
#a.clean_the_mess(lammpstrj = "NO")
#pylab.plot([item-min(list(a.structure_energies['annealer_0.reax'].values())) for item in a.structure_energies['annealer_0.reax'].values()])

##For CL
#ff_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/ffieldoriginal.txt'
#ParamSelect_filePath = '/home/sarashs/Python-forcefield-optimizer/tests/params'
#Training_file = 'Trainingfile_2.txt'
#Input_structure_file = '/home/sarashs/Python-forcefield-optimizer/tests/Inputstructurefile.txt'
#output_path = '/home/sarashs/Python-forcefield-optimizer/tests/'
#geofilecreator(Input_structure_file, output_path)
#
#a = SA_REAX_FF(ff_filePath, output_path, ParamSelect_filePath, Training_file, Input_structure_file, T=0.2, T_min=0.1, Temperature_decreasing_factor=0.1, max_iter=3, number_of_points=1, min_style = 'cg', processors = 6)
#startTime = datetime.now()
#for i in range(5):
#    a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'NO')
###g = GA_REAX_FF.from_forcefield_list(a.sol_, a.cost_, output_path)
###g.next_generation(5, Keep_the_best = "YES", mode = "average")
#    a.T = 0.2
###a.anneal(record_costs = "YES", repelling_weight = 0, parallel = 'NO')
#a.clean_the_mess(lammpstrj = "YES")
#print(datetime.now() - startTime)
#listaa = [item['annealer_0.reax'] for item in a.costs]
#pylab.plot(listaa)

#create the xyz files
#import os
#path = "/home/sarashs/Python-forcefield-optimizer/tests/Zr_Si_forcefield/OH3_Si_O_Zr_OH3_B3LYP_disperssion/"
#for file_name in os.listdir(path):
#    if file_name.endswith(".log"):
#        gaussian_xyz_extractor(path, file_name)
#
#
##create the input_structure_file
#for file_name in os.listdir(path):
#    if file_name.endswith(".xyz"):
#        append_2structure_file(path, "/home/sarashs/Python-forcefield-optimizer/tests/", file_name, "OH3_Si_O_Zr_OH3_structure", box_dim = [100, 100, 100],  restrain = ["bond 1 5 2000 2000 x\n"])
#  
##extract the energies     
#for file_name in os.listdir(path):
#    if file_name.endswith(".log"):
#        gaussian_energy_extractor(path, "/home/sarashs/Python-forcefield-optimizer/tests/", file_name, "OH3_Si_O_Zr_OH3_energies")
