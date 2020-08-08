#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:06:23 2020

@author: sarashs
"""

import sys
from os import getcwd, path
from SA import SA_REAX_FF 
from GA import GA_REAX_FF
from LAMMPS_Utils import geofilecreator
from datetime import datetime
import pylab

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

cwd = getcwd()
input_file_path = path.join(cwd, sys.argv[1]) 
arguments ={'Pyfield_parameters': {'structure_file_path' : '', 'forecfield_file_path' : '', 'parameter_file_path' : '', 'output_path' : '',\
                'Training_file_path' : '', 'log_everything' : '', 'save_lammps_trajectory' : '', 'Parallel' : 'NO', 'Number_of_processors' : 1 },\
                'Optimizatoin_parameters' : {'simulated_annealing_parameters' : {'simulated_annealing' : 'YES', 'number_of_annealers' : 1, 'coulomb_repelling_force_between_annealers' : 'OFF',\
                                            'Initial_temperature' : 10, 'Final_temperature' : 0.0001, 'Temperature_decreasing_factor' : 0.1, 'Maximum_number_of_iterations' : 3},\
                                            'genetic_algorithm parametes' : {'Genetic_Algorithm' : 'NO', 'keep_the_best' : 'YES', 'mode' : 'average', 'number_of_generations' : 5,\
                                            'initial_temp_list' : [], 'number_of_parameters' : 0}}, 'LAMMPS_parameters' : {'minimization' : 'cg', 'box_dimentions': [100, 100, 100]}}

try:
    input_file = open(input_file_path)
except OSError:
    print('Ooops! You f**ked up bro/sis! The file does not seem to exist.')
else:
    print('\n ***Do not forget to cite our paper wink wink*** \n\n')
    lines = input_file.readlines()
    print('Your input file has %d lines' %len(lines))
input_file.close()

key_3_plus = 'Pyfield_parameters'
key_2_plus = 'simulated_annealing_parameters'
for item in lines:
    cleaned_item = item.strip()
    if cleaned_item.startswith('#'):
        pass
    elif cleaned_item.startswith('+++'):
        if cleaned_item[3:].startswith('Pyfield_parameters'):
            key_3_plus = 'Pyfield_parameters'
        elif cleaned_item[3:].startswith('Optimizatoin_parameters'):
            key_3_plus = 'Optimizatoin_parameters'
        elif cleaned_item[3:].startswith('LAMMPS_parameters'):
            key_3_plus = 'LAMMPS_parameters'
    elif cleaned_item.startswith('++'):
        if cleaned_item[2:].startswith('simulated_annealing_parameters'):
            key_2_plus = 'simulated_annealing_parameters'
        elif cleaned_item[2:].startswith('genetic_algorithm parametes'):
            key_2_plus = 'genetic_algorithm parametes'
    elif key_3_plus == 'Optimizatoin_parameters':
        if 'initial_temp_list' in cleaned_item.split(' ')[0]:
            temp = cleaned_item.split(' ')[1].split(',')
            arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = [float(i) for i in temp]
        else:
            if cleaned_item.split(' ')[1].isdigit():
                arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = int(cleaned_item.split(' ')[1])
            elif isfloat(cleaned_item.split(' ')[1]):
                arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = float(cleaned_item.split(' ')[1])
            else:
                arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = cleaned_item.split(' ')[1]
    else:
        if 'box_dimentions' in cleaned_item.split(' ')[0]:
            temp = cleaned_item.split(' ')[1].split(',')
            arguments[key_3_plus][cleaned_item.split(' ')[0]] = [int(i) for i in temp]
        else:
            if cleaned_item.split(' ')[1].isdigit():
                arguments[key_3_plus][cleaned_item.split(' ')[0]] = int(cleaned_item.split(' ')[1])
            elif isfloat(cleaned_item.split(' ')[1]):
                arguments[key_3_plus][cleaned_item.split(' ')[0]] = float(cleaned_item.split(' ')[1])
            else:
                arguments[key_3_plus][cleaned_item.split(' ')[0]] = cleaned_item.split(' ')[1]
print("Your input parameters: \n\n")
print(arguments)
print("\n\n")
startTime = datetime.now()
# for annealing
if arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['simulated_annealing'].upper() == 'YES':  
    geofilecreator(arguments['Pyfield_parameters']['structure_file_path'], arguments['Pyfield_parameters']['output_path'])
    
    a = SA_REAX_FF(arguments['Pyfield_parameters']['forecfield_file_path'] , arguments['Pyfield_parameters']['output_path'], arguments['Pyfield_parameters']['parameter_file_path'],
                   arguments['Pyfield_parameters']['Training_file_path'], arguments['Pyfield_parameters']['structure_file_path'],
                   T = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['Initial_temperature'],
                   T_min =  arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['Final_temperature'],
                   Temperature_decreasing_factor = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['Temperature_decreasing_factor'],
                   max_iter = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['Maximum_number_of_iterations'],
                   number_of_points = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['number_of_annealers'],
                   min_style = arguments['LAMMPS_parameters']['minimization'], processors = arguments['Pyfield_parameters']['Number_of_processors'])
    a.anneal(record_costs = arguments['Pyfield_parameters']['log_everything'].upper(), repelling_weight = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['repelling_weight'], parallel = arguments['Pyfield_parameters']['Parallel'].upper())
## For genetic algorithm
if arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['Genetic_Algorithm'].upper() == 'YES': 
    g = GA_REAX_FF.from_forcefield_list(a.sol_, a.cost_, arguments['Pyfield_parameters']['output_path'])
    if len(arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['initial_temp_list']) == arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['number_of_generations']:
        
        print("The GA initial temperatures are being read from the input list.\n")
        
        for i in arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['initial_temp_list']:
            g.next_generation(arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['number_of_parameters'], Keep_the_best = arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['keep_the_best'], mode = arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['mode'])
            a.T = i
            a.anneal(record_costs = arguments['Pyfield_parameters']['log_everything'].upper(), repelling_weight = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['repelling_weight'], parallel = arguments['Pyfield_parameters']['Parallel'].upper())
    else:
        
        print("The GA initial temperatures are set by the software.\n")
        
        for i in range(arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['number_of_generations']):
            g.next_generation(arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['number_of_parameters'], Keep_the_best = arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['keep_the_best'], mode = arguments['Optimizatoin_parameters']['genetic_algorithm parametes']['mode'])
            a.T = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['Initial_temperature']
            a.anneal(record_costs = arguments['Pyfield_parameters']['log_everything'].upper(), repelling_weight = arguments['Optimizatoin_parameters']['simulated_annealing_parameters']['repelling_weight'], parallel = arguments['Pyfield_parameters']['Parallel'].upper())
#cleaning the mess
if arguments['Pyfield_parameters']['log_everything'].upper() == 'NO':
    a.clean_the_mess(lammpstrj = arguments['Pyfield_parameters']['save_lammps_trajectory'].upper())
print('Your simulation took: ')
print(datetime.now() - startTime)