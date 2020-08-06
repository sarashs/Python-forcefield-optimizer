#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:06:23 2020

@author: sarashs
"""

import sys
from os import getcwd, path

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

cwd = getcwd()
input_file_path = path.join(cwd, sys.argv[1]) 
arguments ={'Pyfield_parameters': {'structure_file_path' : '', 'forecfield_file_path' : '', 'parameter_file_path' : '', 'output_path' : '',\
                'log_everything' : '', 'save_lammps_trajectory' : '', 'Parallel' : 'NO', 'Number_of_processors' : 1 },\
                'Optimizatoin_parameters' : {'simulated_annealing_parameters' : {'number_of_annealers' : 1, 'coulomb_repelling_force_between_annealers' : 'OFF',\
                                            'Initial_temperature' : 10, 'Final_temperature' : 0.0001}, 'genetic_algorithm parametes' : {'keep_the_best' : 'YES'},\
                                            'mode' : 'average', 'number_of_generations' : 5, 'initial_temp_list' : [1, 0.5, 0.1, 0.05, 0.05]},\
                'LAMMPS_parameters' : {'minimization' : 'cg', 'box_dimentions': [100, 100, 100]}}

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
            arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = [isfloat(i) for i in temp]
        else:
            if cleaned_item.split(' ')[1].isdigit():
                arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = int(cleaned_item.split(' ')[1])
            elif isfloat(cleaned_item.split(' ')[1]):
                arguments[key_3_plus][key_2_plus][cleaned_item.split(' ')[0]] = float(cleaned_item.split(' ')[1])
    else:
        if 'box_dimentions' in cleaned_item.split(' ')[0]:
            temp = cleaned_item.split(' ')[1].split(',')
            arguments[key_3_plus][cleaned_item.split(' ')[0]] = [int(i) for i in temp]
        else:
            if cleaned_item.split(' ')[1].isdigit():
                arguments[key_3_plus][cleaned_item.split(' ')[0]] = int(cleaned_item.split(' ')[1])
            elif isfloat(cleaned_item.split(' ')[1]):
                arguments[key_3_plus][cleaned_item.split(' ')[0]] = float(cleaned_item.split(' ')[1])