#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:06:23 2020

@author: sarashs
"""

import sys
from os import getcwd, path

cwd = getcwd()
input_file_path = path.join(cwd, sys.argv[1]) 
flags_dict = {'Pyfield_parameters' : False, 'Optimizatoin_parameters' : False, 'LAMMPS_parameters' : False}
Pyfield_args = {'structure_file_path' : '', 'forecfield_file_path' : '', 'parameter_file_path' : '', 'output_path' : '',\
                'log_everything' : '', 'save_lammps_trajectory' : '', 'Parallel' : 'NO', 'Number_of_processors' : 1 }
Optimization_args = {} #do these ones as well
LAMMPS_args = {}

try:
    input_file = open(input_file_path)
except OSError:
    print('Ooops! You f**ked up bro/sis! The file does not seem to exist.')
else:
    print('\n ***Do not forget to cite our paper wink wink*** \n\n')
    lines = input_file.readlines()
    print('Your input file has %d lines' %len(lines))
input_file.close()

for item in lines:
    cleaned_item = item.strip()
    if cleaned_item[0] == '#':
        pass
    elif cleaned_item.startswith('+++'):
        if cleaned_item[3:].startswith('Pyfield_parameters'):
            flags_dict = {'Pyfield_parameters' : True, 'Optimizatoin_parameters' : False, 'LAMMPS_parameters' : False}
        elif cleaned_item[3:].startswith('Optimizatoin_parameters'):
            flags_dict = {'Pyfield_parameters' : False, 'Optimizatoin_parameters' : True, 'LAMMPS_parameters' : False}
        elif cleaned_item[3:].startswith('LAMMPS_parameters'):
            flags_dict = {'Pyfield_parameters' : False, 'Optimizatoin_parameters' : False, 'LAMMPS_parameters' : True}
    if flags_dict['Pyfield_parameters']:
        pass
    if flags_dict['Optimizatoin_parameters']:
        pass
    if flags_dict['LAMMPS_parameters']: