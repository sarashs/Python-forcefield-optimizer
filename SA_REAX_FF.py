#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:37:58 2020

@author: sarashs
This is the reax forcefield simulated annealing class. The energy functions and data types are defined per reax forcefield.
"""
from REAX_FF import REAX_FF 
from copy import deepcopy
from LAMMPS_Utils import lammps_input_creator
from SA import SA

class SA_REAX_FF(SA):
    def __init__(self,forcefield_path, output_path, params_path, Training_file, Input_structure_file, T=1, T_min=0.00001, Temperature_decreasing_factor=0.1, max_iter=50, number_of_points=1):
        super().__init__(forcefield_path, output_path, params_path, Training_file, Input_structure_file, T, T_min, Temperature_decreasing_factor, max_iter, number_of_points)
        # Initial forcefield (initial annealer(s))
        temp_init = REAX_FF(forcefield_path,params_path)
        temp_init.parseParamSelectionFile()
        self.init_ff = [temp_init]
        output_path = self.general_output_path + str(0) + "output_FF.reax"
        self.init_ff[0].write_forcefield(output_path)
        self.lammps_file_list = [0] * number_of_points
        self.lammps_file_list[0] = lammps_input_creator(self.Input_structure_file, output_path, 'reax', self.general_output_path)
        for i in range(1, number_of_points):
            output_path = self.general_output_path + str(i)
            self.lammps_file_list[i] = lammps_input_creator(self.Input_structure_file, output_path, 'reax', self.general_output_path)
            self.init_ff.append(self.input_generator(temp_init, output_path))
    def input_generator(self, input_forcefield, output_path):
        """Generates the next solution.

        Returns
        -------
        Updated forcefield of forcefield type

        """
        output_forcefield = deepcopy(input_forcefield)
        # generate values for selected params
        for param_tuple in input_forcefield.param_min_max_delta.keys():
            while True:
                output_forcefield.params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = input_forcefield.params[param_tuple[0]][param_tuple[1]][param_tuple[2]] + random.choice([-1, 1]) * input_forcefield.param_min_max_delta['delta']
                if output_forcefield.params[param_tuple[0]][param_tuple[1]][param_tuple[2]] >= input_forcefield.param_min_max_delta['min'] and output_forcefield.params[param_tuple[0]][param_tuple[1]][param_tuple[2]] <= input_forcefield.param_min_max_delta['max']:
                    break
        ####
        # save the new forcefield file
        output_forcefield.write_forcefield(output_path)
        # use the same name for the input structure file
        lammps_input_creator(self.Input_structure_file, output_path)
        # I think these two neet to be removed
        #self._Input_data_file_list = list_of_structures(self.Input_structure_file)
        #self.Training_data = Training_data(self.Training_file)