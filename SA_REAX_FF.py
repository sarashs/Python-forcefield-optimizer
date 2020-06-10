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
from lammps import lammps
from mpi4py import MPI
from SA import SA
import random

class SA_REAX_FF(SA):
    def __init__(self,forcefield_path, output_path, params_path, Training_file, Input_structure_file, T=1, T_min=0.00001, Temperature_decreasing_factor=0.1, max_iter=50, number_of_points=1):
        super().__init__(forcefield_path, output_path, params_path, Training_file, Input_structure_file, T, T_min, Temperature_decreasing_factor, max_iter, number_of_points)
        # Initial forcefield (initial annealer(s))
        temp_init = REAX_FF(forcefield_path,params_path)
        temp_init.parseParamSelectionFile()
        forcefield_name = "anenaler_" + str(0) + ".reax"
        self.sol_[forcefield_name] = deepcopy(temp_init)
        self.sol_[forcefield_name].ff_filePath = self.general_output_path + forcefield_name
        self.sol_[forcefield_name].write_forcefield(self.general_output_path + forcefield_name)
        self.lammps_file_list = {} #dictionary keys: forcefield tag, values: list of lammps files
        self.lammps_file_list[forcefield_name] = lammps_input_creator(self.Input_structure_file, forcefield_name, 'reax', self.general_output_path)
        for i in range(1, number_of_points):
            forcefield_name = "anenaler_" + str(i) + ".reax"
            self.sol_[forcefield_name] = deepcopy(temp_init)
            self.sol_[forcefield_name].ff_filePath = self.general_output_path + forcefield_name
            self.input_generator(forcefield_name, update = "YES")
    def input_generator(self, forcefield_name, update = "YES"):
        """Generates the next solution.

        Returns
        -------
        Nothing

        """
        if "YES" in update:
            # generate values for selected params
            for param_tuple in self.sol_[forcefield_name].param_min_max_delta.keys():
                while True:
                    self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] + random.choice([-1, 1]) * self.sol_[forcefield_name].param_min_max_delta[param_tuple]['delta']
                    if self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] >= self.sol_[forcefield_name].param_min_max_delta[param_tuple]['min'] and self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] <= self.sol_[forcefield_name].param_min_max_delta[param_tuple]['max']:
                        break
            ####
            # save the new forcefield file
            self.sol_[forcefield_name].write_forcefield(self.Input_structure_file + forcefield_name)
            # I think these two neet to be removed
            #self._Input_data_file_list = list_of_structures(self.Input_structure_file)
            #self.Training_data = Training_data(self.Training_file)
        elif "NO" in update:
            pass
        else:
            raise ValueError("update value for inpute_generator takes YES or NO only!")
        # use the same name for the input structure file
        self.lammps_file_list[forcefield_name] = lammps_input_creator(self.Input_structure_file, forcefield_name, 'reax', self.general_output_path)
    def __Individual_Energy(self, parallel = "NO"):
        """
        Computes the Energy for ALL of the annealers and for ALL input file
        This is a private method that is called by objective function calculator
        :return: float Energy
        """
        #Running lammps and python in serial
        lmp = {}
        etotal = {}
        if "NO" in parallel:
            pass
        elif "YES" in parallel:
            pass
        else:
            raise ValueError("parallel value for __Individual_Energy takes YES or NO only!")
#        for item in self.sol_.keys():
#            lmp[item] = lammps()
#            lmp[item].file(self.lammps_file_list[item])
#            etotal[item] = lmp[item].get_thermo("etotal")
#            #pe = lmp.get_thermo("pe")
#            lmp[item].close()
        return etotal