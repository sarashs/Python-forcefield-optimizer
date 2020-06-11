#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:37:58 2020

@author: sarashs
This is the reax forcefield simulated annealing class. The energy functions and data types are defined per reax forcefield.

    Attributes
    --------------
    structure_energies : dict of dict
        energy calculated per annealer (forcefield) per structure file.
        [forcefield_name][structure_name] = energy
    structure_charges : dict of dict
        charge calculated per annealer (forcefield) per structure file.
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
        self.structure_charges = {} 
        forcefield_name = "anenaler_" + str(0) + ".reax"
        self.sol_[forcefield_name] = deepcopy(temp_init)
        self.sol_[forcefield_name].ff_filePath = self.general_output_path + forcefield_name
        self.sol_[forcefield_name].write_forcefield(self.general_output_path + forcefield_name)
        self.lammps_file_list = {} #dictionary keys: forcefield tag, values: list of lammps files
        self.lammps_file_list[forcefield_name] = lammps_input_creator(self.Input_structure_file, forcefield_name, 'reax', self.general_output_path)
        self.structure_energies[forcefield_name] = {} 
        self.structure_charges[forcefield_name] = {} 
        for i in range(1, number_of_points):
            forcefield_name = "anenaler_" + str(i) + ".reax"
            self.sol_[forcefield_name] = deepcopy(temp_init)
            self.sol_[forcefield_name].ff_filePath = self.general_output_path + forcefield_name
            self.input_generator(forcefield_name, update = "YES")
            self.structure_energies[forcefield_name] = {}
            self.structure_charges[forcefield_name] = {} 
    def input_generator(self, forcefield_name, update = "YES"):
        """Generates the next solution.

        Returns
        -------
        self : object

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
        :return: 
        --------
        self : object
        
        """
#####Running lammps and python in serial        
        if "NO" in parallel:
            for item in self.sol_.keys():
                for a_file in self.lammps_file_list[item]:
                    lmp = lammps()
                    lmp.file(self.general_output_path + a_file)
                    self.structure_energies[item][a_file] = lmp.get_thermo("etotal")
                    #pe = lmp.get_thermo("pe")
                    lmp.close()
        elif "YES" in parallel:
            pass
        else:
            raise ValueError("parallel value for __Individual_Energy takes YES or NO only!")
    def cost_function(self):
        """Computes the cost function.
        Returns
        -------
        self : object
        
        """
        # decide whether or not to do the charge based on self.training_charge_weight= 0
        for item in self.sol_.keys():
            ##### Cost calculation: For now mean square
            ##### Computing energy
            self.cost_ = self.training_energy_weight * sum([trainee[0] * (trainee[1] * self.structure_energies[item][trainee[2]]+ trainee[3] * self.structure_energies[item][trainee[4]] - trainee[5]) ** 2 for trainee in self.Training_info.training_energy])
            ##### Computing charge 
            