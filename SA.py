import os
from ForceField import REAX_FF
import numpy as np
from Training_data import Training_data
from LAMMPS_Utils import energy_charge
from multiprocessing import Pool, cpu_count

class SA(object):
    """ Simulated Annealing optimizer.
    I have implemented a simple simulated annealing algorithm which will run on "number_of_point" different annealers and finally picks the best among those.

    Parameters
    --------------
    T: float
        starting temperature.
    T_min: float
        final temperature.
    alpha: float
        temperature scaling factor.
    max_iter: int
        maximum number of iterations per temperature.
    number_of_points: int
        number of annealers (lammps instances at once)
    Attributes
    --------------
    cost_ : list of float
        Cost calculated by the cost function, needs to be minimized.
    sol_ : list of solution objects for each annealer
        Solution to the cost function.
    costs : list of dict, 
            Costs over time
    single_best_solution: list,
        Contains the single best solution rom the last set of annealers.
    """
    def __init__(self,forcefield_path,output_path,params_path,Training_file,Input_structure_file,T=1,T_min=0.00001,Temperature_decreasing_factor=0.1,max_iter=50, number_of_points=1, min_style = 'cg', processors=0):
        self.general_path = output_path
        self.min_style = min_style
        self.single_best_solution = None
        self.reppeling_cost_ = {}
        self.charge_cost_ = {}
        self.energy_cost_ = {}
        self.T=T
        self.T_min = T_min
        self.alpha = Temperature_decreasing_factor
        self.max_iter = max_iter
        self.Input_structure_file = Input_structure_file
        # Training data for the Energy calculation
        self.Training_info = Training_data(Training_file)
        self.number_of_points = number_of_points
        #This should be defined for each forcefield separately
        #self.init_ff= {}
        self.cost_= {}#[0] * number_of_points
        ###### take them to SA REAX
        # this is a list of solutions
        self.sol_= {}#[0] * number_of_points #self.init_ff.selected_parameters_value
        self.costs = []
        #Energy per annealer per structure
        self.structure_energies = {}
        self.structure_charges = {} 
        self.number_of_processors = processors
    
    def input_generator(self):
        """Generates the next solution.

        Returns
        -------
        self : object

        """
        pass
    def cost_function(self, repelling_weight = 0):
        """Computes the cost function.
        Returns
        -------
        self : object
        
        """
        epsilon = 1e-10
        # Normalization factor must be update if the cost function calculation scheme changes
        Normalization_factor = self.Training_info.training_energy_weight * sum([trainee[0] * trainee[5]**2 for trainee in self.Training_info.training_energy]) ** 0.5
        for trainee in self.Training_info.training_charge: 
            Normalization_factor += self.Training_info.training_charge_weight * sum([trainee[0] * trainee[2][ID]**2 for ID in trainee[2].keys()]) ** 0.5
        for item in self.sol_.keys():
            ##### Cost calculation: For now mean square
            ##### Computing energy
            self.energy_cost_[item] = self.Training_info.training_energy_weight * sum([trainee[0] * (trainee[1] * self.structure_energies[item][trainee[2] + '.dat'] + trainee[3] * self.structure_energies[item][trainee[4] + '.dat'] - trainee[5]) ** 2 for trainee in self.Training_info.training_energy]) / len(self.Training_info.training_energy)
            self.cost_[item] = self.energy_cost_[item]
            ##### Computing charge 
            for trainee in self.Training_info.training_charge:
                temp_sum = self.Training_info.training_charge_weight * sum([(trainee[0] * (self.structure_charges[item][trainee[1] + '.dat'][ID - 1] - trainee[2][ID])) ** 2 for ID in trainee[2].keys()]) / (len(self.Training_info.training_charge) * len(trainee[2]))
                self.charge_cost_[item] += temp_sum
                self.cost_[item] += temp_sum
            self.cost_[item] /= Normalization_factor
            ####Applying a repelling potential sum(x**2) where x is the set of optimized parameter
            if repelling_weight != 0:
                #The potential is not applied to the zeroth annealer but to others'
                if "0" in item:
                    self.reppeling_cost_[item] = 0
                else:
                    self.reppeling_cost_[item] = 0
                    for item2 in self.sol_.keys():
                        if (item2 != item):
                            distance = 0
                            for param_tuple in self.sol_[item2].param_min_max_delta.keys():
                                # divide by max(abs(max,min)) to normalize them
                                X_1 = self.sol_[item].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]/max(abs(self.sol_[item].param_min_max_delta[param_tuple]['min']),abs(self.sol_[item].param_min_max_delta[param_tuple]['max']))
                                X_2 = self.sol_[item2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]/max(abs(self.sol_[item].param_min_max_delta[param_tuple]['min']),abs(self.sol_[item].param_min_max_delta[param_tuple]['max']))
                                distance += (X_1 - X_2)**2
                            # to prevent division by zero we add epsilon
                            self.reppeling_cost_[item] += repelling_weight * 1 / (distance + epsilon)
                    self.cost_[item] +=  self.reppeling_cost_[item]
    def Individual_Energy(self, parallel="NO", p=None):
        """
        Computes the Energy for ALL of the annealers and for ALL input file
        This is a public method that is called by objective function calculator
        :return: 
        --------
        self : object
        
        """               
        if "NO" in parallel:
            ####Running lammps and python in serial  
            for item in self.sol_.keys():
                for a_file in self.lammps_file_list[item]:
                    self.structure_energies[item][a_file], self.structure_charges[item][a_file] = energy_charge(self.general_path + a_file.replace('.dat', item.replace('.reax','') + '.dat'))
        elif "YES" in parallel:
            ####Running lammps in serial but in multiple instances on each processor  
            for item in self.sol_.keys():
                list_of_files = [self.general_path + a_file.replace('.dat', item.replace('.reax','') + '.dat') for a_file in self.lammps_file_list[item]]
                #p = Pool(processes=self.number_of_processors)
                output = p.map(energy_charge, list_of_files)
                self.structure_energies[item] = dict(zip(self.lammps_file_list[item], [i[0] for i in output]))
                self.structure_charges[item] = dict(zip(self.lammps_file_list[item], [i[1] for i in output]))
                #p.close()
                #p.join()
        else:
            raise ValueError("parallel value for Individual_Energy takes YES or NO only!")

    def accept_prob(self,c_old,c_new):
        """Computes the acceptance probability.

        Returns
        -------
        self : object

        """
        ap = {item : np.exp(- (c_new[item] - c_old[item] ) / self.T) for item in c_old.keys()}
        return ap
    def clean_the_mess(self, lammpstrj = "NO"):
        """finds the single best solution.
        names : list 
        contains the name classes that we want to delete
        
        Returns
        -------
        self : object

        """
        for item in self.sol_.keys():
            command = "rm " + self.general_path + item
            os.system(command)   
        command = "rm " + self.general_path + "*.data"
        os.system(command) 
        command = "rm " + self.general_path + "*.dat"
        os.system(command)
        if "YES" in lammpstrj:
            command = "rm " + self.general_path + "*.lammpstrj"
            os.system(command)  
    def anneal(self, record_costs = "NO"):
        pass

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
from copy import deepcopy
from LAMMPS_Utils import lammps_input_creator
#from mpi4py import MPI
import random
#import time

class SA_REAX_FF(SA):
    def __init__(self,forcefield_path, output_path, params_path, Training_file, Input_structure_file, T=1, T_min=0.00001, Temperature_decreasing_factor=0.1, max_iter=50, number_of_points=1, min_style = 'cg', processors = 0):
        super().__init__(forcefield_path, output_path, params_path, Training_file, Input_structure_file, T, T_min, Temperature_decreasing_factor, max_iter, number_of_points, min_style, processors)
        # Initial forcefield (initial annealer(s))
        temp_init = REAX_FF(forcefield_path,params_path)
        temp_init.parseParamSelectionFile()
        forcefield_name = "annealer_" + str(0) + ".reax"
        self.sol_[forcefield_name] = deepcopy(temp_init)
        self.sol_[forcefield_name].ff_filePath = self.general_path + forcefield_name
        self.sol_[forcefield_name].write_forcefield(self.general_path + forcefield_name)
        self.reppeling_cost_[forcefield_name] = 0
        self.lammps_file_list = {} #dictionary keys: forcefield tag, values: list of lammps files
        self.lammps_file_list[forcefield_name] = lammps_input_creator(self.Input_structure_file, forcefield_name, self.min_style, 'reax', self.general_path)
        ### update the number of processors accordingly
        if processors > len(self.lammps_file_list[forcefield_name]):
            self.number_of_processors = len(self.lammps_file_list[forcefield_name])
            print('The requested number of processors was reduced to %d for efficiency.' %self.number_of_processors)
        ###
        self.structure_energies[forcefield_name] = {} 
        self.structure_charges[forcefield_name] = {} 
        for i in range(1, number_of_points):
            forcefield_name = "annealer_" + str(i) + ".reax"
            self.sol_[forcefield_name] = deepcopy(temp_init)
            self.sol_[forcefield_name].ff_filePath = self.general_path + forcefield_name
            self.input_generator(forcefield_name, update = "YES")
            self.reppeling_cost_[forcefield_name] = 0
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
                    self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = round(self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] + random.uniform(-1, 1) * self.sol_[forcefield_name].param_min_max_delta[param_tuple]['delta'], 4)
                    if self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] >= self.sol_[forcefield_name].param_min_max_delta[param_tuple]['min'] and self.sol_[forcefield_name].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] <= self.sol_[forcefield_name].param_min_max_delta[param_tuple]['max']:
                        break
            ####
            # save the new forcefield file
            self.sol_[forcefield_name].write_forcefield(self.general_path + forcefield_name)
        elif "NO" in update:
            pass
        else:
            raise ValueError("update value for inpute_generator takes YES or NO only!")
        # use the same name for the input structure file
        self.lammps_file_list[forcefield_name] = lammps_input_creator(self.Input_structure_file, forcefield_name, self.min_style, 'reax', self.general_path)

    def anneal(self, record_costs = "NO", repelling_weight = 0, parallel = 'NO'):
        #Automatic temperature rate control initialize
        tmp_ctrl_step = 0
        total_accept = 0
        accept_rate = 0
        ###
        # Parallezization setup
        if 'YES' in parallel:
            p = Pool(processes=self.number_of_processors)
        else:
            p = None
        ###
        self.Individual_Energy(parallel, p)
        self.cost_function(repelling_weight=repelling_weight)
        current_sol = deepcopy(self.sol_)
        cost_old = deepcopy(self.cost_)
        reppeling_cost_old = deepcopy(self.reppeling_cost_)
        if "YES" in record_costs:
            self.costs.append({temp_key:cost_old[temp_key] - reppeling_cost_old[temp_key] for temp_key in cost_old.keys()})
        while self.T > self.T_min:
            i = 1
            while i <= self.max_iter:
                for item in self.sol_.keys():
                    self.input_generator(item, update = "YES")
                self.Individual_Energy(parallel, p)
                self.cost_function(repelling_weight=repelling_weight)
                cost_new = deepcopy(self.cost_)
                ap = self.accept_prob(cost_old, cost_new)
                # counting the total number of steps for all of the annealers
                tmp_ctrl_step += 1
                for item in self.sol_.keys():
                    if ap[item] > random.random():
                        current_sol[item] = deepcopy(self.sol_[item])
                        cost_old[item] = deepcopy(cost_new[item])
                        reppeling_cost_old[item] = deepcopy(self.reppeling_cost_[item])
                        # counting total acceptance
                        total_accept += 1
                        #
                    else:
                        self.reppeling_cost_[item] = deepcopy(reppeling_cost_old[item])
                        self.cost_[item] = deepcopy(cost_old[item])
                        self.sol_[item] = deepcopy(current_sol[item])
                #  check the acceptance rates at every 100 steps
                if tmp_ctrl_step == 100:
                    accept_rate = total_accept / (self.number_of_points)
                    tmp_ctrl_step = 0
                    total_accept = 0
                    if accept_rate > 70:
                        self.alpha *= 1.2
#                    elif accept_rate < 1:
#                        self.alpha /= 1.1
                i += 1
                if "YES" in record_costs:                  
                    self.costs.append({temp_key:cost_old[temp_key] - reppeling_cost_old[temp_key] for temp_key in cost_old.keys()})
            self.T = self.T * (1 - self.alpha)
            ## debug
            print(self.T, self.cost_)
            ##
        # Parallezization setup
        if 'YES' in parallel:
            p.close()
            p.join()
        ###
        ### writing the best output
        ###removing the repelant costs first
        if repelling_weight != 0:
            for item in self.cost_.keys():
                self.cost_[item] -= self.reppeling_cost_[item]
        bestFF_key = min(self.cost_, key = self.cost_.get)
        self.single_best_solution = self.sol_[bestFF_key]
        self.sol_[bestFF_key].write_forcefield(self.general_path + bestFF_key)
        self.single_best_solution.write_forcefield(self.general_path + "bestFF.reax")