import os
from REAX_FF import *
import numpy as np
from Training_data import Training_data
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
    def __init__(self,forcefield_path,output_path,params_path,Training_file,Input_structure_file,T=1,T_min=0.00001,Temperature_decreasing_factor=0.1,max_iter=50, number_of_points=1, min_style = 'cg'):
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
        self.Training_info = Training_data(output_path + Training_file)
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

    def Individual_Energy(self, parallel = "NO"):
        """
        Computes the Energy for all members of population and for all input file
        This is a private method that is called by objective function calculator
        :return: float Energy
        """
        pass

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
