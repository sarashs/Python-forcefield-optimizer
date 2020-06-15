import random
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
    def __init__(self,forcefield_path,output_path,params_path,Training_file,Input_structure_file,T=1,T_min=0.00001,Temperature_decreasing_factor=0.1,max_iter=50, number_of_points=1):
        self.general_output_path = output_path
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
        self.single_best_solution = []
    def input_generator(self):
        """Generates the next solution.

        Returns
        -------
        self : object

        """
        pass
    def cost_function(self):
        """Computes the cost function.
        Returns
        -------
        self : object
        
        """
        pass

    def __Individual_Energy(self, parallel = "NO"):
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
    def best_answer_calculator(self):
        """finds the single best solution.
        
        Returns
        -------
        self : object

        """
        item = min(self.cost_, key = self.cost_.get)
        self.single_best_solution = self.sol_[item]
    def anneal(self, record_costs = "NO"):
        pass
