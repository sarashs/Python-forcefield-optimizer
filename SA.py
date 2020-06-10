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
    costs : dictionary, 
        -Keys = iteration
        -Values = [list of costs]
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
        self.costs = {}
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

    def __Individual_Energy(self):
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
        ap = [np.exp(-(i_c_new-i_c_old)/self.T) for i_c_new,i_c_old in zip(c_old,c_new)]
        return ap
    def best_answer_calculator(self):
        """finds the single best solution.
        
        Returns
        -------
        self : object

        """
        item = self.cost_.index(min(self.cost_))
        self.single_best_solution = self.sol_[item]
    def anneal(self):
        #Automatic temperature rate control initialize
        tmp_ctrl_step = 0
        total_accept = 0
        accept_rate = 0
        ###
        self.cost_function()
        current_sol = self.sol_
        cost_old = self.cost_
        self.costs[0] = cost_old
        while self.T > self.T_min:
            i = 1
            while i <= self.max_iter:
                self.costs[i] = cost_old
                self.input_generator()
                self.cost_function()
                cost_new = self.cost_
                ap=self.accept_prob(cost_old, cost_new)
                # counting the total number of steps for all of the annealers
                tmp_ctrl_step += 1
                for item in range(self.number_of_points):
                    if ap[item] > random():
                        current_sol[item] = self.sol_[item]
                        cost_old[item] = cost_new[item]
                        self.costs[i][item] = cost_new[item]
                        # counting total acceptance
                        total_accept += 1
                        #
                    else:
                        self.cost_[item] = cost_old[item]
                        self.sol_[item] = current_sol[item]
                #  check the acceptance rates at every 100 steps
                if tmp_ctrl_step == 100:
                    accept_rate = total_accept / self.number_of_points
                    tmp_ctrl_step = 0
                    total_accept = 0
                    if accept_rate > 70:
                        self.alpha *= 1.2
                    elif accept_rate < 30:
                        self.alpha /= 1.2
                i += 1
            self.T = self.T * (1 - self.alpha)
