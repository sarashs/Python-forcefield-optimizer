from random import random
from REAX_FF import *
import numpy as np
from Training_data import *
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
    def __init__(self,forcefield_path,params_path,Training_file,Input_structure_file,T=1,T_min=0.00001,alpha=0.9,max_iter=50, number_of_points=1):
        self.T=T
        self.T_min=T_min
        self.alpha=alpha
        self.max_iter=max_iter
        self.Input_structure_file=Input_structure_file
        self.Training_file=Training_file
        self.number_of_points = number_of_points
        ###### take them to SA REAX
        self.init_ff= [0] * number_of_points #REAX_FF(forcefield_path,params_path)
        #self.init_ff.parseParamSelectionFile()
        self.cost_= [0] * number_of_points
        ###### take them to SA REAX
        # this is a list of solutions
        self.sol_= [0] * number_of_points #self.init_ff.selected_parameters_value
        self.costs = {}
        self.single_best_solution = []
    def input_generator(self):
        """Generates the next solution.

        Returns
        -------
        self : object

        """
        pass
#        for j in range(self.init_ff.param_selected):
#            while True:
#                sol_=self.sol_[j]+ self.init_ff.param_range[j][0]*random()-self.init_ff.param_range[j][0]
#                if sol_ > self.init_ff.param_range[j][1] and sol_ < self.init_ff.param_range[j][2]:
#                    break
#            self.sol_[j]=sol_
#            self.init_ff.params[self.init_ff.param_selection[j][0]][self.init_ff.param_selection[j][1]][self.init_ff.param_selection[j][2]]=self.sol_[j]
#        self.init_ff.write_forcefield("ffsolution.reax")
#        lammps_input_creator(self.Input_structure_file,"ffsolution.reax")
#        self._Input_data_file_list=list_of_structures(self.Input_structure_file)
#        self.Training_data=Training_data(self.Training_file)
    def cost_function(self):
#        for i in self._Input_data_file_list:
#            self.energies.update({i: self.__Individual_Energy(i)})
#        for i in self.Training_data:
#            self.cost_+=float(i[0])*(float(i[1])*self.energies[i[2]]-float(i[3],lammps_input_file_name))*self.energies[i[4]]-float(i[5]))**2
        pass

    def __Individual_Energy(self):
        """
        Computes the Energy for individual members of population and for individual input file
        This is a private method that is called by objective function calculator
        :return: float Energy
        """
        pass
#        #Running lammps and python in serial
#        from lammps import lammps
#        lmp = lammps()
#        lmp.file(lammps_input_file_name)
#        etotal = lmp.get_thermo("etotal")
#        #pe = lmp.get_thermo("pe")
#        lmp.close()
#        return etotal

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
                for item in range(self.number_of_points):
                    if ap[item] > random():
                        current_sol[item] = self.sol_[item]
                        cost_old[item] = cost_new[item]
                        self.costs[i][item] = cost_new[item]
                    else:
                        self.cost_[item] = cost_old[item]
                        self.sol_[item] = current_sol[item]
                i += 1
            self.T = self.T*self.alpha
