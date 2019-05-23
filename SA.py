from random import random
from REAX_FF import *
import numpy as np
from Training_data import *
class SA(object):
    """ Simulated Annealing optimizer.

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

    Attributes
    --------------
    cost_ : float
        Cost calculated by the cost function, needs to be minimized.
    sol_ : float
        Solution to the cost function.
    costs : list
        List of costs over time
    """
    def __init__(self,forcefield_path,params_path,Training_file,Input_structure_file,T=1,T_min=0.00001,alpha=0.9,max_iter=50):
        self.T=T
        self.T_min=T_min
        self.alpha=alpha
        self.max_iter=max_iter
        self.Input_structure_file=Input_structure_file
        self.Training_file=Training_file
        self.init_ff=REAX_FF(forcefield_path,params_path)
        self.init_ff.parseParamSelectionFile()
        self.cost_=0
        self.sol_=self.init_ff.selected_parameters_value
        self.energies={}
    def input_generator(self):
        """Generates the next solution.

        Returns
        -------
        self : object

        """
        for j in range(self.init_ff.param_selected):
            while True:
                sol_=self.sol_[j]+ self.init_ff.param_range[j][0]*random()-self.init_ff.param_range[j][0]
                if sol_ > self.init_ff.param_range[j][1] and sol_ < self.init_ff.param_range[j][2]:
                    break
            self.sol_[j]=sol_
            self.init_ff.params[self.init_ff.param_selection[j][0]][self.init_ff.param_selection[j][1]][self.init_ff.param_selection[j][2]]=self.sol_[j]
        self.init_ff.write_forcefield("ffsolution.reax")
        lammps_input_creator(self.Input_structure_file,"ffsolution.reax")
        self._Input_data_file_list=list_of_structures(self.Input_structure_file)
        self.Training_data=Training_data(self.Training_file)
    def cost_function(self):
        error=0
        for i in self._Input_data_file_list:
            self.energies.update({i: self.__Individual_Energy(i)})
        for i in self.Training_data:
            self.cost_+=float(i[0])*(float(i[1])*self.energies[i[2]]-float(i[3])*self.energies[i[4]]-float(i[5]))**2

    def __Individual_Energy(self,lammps_input_file_name):
        """
        Computes the Energy for individual members of population and for individual input file
        This is a private method that is called by objective function calculator
        :return: float Energy
        """

        #Running lammps and python in serial
        from lammps import lammps
        lmp = lammps()
        lmp.file(lammps_input_file_name)
        etotal = lmp.get_thermo("etotal")
        #pe = lmp.get_thermo("pe")
        lmp.close()
        return etotal

    def accept_prob(self,c_old,c_new):
        """Computes the acceptance probability.

        Returns
        -------
        self : object

        """
        if -(c_new-c_old)/self.T > 0:
            ap=1.1 # to deal with
        else:
            ap = np.exp(-(c_new-c_old)/self.T)
        return ap
    def anneal(self):
        self.cost_function()
        best_sol=self.sol_
        cost_old=self.cost_
        self.costs=[cost_old]
        while self.T > self.T_min:
            i = 1
            while i <= self.max_iter:
                self.input_generator()
                self.cost_function()
                cost_new=self.cost_
                ap=self.accept_prob(cost_old, cost_new)
                if ap > random():
                    best_sol=self.sol_
                    cost_old=cost_new
                    self.costs.append(cost_new)
                else:
                    self.cost_=cost_old
                    self.sol_=best_sol
                i += 1
            self.T = self.T*self.alpha
