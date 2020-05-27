import random
import re
from REAX_FF import *
class GA:
    """
    Genetic Algorithm optimizer.

    Parameters
    --------------
    population: float list of size params size [number of members][Selected parameters]
        starting population.

    Attributes
    --------------

    """
    def __init__(self, Train_data_filepath):
        self.init_ff=None
        self.population=[]
        self.fitness=None
        self.epochs=0
        self.mutation=0
        self.crossover=0
        self.number_of_population=20
   # def fitness_funtion(self,params,param_selection):
#        for i in self.Input_data_file_list
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


    def population_init(self,number_of_population,forcefield_path,params_path,input_member_flag= True):
        """Creates the initial self.population and assigns self.fitness to them.
        You can also add your own input members if there are population members (a local minimum that you are aware of) that have to be in the batch.
        The self.population variable have the form [][] where [Individual ID][selected params]"""
        self.number_of_population=number_of_population
        #self.population=[0]*self.number_of_population
        self.init_ff=REAX_FF(forcefield_path,params_path)
        self.init_ff.parseParamSelectionFile()
        # Keeps the input file as one of the members
        if input_member_flag is True:
            self.population.append(self.init_ff.selected_parameters_value)
            self.init_ff.write_forcefield("member_0")
        # Creating the initial population
        for i in range(1,number_of_population):
            temp_individual=[]
            for j in range(self.init_ff.param_selected):
                temp_individual.append(random.uniform(self.init_ff.param_range[j][1],self.init_ff.param_range[j][2]))
                self.init_ff.params[self.init_ff.param_selection[j][0]][self.init_ff.param_selection[j][1]][self.init_ff.param_selection[j][2]]=temp_individual[-1]
            self.init_ff.write_forcefield("member_%d" %i)
            self.population.append(temp_individual)
    def cross_over(self,parrent_1,parrent_2):
        cut_off=random.randint(0,self.init_ff.param_selected)
        daughter1=parrent_1[:cut_off]+parrent_2[cut_off:]
        daughter2=parrent_2[:cut_off]+parrent_1[cut_off:]
        return daughter1,daughter2
    def mutation(self,parrent):
        gene=random.randint(0,self.init_ff.param_selected)
        parrent[gene]=random.uniform(self.init_ff.param_range[gene][1],self.init_ff.param_range[gene][2])
        daughter=parrent
        return daughter
    def cleanup(self):
        import os
        for j in range(self.number_of_population):
            os.remove("member_%d" %j)
