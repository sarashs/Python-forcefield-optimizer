class GA(object):
    """
    Genetic Algorithm optimizer base class. 
    

    Parameters
    --------------
    population: list of forcefield objects
        starting population.

    Attributes
    --------------

    """
    def __init__(self, number_of_population, general_path):
        self.population = {}
        self._cost = {}
        self.general_path = general_path
        self.number_of_population = number_of_population
   # def fitness_funtion(self,params,param_selection):
#        for i in self.Input_data_file_list

    def __fitness(self):
        """
        Computes the fitness for the members of the population.
        """

        pass

    def population_init(self):
        """Creates the initial population and assigns fitness to them."""
        
        pass
    
    def cross_over(self,parrent_1, parrent_2):
        """Creates a pair of siblings from the parrents."""
        
        pass
    
    def mutation(self,parrent):
        """A mutation function can be preceived"""

"""
Created on Sat Jul 18 06:30:17 2020

@author: sarashs
This is the ReaxFF genetic algorithm. Our version of generic algorithm is different than what is typical.
I only give birth to new children when the parent became adults (lived through a simulated annealing cycle.)
    Parameters
    --------------
    population: list of forcefield objects
        starting population.

    Attributes
    --------------
"""
#from SA import SA_REAX_FF
from copy import deepcopy
from random import choices
class GA_REAX_FF(GA):
    def __init__(self, number_of_population, general_path):
        super().__init__(number_of_population, general_path)
    @classmethod
    def from_forcefield_list(cls, dict_of_forcefields, dict_of_costs, general_path): #inputs are simply SA.sol_ and SA.cost_
        number_of_population = len(dict_of_forcefields)
        GA_object = cls(number_of_population, general_path)
        GA_object.population = dict_of_forcefields
        GA_object.cost_ = dict_of_costs
        return GA_object
    def cross_over(self, parent_ID1, parent_ID2, child_ID1, child_ID2, cross_over_point, mode = "swap" ):
        """This function performs the crossover
        :param parent_ID1: string forcefield name of the first parent
        :param parent_ID2: string forcefield name of the second parent
        :param cross_over_point: int the point after(and including) which the active 
        :param mode: str swap or average
        parameters of the forcefield are swapped
        :return:
        self : object
        """
        temp_cross = 1
        for param_tuple in self.population[parent_ID1].param_min_max_delta.keys():
            if temp_cross >= cross_over_point:
                tempa = self.population[parent_ID1].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]
                tempb = self.population[parent_ID2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]
                if "swap" in mode:
                    self.population[child_ID1].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = tempb
                    self.population[child_ID2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = tempa
                elif "average" in mode:
                    self.population[child_ID1].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = round((tempb + tempa)/2, 4)
                    self.population[child_ID2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = round((tempb + tempa)/2, 4)
            temp_cross += 1
    def next_generation(self, size_of_DNA, Keep_the_best="NO", mode = "swap"):
        """This function performs the mating between the population members
        :size_of_DNA: integer
        :Keep_the_best: string YES/NO decides if the best of the previous generation must be kept
        :return:
        self : object
        """
        best_sol = list(self.population.keys())[0]
        old_generation = deepcopy(self.population) 
        accumulation = 0
        for item in self.population.keys():
            if self.cost_[item] < self.cost_[best_sol]:
                best_sol = item
            accumulation += self.cost_[item]
        raw = {item:(accumulation - self.cost_[item])/accumulation for item in self.population.keys()}
        for item1, item2 in zip(list(old_generation.keys())[:-1], list(old_generation.keys())[1:]):
            probabilities = {item:float(raw[item])/sum(list(raw.values())) for item in self.population.keys()}
            parent_ID1 = choices(list(self.population.keys()), probabilities.values())[0] #It returns a list of length 1 but we want a string
            v = list(self.population.keys())
            del probabilities[parent_ID1]
            v.remove(parent_ID1)
            parent_ID2 = choices(v, list(probabilities.values()))[0]
            self.cross_over(parent_ID1, parent_ID2, item1, item2, choices(list(range(1,size_of_DNA)))[0], mode)
            self.population[item1].write_forcefield(self.general_path + item1)
            self.population[item2].write_forcefield(self.general_path + item2)
        if "YES" in Keep_the_best:
            self.population[best_sol] = deepcopy(old_generation[best_sol])
            self.population[best_sol].write_forcefield(self.general_path + best_sol)