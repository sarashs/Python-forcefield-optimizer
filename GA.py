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
