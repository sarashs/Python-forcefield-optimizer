#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from SA_REAX_FF import SA_REAX_FF
from GA import GA
class GA_REAX_FF(GA):
    def __init__(self, number_of_population):
        super.__init__(number_of_population)
    @classmethod
    def from_forcefield_list(cls, dict_of_forcefields, dict_of_costs): #inputs are simply SA.sol_
        number_of_population = len(dict_of_forcefields)
        GA_object = cls(number_of_population)
        GA_object.population = dict_of_forcefields
        GA_object._cost = dict_of_costs
        return GA_object
def cross_over(self, parent_ID1, parent_ID2, cross_over_point):
    """This function performs the crossover
    :param parent_ID1: string forcefield name of the first parent
    :param parent_ID2: string forcefield name of the second parent
    :param cross_over_point: int the point after(and including) which the active 
    parameters of the forcefield are swapped
    :return:
    self : object
    """
    temp_cross = 1
    for param_tuple in self.population[parent_ID1].param_min_max_delta.keys():
        if temp_cross >= cross_over_point:
            tempa = self.population[parent_ID1].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]
            tempb = self.population[parent_ID2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]]
            self.population[parent_ID1].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = tempb
            self.population[parent_ID2].params[param_tuple[0]][param_tuple[1]][param_tuple[2]] = tempa
        param_tuple += 1
def next_generation(self):
    """This function performs the mating between the population members
    :param parent_ID1: string forcefield name of the first parent
    :param parent_ID2: string forcefield name of the second parent
    :param cross_over_point: int the point after(and including) which the active 
    parameters of the forcefield are swapped
    :return:
    self : object
    """
    pass