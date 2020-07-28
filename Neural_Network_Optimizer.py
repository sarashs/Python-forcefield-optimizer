#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:19:24 2020

    Neural_Network_Optimizer base class
    
    Neural_Network_Optimizer optimizes a neural network for a given molecular structure such that it predicts the system potential energy
    for a certain range of structural parameters (such as bond length and bond angle) and certain range of forcefield parameters

@author: sarashs
"""

class Neural_Network_Optimizer(object):
    def __init__(self):
        self.molecular_system = {}
        for item in self.molecular_system.keys():
            self.molecular_system['training_data'] = {'input' : [], 'target' : []}
            self.molecular_system['test_data'] = {'input' : [], 'target' : []}
    def prepare_data(self, Number_of_data_points):
        pass
    def prepare_network(self, type_of_Network, number_of_atoms_in_system):
        pass
    def train_network(self):
        pass