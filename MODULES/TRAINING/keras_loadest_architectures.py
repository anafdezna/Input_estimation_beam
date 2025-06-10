#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:53:45 2025

@author: afernandez
"""

import keras as K 

# architecture for multinode load (although we will start using it for one loaded node)
def Fully_connected_arch_multinode(num_points_sim, n_modes, n_loadnodes):
    input1 = K.Input(shape =(num_points_sim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense( 150*n_loadnodes, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer= K.regularizers.l2(0.001))(input1) #Intermediate layers
    lay2 = K.layers.Dense( 1000, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay2',
                      kernel_regularizer= K.regularizers.l2(0.001))(lay1) #Intermediate layers

    Qpred_output = K.layers.Dense(num_points_sim*n_loadnodes, activation = 'linear')(lay2)
    return K.Model(inputs = input1, outputs = Qpred_output)



# Load Estimator Architecture architecture: 
def Fully_connected_arch(num_points_sim, n_modes):
    input1 = K.Input(shape =(num_points_sim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(100, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer=K.regularizers.l2(0.001))(input1) #Intermediate layers
    lay2 = K.layers.Dense(200, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay1) #Intermediate layers
    lay2 = K.layers.Dense(500, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers

    Qpred_output = K.layers.Dense(num_points_sim*n_modes, activation = 'linear')(lay2)
    return K.Model(inputs = input1, outputs = Qpred_output)

# Load Estimator Architecture architecture: 
def Fully_connected_arch_singlenode(num_points_sim, n_modes):
    input1 = K.Input(shape =(num_points_sim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense( 50, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer = K.regularizers.l2(0.001))(input1) #Intermediate layers
    # lay2 = K.layers.Dense(500, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay1) #Intermediate layers
    # lay2 = K.layers.Dense(1000, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(500, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(5, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay1) #Intermediate layers

    Qpred_output = K.layers.Dense(num_points_sim, activation = 'linear')(lay1)
    return K.Model(inputs = input1, outputs = Qpred_output)