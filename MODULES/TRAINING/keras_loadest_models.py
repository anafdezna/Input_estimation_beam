#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:49:00 2025

@author: afernandez
"""


import os
import tensorflow as tf
import tensorflow.keras as K 
from MODULES.Load_estimation.keras_loadest_architectures import  Fully_connected_arch, Fully_connected_arch_singlenode
from MODULES.Load_estimation.keras_loadest_functions import Newmark_beta_solver

# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations
tf.keras.backend.set_floatx('float64')
K.utils.set_random_seed(1234)


class Modal_force_estimator(K.Model):
    def __init__(self, num_points_sim, n_modes, n_steps, n_dof, Phi, m_col, c_col, k_col, **kwargs):
        super(Modal_force_estimator, self).__init__()
        self.num_points_sim = num_points_sim
        self.n_modes = n_modes
        self.n_steps = n_steps
        self.n_dof = n_dof

        self.Phi = Phi # Fixed truncated modal matrix for n_modes mode shapes in the columns. 
        self.m_col = m_col # Fixed diagonal modal mass matrix with dims (n_modes, 1). Same for damping and stiffness.
        self.c_col = c_col
        self.k_col = k_col
        # self.fullyconnected_nodeloadestimator = Fully_connected_arch_singlenode(num_points_sim, n_modes)
        self.fullyconnected_loadestimator = Fully_connected_arch(num_points_sim, n_modes)
        
        

    def call(self, inputs):
        self.t_vector, self.u_true = inputs 
        
        # nodeF = self.fullyconnected_nodeloadestimator(self.t_vector) #shape [batch_size, num_points_sim] as only for one node we estimate the modal force. 
        # F_at_node_load_reshaped = tf.expand_dims(nodeF, axis=-1)
        # #Masking vector with zero in all locations except at node = node_load, where we assume the laod is unknonwn. 
        # one_hot_vector = tf.one_hot(
        # indices = 5, #specifies the position with the laod. 
        # depth=self.n_dof,
        # dtype=F_at_node_load_reshaped.dtype) # Match the dtype of the force predictions
        # one_hot_for_broadcast = tf.reshape(one_hot_vector, (1, 1, self.n_dof))
        # Fpred = F_at_node_load_reshaped * one_hot_for_broadcast # shape (batch_size, time_points, n_dofs)
        # # Q = Phi_transp * F(t) 
        # Phi_transp = tf.transpose(self.Phi) #shape = (n_modes, n_dof)
        # ## for the einsum: m= n_modes, d = n_dof, b  =batch_size, t = num_points_sim 
        # self.Qpred = tf.einsum('md,btd->btm', Phi_transp, Fpred)
        
          
        # # when estimatng all the modal forces (one time-domain vector for each mode)
        fullyQ = self.fullyconnected_loadestimator(self.t_vector)
        self.Qpred = tf.reshape(fullyQ, [-1, self.num_points_sim, self.n_modes])

        uddot_pred = Newmark_beta_solver(self.Qpred,self.t_vector,self.Phi, self.m_col, self.c_col, self.k_col, self.n_steps)
        
        ## TODO missing a function to retain only the sensor locations. But you can do it now for the 11 dofs. 
        # sensor_locs = [3,5,7] #indicates the node id where you can compare the response as there was a sensor measuring.  It is not a distance in meters but a ID for the DOF corresponding to the sensor emplacement. 
        # uddot_pred = uddot_pred[:,:, sensor_locs] #should be now shape (batch_size, num_sim_points, 3) for the example with 3 sensors. 
        
        return {"acceleration_output": uddot_pred, "modal_force_output": self.Qpred}
    
    def udata_loss(self, y_true, y_pred_acc):

        response_squared_error = tf.square(y_true - y_pred_acc)
        Loss_data = tf.math.reduce_mean(response_squared_error, axis = None)
        return Loss_data
    
    # get_config and from_config methods (ensure they are complete and handle all necessary attributes)
    def get_config(self):
        config = {
            'num_points_sim': self.num_points_sim,
            'n_modes': self.n_modes,
            'n_steps': self.n_steps,
            'n_dof': self.n_dof,
            'Phi': self.Phi.numpy().tolist() if tf.is_tensor(self.Phi) else self.Phi,
            'm_col': self.m_col.numpy().tolist() if tf.is_tensor(self.m_col) else self.m_col,
            'c_col': self.c_col.numpy().tolist() if tf.is_tensor(self.c_col) else self.c_col,
            'k_col': self.k_col.numpy().tolist() if tf.is_tensor(self.k_col) else self.k_col,
            'node_load_idx': self.node_load_idx 
        }
        base_config = super(Modal_force_estimator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        for key in ['Phi', 'm_col', 'c_col', 'k_col']:
            if key in config and isinstance(config[key], list):
                config[key] = tf.convert_to_tensor(config[key], dtype=tf.float32)
        # Ensure all __init__ args are present in config or handled
        return cls(**config)
  