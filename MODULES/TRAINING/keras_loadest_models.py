#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:49:00 2025

@author: afernandez
"""

import os
import tensorflow.keras as K 
from MODULES.TRAINING.keras_loadest_architectures import Fully_connected_arch_multinode
from MODULES.TRAINING.keras_loadest_functions import Newmark_beta_solver_keras

# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations
K.backend.set_floatx('float64')
K.utils.set_random_seed(1234)

class Modal_multinode_force_estimator(K.Model):
    def __init__(self, num_points_sim, n_modes, n_steps, n_dof, Phi, m_col, c_col, k_col, sensor_locs, load_locs, n_loadnodes, **kwargs):
        super().__init__(**kwargs) # Python 3 super()
        self.num_points_sim = num_points_sim
        self.n_modes = n_modes
        self.n_steps = n_steps
        self.n_dof = n_dof
        
        # Store load_locs and n_loadnodes
        self.load_locs = load_locs # list/array of indices
        self.nload_nodes = n_loadnodes # number of nodes with load

        # Convert fixed matrices and vectors to Keras tensors with appropriate dtype
        self.Phi_matrix = K.ops.convert_to_tensor(Phi, dtype='float64')
        self.m_modal = K.ops.convert_to_tensor(m_col, dtype='float64')
        self.c_modal = K.ops.convert_to_tensor(c_col, dtype='float64')
        self.k_modal = K.ops.convert_to_tensor(k_col, dtype='float64')
        
        # Sensor and load locations are indices, ensure they are in a format usable by keras.ops.take
        self.sensor_indices = K.ops.convert_to_tensor(sensor_locs, dtype='int32')
        self.load_indices = K.ops.convert_to_tensor(load_locs, dtype='int32')


        self.fullyconnected_multinodeloadestimator = Fully_connected_arch_multinode(
            num_points_sim, n_modes, n_loadnodes
        )
        
    def call(self, inputs):
        # Inputs are expected to be (t_vector, u_true_accelerations_at_all_dofs)
        # Ensure inputs also conform to the float64 policy
        t_vector_input, u_true_all_dofs = inputs
        
        t_vector = K.ops.cast(t_vector_input, dtype='float64')
        u_true_all_dofs = K.ops.cast(u_true_all_dofs, dtype='float64')

        # Predict nodal loads (e.g., at specific DOFs) using the NN
        predicted_load_magnitudes = self.fullyconnected_multinodeloadestimator(t_vector)
        nodeloads = K.ops.reshape(predicted_load_magnitudes, [-1, self.num_points_sim, self.nload_nodes])

        # Create the full force vector F(t) by mapping nodal loads to global DOFs
        # creates a matrix of shape (len(self.load_locs), self.n_dof).
        one_hot_mapping_matrix = K.ops.one_hot(
            self.load_indices,       # Pass self.load_indices as the first argument
            num_classes=self.n_dof,  # num_classes is correct
            dtype=nodeloads.dtype    # dtype is correct
        ) # Expected Shape: (len(self.load_indices) or nload_nodes, n_dof)

      
        # one_hot_mapping_matrix: (L, D) where D is n_dof
        # Result Fpred: (B, T, D)
        Fpred = K.ops.einsum('btl,ld->btd', nodeloads, one_hot_mapping_matrix)
        
        # Transform physical forces F(t) to modal forces Q(t)
        Phi_transp = K.ops.transpose(self.Phi_matrix)
        # Fpred: (batch_size, num_points_sim, n_dof)
        # Qpred: (batch_size, num_points_sim, n_modes) using einsum: 'md,btd->btm'
        Qpred = K.ops.einsum('md,btd->btm', Phi_transp, Fpred)
        self.Qpred_history = Qpred # Storing for potential access in loss/metrics

        # Solve the system of uncoupled ODEs in modal coordinates using Newmark-beta
        uddot_pred_full = Newmark_beta_solver_keras(
            Qpred,
            t_vector, # Pass the casted t_vector
            self.Phi_matrix,
            self.m_modal,
            self.c_modal,
            self.k_modal,
            self.n_steps # n_steps is num_points_sim - 1
        )
        
        
        # Example: if u_true_all_dofs is (B, T, D) and sensor_indices is (S), output is (B, T, S)
        self.uddot_true = K.ops.take(u_true_all_dofs, self.sensor_indices, axis=2)
        uddot_pred_at_sensors = K.ops.take(uddot_pred_full, self.sensor_indices, axis=2)
        
        return {"acceleration_output": uddot_pred_at_sensors, "modal_force_output": Qpred}
        
    def udata_loss(self, y_true, y_pred_acc):

        response_squared_error = K.ops.square(self.uddot_true - y_pred_acc)
        Loss_data = K.ops.mean(response_squared_error, axis = None)
        return Loss_data
    

    # get_config and from_config methods
    def get_config(self):
        config = {
            'num_points_sim': self.num_points_sim,
            'n_modes': self.n_modes,
            'n_steps': self.n_steps,
            'n_dof': self.n_dof,
            # Use keras.ops.convert_to_numpy for robust serialization
            'Phi': K.ops.convert_to_numpy(self.Phi_matrix).tolist(),
            'm_col': K.ops.convert_to_numpy(self.m_modal).tolist(),
            'c_col': K.ops.convert_to_numpy(self.c_modal).tolist(),
            'k_col': K.ops.convert_to_numpy(self.k_modal).tolist(),
            'sensor_locs': K.ops.convert_to_numpy(self.sensor_indices).tolist(), # Save sensor_locs
            'load_locs': K.ops.convert_to_numpy(self.load_indices).tolist(),     # Save load_locs
            'n_loadnodes': self.nload_nodes                                          # Save n_loadnodes
        }
        base_config = super().get_config()
        return {**base_config, **config} # Python 3.5+ dict merging

    @classmethod
    def from_config(cls, config):
        # Tensors should be recreated with the correct dtype ('float64')
        # Pop items that are handled by the constructor from config copy for **kwargs
        phi_list = config.pop('Phi', None)
        m_col_list = config.pop('m_col', None)
        c_col_list = config.pop('c_col', None)
        k_col_list = config.pop('k_col', None)
        
        # Reconstruct using the main constructor arguments explicitly
        # This ensures all required arguments are present.
        return cls(
            num_points_sim=config.get('num_points_sim'),
            n_modes=config.get('n_modes'),
            n_steps=config.get('n_steps'),
            n_dof=config.get('n_dof'),
            Phi=phi_list, # Pass the lists, constructor will convert
            m_col=m_col_list,
            c_col=c_col_list,
            k_col=k_col_list,
            sensor_locs=config.get('sensor_locs'),
            load_locs=config.get('load_locs'),
            n_loadnodes=config.get('n_loadnodes'),
            name=config.get('name'), # from base_config
            trainable=config.get('trainable', True) # from base_config
            # Add other relevant base config items if needed by your specific Keras version/setup
        )