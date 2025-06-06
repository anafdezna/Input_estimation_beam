#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:04:25 2025

@author: afernandez
"""

import os
import keras as K 
import numpy as np 
from MODULES.TRAINING.keras_loadest_models import Modal_multinode_force_estimator
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration

plot_configuration()
K.backend.set_floatx('float64')
# tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
K.utils.set_random_seed(1234)

system_info_path = os.path.join("Data", "System_info_9modes.npy")
system_info = np.load(system_info_path, allow_pickle = True).item()
n_modes, Phi, m_col, c_col, k_col, uddot_true, t_vector, F_true = system_info['n_modes'], system_info['Phi'], system_info['m_col'], system_info['c_col'], system_info['k_col'], system_info['uddot_true'], system_info['t_vector'], system_info['F_true']


folder_name  = 'Jun05ES_singleloadednode5_sensorsatnodes1_400timepoints_1sensors_9modes_0.001LR_1000000epochs'
folder_path = os.path.join("Output", "Preliminary_results", folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

Problem_info = np.load(os.path.join(folder_path, "Problem_info.npy"), allow_pickle  = True).item()
ntime_points, n_epochs, LR, n_sensors = Problem_info['ntime_points'], Problem_info['epochs'], Problem_info['LR'], Problem_info['n_sensors']
n_steps   = ntime_points -1
sensor_locs = [1] 
load_locs = [5]
n_loadnodes = len(load_locs)

t_vector = K.ops.expand_dims(t_vector[0:ntime_points], axis=0) # Shape: (1, num_points_sim_example)
uddot_true = K.ops.expand_dims(K.ops.transpose(uddot_true[:, 0:ntime_points]), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.    
n_dof = uddot_true.shape[2]

model = Modal_multinode_force_estimator(ntime_points, n_modes, n_steps, n_dof,Phi,m_col, c_col, k_col, sensor_locs, load_locs, n_loadnodes)
model.build(input_shape = ())
weights_path = os.path.join(folder_path, "Weights_ae.weights.h5" )
model.load_weights(weights_path)


predictions_dict  = model.predict([t_vector, uddot_true])
uddot_pred = predictions_dict['acceleration_output']
Q_pred = predictions_dict['modal_force_output']       
    
# ACTIVATE THIS ONLY WHEN YOU ARE RESTRINCTING THE NUMBER OF SENSORS TO CONSIDER (LIMITED INSTRUMENTATION)
# Be careful here since you must mask the true uddot before plotting in order to match dimensions with the predicted vector. 
uddot_true = K.ops.take(uddot_true, sensor_locs, axis=2)

from matplotlib import pyplot as plt
i = 7
Dt = 0.001 # Your time step

# 1. Apply a style for overall aesthetics (optional, kept as in your original)
# plt.style.use('seaborn-v0_8-whitegrid')

# 2. Create the figure and axes for a single plot
fig, ax = plt.subplots(figsize=(12, 7))

# --- Create the time vector ---
# Determine the number of time points from your data's shape
# Assuming the second dimension of your data arrays is time points
num_time_points = uddot_true.shape[1]
# Create the time vector: t = [0, Dt, 2*Dt, ..., (N-1)*Dt]
time_vector = np.arange(num_time_points) * Dt
# --- End of time vector creation ---

# 3. Plot the data for the specified 'i' using the time_vector for x-axis
# True data: solid line
ax.plot(time_vector, uddot_true[0, :, i], color='springgreen', alpha=0.5, linestyle='-', linewidth=5.5, label=f'True')

# Prediction data: dashed line
ax.plot(time_vector, uddot_pred[0, :, i], color='red', linestyle='--', linewidth=2.5, label=f'Predicted')

# 4. Improve labels and title
ax.set_xlabel('Time (s)') # Updated x-axis label
ax.set_ylabel(r'ü(t) [m/s²] at node 7') # y-label remains appropriate

# 5. Customize the legend
ax.legend(frameon=True, loc='best', shadow=True)

# 6. Add a grid
ax.grid(True, linestyle=':', alpha=0.7)

# 7. Adjust tick parameters
ax.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f'response_prediction_node{i}.png'),dpi = 500, bbox_inches='tight')
plt.show()

import tensorflow as tf
## trying to see the recovery of the load: 
Phi_T = K.ops.transpose(Phi)
pseudoinv_Phi = tf.linalg.pinv(Phi_T)
F_pred = K.ops.einsum('dm,btm->btd', pseudoinv_Phi, Q_pred)

# Comparison of the estimated vs the true load at the known loaded node. (or nodes when it is the case)
i = 1
Fp   = F_pred[0,0:ntime_points,i]
Ft = F_true[i,0:ntime_points]
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(time_vector, Ft, color='orange', alpha=0.5, linestyle='-', linewidth=5.5, label=f'True')
ax.plot(time_vector, Fp, color='blue', linestyle='--', linewidth=2.5, label=f'Predicted')

# 4. Improve labels and title
ax.set_xlabel('Time [s]')
ax.set_ylabel(f' f(t) at node {i}')
# 5. Customize the legend
ax.legend(frameon=True, loc='best', shadow=True)
# 6. Add a grid
ax.grid(True, linestyle=':', alpha=0.7)
# 7. Adjust tick parameters
ax.tick_params(axis='both', which='major')

plt.tight_layout()
plt.savefig(os.path.join(folder_path, f'load_comparison_node{i}.png'),dpi = 500, bbox_inches='tight')
plt.show()
