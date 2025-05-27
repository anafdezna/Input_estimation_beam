#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:00:20 2025

@author: afernandez
"""


import os
import tensorflow as tf
import tensorflow.keras as K 
import numpy as np 
from MODULES.Load_estimation.keras_loadest_functions import Newmark_beta_solver
from MODULES.Load_estimation.keras_loadest_models import Modal_force_estimator
# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations. We are trying with 32 to see velocity. 

K.backend.set_floatx('float64')
print(f"TensorFlow version: {tf.__version__}")
tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
K.utils.set_random_seed(1234)

# --- Example Usage ---
system_info_path = os.path.join("MODULES", "Load_estimation", "System_info_8modes.npy")
system_info = np.load(system_info_path, allow_pickle = True).item()
n_modes, Phi, m_col, c_col, k_col, uddot_true, t_vector, F_true = system_info['n_modes'], system_info['Phi'], system_info['m_col'], system_info['c_col'], system_info['k_col'], system_info['uddot_true'], system_info['t_vector'], system_info['F_true']
# Phi, m_col, c_col, k_col, uddot_true, t_vector, F_true = tf.constant(Phi, dtype = tf.float32), tf.constant(m_col, dtype = tf.float32), tf.constant(c_col, dtype = tf.float32),  tf.constant(k_col, dtype = tf.float32),  tf.constant(uddot_true, dtype = tf.float32),  tf.constant(t_vector, dtype = tf.float32), tf.constant(F_true, dtype = tf.float32) 


# num_points_sim_example = F_true.shape[1] # Example value from your Newmark_beta_solver context
num_points_sim_example = 500
# 1. Prepare your t_vector:
# It should be a 1D NumPy array or TensorFlow tensor of shape (num_points_sim_example,)
# 2. If you have a single t_vector for prediction, add a batch dimension:
# pos0 = 850
t_vector = K.ops.expand_dims(t_vector[0:num_points_sim_example], axis=0) # Shape: (1, num_points_sim_example)
uddot_true = K.ops.expand_dims(K.ops.transpose(uddot_true[:, 0:num_points_sim_example]), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.
n_dof = uddot_true.shape[2]

# t_vector = tf.expand_dims(t_vector, axis=0) # Shape: (1, num_points_sim_example)
# uddot_true = tf.expand_dims(tf.transpose(uddot_true), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.
# we expand dimensions to add the batch size = 1 

LR = 0.0001
batch_size = 1
n_epochs = 5000
n_steps   = num_points_sim_example -1
model = Modal_force_estimator(num_points_sim_example, n_modes, n_steps, n_dof, Phi, m_col, c_col, k_col)
# model.compile(optimizer = K.optimizers.Adam(learning_rate = LR), loss = model.udata_loss)

# Assuming 'model' is an instance of your modified Modal_force_estimator
model.compile(optimizer=K.optimizers.Adam(learning_rate = LR), # Or your preferred optimizer
              loss={'acceleration_output': model.udata_loss}, # Apply udata_loss to this specific output
              loss_weights={'acceleration_output': 1.0, 'modal_force_output': 0.0} # Only train on accel loss
             )


model_history = model.fit(x = [t_vector, uddot_true],
  y = uddot_true,
  batch_size = batch_size,
  epochs = n_epochs,
  shuffle = True,
  validation_data = ([t_vector, uddot_true], uddot_true),
  callbacks = [])



from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
plot_configuration()
folder_path = os.path.join("MODULES","Load_estimation")

loss_histories = model_history.history
from matplotlib import pyplot as plt
plt.plot(loss_histories['loss'], 'red')
plt.yscale('log')
# plt.xscale('log')

predictions_dict  = model.predict([t_vector, uddot_true])
uddot_pred = predictions_dict['acceleration_output']
Q_pred = predictions_dict['modal_force_output']       
    

i =3
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
ax.set_ylabel('ü(t) [m/s²] at L/3') # y-label remains appropriate

# 5. Customize the legend
ax.legend(frameon=True, loc='best', shadow=True)

# 6. Add a grid
ax.grid(True, linestyle=':', alpha=0.7)

# 7. Adjust tick parameters
ax.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, '30Hz_result_.png'),dpi = 500, bbox_inches='tight')
plt.show()


## trying to see the recovery of the load: 
Phi_T = tf.transpose(Phi)
pseudoinv_Phi = tf.linalg.pinv(Phi_T)
F_pred = tf.einsum('dm,btm->btd', pseudoinv_Phi, Q_pred)

i = 0
Fp   = F_pred[0,0:num_points_sim_example,i]
Ft = F_true[i,0:num_points_sim_example]
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(time_vector, Ft, color='orange', alpha=0.5, linestyle='-', linewidth=5.5, label=f'True')
ax.plot(time_vector, Fp, color='blue', linestyle='--', linewidth=2.5, label=f'Predicted')

# 4. Improve labels and title
ax.set_xlabel('Time point')
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




