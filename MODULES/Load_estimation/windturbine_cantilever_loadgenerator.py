#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 18:56:51 2025

@author: afernandez
"""

import os
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

# This is the correct way to check for GPUs with TensorFlow.
# Keras uses TensorFlow as a backend, so this check applies.
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
K.utils.set_random_seed(1234)

# Assuming MODULES.POSTPROCESSING.postprocessing_tools.plot_configuration exists
# from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
# plot_configuration() # You might need to adjust this or comment out if not available

#%%######################################################################################
# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations
tf.keras.backend.set_floatx('float64')
print(f"TensorFlow version: {tf.__version__}")
# For detailed NaN/Inf checks during debugging, you can uncomment the next line:
# tf.debugging.enable_check_numerics()

# --- 1. Beam Properties and Element Definitions (Adapted for Wind Turbine Tower) ---
print("\n--- 1. Beam Properties (Wind Turbine Tower Approximation) ---")
# Define the number of finite elements and the number of modes to use in the analysis
n_elements = 10  # Number of beam elements

# Material and Geometric Properties of the beam (Approximation for a steel wind turbine tower)
E = 210e9       # Young's modulus in Pascals (Pa) - Steel
rho = 7850      # Density in kg/m^3 - Steel
L_total = 100.0 # Total length/height of the beam in meters (m) - Wind Turbine Tower Height

# Approximated cross-sectional properties for a hollow circular steel tower
# Assuming average outer diameter ~4m, thickness ~4cm
A = 0.5         # Cross-sectional area in m^2 (approx. for D_o=4m, t=4cm)
I = 0.98        # Second moment of area in m^4 (approx. for D_o=4m, t=4cm)
print(f"Approximated Tower Properties: L={L_total}m, A={A}m^2, I={I}m^4")

L_e = L_total / n_elements  # Length of each individual element

# Degrees of Freedom (DOF) calculations
n_nodes = n_elements + 1  # Total number of nodes
n_dof_full = 2 * n_nodes    # Total DOFs considering vertical displacement (v) and rotation (theta) at each node
n_dof_vertical = n_nodes  # Total DOFs if only considering vertical displacements (v_0, v_1, ..., v_{n_nodes-1})

n_m = 9 # Number of modes to retain for modal superposition
n_modes = n_m

# Print basic model information
print(f"Number of elements: {n_elements}")
print(f"Number of nodes: {n_nodes}")
print(f"Length of each element (L_e): {L_e} m")
print(f"Total vertical DOFs (n_dof_vertical): {n_dof_vertical}")
print(f"Total DOFs including rotations (n_dof_full): {n_dof_full}")


# Element Stiffness Matrix (Ke) for a 2D Euler-Bernoulli beam element
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j] for an element connecting node i and j
Ke = E * I / L_e**3 * tf.constant([
    [12,      6*L_e,    -12,      6*L_e],
    [6*L_e,   4*L_e**2, -6*L_e,   2*L_e**2],
    [-12,    -6*L_e,     12,     -6*L_e],
    [6*L_e,   2*L_e**2, -6*L_e,   4*L_e**2]
], dtype=tf.float64)

# Element Mass Matrix (Me) - Consistent Mass Matrix for a 2D beam element
# Includes terms related to Timoshenko beam theory (shear deformation and rotatory inertia)
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j]
Me = (rho * A * L_e) * tf.constant([
    [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
    [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
    [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
    [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
], dtype=tf.float64)


#%%%%%%%%%%%%%%%%%%%%%%%#############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- 2. Assemble Global Matrices (Full - including rotations) ---
K_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)
M_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)

Ke_flat = tf.reshape(Ke, [-1])
Me_flat = tf.reshape(Me, [-1])

print("\n--- 2. Assembling Full Global K and M matrices ---")
for e in range(n_elements):
    node_i = e
    node_j = e + 1
    idx_map = tf.constant([2*node_i, 2*node_i + 1, 2*node_j, 2*node_j + 1], dtype=tf.int32)
    indices_pairs = tf.reshape(tf.stack(tf.meshgrid(idx_map, idx_map, indexing='ij'), axis=-1), [-1, 2])
    K_global_full = tf.tensor_scatter_nd_add(K_global_full, indices_pairs, Ke_flat)
    M_global_full = tf.tensor_scatter_nd_add(M_global_full, indices_pairs, Me_flat)
print("Full global K and M matrices assembled.")

#%% ###################################################################################################################################
# --- 3. Extract Submatrices for Vertical DOFs Only ---
print("\n--- 3. Extracting Submatrices for Vertical DOFs ---")
vertical_dof_indices_full_system = tf.range(0, n_dof_full, 2, dtype=tf.int32)
K_unconstrained = tf.gather(tf.gather(K_global_full, vertical_dof_indices_full_system, axis=0), vertical_dof_indices_full_system, axis=1)
M_unconstrained = tf.gather(tf.gather(M_global_full, vertical_dof_indices_full_system, axis=0), vertical_dof_indices_full_system, axis=1)
print(f"Shape of K_unconstrained (vertical DOFs): {K_unconstrained.shape}")
print(f"Shape of M_unconstrained (vertical DOFs): {M_unconstrained.shape}")

# --- 4. Apply Boundary Conditions (Cantilever: Fixed at Node 0) ---
print("\n--- 4. Applying Cantilever Boundary Conditions ---")
# For a cantilever beam fixed at node 0:
# - Vertical displacement at node 0 (DOF 0 in the vertical system) is constrained.
# - Rotation at node 0 (DOF 1 in the full system) is also constrained.
# The current script applies BCs to the system of vertical DOFs.
# We constrain the vertical displacement at node 0.
constrained_dofs = tf.constant([0], dtype=tf.int32) # v_0 = 0
all_dofs = tf.range(n_dof_vertical, dtype=tf.int32)

mask_free_dofs = tf.ones(n_dof_vertical, dtype=tf.bool)
for dof in constrained_dofs:
    mask_free_dofs = tf.tensor_scatter_nd_update(mask_free_dofs, [[dof]], [False])
free_dofs = tf.boolean_mask(all_dofs, mask_free_dofs)

print(f"All vertical DOFs indices: {all_dofs.numpy()}")
print(f"Constrained DOFs indices (Cantilever BCs - v_0=0): {constrained_dofs.numpy()}")
print(f"Free DOFs indices: {free_dofs.numpy()}")

K_eff = tf.gather(tf.gather(K_unconstrained, free_dofs, axis=0), free_dofs, axis=1)
M_eff = tf.gather(tf.gather(M_unconstrained, free_dofs, axis=0), free_dofs, axis=1)
print(f"Shape of K_eff (after BCs): {K_eff.shape}")
print(f"Shape of M_eff (after BCs): {M_eff.shape}")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- 5. Solve Generalized Eigenvalue Problem on Effective (Constrained) System ---
print("\n--- 5. Solving Eigenvalue Problem ---")
try:
    M_eff_inv = tf.linalg.inv(M_eff)
    A_eig_problem = M_eff_inv @ K_eff
    eigenvalues_sq_eff, eigenvectors_eff = tf.linalg.eigh(A_eig_problem)
except Exception as e:
    print(f"Standard eigenvalue solution failed: {e}. Trying Cholesky decomposition approach.")
    L_chol = tf.linalg.cholesky(M_eff)
    L_chol_inv = tf.linalg.inv(L_chol)
    A_cholesky = L_chol_inv @ K_eff @ tf.linalg.inv(tf.transpose(L_chol))
    eigenvalues_sq_eff, y_eig = tf.linalg.eigh(A_cholesky)
    eigenvectors_eff = tf.linalg.solve(tf.transpose(L_chol), y_eig)

omegas_sq_eff = tf.maximum(eigenvalues_sq_eff, 0.0)
omegas_eff = tf.sqrt(omegas_sq_eff)

print(f"Effective system omegas_sq (rad^2/s^2, first few): {omegas_sq_eff.numpy()[:min(n_m*2, len(omegas_sq_eff))]}")
print(f"Effective system omegas (rad/s, first few): {omegas_eff.numpy()[:min(n_m, len(omegas_eff))]}")
print(f"Effective system frequencies (Hz, first few): {(omegas_eff / (2 * np.pi)).numpy()[:min(n_m, len(omegas_eff))]}")

#%% ###########################################################################################################
# --- 6. Select Modes, Normalize, and Define Modal Damping ---
print("\n--- 6. Mode Selection, Normalization, and Damping ---")
Phi_eff = eigenvectors_eff[:, :n_m]
selected_omegas_sq = omegas_sq_eff[:n_m]
selected_omegas = omegas_eff[:n_m]

M_r_diag_eff = tf.einsum('ji,jk,kl->il', Phi_eff, M_eff, Phi_eff)
norm_factors = 1.0 / tf.sqrt(tf.maximum(tf.linalg.diag_part(M_r_diag_eff), 1e-12))
Phi_eff_normalized = Phi_eff * norm_factors

Phi = tf.Variable(tf.zeros((n_dof_vertical, n_m), dtype=tf.float64))
scatter_indices = tf.expand_dims(free_dofs, axis=1)
Phi = tf.scatter_nd(scatter_indices, Phi_eff_normalized, shape=(n_dof_vertical, n_m))

print(f"Selected natural frequencies (rad/s) after BCs and normalization: {selected_omegas.numpy()}")
if tf.reduce_any(selected_omegas <= 1e-6):
    print("WARNING: Near-zero frequencies detected among selected modes. This might indicate issues (e.g. rigid body motion).")

k_q = selected_omegas_sq

# Define modal damping ratios (zeta) - Adjusted for wind turbine (steel structure)
zeta_val = tf.constant(0.02, dtype=tf.float64)  # Assume 2% damping ratio for all selected modes
zeta_modal = tf.fill((n_m,), zeta_val)
print(f"Modal damping ratio (zeta) for all modes: {zeta_val.numpy()}")

c_q = 2.0 * zeta_modal * selected_omegas

k_col = tf.reshape(k_q, (n_m, 1))
c_col = tf.reshape(c_q, (n_m, 1))
m_col = tf.ones_like(k_col)

#%%#######################################################################################
# --- 7. Define Load and Time Parameters ---
print("\n--- 7. Load and Time Parameters ---")
force_amplitude = tf.constant(10000.0, dtype=tf.float64) # Amplitude of the sinusoidal force (e.g., 10 kN)
forcing_frequency_hz = tf.constant(0.2, dtype=tf.float64) # Frequency of the force in Hz (e.g., low freq for wind)
forcing_frequency_rad = 2.0 * np.pi * forcing_frequency_hz
Omega_force = forcing_frequency_rad

t_max_sim = tf.constant(100.0, dtype=tf.float64) # Total simulation time (e.g., 100s)
dt = tf.constant(0.01, dtype=tf.float64)    # Time step for integration
load_start_time = tf.constant(1.0, dtype=tf.float64)
load_end_time = tf.constant(t_max_sim -1.0, dtype=tf.float64) # Apply load for most of the simulation

n_steps = tf.cast(tf.round(t_max_sim / dt), dtype=tf.int32)
start_time_sim = tf.constant(0.0, dtype=tf.float64)
num_points_sim = n_steps + 1
t_vector = tf.linspace(start_time_sim, t_max_sim, num_points_sim)
actual_dt = t_vector[1] - t_vector[0]
print(f"Time vector from {t_vector[0]:.3f}s to {t_vector[-1]:.3f}s with {num_points_sim} points, actual_dt = {actual_dt:.6f}s.")

# Define Load Application Point: Vertical DOF at the free end of the cantilever
# Fixed end is node 0 (DOF 0 for vertical displacement).
# Free end is node n_elements (DOF n_elements, which is n_dof_vertical - 1).
load_application_dof_index = n_dof_vertical - 1 # Vertical DOF at the tip (free end)

# Check if the load application DOF is inadvertently constrained (should not happen for cantilever free end)
if load_application_dof_index in constrained_dofs.numpy():
    raise ValueError(f"ERROR: Load application DOF {load_application_dof_index} is constrained! This should not happen for the free end of a cantilever.")
print(f"Load applied at vertical DOF index (free end): {load_application_dof_index}")

load_vector_p_physical = np.zeros(n_dof_vertical)
load_vector_p_physical[load_application_dof_index] = 1.0
p_force_location = tf.constant(load_vector_p_physical, shape=(n_dof_vertical, 1), dtype=tf.float64)

F_t_sinusoidal = force_amplitude * tf.sin(Omega_force * t_vector)
mask_load_active = tf.logical_and(t_vector >= load_start_time, t_vector <= load_end_time)
time_mask_for_load = tf.cast(mask_load_active, dtype=tf.float64)
F_t_magnitude = F_t_sinusoidal * time_mask_for_load

F_t_physical = p_force_location @ tf.expand_dims(F_t_magnitude, axis=0)
Q_t_modal = tf.transpose(Phi) @ F_t_physical

print(f"Shape of modal force Q_t_modal: {Q_t_modal.shape}") # Expected: [n_m, num_points_sim]
print("Setup complete for cantilever beam (wind turbine tower approximation).")

# --- 8. (Placeholder for Newmark Integration or other time history analysis) ---
# The rest of your script would presumably go here, performing time integration
# using the modal properties (m_col, c_col, k_col) and modal forces (Q_t_modal).

# Example: Plotting the first mode shape (vertical displacements)
plt.figure(figsize=(10, 6))
node_positions = np.linspace(0, L_total, n_dof_vertical)
plt.plot(node_positions, Phi[:, 2].numpy(), marker='o', label=f"Mode 1 ({selected_omegas[0]/(2*np.pi):.2f} Hz)")
plt.title("First Mode Shape of the Cantilever Tower (Vertical Displacements)")
plt.xlabel("Position along beam (m)")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.ylim([-tf.reduce_max(tf.abs(Phi[:,0])).numpy()*1.1, tf.reduce_max(tf.abs(Phi[:,0])).numpy()*1.1]) # Symmetrical y-axis
plt.axhline(0, color='black', lw=0.5)
plt.legend()
plt.show()

# Example: Plotting the applied force magnitude over time
plt.figure(figsize=(10, 4))
plt.plot(t_vector.numpy(), F_t_magnitude.numpy())
plt.title("Force Magnitude F(t) Applied at the Free End")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.show()