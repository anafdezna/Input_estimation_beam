#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:21:16 2025

@author: afernandez
"""

import tensorflow as tf
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
plot_configuration()


#%%#######################################################################################
# def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col,
#                                   beta_newmark_val=tf.constant(0.25, dtype=tf.float64),
#                                   gamma_newmark_val=tf.constant(0.50, dtype=tf.float64)):
#     """
#     Solves ODEs using Newmark-beta, with explicit while_loops for batch and time.
#     Designed for JIT compilation.

#     Args:
#         Qpred: (batch_size, num_points_sim, n_m)
#         t_vector: (batch_size, num_points_sim)
#         Phi: (n_dofs, n_m)
#         m_col, c_col, k_col: (n_m, 1)
#         beta_newmark_val, gamma_newmark_val: Newmark parameters.

#     Returns:
#         Tensor of accelerations (batch_size, num_points_sim, n_dofs)
#     """

#     # _internal_core_solver: Processes ONE batch item, loops over time steps.
#     # It captures Phi, m_col, c_col, k_col, beta_newmark_val, gamma_newmark_val
#     # from the enclosing scope of Newmark_beta_solver_jitted_v3.
#     def _internal_core_solver(single_Qpred, single_t_vector):
#         # single_Qpred shape: (num_points_sim, n_m)
#         # single_t_vector shape: (num_points_sim)

#         actual_dt = single_t_vector[1] - single_t_vector[0]
#         dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)

#         n_m_item = tf.shape(single_Qpred)[1]
#         num_points_sim_item = tf.shape(single_Qpred)[0]
#         num_time_steps_loop = num_points_sim_item - 1

#         q_curr_t0 = tf.zeros((n_m_item, 1), dtype=tf.float64)
#         qdot_curr_t0 = tf.zeros((n_m_item, 1), dtype=tf.float64)
#         Q_at_t0 = tf.transpose(single_Qpred[0:1, :])
#         qddot_curr_t0 = (Q_at_t0 - c_col * qdot_curr_t0 - k_col * q_curr_t0) / (m_col + 1e-12)

#         LHS_factor_modal = m_col + gamma_newmark_val * dt_val_newmark * c_col + \
#                            beta_newmark_val * dt_val_newmark**2 * k_col
#         LHS_factor_modal = LHS_factor_modal + 1e-12

#         qddot_history_ta = tf.TensorArray(
#             dtype=tf.float64,
#             size=num_points_sim_item,
#             dynamic_size=False,
#             element_shape=tf.TensorShape([None, 1]), # n_m_item can vary
#             clear_after_read=False
#         )
#         qddot_history_ta = qddot_history_ta.write(0, qddot_curr_t0)

#         time_idx = tf.constant(0, dtype=tf.int32)
#         q_loop_state = q_curr_t0
#         qdot_loop_state = qdot_curr_t0
#         qddot_loop_state = qddot_curr_t0

#         loop_vars_time = (time_idx, q_loop_state, qdot_loop_state, qddot_loop_state, qddot_history_ta)

#         def condition_time(t_idx, _q, _qd, _qdd, _ta):
#             return t_idx < num_time_steps_loop

#         def body_time(t_idx, q_prev, qd_prev, qdd_prev, ta_time_body):
#             Q_modal_ti1 = tf.transpose(single_Qpred[t_idx + 1 : t_idx + 2, :])
#             q_predictor = q_prev + dt_val_newmark * qd_prev + \
#                           (0.5 - beta_newmark_val) * dt_val_newmark**2 * qdd_prev
#             qd_predictor = qd_prev + (1.0 - gamma_newmark_val) * dt_val_newmark * qdd_prev
#             RHS_force = Q_modal_ti1 - (c_col * qd_predictor + k_col * q_predictor)
#             qdd_next = RHS_force / LHS_factor_modal
#             qd_next = qd_predictor + gamma_newmark_val * dt_val_newmark * qdd_next
#             q_next = q_predictor + beta_newmark_val * dt_val_newmark**2 * qdd_next
#             ta_time_updated = ta_time_body.write(t_idx + 1, qdd_next)
#             return (t_idx + 1, q_next, qd_next, qdd_next, ta_time_updated)

#         final_loop_vars_time = tf.while_loop(
#             cond=condition_time,
#             body=body_time,
#             loop_vars=loop_vars_time,
#             shape_invariants=(
#                 tf.TensorShape([]), tf.TensorShape([None, 1]), tf.TensorShape([None, 1]),
#                 tf.TensorShape([None, 1]), tf.TensorShape([])
#             ),
#             parallel_iterations=1
#         )

#         final_qddot_ta = final_loop_vars_time[4]
#         qddot_stacked = final_qddot_ta.stack()
#         qddot_modal_history = tf.transpose(tf.squeeze(qddot_stacked, axis=-1), perm=[1, 0])
#         uddot_physical_single_item = tf.matmul(Phi, qddot_modal_history)
#         return uddot_physical_single_item # Shape: (n_dofs, num_points_sim_item)

#     # --- Main logic of Newmark_beta_solver_jitted_v3: Iterate over batch ---
#     batch_size = tf.shape(Qpred)[0]
    
#     # Get representative shapes for TensorArray element_shape.
#     # These might be symbolic (None) if input_signature allows variability.
#     # It's safer to use more general [None, None] if these can indeed vary across calls.
#     # However, for a single call, n_dofs and num_points_sim (from Qpred) are fixed.
#     n_dofs_static = tf.shape(Phi)[0]
#     num_points_sim_static = tf.shape(Qpred)[1]

#     # TensorArray to store results for each batch item.
#     # Each element will be uddot_physical_single_item: shape (n_dofs, num_points_sim_item)
#     batch_results_ta = tf.TensorArray(
#         dtype=tf.float64,
#         size=batch_size,
#         dynamic_size=False,
#         element_shape=tf.TensorShape([None, None]), # (n_dofs, num_points_sim)
#         # To be more specific, if n_dofs_static and num_points_sim_static are suitable:
#         # element_shape=tf.TensorShape([n_dofs_static, num_points_sim_static]),
#         clear_after_read=False
#     )

#     batch_idx = tf.constant(0, dtype=tf.int32)
#     loop_vars_batch = (batch_idx, batch_results_ta)

#     def condition_batch(b_idx, _ta_results):
#         return b_idx < batch_size

#     def body_batch(b_idx, ta_results_body):
#         # Slice inputs for the current batch item
#         qpred_item = Qpred[b_idx, :, :]      # Shape: (num_points_sim, n_m)
#         t_vector_item = t_vector[b_idx, :] # Shape: (num_points_sim)

#         # Call the internal solver for this single batch item
#         uddot_item_result = _internal_core_solver(qpred_item, t_vector_item)
#         # uddot_item_result has shape (n_dofs, num_points_sim_item)

#         ta_results_updated = ta_results_body.write(b_idx, uddot_item_result)
#         return (b_idx + 1, ta_results_updated)

#     final_loop_vars_batch = tf.while_loop(
#         cond=condition_batch,
#         body=body_batch,
#         loop_vars=loop_vars_batch,
#         shape_invariants=(
#             tf.TensorShape([]),       # batch_idx
#             tf.TensorShape([])        # batch_results_ta handle
#         ),
#         parallel_iterations=1
#     )

#     final_batch_results_ta = final_loop_vars_batch[1]
#     uddots_batched = final_batch_results_ta.stack()
#     # uddots_batched shape: (batch_size, n_dofs, num_points_sim)

#     # Transpose to match original function's output: (batch_size, num_points_sim, n_dofs)
#     uddots_final_transposed = tf.transpose(uddots_batched, perm=[0, 2, 1])
    
#     return uddots_final_transposed

# It's good practice to define constants or ensure inputs have the correct dtype
# For Newmark solver, float64 is often preferred for numerical stability.
DEFAULT_DTYPE = tf.float64

def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
                        beta_newmark=tf.constant(0.25, dtype=DEFAULT_DTYPE),
                        gamma_newmark=tf.constant(0.50, dtype=DEFAULT_DTYPE)):
    ''' 
    This function solves the system of n_m uncoupled ODEs to produce the response 
    in physical coordinates for the time vector.
    
    Inputs:
        Qpred: tensor of predicted modal forces. 
               Shape (batch_size, num_points_sim, n_m), 
               where num_points_sim = n_steps + 1 = time_vector.shape[0]
        t_vector: time vector. Shape (batch_size, num_points_sim)
        Phi: Truncated mode shape matrix. Shape (n_dofs, n_m).
        m_col, c_col, k_col: Entries of the diagonal matrices of mass, damping, 
                             and stiffness as column vectors. Shape (n_m, 1).
                             Ensure these are tf.Tensor with dtype=DEFAULT_DTYPE.
        beta_newmark, gamma_newmark: Parameters for the Newmark method.
        n_steps: Number of time steps (integer).
            
    Outputs:
        uddots_final_perm: Predicted accelerations in physical coordinates.
                           Shape (batch_size, num_points_sim, n_dofs)
                           Note: Original docstring mentioned [u, udot, uddot]. 
                           This implementation currently only returns accelerations.
    '''

    # Ensure inputs that are assumed constant across batches or used in calculations
    # have the correct dtype.
    Phi = tf.cast(Phi, dtype=DEFAULT_DTYPE)
    m_col = tf.cast(m_col, dtype=DEFAULT_DTYPE)
    c_col = tf.cast(c_col, dtype=DEFAULT_DTYPE)
    k_col = tf.cast(k_col, dtype=DEFAULT_DTYPE)

    # This internal function will be JIT-compiled.
    # It processes a single item from the batch.
    @tf.function(jit_compile=True)
    def _internal_jitted_core_solver(batch_Qpred_single, batch_t_vector_single):
        # batch_Qpred_single shape: (num_points_sim, n_m)
        # batch_t_vector_single shape: (num_points_sim,)

        # Calculate dt from the time vector. Assumes a constant time step.
        # Using tf.constant for indices to aid static graph analysis.
        actual_dt = batch_t_vector_single[tf.constant(1)] - batch_t_vector_single[tf.constant(0)]
        dt_val_newmark = tf.cast(actual_dt, dtype=DEFAULT_DTYPE)
        
        # n_m is dynamically inferred from the shape of Qpred for this batch item.
        n_m = tf.shape(batch_Qpred_single)[1] # num_modes for this item

        # Initial conditions in modal coordinates
        q_initial = tf.zeros((n_m, 1), dtype=DEFAULT_DTYPE)
        qdot_initial = tf.zeros((n_m, 1), dtype=DEFAULT_DTYPE)
        
        # Q_at_t0: Modal force at t=0. Shape (n_m, 1)
        Q_at_t0 = tf.transpose(batch_Qpred_single[0:1, :]) 
        
        # Calculate initial modal acceleration: m*qddot + c*qdot + k*q = Q
        # qddot_initial = m_inv * (Q_at_t0 - c_col * qdot_initial - k_col * q_initial)
        # Assuming m_col contains mass values, so direct division is 1/m.
        qddot_initial = (Q_at_t0 - c_col * qdot_initial - k_col * q_initial) / m_col
        qddot_initial = tf.ensure_shape(qddot_initial, (None, 1)) # Aid shape inference

        # LHS factor for Newmark implicit formulation (constant for all time steps)
        # LHS_factor_modal = M + gamma*dt*C + beta*dt^2*K
        LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + \
                           beta_newmark * dt_val_newmark**2 * k_col

        # This function is executed for each step in tf.scan
        # It must consist of TensorFlow operations for JIT compatibility.
        def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step_scan):
            # previous_state_tuple: (q_current, qdot_current, qddot_current)
            # Q_force_for_current_target_step_scan: Modal force Q(t_i+1). Shape (n_m, 1)
            q_current, qdot_current, qddot_current = previous_state_tuple
            
            Q_modal_ti1 = Q_force_for_current_target_step_scan # Already (n_m, 1)

            # Predictor step
            q_predictor = q_current + dt_val_newmark * qdot_current + \
                          (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
            qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current
            
            # Corrector step
            # Solve for qddot_next: LHS_factor_modal * qddot_next = RHS_force_modal
            RHS_force_modal = Q_modal_ti1 - (c_col * qdot_predictor + k_col * q_predictor)
            qddot_next = RHS_force_modal / LHS_factor_modal # Element-wise division as matrices are diagonal
            
            qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
            q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next
            
            return (q_next, qdot_next, qddot_next)

        # Initial state for tf.scan (at t=0)
        initial_scan_state = (q_initial, qdot_initial, qddot_initial)
        
        # Elements to scan over: modal forces from t=1 to t=N
        # batch_Qpred_single[1:, :] has shape (n_steps, n_m)
        # tf.expand_dims adds a trailing dimension for compatibility with (n_m, 1) state vectors.
        elems_for_scan = tf.expand_dims(batch_Qpred_single[1:, :], axis=-1) # Shape: (n_steps, n_m, 1)

        # tf.scan performs the iterative Newmark-beta steps
        # It iterates n_steps times.
        scan_results_tuple = tf.scan(
            fn=newmark_scan_step,
            elems=elems_for_scan,
            initializer=initial_scan_state,
            name="newmark_beta_scan_loop" # JIT prefix will be added by TF
        )
        # scan_results_tuple contains (q_history, qdot_history, qddot_history) for t_1 to t_N
        # Each history tensor has shape (n_steps, n_m, 1)

        q_scan_output = scan_results_tuple[0]
        qdot_scan_output = scan_results_tuple[1]
        qddot_scan_output = scan_results_tuple[2]

        # Concatenate initial states with scan outputs to get full history [t_0, t_1, ..., t_N]
        # qddot_initial has shape (n_m, 1). Add a leading dimension to match scan output.
        qddot_initial_expanded = tf.expand_dims(qddot_initial, axis=0) # Shape (1, n_m, 1)
        qddot_history_full = tf.concat([qddot_initial_expanded, qddot_scan_output], axis=0)
        # qddot_history_full shape: (num_points_sim, n_m, 1)

        # Squeeze the trailing dimension and transpose for matrix multiplication:
        # Result shape: (n_m, num_points_sim)
        qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1, 0])
        
        # Transform modal accelerations to physical coordinates: uddot = Phi * qddot
        # Phi: (n_dofs, n_m), qddot_history_final: (n_m, num_points_sim)
        # uddot_physical: (n_dofs, num_points_sim)
        uddot_physical = tf.matmul(Phi, qddot_history_final)
        
        # If you need displacements (u) and velocities (udot) as well:
        # Similar concatenation and processing would be needed for q_scan_output and qdot_scan_output.
        # For example:
        # q_initial_expanded = tf.expand_dims(q_initial, axis=0)
        # q_history_full = tf.concat([q_initial_expanded, q_scan_output], axis=0)
        # q_history_final = tf.transpose(tf.squeeze(q_history_full, axis=-1), perm=[1, 0])
        # u_physical = tf.matmul(Phi, q_history_final)
        # ... and similarly for udot_physical.
        # Then return (u_physical, udot_physical, uddot_physical)

        return uddot_physical # Currently returning only accelerations

    # Use tf.vectorized_map to apply the JIT-compiled solver to each batch element.
    # It requires the mapped function to have inputs and outputs with consistent structures.
    # (Qpred, t_vector) are inputs to the lambda, matching `elems` for vectorized_map.
    # Qpred shape: (batch_size, num_points_sim, n_m)
    # t_vector shape: (batch_size, num_points_sim)
    uddots = tf.vectorized_map(
        lambda packed_slices: _internal_jitted_core_solver(packed_slices[0], packed_slices[1]),
        elems=(Qpred, t_vector) # Slices along axis 0 (batch dimension)
    )
    # uddots shape from vectorized_map: (batch_size, n_dofs, num_points_sim)

    # Transpose to final desired shape: (batch_size, num_points_sim, n_dofs)
    uddots_final_perm = tf.transpose(uddots, perm=[0, 2, 1])
    
    return uddots_final_perm




# def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
#                         beta_newmark=tf.constant(0.25, dtype=tf.float64),
#                         gamma_newmark=tf.constant(0.50, dtype=tf.float64)):
    
#     ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
#     #     The inputs are: 
#     #         Qpred: tensor of predicted modal forces. Shape (batch_size=1, num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
#     #         t_vector: time vector according to step Deltat and final time tf. Shape (batch_size =1, n_steps+ 1 = num_points_sim)
#     #         Phi: Truncated mode shape matrix. Shape (n_dofs, n_m) . It is a rectangular matrix after truncation. 
#     #         m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
#     #         beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
        
#     #     The outputs are: 
#     #         [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
#     #         The three tensors in the tuple have shape (batch_size, n_dof, num_points_sim)
#     #         '''

#     # @tf.function(jit_compile=True)
#     def _internal_jitted_core_solver(batch_Qpred_single, batch_t_vector_single):
#         actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
#         dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)
#         n_m = tf.shape(batch_Qpred_single)[1]

#         q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         Q_at_t0 = tf.transpose(batch_Qpred_single[0:1, :])
#         qddot_initial = (Q_at_t0 - c_col * qdot_initial - k_col * q_initial) / m_col

#         LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + \
#                             beta_newmark * dt_val_newmark**2 * k_col

#         def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step_scan):
#             q_current, qdot_current, qddot_current = previous_state_tuple
#             Q_modal_ti1 = Q_force_for_current_target_step_scan
#             q_predictor = q_current + dt_val_newmark * qdot_current + \
#                             (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
#             qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current
#             RHS_force_modal = Q_modal_ti1 - (c_col * qdot_predictor + k_col * q_predictor)
#             qddot_next = RHS_force_modal / LHS_factor_modal
#             qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
#             q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next
#             return (q_next, qdot_next, qddot_next)

#         initial_scan_state = (q_initial, qdot_initial, qddot_initial)
#         elems_for_scan = tf.expand_dims(batch_Qpred_single[1:, :], axis=-1)

#         scan_results_tuple = tf.scan(
#             fn=newmark_scan_step,
#             elems=elems_for_scan,
#             initializer=initial_scan_state,
#             name="newmark_beta_scan_loop_jit_internal"
#         )

#         qddot_scan_output = scan_results_tuple[2]
#         qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)


#         qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1, 0])
#         uddot_physical = tf.matmul(Phi, qddot_history_final)
#         return uddot_physical

#     uddots = tf.vectorized_map(
#         lambda packed_slices: _internal_jitted_core_solver(packed_slices[0], packed_slices[1]),
#         elems=(Qpred, t_vector)
#     )

#     uddots_final_perm = tf.transpose(uddots, perm=[0, 2, 1])
#     return uddots_final_perm

















#################################################################################################################################################################################3
#####################################################################################################################################################################
#################################################################################################################################################################################3
#################################################################################################################################################################################3
#################################################################################################################################################################################3

## old without jit-friendly form 
# def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps, beta_newmark = tf.constant(0.25, dtype=tf.float64), gamma_newmark = tf.constant(0.50, dtype=tf.float64)):
#     ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
#     The inputs are: 
#         Qpred: tensor of predicted modal forces. Shape (batch_size=1, num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
#         t_vector: time vector according to step Deltat and final time tf. Shape (batch_size =1, n_steps+ 1 = num_points_sim)
#         Phi: Truncated mode shape matrix. Shape (batch_size, n_dofs, n_m) . It is a rectangular matrix after truncation. 
#         m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
#         beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
    
#     The outputs are: 
#         [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
#         The three tensors in the tuple have shape (batch_size, n_dof, num_points_sim)
#         '''
#     def batch_NewmarkBeta(batch_Qpred, batch_t_vector):
#         actual_dt = batch_t_vector[1] - batch_t_vector[0]                  # Actual time step used by linspace
#         dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)   
#         n_m = Phi.shape[1]    
#         # Initial conditions
#         q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         Q_at_t0 = tf.transpose(batch_Qpred[0:1, :]) # since Qpred has shape (num_points_sim, n_m)
#         qddot_initial = Q_at_t0 - c_col * qdot_initial - k_col * q_initial
        
#         # LHS Factor
#         LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + beta_newmark * dt_val_newmark**2 * k_col

#         # @tf.function(jit_compile=True)
#         def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step):
#             # previous_state_tuple: (r_prev, rdot_prev, rddot_prev) from time t_i
#             # Q_r_force_for_current_target_step: Modal force Q_r(t_{i+1}), shape [n_m, 1]

#             q_current, qdot_current, qddot_current = previous_state_tuple

#             # Predictor terms based on state at time t_i
#             q_predictor = q_current + dt_val_newmark * qdot_current + \
#                           (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
#             qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current

#             # Calculate effective modal force for solving acceleration at t_{i+1}
#             RHS_force_modal = Q_force_for_current_target_step - \
#                               c_col * qdot_predictor - k_col * q_predictor

#             # Solve for modal acc^eleration at time t_{i+1}
#             qddot_next = RHS_force_modal / LHS_factor_modal

#             # Correct modal velocity and displacement at time t_{i+1}
#             qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
#             q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next

#             # The function returns the new state (r_next, rdot_next, rddot_next)
#             # This tuple will be passed as 'previous_state_tuple' to the next iteration and also collected by tf.scan as the output for this step.
#             return (q_next, qdot_next, qddot_next)

#         # Initial state for the scan (state at t=0)
#         initial_scan_state = (q_initial, qdot_initial, qddot_initial)

#         # Prepare the sequence of modal forces for tf.scan. (Remove the initial time t= 0 since that is the initial condition. Then we will add it). 
#         elems_for_scan = tf.expand_dims(batch_Qpred[1:, :], axis = -1) 
#         # It seems that the expanded additional dimension is required for the structure management of tf.scan. 

#         # The output will be a tuple of three tensors:  (stacked_q_next, stacked_qdot_next, stacked_qddot_next)
#         # Each tensor will have shape [n_steps, n_m, 1], representing states from t_1 to t_{n_steps}
#         scan_results_tuple = tf.scan(
#             fn=newmark_scan_step,
#             elems=elems_for_scan,
#             initializer=initial_scan_state,
#             name="newmark_beta_solver"
#         )

#         # Unpack results from scan. These are histories from t_1 to t_{n_steps} (n_steps items)
#         # q_scan_output = scan_results_tuple[0]    # Shape [n_steps, n_m, 1] # We do not use it in the loss
#         # qdot_scan_output = scan_results_tuple[1] # Shape [n_steps, n_m, 1] # We do not use it in the loss
#         qddot_scan_output = scan_results_tuple[2]# Shape [n_steps, n_m, 1]
#         qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)

#         # Reshape results to [n_m, num_points_sim] for easier use (same as original code)
#         # Current shape is [num_points_sim, n_m, 1]
#         qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1,0])
#         # Transform modal accelerations back to physical accelerations
#         uddot = tf.matmul(Phi, qddot_history_final) # Shape [n_dof_vertical, num_points_sim]
#         return uddot
    
#     uddots = tf.vectorized_map(
#         lambda args: batch_NewmarkBeta(*args),
#         (Qpred, t_vector),)
    
#     uddots = tf.transpose(uddots, perm = [0,2,1])
    
#     return uddots



# # no hace falta la layer

# # Define a layer that applies the function for newmark calculation 
# class Newmark_beta_solver_layer(tf.keras.layers.Layer):
#     def __init__(self, num_points_sim, n_modes, n_steps, Phi, m_col, c_col, k_col, **kwargs):
#         super(Newmark_beta_solver_layer, self).__init__(**kwargs)
#         self.num_points_sim = num_points_sim
#         self.n_modes = n_modes
#         self.n_steps = n_steps
#         self.Phi = Phi
#         self.m_col = m_col
#         self.c_col = c_col 
#         self.k_col = k_col
        
        
#     def call(self,inputs): 
#         Qpred, t_vector = inputs         
#         uddot_pred = Newmark_beta_solver(Qpred, t_vector, self.Phi, self.m_col, self.c_col, self.k_col, self.n_steps)
        
#         return uddot_pred
        



def Newmark_beta_solver_singleb(Qpred, Phi, m_col, c_col, k_col, t_vector, n_steps, beta_newmark = tf.constant(0.25, dtype=tf.float64), gamma_newmark = tf.constant(0.50, dtype=tf.float64)):
    ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
    The inputs are: 
        Qpred: tensor of predicted modal forces. Shape (num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
        Phi: Truncated mode shape matrix. Shape ( n_dofs, n_m) . It is a rectangular matrix after truncation. 
        m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
        t_vector: time vector according to step Deltat and final time tf. Shape (n_steps+ 1 = num_points_sim)
        beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
    
    The outputs are: 
        [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
        The three tensors in the tuple have shape (n_dof, num_points_sim)
        '''
    
    actual_dt = t_vector[1] - t_vector[0]                  # Actual time step used by linspace
    dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)   
    n_m = Phi.shape[1]    
    # Initial conditions
    q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
    qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
    Q_at_t0 = tf.transpose(Qpred[0:1, :]) # since Qpred has shape (num_points_sim, n_m)
    qddot_initial = Q_at_t0 - c_col * qdot_initial - k_col * q_initial
    
    # LHS Factor
    LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + beta_newmark * dt_val_newmark**2 * k_col
    # if tf.reduce_any(LHS_factor_modal <= 1e-12):
    #     tf.print(f"ERROR: LHS_factor_modal is too small or zero")
    
    def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step):
        # previous_state_tuple: (r_prev, rdot_prev, rddot_prev) from time t_i
        # Q_r_force_for_current_target_step: Modal force Q_r(t_{i+1}), shape [n_m, 1]

        q_current, qdot_current, qddot_current = previous_state_tuple

        # Predictor terms based on state at time t_i
        q_predictor = q_current + dt_val_newmark * qdot_current + \
                      (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
        qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current

        # Calculate effective modal force for solving acceleration at t_{i+1}
        RHS_force_modal = Q_force_for_current_target_step - \
                          c_col * qdot_predictor - k_col * q_predictor

        # Solve for modal acc^eleration at time t_{i+1}
        qddot_next = RHS_force_modal / LHS_factor_modal

        # Correct modal velocity and displacement at time t_{i+1}
        qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
        q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next

        # The function returns the new state (r_next, rdot_next, rddot_next)
        # This tuple will be passed as 'previous_state_tuple' to the next iteration and also collected by tf.scan as the output for this step.
        return (q_next, qdot_next, qddot_next)

    # Initial state for the scan (state at t=0)
    initial_scan_state = (q_initial, qdot_initial, qddot_initial)

    # Prepare the sequence of modal forces for tf.scan. (Remove the initial time t= 0 since that is the initial condition. Then we will add it). 
    elems_for_scan = tf.expand_dims(Qpred[1:, :], axis = -1) 
    # It seems that the expanded additional dimension is required for the structure management of tf.scan. 

    tf.print(f"Starting Newmark integration using tf.scan for  {n_steps} steps ...")
    # The output will be a tuple of three tensors:  (stacked_q_next, stacked_qdot_next, stacked_qddot_next)
    # Each tensor will have shape [n_steps, n_m, 1], representing states from t_1 to t_{n_steps}
    scan_results_tuple = tf.scan(
        fn=newmark_scan_step,
        elems=elems_for_scan,
        initializer=initial_scan_state,
        name="newmark_beta_solver"
    )

    # Unpack results from scan. These are histories from t_1 to t_{n_steps} (n_steps items)
    # q_scan_output = scan_results_tuple[0]    # Shape [n_steps, n_m, 1] # We do not use it in the loss
    # qdot_scan_output = scan_results_tuple[1] # Shape [n_steps, n_m, 1] # We do not use it in the loss
    qddot_scan_output = scan_results_tuple[2]# Shape [n_steps, n_m, 1]
    qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)
    tf.print("Newmark integration with tf.scan finished.")

    # Optional: Check for NaNs in the final results (as an example)
    if tf.reduce_any(tf.math.is_nan(qddot_history_full)):
        tf.print("NaN detected in rddot_history_full. Check parameters and matrices.")

    # Reshape results to [n_m, num_points_sim] for easier use (same as original code)
    # Current shape is [num_points_sim, n_m, 1]
    qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1,0])
    # Transform modal accelerations back to physical accelerations
    uddot = tf.matmul(Phi, qddot_history_final) # Shape [n_dof_vertical, num_points_sim]
    return uddot


    