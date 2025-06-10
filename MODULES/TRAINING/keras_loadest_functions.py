#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:38:08 2025

@author: afernandez
"""
import keras as K 
# # Ensure Keras 3 features are available
# if not hasattr(K, 'ops'):
#     raise ImportError("This implementation requires Keras 3 with keras.ops module.")

#%%#######################################################################################

class NewmarkStepCell(K.layers.Layer):
    def __init__(self, m_col, c_col, k_col, beta_newmark, gamma_newmark, n_m, **kwargs):
        super().__init__(**kwargs) # This sets self.compute_dtype based on global policy or dtype in kwargs
        
        # Use self.dtype which is set by Layer.__init__ from kwargs['dtype']
        # In Keras 3, self.compute_dtype is preferred, but self.dtype is safer with tf.keras
        current_dtype = self.dtype # Or self.compute_dtype if using pure Keras 3

        # These are modal properties, constant for the cell instance
        self.m_col_p = K.ops.cast(m_col, dtype=current_dtype) # (n_m, 1)
        self.c_col_p = K.ops.cast(c_col, dtype=current_dtype) # (n_m, 1)
        self.k_col_p = K.ops.cast(k_col, dtype=current_dtype) # (n_m, 1)
        
        self.beta_newmark_p = K.ops.cast(beta_newmark, dtype=current_dtype) # scalar
        self.gamma_newmark_p = K.ops.cast(gamma_newmark, dtype=current_dtype) # scalar
        self.n_m = n_m # Store n_m for slicing in call
        
        # State consists of modal displacement, velocity, acceleration
        # Each state has shape (batch_size_for_rnn, n_m, 1)
        self.state_size = [(self.n_m, 1), (self.n_m, 1), (self.n_m, 1)]
        # Output is the next state
        self.output_size = [(self.n_m, 1), (self.n_m, 1), (self.n_m, 1)]

    def call(self, inputs_at_step_combined, states, training=None):
        # inputs_at_step_combined shape: (batch_size_for_rnn, n_m + 1)
        # It's a concatenation of Q_input_current_step and dt_for_this_item
        
        # Split the combined input
        Q_input_current_step = inputs_at_step_combined[:, :self.n_m] # Shape: (batch_size_for_rnn, n_m)
        dt_val_newmark_item = inputs_at_step_combined[:, self.n_m:]  # Shape: (batch_size_for_rnn, 1)
        
        # Current states (modal displacement, velocity, acceleration)
        # Shapes: (batch_size_for_rnn, n_m, 1)
        q_current, qdot_current, qddot_current = states[0], states[1], states[2]

        # Reshape Q_input for calculations: (batch_size_for_rnn, n_m, 1)
        Q_modal_ti1 = K.ops.expand_dims(Q_input_current_step, axis=-1)
        
        # Reshape dt_val_newmark_item for broadcasting with modal properties: (batch_size_for_rnn, 1, 1)
        dt_val = K.ops.expand_dims(dt_val_newmark_item, axis=1)

        # Expand modal properties for broadcasting with batch dimension: (1, n_m, 1)
        # These are already (n_m, 1), so expand_dims(axis=0) makes them (1, n_m, 1)
        m_col_b = K.ops.expand_dims(self.m_col_p, axis=0)
        c_col_b = K.ops.expand_dims(self.c_col_p, axis=0)
        k_col_b = K.ops.expand_dims(self.k_col_p, axis=0)

        # LHS_factor_modal calculation (effective stiffness for the increment)
        # All terms will broadcast to (batch_size_for_rnn, n_m, 1)
        LHS_factor_modal = m_col_b + \
                           self.gamma_newmark_p * dt_val * c_col_b + \
                           self.beta_newmark_p * K.ops.power(dt_val, 2) * k_col_b

        # Predictor step
        q_predictor = q_current + dt_val * qdot_current + \
                      (0.5 - self.beta_newmark_p) * K.ops.power(dt_val, 2) * qddot_current
        qdot_predictor = qdot_current + (1.0 - self.gamma_newmark_p) * dt_val * qddot_current
        
        # Corrector step
        RHS_force_modal = Q_modal_ti1 - (c_col_b * qdot_predictor + k_col_b * q_predictor)
        qddot_next = RHS_force_modal / LHS_factor_modal # Element-wise division
        
        # Update velocity and displacement
        qdot_next = qdot_predictor + self.gamma_newmark_p * dt_val * qddot_next
        q_next = q_predictor + self.beta_newmark_p * K.ops.power(dt_val, 2) * qddot_next
        
        # Return new states
        return (q_next, qdot_next, qddot_next), [q_next, qdot_next, qddot_next]

def Newmark_beta_solver_keras(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
                              beta_newmark_val=0.25, gamma_newmark_val=0.50):
    """
    Solves the system of n_m uncoupled ODEs using Newmark-beta method with Keras RNN.
    Avoids map_fn and tuple inputs to RNN by concatenating inputs for the cell.
    """
    dtype = "float64" # Consistent with your model's float policy

    # Convert inputs to Keras tensors
    Qpred_tensor = K.ops.convert_to_tensor(Qpred, dtype=dtype)
    t_vector_tensor = K.ops.convert_to_tensor(t_vector, dtype=dtype)
    Phi_tensor = K.ops.convert_to_tensor(Phi, dtype=dtype)
    # m_col, c_col, k_col will be passed to the cell and cast there.

    # Newmark parameters
    beta_newmark_k = K.ops.cast(beta_newmark_val, dtype=dtype)
    gamma_newmark_k = K.ops.cast(gamma_newmark_val, dtype=dtype)

    # Get dimensions
    batch_size = K.ops.shape(Qpred_tensor)[0]
    # num_points_sim = K.ops.shape(Qpred_tensor)[1] # This is n_steps + 1
    n_m = K.ops.shape(Qpred_tensor)[2] # Number of modes

    # Calculate dt for each batch item. Assuming dt is constant throughout one simulation.
    actual_dt_per_item = t_vector_tensor[:, 1] - t_vector_tensor[:, 0] # Shape: (batch_size,)
    
    # Expand dt to be (batch_size, n_steps, 1)
    dt_for_rnn_input_temp = K.ops.expand_dims(actual_dt_per_item, axis=-1) # (batch_size, 1)
    dt_for_rnn_input_temp = K.ops.expand_dims(dt_for_rnn_input_temp, axis=-1) # (batch_size, 1, 1)
    dt_for_rnn_input = K.ops.tile(dt_for_rnn_input_temp, [1, n_steps, 1]) # (batch_size, n_steps, 1)

    # Initial conditions (modal displacement, velocity, acceleration)
    q_initial = K.ops.zeros((batch_size, n_m, 1), dtype=dtype)
    qdot_initial = K.ops.zeros((batch_size, n_m, 1), dtype=dtype)
    
    Q_at_t0 = K.ops.transpose(Qpred_tensor[:, 0:1, :], axes=[0, 2, 1]) # (batch_size, n_m, 1)
    
    m_col_t_init = K.ops.cast(m_col, dtype=dtype)
    c_col_t_init = K.ops.cast(c_col, dtype=dtype)
    k_col_t_init = K.ops.cast(k_col, dtype=dtype)

    m_col_b_init = K.ops.expand_dims(m_col_t_init, axis=0)
    c_col_b_init = K.ops.expand_dims(c_col_t_init, axis=0)
    k_col_b_init = K.ops.expand_dims(k_col_t_init, axis=0)

    qddot_initial = (Q_at_t0 - c_col_b_init * qdot_initial - k_col_b_init * q_initial) / m_col_b_init
    initial_rnn_states = [q_initial, qdot_initial, qddot_initial]

    # Prepare input sequence for RNN
    Qpred_sequence_for_rnn = Qpred_tensor[:, 1:, :] # Shape: (batch_size, n_steps, n_m)

    # Concatenate Q_forces and dt_values along the feature axis for RNN input
    # rnn_combined_input shape: (batch_size, n_steps, n_m + 1)
    rnn_combined_input = K.ops.concatenate([Qpred_sequence_for_rnn, dt_for_rnn_input], axis=2)

    # Instantiate the RNN cell and layer
    cell = NewmarkStepCell(m_col, c_col, k_col, 
                           beta_newmark_k, gamma_newmark_k, n_m, dtype=dtype)
    rnn_layer = K.layers.RNN(cell, return_sequences=True, return_state=False, unroll = False)
    
    # Execute RNN with the combined input
    scan_results_tuple = rnn_layer(rnn_combined_input, initial_state=initial_rnn_states)
    
    qddot_history_scan = scan_results_tuple[2] 
    
    qddot_initial_expanded_for_concat = K.ops.expand_dims(qddot_initial, axis=1)
    qddot_history_full = K.ops.concatenate([qddot_initial_expanded_for_concat, qddot_history_scan], axis=1)
    
    qddot_history_final_modal = K.ops.transpose(K.ops.squeeze(qddot_history_full, axis=-1), axes=[0, 2, 1])
    
    uddots_batched = K.ops.einsum('dm,bmt->bdt', Phi_tensor, qddot_history_final_modal)
    uddots_final_perm = K.ops.transpose(uddots_batched, axes=[0, 2, 1])
    
    return uddots_final_perm






# def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
#                         beta_newmark=K.backend.constant(0.25, dtype='float64'),
#                         gamma_newmark=K.backend.constant(0.50, dtype='float64')):
    
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


    
#     def _internal_jitted_core_solver_keras_ops(batch_Qpred_single, batch_t_vector_single):
#         actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
#         dt_val_newmark = K.ops.cast(actual_dt, dtype='float64')
#         n_m = K.ops.shape(batch_Qpred_single)[1]
    
#         q_initial = K.ops.zeros((n_m, 1), dtype='float64')
#         qdot_initial = K.ops.zeros((n_m, 1), dtype='float64')
#         Q_at_t0 = K.ops.transpose(batch_Qpred_single[0:1, :])

#         qddot_initial = (Q_at_t0 - c_col * qdot_initial - k_col * q_initial) / m_col
    
#         LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + \
#                            beta_newmark * dt_val_newmark**2 * k_col
    
#         # Modified loop function for keras.ops.scan
#         def newmark_keras_scan_step(current_state_tuple, current_Q_force_element):
#             q_current, qdot_current, qddot_current = current_state_tuple
#             Q_modal_ti1 = current_Q_force_element # This is an element from elems_for_scan
    
#             q_predictor = q_current + dt_val_newmark * qdot_current + \
#                           (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
#             qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current
#             RHS_force_modal = Q_modal_ti1 - (c_col * qdot_predictor + k_col * q_predictor)
#             qddot_next = RHS_force_modal / LHS_factor_modal
#             qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
#             q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next
            
#             next_state = (q_next, qdot_next, qddot_next)
#             # Since tf.scan collects the full state, we output the full next_state here as well
#             output_this_step = next_state 
#             return next_state, output_this_step
    
#         initial_scan_state = (q_initial, qdot_initial, qddot_initial)
#         # Q_force inputs for each step (excluding the Q at t0 which is used for qddot_initial)
#         elems_for_scan = K.ops.expand_dims(batch_Qpred_single[1:, :], axis=-1)
    
#         # Using K.ops.scan
#         # The 'name' argument is not available in K.ops.scan
#         final_state, scan_outputs_stacked = K.ops.scan(
#             loop_fn=newmark_keras_scan_step,
#             initial_state=initial_scan_state,
#             sequence=elems_for_scan
#         )
        
#         # scan_outputs_stacked will be a tuple: (all_q_next, all_qdot_next, all_qddot_next)
#         # final_state will be the state at the very last step: (q_final, qdot_final, qddot_final)
        
#         # You might want to return scan_outputs_stacked, which corresponds to tf.scan's output
#         return scan_outputs_stacked