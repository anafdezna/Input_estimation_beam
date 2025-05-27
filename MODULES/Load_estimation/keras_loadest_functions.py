#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:38:08 2025

@author: afernandez
"""
import tensorflow.keras as K 

#%%#######################################################################################

# # --- JIT-COMPILE TRUE VERSION OF Newmark Beta Solver with XLA-compatible NaN check ---
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

#     # @tf.function(jit_compile=True)
#     def _internal_jitted_core_solver(batch_Qpred_single, batch_t_vector_single):
#         actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
#         dt_val_newmark = K.ops.cast(actual_dt, dtype='float64')
#         n_m = K.ops.shape(batch_Qpred_single)[1]

#         q_initial = K.ops.zeros((n_m, 1), dtype='float64')
#         qdot_initial = K.ops.zeros((n_m, 1), dtype='float64')
#         Q_at_t0 = K.ops.transpose(batch_Qpred_single[0:1, :])
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
#         elems_for_scan = K.ops.expand_dims(batch_Qpred_single[1:, :], axis=-1)

#         scan_results_tuple = K.ops.scan(
#             f=newmark_scan_step,
#             elems=elems_for_scan,
#             initializer=initial_scan_state,
#             name="newmark_beta_scan_loop_jit_internal"
#         )

#         qddot_scan_output = scan_results_tuple[2]
#         qddot_history_full = K.backend.concatenate([K.ops.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)

#         qddot_history_final = K.ops.transpose(K.ops.squeeze(qddot_history_full, axis=-1), perm=[1, 0])
#         uddot_physical = K.ops.matmul(Phi, qddot_history_final)
#         return uddot_physical

#     uddots = K.ops.vectorized_map(
#         lambda packed_slices: _internal_jitted_core_solver(packed_slices[0], packed_slices[1]),
#         elements=(Qpred, t_vector)
#     )

#     uddots_final_perm = K.ops.transpose(uddots, perm=[0, 2, 1])
#     return uddots_final_perm

















def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
                        beta_newmark=K.backend.constant(0.25, dtype='float64'),
                        gamma_newmark=K.backend.constant(0.50, dtype='float64')):
    
    ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
    #     The inputs are: 
    #         Qpred: tensor of predicted modal forces. Shape (batch_size=1, num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
    #         t_vector: time vector according to step Deltat and final time tf. Shape (batch_size =1, n_steps+ 1 = num_points_sim)
    #         Phi: Truncated mode shape matrix. Shape (n_dofs, n_m) . It is a rectangular matrix after truncation. 
    #         m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
    #         beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
        
    #     The outputs are: 
    #         [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
    #         The three tensors in the tuple have shape (batch_size, n_dof, num_points_sim)
    #         '''


    
    def _internal_jitted_core_solver_keras_ops(batch_Qpred_single, batch_t_vector_single):
        actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
        dt_val_newmark = K.ops.cast(actual_dt, dtype='float64')
        n_m = K.ops.shape(batch_Qpred_single)[1]
    
        q_initial = K.ops.zeros((n_m, 1), dtype='float64')
        qdot_initial = K.ops.zeros((n_m, 1), dtype='float64')
        Q_at_t0 = K.ops.transpose(batch_Qpred_single[0:1, :])

        qddot_initial = (Q_at_t0 - c_col * qdot_initial - k_col * q_initial) / m_col
    
        LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + \
                           beta_newmark * dt_val_newmark**2 * k_col
    
        # Modified loop function for keras.ops.scan
        def newmark_keras_scan_step(current_state_tuple, current_Q_force_element):
            q_current, qdot_current, qddot_current = current_state_tuple
            Q_modal_ti1 = current_Q_force_element # This is an element from elems_for_scan
    
            q_predictor = q_current + dt_val_newmark * qdot_current + \
                          (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
            qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current
            RHS_force_modal = Q_modal_ti1 - (c_col * qdot_predictor + k_col * q_predictor)
            qddot_next = RHS_force_modal / LHS_factor_modal
            qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
            q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next
            
            next_state = (q_next, qdot_next, qddot_next)
            # Since tf.scan collects the full state, we output the full next_state here as well
            output_this_step = next_state 
            return next_state, output_this_step
    
        initial_scan_state = (q_initial, qdot_initial, qddot_initial)
        # Q_force inputs for each step (excluding the Q at t0 which is used for qddot_initial)
        elems_for_scan = K.ops.expand_dims(batch_Qpred_single[1:, :], axis=-1)
    
        # Using K.ops.scan
        # The 'name' argument is not available in K.ops.scan
        final_state, scan_outputs_stacked = K.ops.scan(
            loop_fn=newmark_keras_scan_step,
            initial_state=initial_scan_state,
            sequence=elems_for_scan
        )
        
        # scan_outputs_stacked will be a tuple: (all_q_next, all_qdot_next, all_qddot_next)
        # final_state will be the state at the very last step: (q_final, qdot_final, qddot_final)
        
        # You might want to return scan_outputs_stacked, which corresponds to tf.scan's output
        return scan_outputs_stacked