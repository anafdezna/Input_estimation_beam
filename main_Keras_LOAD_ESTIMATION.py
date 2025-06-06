#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:00:20 2025

@author: afernandez
"""

def main():
    import os
    os.environ["KERAS_BACKEND"] = "jax" # esto yo creo que no me lo está usando
    import keras as K
    # import tensorflow.keras as K 
    import numpy as np 
    from MODULES.TRAINING.keras_loadest_models import Modal_multinode_force_estimator
    
    import time  # Import the time module
    K.backend.set_floatx('float64')
    
    if K.backend.backend() == "jax" and K.backend.floatx()=='float64':
        os.environ["JAX_ENABLE_X64"] = "True"
        print('jax con float64')
        
    # print(f"TensorFlow version: {tf.__version__}")
    # tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
    K.utils.set_random_seed(1234)

    # --- Load dataset ---
    system_info_path = os.path.join("Data", "System_info_9modes_numpy.npy")
    system_info = np.load(system_info_path, allow_pickle = True).item()
    n_modes, Phi, m_col, c_col, k_col, uddot_true, t_vector, F_true = system_info['n_modes'], system_info['Phi'], system_info['m_col'], system_info['c_col'], system_info['k_col'], system_info['uddot_true'], system_info['t_vector'], system_info['F_true']
    
    # --- Specify time and loading details
    # num_points_sim_example = F_true.shape[1] # Example value from your Newmark_beta_solver context
    ntime_points = 400 # I think this value is the key that prevents a faster training. We need to solve the newmark method sequentially for a time vector with all these points. 
    # If you have a single t_vector for prediction, add a batch dimension:
    t_vector = K.ops.expand_dims(t_vector[0:ntime_points], axis=0) # Shape: (1, num_points_sim_example)
    uddot_true = K.ops.expand_dims(K.ops.transpose(uddot_true[:, 0:ntime_points]), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.    
    
    #We need to convert the data to flotat64 if needed
    
    n_dof = uddot_true.shape[2]
    # Loaded nodes: 
    load_locs = [5] # Specify the DOF of the loaded nodes. It will be only one or two. 
    n_loadnodes = len(load_locs)
           

    # Training specifications        
    LR = 0.001
    batch_size = 1
    n_epochs = 10
    n_steps   = ntime_points -1
    # sensor_locs_tensor = [0,1,2,3,4,5,6,7,8,9,10]    # for all dOFS instrumented (there should be 9 but till I can fix this in the data generator we will keep the 11)
    # sensor_locs_tensor = [1,3,5,7,9] #for three specific DOFS instruementd
    sensor_locs_tensor = [1] # specify the location of the sensor(s).

    n_sensors =len(sensor_locs_tensor)
    
    
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10000,         # Number of epochs with no improvement after which training will be stopped
        min_delta=0.000001,    # Minimum change in the monitored quantity to qualify as an improvement
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
        verbose=0,
        mode = 'min'
    )
    
    heading ='Jun05_Shorter_ESsmall_singleloadednode5_sensorsatnodes1'
    filename = f'{heading}_{ntime_points}timepoints_{n_sensors}sensors_{n_modes}modes_{LR}LR_{n_epochs}epochs'
    folder_path = os.path.join('Output', 'Preliminary_results', filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    Problem_info = {
            'ntime_points': ntime_points,
            'epochs': n_epochs,
            'LR': LR,
            'n_sensors': n_sensors,
            'sensor_locs': sensor_locs_tensor,
            'load_locations':load_locs
            }
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
        
    model = Modal_multinode_force_estimator(ntime_points, n_modes, n_steps, n_dof, Phi, m_col, c_col, k_col, sensor_locs_tensor, load_locs, n_loadnodes)
    model.compile(optimizer=K.optimizers.RMSprop(learning_rate = LR), 
                  loss={'acceleration_output': model.udata_loss}, # Apply udata_loss to this specific output
                  loss_weights={'acceleration_output': 1.0} # Only train on accel loss
                 )

    start_time = time.time()  # Record the starting time just before training
    model_history = model.fit(x = [t_vector, uddot_true],
      y = uddot_true,
      batch_size = batch_size,
      epochs = n_epochs,
      shuffle = True,
      validation_data = ([t_vector, uddot_true], uddot_true),
      callbacks = [early_stopping])


    model.save_weights(os.path.join(folder_path, "Weights_ae.weights.h5"), overwrite = True)                        
    
    end_time = time.time()  # Record the ending time
    training_time = end_time - start_time  # Calculate the training time
    
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Save the training time
    time_info = {'training_time': training_time}
    np.save(os.path.join(folder_path, 'training_time.npy'), time_info, allow_pickle=True)
    
    history_path = os.path.join(folder_path, 'model_history.npy')
    np.save(os.path.join(history_path), model_history.history, allow_pickle=True)
    

    from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
    plot_configuration()
    loss_histories = model_history.history
    from matplotlib import pyplot as plt
    plt.plot(loss_histories['loss'], 'red')
    plt.plot(loss_histories['val_loss'], 'blue')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'Loss_evolution.png'),dpi = 500, bbox_inches='tight')
    # plt.show()
    # plt.close()

    predictions_dict  = model.predict([t_vector, uddot_true])
    uddot_pred = predictions_dict['acceleration_output']
    Q_pred = predictions_dict['modal_force_output']       
        
    uddot_true = K.ops.take(uddot_true,sensor_locs_tensor, axis=2)

    
    ## plot the response at a certain sensor (at a certain node) 
    i = 0
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
    ax.set_ylabel(r'ü(t) [m/s²] at sensor $s_{9}$') # y-label remains appropriate

    # 5. Customize the legend
    ax.legend(frameon=True, loc='best', shadow=True)

    # 6. Add a grid
    ax.grid(True, linestyle=':', alpha=0.7)

    # 7. Adjust tick parameters
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'response_prediction_sensor{i+1}.png'),dpi = 500, bbox_inches='tight')
    plt.show()

    import tensorflow as tf # TODO I need to find the expression for pseudo-inverse in keras, which I could not find yet. 
    ## trying to see the recovery of the load: 
    Phi_T = K.ops.transpose(Phi)
    pseudoinv_Phi = tf.linalg.pinv(Phi_T)
    F_pred = K.ops.einsum('dm,btm->btd', pseudoinv_Phi, Q_pred)

    # Comparison of the estimated vs the true load at the known loaded node. (or nodes when it is the case)
    i = 5
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



############## This operation allows to exeute functions in the script     
if __name__ == "__main__":
    
    main()




