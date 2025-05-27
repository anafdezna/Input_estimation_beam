#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:51:11 2024

@author: afernandez
"""

import os
import numpy as np
import json
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
import matplotlib.patches as mpatches
from tensorflow.keras import optimizers, models, utils
from MODULES.TRAINING.full_multivariate_models import My_InverseForward, ForwardModel
from MODULES.POSTPROCESSING.multivariate_results_tools import save_deterministic_test_solutions , plot_test_contourplots, plot_test_Dataloss_contourplots
from MODULES.POSTPROCESSING.postprocessing_tools import custom_plot_loss
from MODULES.POSTPROCESSING.postprocessing import plot_configuration
tfd = tfp.distributions
plot_configuration()
# # 

# import seaborn as sns
# sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
# sns.set(style = "white")
#####################################################################################################################
# foldername = '13May_Roll_2Props_0.036Beta_InverseForwardELBOloss_2mixtures_5gaussians_50Samples_1e-05LR_120epochs'
# # foldername = '30May_Roll_2Props_0.07Beta_InverseForwardELBOloss_2mixtures_5gaussians_50Samples_1e-05LR_200epochs'
def load_trained_model():
    
    # foldername = 'P10_Test_Cond_11Dec_diag_Roll_2Props_0.08Beta_InverseForward_3Gaussians_1Samples_0.0001LR_3000epochs_8192batch'
    foldername = 'P100_Test_Cond_16Dec_full_Roll_2Props_0.08Beta_InverseForward_9Gaussians_1Samples_0.0001LR_10000epochs_8192batch'
    # folder_path = os.path.join('Output', 'New_Diag_solutions', foldername)

    folder_path = os.path.join('Output', foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # # LOAD THE DATASETS AND PROBLEM INFORMATION (TRAINING, ENCODING ...):
    
    Problem_info = np.load(os.path.join(folder_path, 'Problem_info.npy'), allow_pickle = True).item()
    input_dim_encoder, input_dim_decoder, num_mixtures, num_gaussians, num_samples_per_mixture, LR, selected_features, beta = Problem_info['input_dim_enc'], Problem_info['input_dim_dec'], Problem_info['n_mixtures'], Problem_info['n_gaussians'], Problem_info['n_samples'], Problem_info['LR'], Problem_info['selected_features'], Problem_info['beta']
    Data_info = np.load(os.path.join(folder_path, 'Data_info.npy'), allow_pickle = True).item()
    u_val, r_val, u_test, r_test, p_val, p_test = Data_info['u_val'], Data_info['r_val'], Data_info['u_test'], Data_info['r_test'], Data_info['p_val'], Data_info['p_test']
    ## LOAD AND BUILD THE TRAINED AUTOENCODER MODEL 
    output_dim = u_val.shape[1]
    forward_path = os.path.join("Output","23Apr_2Prop_Forward", "model_forward_2mix")
    model_forward = tf.saved_model.load(forward_path)
    
    num_samples_per_mixture  = 1 # you can fix this to one for the diagnostis step 
    full_cov = True
    s_lb = 0.000001
    s_ub = 1.
    model = My_InverseForward(input_dim_encoder, input_dim_decoder, output_dim, num_mixtures, num_gaussians, num_samples_per_mixture, model_forward, selected_features, beta, full_cov, s_ub, s_lb)
    opt = optimizers.Adam(learning_rate = LR)
    model.build(input_shape = ())
    ae_weights_path = os.path.join(folder_path, "Weights_ae.h5" )
    model.load_weights(ae_weights_path)
    
    ################ plotting loss functions to compare with Diagonal covariance:
    history_path = os.path.join(folder_path, 'model_history.npy')
    model_history  = np.load(history_path, allow_pickle = True)
    return  folder_path, u_test, r_test, p_test, model, model_history

def compare_losses(model_history, folder_path):
    losses = model_history.item()
    names = [ 'loss', 'Mixture_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Mixture_dens_term', 'val_Conditional_likelihood_term']

    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(losses[names[1]], label=r'$\mathcal{L}_{GMM}$', color='tomato', linestyle='-', linewidth=3.)
    plt.plot(losses[names[2]], label=r'$\mathcal{L}_{Data}$', color='blue', linestyle='-', linewidth=3.)
    plt.plot(losses[names[0]], label=r'$\mathcal{L}_{ELBO}$', color='black', linestyle='-', linewidth=3.)
    print(losses[names[0]][-1])
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(0,0.1)
    # Add a legend
    plt.legend(loc='upper right', fontsize=22)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'All_losses.png'),dpi = 500, bbox_inches='tight')
    plt.show()



def compare_GMM_losses(model_history, folder_path):
    losses = model_history.item()
    names = [ 'loss', 'Mixture_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Mixture_dens_term', 'val_Conditional_likelihood_term']   
    # diag_history_path = os.path.join("Output", "New_Diag_solutions", "P51_Test_Cond_16Dec_diag_Roll_2Props_0.075Beta_InverseForward_6Gaussians_1Samples_1e-05LR_20000epochs_8192batch", "loss_autoencoder.csv")
    diag_history_path = os.path.join("Output", "P101_Test_Cond_16Dec_diag_Roll_2Props_0.08Beta_InverseForward_9Gaussians_1Samples_0.0001LR_10000epochs_8192batch", "loss_autoencoder.csv")
    # diag_history_path = os.path.join("Output", "P102_Test_Cond_16Dec_diag_Roll_2Props_0.055Beta_InverseForward_3Gaussians_1Samples_0.0001LR_20000epochs_8192batch", "loss_autoencoder.csv")
    import pandas as pd
    diag_losses = pd.read_csv(diag_history_path).to_numpy()
     
    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(diag_losses[:,1], label=r'$\mathcal{L}_{GMM}^{diag}$', color='tomato', linestyle='-', linewidth=3.)
    plt.plot(losses[names[1]], label=r'$\mathcal{L}_{GMM}^{full}$', color='maroon', linestyle='-', linewidth=3.)
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')
    # plt.yscale('log')
    # Add a legend
    plt.legend(loc='lower right', fontsize=22)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'Comparing_LGMM.png'),dpi = 500, bbox_inches='tight')
    plt.show()

def compare_Data_losses(model_history, folder_path):
    losses = model_history.item()
    names = [ 'loss', 'Mixture_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Mixture_dens_term', 'val_Conditional_likelihood_term']   
    # diag_history_path = os.path.join("Output", "New_Diag_solutions", "P51_Test_Cond_16Dec_diag_Roll_2Props_0.075Beta_InverseForward_6Gaussians_1Samples_1e-05LR_20000epochs_8192batch", "loss_autoencoder.csv")
    diag_history_path = os.path.join("Output", "P101_Test_Cond_16Dec_diag_Roll_2Props_0.08Beta_InverseForward_9Gaussians_1Samples_0.0001LR_10000epochs_8192batch", "loss_autoencoder.csv")
    import pandas as pd
    diag_losses = pd.read_csv(diag_history_path).to_numpy()
    
    
    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(diag_losses[:,2], label=r'$\mathcal{L}_{\mathcal{M}}^{diag}$', color='lime', linestyle='-', linewidth=3.0)
    plt.plot(losses[names[2]], label=r'$\mathcal{L}_{\mathcal{M}}^{full}$', color='darkolivegreen', linestyle='-', linewidth=3.0)

    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')
    plt.yscale('log')
    # Add a legend
    plt.legend(loc='upper right', fontsize=22)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'Comparing_LData.png'),dpi = 500, bbox_inches='tight')
    plt.show()

        

def plot_test_contourplots_gpu(model, selected_features, u_test, r_test, p_test, num_mixtures, num_gaussians, full_cov, folder_path):
    ## Load the deterministic solutions according to the selected DOFs 
    ### UNCOMMENT THE LINE THAT CORRESPONDS TO THE PATH OF THE DOFS USED (3 OR 1)
    # deterministic_path = os.path.join("Results","Results08May_RollPitchYaw_2Props_1e-15Beta_InverseForwardDataLoss_2mixtures_5gaussians_10Samples_1e-05LR_500epochs" ) ### FOR THE THRE DOFs
    deterministic_path = os.path.join("Results","Results08May_Roll_2Props_1e-15Beta_InverseForwardDataloss_2mixtures_5gaussians_10Samples_1e-05LR_500epochs" ) ### FOR ROLL ONLY
    det_solutions = np.load(os.path.join(deterministic_path, "deterministc_test_solutions.npy") )
    
    positions = [12, 4648, 7274, 2365, 243, 5015, 2458, 9659, 35,410,70,2564] # LATEST SELECTION: IF WE CHANGE THE FIFTH CASE FROM 9657 TO 243.

    deterministic_solutions_file = "Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.001Beta_InverseForward_1Gaussians_1Samples_0.0001LR_5000epochs_8192batch"
    det_path = os.path.join("Output", deterministic_solutions_file, "Test_results_info.npy")
    deterministic_test_sols = np.load(os.path.join(det_path), allow_pickle = True).item()
    det_z_test = deterministic_test_sols['test_p_predicted']
    det_solutions = det_z_test[positions,:]
    
    ######### WARNING: USE THIS ONLY IF WE CHANGE THE TEST CASE 5 FOR THE NEW ONE. OTHERWISE  COMMENT BOTH LINES:
    # det_solutions[4,:] = [0.1267,0.61646] ## For RollPitchYaw: We have changed one test case for position 243 (the fifth case) becasue of a strange discrepancy between truth and deterministic that was greater in the case of RollPitchYaw rather than with only Roll :() 
    # det_solutions[4,:] = [0.1048003,0.6897715] ## For Roll: We have changed one test case for position 243 (the fifth case) becasue of a strange discrepancy between truth and deterministic that was greater in the case of RollPitchYaw rather than with only Roll :() 
    
    ## SELECTED  TEST CASES: 
    # positions = [12, 4648, 7274, 2365, 9657, 5015, 2458, 9659] # OLD SELECTION
    # positions = [12, 4648, 7274, 2365, 243, 5015, 2458, 9659] # NEW SELECTION: IF WE CHANGE THE FIFTH CASE FROM 9657 TO 243.


    ########################################################################################################################################################################################################################33
    ############################ PLOTTING THE RESULTS DURING TESTING ########################################   
    inverse_model = model.Encoder_model
    GMMprops_test = inverse_model.predict([u_test[:,selected_features], r_test])
    

    test_means, test_sigmas_diag, test_weights, test_phi_angles = tf.split(GMMprops_test, [num_mixtures * num_gaussians,
                                                    num_mixtures * num_gaussians,
                                                    num_gaussians,
                                                    num_gaussians], axis=-1)
    #Reshaping required to accomodate for the batch size (, num_mixtures*num_gaussians) during training
    test_means = tf.reshape(test_means, (-1, num_mixtures, num_gaussians))
    test_sigmas_diag = tf.reshape(test_sigmas_diag, (-1, num_mixtures, num_gaussians))
    test_weights = tf.reshape(test_weights, (-1, num_gaussians))
    phi_angles = tf.reshape(test_phi_angles, (-1, num_gaussians))
    
    rot_matrix = tf.transpose(tf.stack([[tf.cos(phi_angles), -tf.sin(phi_angles)], [tf.sin(phi_angles), tf.cos(phi_angles)]], axis=0), perm= [2,3,0,1])
    sigmas_diagonal = tf.transpose(test_sigmas_diag, perm = [0,2,1]) #We do this reshaping to see better the operations
    RS_matrices = tf.einsum('BGDK,BGK -> BGDK', rot_matrix, sigmas_diagonal) # Multiply the rotation matrix times the scaling matrix to obtain the linear Transformation mtrix T
    RS_transp_matrices = tf.einsum('BGDK->BGKD', RS_matrices) #Obtain the transpose of the transformation matrix T = RS
    Sigma_matrices = tf.einsum('BGDK, BGKL -> BGDL', RS_matrices, RS_transp_matrices) 
    LT_matrices = tf.linalg.cholesky(Sigma_matrices) #Apply Cholesk
    
        
    for i in range(len(positions)):
        pos  = positions[i]            
        diag_components = []
        for j in range(num_gaussians):
            component = tfp.distributions.MultivariateNormalDiag(loc=test_means[pos,:, j], scale_diag=test_sigmas_diag[pos, :, j])
            diag_components.append(component)
            
        full_components = [] 
        for j in range(num_gaussians):
            LT_matrix = LT_matrices[pos,j,:,:]
            component2 = tfp.distributions.MultivariateNormalTriL(loc= test_means[pos,:,j], scale_tril=LT_matrix)
            full_components.append(component2)

            
        if full_cov: 
            Components = full_components
        else: 
            Components = diag_components 

        cat = tfp.distributions.Categorical(probs=test_weights[pos,:])
        mixture = tfp.distributions.Mixture(cat=cat, components=Components)     
        
        num_points = 200000
        x_values = np.random.uniform(0, 1, num_points)
        y_values = np.random.uniform(0, 1, num_points)
        mesh_points = np.column_stack([x_values, y_values])
        
        # Evaluate the probability at each point on the grid
        probabilities = mixture.prob(mesh_points)
        lb = 0.0005
        ub = 1500.0
        filtered_indices = (probabilities > lb) & (probabilities < ub)
        
        mesh_points = mesh_points[filtered_indices]
        probabilities = probabilities[filtered_indices]
        # Create a smooth contour plot
        plt.figure(figsize=(10, 8))
        plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], probabilities, cmap='viridis', levels = 100)
        plt.colorbar(label='density')
        plt.scatter(p_test[pos,0], p_test[pos,1], c='red', marker='*', edgecolor = 'k', s=200, label = 'True solution')
        plt.scatter(det_solutions[i,0], det_solutions[i,1], c='red', edgecolor = 'k', s=100, marker='s', label='Deterministic solution')
        plt.legend()
        # Customize legend with background and text color
        # legend = plt.legend(frameon=True, loc='lower left')
        legend = plt.legend(frameon=True, loc='upper right')
        
        legend.get_frame().set_facecolor('lightgray')  # Set background color
        legend.get_frame().set_edgecolor('black')      # Set edge color
        
        # Set the color of the legend text
        for text in legend.get_texts():
            text.set_color("black")  # Set text color
            
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(folder_path, 'ContourPlot_Density_case_'+ str(i)+'.png'), dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close()
        
        

def plot_test_Dataloss_contourplots(input_dim_decoder, u_test, r_test, p_test, num_mixtures, num_gaussians, num_samples_per_mixture, selected_features, positions, det_solutions, folder_path ):
    foldername = 'Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.001Beta_InverseForward_1Gaussians_1Samples_0.0001LR_5000epochs_8192batch'
    folder_path = os.path.join('Output', foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    Problem_info = np.load(os.path.join(folder_path, 'Problem_info.npy'), allow_pickle = True).item()
    input_dim_encoder, input_dim_decoder, num_mixtures, num_gaussians, num_samples_per_mixture, LR, selected_features, beta = Problem_info['input_dim_enc'], Problem_info['input_dim_dec'], Problem_info['n_mixtures'], Problem_info['n_gaussians'], Problem_info['n_samples'], Problem_info['LR'], Problem_info['selected_features'], Problem_info['beta']
    Data_info = np.load(os.path.join(folder_path, 'Data_info.npy'), allow_pickle = True).item()
    u_val, r_val, u_test, r_test, p_val, p_test = Data_info['u_val'], Data_info['r_val'], Data_info['u_test'], Data_info['r_test'], Data_info['p_val'], Data_info['p_test']
    ## LOAD AND BUILD THE TRAINED AUTOENCODER MODEL 
    output_dim = u_val.shape[1]
    forward_path = os.path.join("Output","23Apr_2Prop_Forward", "model_forward_2mix")
    model_forward = tf.saved_model.load(forward_path)  
    
    deterministic_solutions_file = "Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.001Beta_InverseForward_1Gaussians_1Samples_0.0001LR_5000epochs_8192batch"
    det_path = os.path.join("Output", deterministic_solutions_file, "Test_results_info.npy")
    deterministic_test_sols = np.load(os.path.join(det_path), allow_pickle = True).item()
    det_z_test = deterministic_test_sols['test_p_predicted']
    det_solutions = det_z_test[positions,:]


    for i in range(len(positions)):
        pos  = positions [i]
        N = 200000
        u_true   = u_test[pos,:].reshape(1,u_test.shape[1]) #Case that we want to exlpore as input
        utrue =  np.array(tf.repeat(u_true, N, axis =0)) #now we have it 1,000 times to associate it to 1,000 different ps
        r_true = r_test[pos,:].reshape(1,r_test.shape[1])
        rtrue =  np.array(tf.repeat(r_true,N,axis = 0))
                
        import random
        ptest = []
        for j in range(N):
            p1 = random.uniform(0,1)
            p2 = random.uniform(0,1)
            p = [p1,p2]
            ptest.append(p)
        
        ptest = np.array(ptest)
        
        ### BUILD AND LOAD THE TRAININD FORWARD WEIGHTS  AND MODEL
        num_samples_per_mixture = 1
        
        output_dim = u_test.shape[1]
        forward_path = os.path.join("Output", "23Apr_2Prop_Forward")
        model_forward = ForwardModel(input_dim_decoder, output_dim, num_mixtures)
        model_forward.build(input_shape = ())
        weights_path = os.path.join(forward_path, "model_forward_weights.h5" )
        model_forward.load_weights(weights_path)
        
        upred = tf.cast(model_forward.predict(tf.concat([ptest, rtrue],axis = 1)), tf.float32)
        utrue = tf.cast(utrue, tf.float32)
        
        def ColumnSliceLayer(inputs, selected_features):
            selected_features = tf.constant(selected_features
        , dtype=tf.int32)
            outputs = tf.gather(inputs, selected_features, axis=1)
            return outputs
        
        def Data_loss(y_true, y_pred, selected_features, num_samples_per_mixture):
            ### este es el antiguo loss antes de tener en cuenta las mixturas, solo mide el error de reconstrucción (la loss del paper original)
            y_true = ColumnSliceLayer(y_true,selected_features)
            y_pred = ColumnSliceLayer(y_pred, selected_features)
            Ytrue = tf.repeat(y_true[:,:,tf.newaxis], num_samples_per_mixture, axis = 2)
            Ytrue = tf.reshape(tf.transpose(Ytrue, perm = [0,2,1]), [-1,y_true.shape[1]])
            # Ydiff = tf.math.square(Ytrue - y_pred)
            Ydiff = tf.math.square((Ytrue - y_pred)/(y_pred+1e-10))
            Data_Loss = tf.math.reduce_mean(Ydiff, axis = 1)
            return Data_Loss
        
        
        # ## PLOT
        
        z1 = ptest[:,0]
        z2 = ptest[:,1]
        
        dloss = Data_loss(utrue,upred, selected_features, num_samples_per_mixture)
        # Filter data based on loss value
        # lb = 0.0000000001
        # ub = 0.006
        lb = 0.00000000001
        ub = 0.004
        filtered_indices = (dloss > lb) & (dloss < ub)
        
        z1_filtered = z1[filtered_indices]
        z2_filtered = z2[filtered_indices]
        Loss_data_filtered = dloss[filtered_indices]
        combined_array = np.column_stack((z1_filtered, z2_filtered, Loss_data_filtered))
        np.savetxt(os.path.join("Output","Example3.csv"), combined_array, delimiter=",", header="Z1,Z2,Data_misfit", comments="", fmt="%g")        
        
        
        
        
        z1_case = p_test[pos,0]
        z2_case = p_test[pos,1]
    
        heading = 'Roll'

        # heading = 'ELBO'+str(num_gaussians)+'g'
        # Create a smooth contour plot
        plt.figure(figsize=(10, 8))
        plt.tricontourf(z1_filtered, z2_filtered, Loss_data_filtered, cmap='viridis_r', levels = 100)
        plt.colorbar(label=r'$\mathcal{L}_{\mathcal{M}}$ value')
        plt.scatter(p_test[pos,0], p_test[pos,1], c='red', marker='*', edgecolor = 'k', s=150, label = 'True solution')
        plt.scatter(det_solutions[i,0], det_solutions[i,1], c='red', edgecolor = 'k', s=50, marker='s', label='Deterministic solution')
        # plt.legend()
        plt.xlabel('z1')
        plt.ylabel('z2')
        # Set x and y limits to [0,1]
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.legend()
        legend = plt.legend(frameon=True, loc='upper right')

        # plt.title(f'Scatter plot of z1 vs z2 ({lb} < Loss Data < {ub})')
        # plt.savefig(os.path.join(folder_path, heading+'DataLoss_ContourPlot_withUB_'+str(ub)+'case_'+ str(i)+'.png'), dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close()
      


# # trying to recover more deterministic solutions  (entire test dataset)
# def determinisict_results():
    
#     foldername = 'Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.07Beta_InverseForward_2Gaussians_1Samples_1e-05LR_2000epochs_8192batch'

#     deterministic_folder_path = os.path.join('Output', foldername)
#     if not os.path.exists(deterministic_folder_path):
#         os.makedirs(deterministic_folder_path)
    
#     # # LOAD THE DATASETS AND PROBLEM INFORMATION (TRAINING, ENCODING ...):
    
#     Problem_info = np.load(os.path.join(deterministic_folder_path, 'Problem_info.npy'), allow_pickle = True).item()
#     input_dim_encoder, input_dim_decoder, num_mixtures, num_gaussians, num_samples_per_mixture, LR, selected_features, beta = Problem_info['input_dim_enc'], Problem_info['input_dim_dec'], Problem_info['n_mixtures'], Problem_info['n_gaussians'], Problem_info['n_samples'], Problem_info['LR'], Problem_info['selected_features'], Problem_info['beta']
#     Data_info = np.load(os.path.join(deterministic_folder_path, 'Data_info.npy'), allow_pickle = True).item()
#     u_val, r_val, u_test, r_test, p_val, p_test = Data_info['u_val'], Data_info['r_val'], Data_info['u_test'], Data_info['r_test'], Data_info['p_val'], Data_info['p_test']
#     ## LOAD AND BUILD THE TRAINED AUTOENCODER MODEL 
#     output_dim = u_val.shape[1]
#     forward_path = os.path.join("Output","23Apr_2Prop_Forward", "model_forward_2mix")
#     model_forward = tf.saved_model.load(forward_path)
    
#     num_samples_per_mixture  = 1 # you can fix this to one for the diagnostis step 
#     full_cov = True
#     s_lb = 0.000001
#     s_ub = 1.
#     model = My_InverseForward(input_dim_encoder, input_dim_decoder, output_dim, num_mixtures, num_gaussians, num_samples_per_mixture, model_forward, selected_features, beta, full_cov, s_ub, s_lb)
#     opt = optimizers.Adam(learning_rate = LR)
#     model.build(input_shape = ())
#     ae_weights_path = os.path.join(deterministic_folder_path, "Weights_ae.h5" )
#     model.load_weights(ae_weights_path)
    
#     test_results = np.load(os.path.join(deterministic_folder_path, 'Test_results_info.npy'), allow_pickle = True).item()
#     test_means = test_results['test_means']
#     test_weights = test_results['test_weights']
    
#     mean_positions = tf.argmax(test_weights, axis = 1)
#     det_solutions_test = tf.gather(test_means, mean_positions, batch_dims=1)



#     positions = [12, 4648, 7274, 2365, 243, 5015, 2458, 9659] # NEW SELECTION: IF WE CHANGE THE FIFTH CASE FROM 9657 TO 243.
#     a  = []
#     for i in range(len(positions)):
#         a.append(det_solutions_test[positions[i]])
        
#     A  = np.array(a)
    
        
#     selected_means = tf.gather(test_means, max_weight_indices, batch_dims=1)
        
#         ################ plotting loss functions to compare with Diagonal covariance:
#         det_history_path = os.path.join(deterministic_folder_path, 'model_history.npy')
#         model_history  = np.load(det_history_path, allow_pickle = True)
#         return  deterministic_folder_path, u_test, r_test, p_test, model, model_history
    




















# means, sigmas, weights = tf.split(GMMprops_test, [num_mixtures * num_gaussians, num_mixtures *num_mixtures * num_gaussians, num_gaussians], axis=-1)
# # #TESTING 
# ### TAKE A LOOK TO MULTIVARIATE outputs. But for one output still requires analysis

# K_dim = int(num_mixtures * (num_mixtures + 1) / 2) - num_mixtures
# test_means, test_sigmas_diag, test_sigmas_off, test_weights = tf.split(GMMprops_test, [num_mixtures * num_gaussians,
#                                                                                        num_mixtures*num_gaussians,
#                                             num_mixtures * num_mixtures * num_gaussians,
#                                             num_gaussians], axis=-1)

# test_means, test_sigmas_diag, test_sigmas_off, test_weights  = test_means.numpy(), test_sigmas_diag.numpy(), test_sigmas_off.numpy(),  test_weights.numpy()

# test_means = np.reshape(test_means, (-1, num_mixtures, num_gaussians))
# test_means = tf.transpose(test_means, perm= [0,2,1])

# test_sigmas = np.reshape(test_sigmas, (-1, num_mixtures*num_mixtures, num_gaussians))
# test_covariance_matrices = tf.reshape(sigmas, (-1,num_mixtures, num_mixtures, num_gaussians))
# test_covariance_matrices = tf.transpose(test_covariance_matrices, perm  = [0,3,1,2])

# test_weights = np.reshape(test_weights, (-1, num_gaussians))


# # WHEN WORKING WITH THE DETERMINISTIC APPROACH, SAVE THE SOLUTIONS FOR FUTURE BAYESIAN TESTS:

# # PLOT THE TEST CONTOURPLOTS INCLUDING THE DETERMINISTIC SOLUTION MARK 
# p_test[positions,:]
# prop_names = [r'$z_{1}$', r'$z_{1}$']
# heading = 'ELBO5g'


# for i in range(len(positions)):
#     pos  = positions[i]
#     # print(x_test_resc[pos,:])
#     means = test_means[pos,:,:]
#     sigmas = test_sigmas[pos,:,:]
#     weights = test_weights[pos,:]
    

#     truncated_components = []
#     cov_matrix = []
#     for j in range(num_gaussians):
#         #We create the covariance matrix for each gaussian conforming the mixture and append them 
#         A = test_covariance_matrices[:, j, :, :]
#         B = tf.tensordot(A, tf.transpose(A),axes = 1)[:,:,:,0]
#         C = num_mixtures * tf.eye(num_mixtures)
#         D = B+C
#         cov_matrix.append(D)
        
#     cov_matrix = tf.stack(cov_matrix, axis = -1)   
#     cov_matrix = tf.convert_to_tensor(cov_matrix, dtype=tf.float32)
#     means = tf.convert_to_tensor(means, dtype=tf.float32)
    
#     truncated_components = []
#     for i in range(num_gaussians):
#         chol_cov_matrix = tf.linalg.cholesky(cov_matrix[:, i, :])
#         component = tfp.distributions.MultivariateNormalTriL(loc=means[i,:], scale_tril=chol_cov_matrix)
#         truncated_components.append(component)
        
#     cat = tfp.distributions.Categorical(probs=weights)
#     mixture = tfp.distributions.Mixture(cat=cat, components=truncated_components)

#     num_points = 200000
#     x_values = np.random.uniform(0, 1, num_points)
#     y_values = np.random.uniform(0, 1, num_points)
#     mesh_points = np.column_stack([x_values, y_values])
    
#     # Evaluate the probability at each point on the grid
#     probabilities = mixture.prob(mesh_points)
    
#     # Create a smooth contour plot
#     plt.figure(figsize=(10, 8))
#     plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], probabilities, cmap='viridis', levels = 100)
#     plt.colorbar(label='density')
#     plt.scatter(p_test[pos,0], p_test[pos,1], c='red', marker='*', edgecolor = 'k', s=150, label = 'True solution')
#     # plt.scatter(det_solutions[i,0], det_solutions[i,1], c='red', edgecolor = 'k', s=50, marker='s', label='Deterministic solution')
#     plt.legend()
#     plt.xlabel('z1')
#     plt.ylabel('z2')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     # plt.savefig(os.path.join(folder_path, heading+'ContourPlot_Density_case_'+ str(i)+'.png'), dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()


# # plot_test_contourplots(test_means, test_sigmas, test_weights, num_gaussians, positions, det_solutions, p_test, folder_path)



def plot_test_contourplots_gpu_full(model, selected_features, u_test, r_test, p_test, num_mixtures, num_gaussians, full_cov, folder_path):
    ## Load the deterministic solutions according to the selected DOFs 
    ### UNCOMMENT THE LINE THAT CORRESPONDS TO THE PATH OF THE DOFS USED (3 OR 1)
    # deterministic_path = os.path.join("Results","Results08May_RollPitchYaw_2Props_1e-15Beta_InverseForwardDataLoss_2mixtures_5gaussians_10Samples_1e-05LR_500epochs" ) ### FOR THE THRE DOFs
    deterministic_path = os.path.join("Results","Results08May_Roll_2Props_1e-15Beta_InverseForwardDataloss_2mixtures_5gaussians_10Samples_1e-05LR_500epochs" ) ### FOR ROLL ONLY
    
    det_solutions = np.load(os.path.join(deterministic_path, "deterministc_test_solutions.npy") )
    
    ######### WARNING: USE THIS ONLY IF WE CHANGE THE TEST CASE 5 FOR THE NEW ONE. OTHERWISE  COMMENT BOTH LINES:
    # det_solutions[4,:] = [0.1267,0.61646] ## For RollPitchYaw: We have changed one test case for position 243 (the fifth case) becasue of a strange discrepancy between truth and deterministic that was greater in the case of RollPitchYaw rather than with only Roll :() 
    # det_solutions[4,:] = [0.1048003,0.6897715] ## For Roll: We have changed one test case for position 243 (the fifth case) becasue of a strange discrepancy between truth and deterministic that was greater in the case of RollPitchYaw rather than with only Roll :() 
    
    ## SELECTED  TEST CASES: 
    # positions = [12, 4648, 7274, 2365, 9657, 5015, 2458, 9659] # OLD SELECTION
    positions = [12, 4648, 7274, 2365, 243, 5015, 2458, 9659] # NEW SELECTION: IF WE CHANGE THE FIFTH CASE FROM 9657 TO 243.
    
    ########################################################################################################################################################################################################################33
    ############################ PLOTTING THE RESULTS DURING TESTING ########################################   
    inverse_model = model.Encoder_model
    GMMprops_test = inverse_model.predict([u_test[:,selected_features], r_test])
    
    test_means, test_sigmas_diag, test_weights, test_angles = tf.split(GMMprops_test, [num_mixtures * num_gaussians,
                                                    num_mixtures * num_gaussians,
                                                      num_gaussians,
                                                      num_gaussians], axis=-1)
    #Reshaping required to accomodate for the batch size (, num_mixtures*num_gaussians) during training
    test_means = tf.reshape(test_means, (-1, num_mixtures, num_gaussians))
    test_sigmas_diag = tf.reshape(test_sigmas_diag, (-1, num_mixtures, num_gaussians))
    test_weights = tf.reshape(test_weights, (-1, num_gaussians))
    angles_phi = tf.reshape(test_angles, (-1, num_gaussians))
    
    rot_matrix = tf.transpose(tf.stack([[tf.cos(angles_phi), -tf.sin(angles_phi)], [tf.sin(angles_phi), tf.cos(angles_phi)]], axis=0), perm= [2,3,0,1])
    RS_matrices = tf.einsum('BGDK,BKG -> BGDK', rot_matrix, test_sigmas_diag)
    RS_transp_matrices = tf.einsum('BGDK->BGKD', RS_matrices) #Obtain the transpose of the transformation matrix T = RS
    Sigma_matrices = tf.einsum('BGDK, BGKL -> BGDL', RS_matrices, RS_transp_matrices) # Build the Covariance matrix as C = T T^{t}
    LT_matrices = tf.linalg.cholesky(Sigma_matrices) #Apply Cholesk
    
        
    for i in range(len(positions)):
        pos  = positions[i]
        if full_cov: 
            Components = []
            
            for j in range(num_gaussians):
                LT_matrix = LT_matrices[pos,j,:,:]
                component = tfp.distributions.MultivariateNormalTriL(loc= test_means[pos, :,j], scale_tril=LT_matrix)
                Components.append(component)
            
        else: 
            Components = []
            for j in range(num_gaussians):
                component = tfp.distributions.MultivariateNormalDiag(loc=test_means[pos, :,j], scale_diag=test_sigmas_diag[pos, j, :])
                Components.append(component)
    
    
        # tf.print('full Covariance matrix examples:', Uchol[:,0,:,:])
        cat = tfp.distributions.Categorical(probs=test_weights[pos,:])
        mixture = tfp.distributions.Mixture(cat=cat, components=Components)
        
        num_points = 200000
        x_values = np.random.uniform(0, 1, num_points)
        y_values = np.random.uniform(0, 1, num_points)
        mesh_points = np.column_stack([x_values, y_values])
        
        # Evaluate the probability at each point on the grid
        probabilities = mixture.prob(mesh_points)
        # Create a smooth contour plot
        plt.figure(figsize=(10, 8))
        plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], probabilities, cmap='viridis', levels = 100)
        plt.colorbar(label='density')
        plt.scatter(p_test[pos,0], p_test[pos,1], c='red', marker='*', edgecolor = 'k', s=200, label = 'True solution')
        plt.scatter(det_solutions[i,0], det_solutions[i,1], c='red', edgecolor = 'k', s=100, marker='s', label='Deterministic solution')
        # plt.legend()
        # Customize legend with background and text color
        legend = plt.legend()
        legend.get_frame().set_facecolor('lightgray')  # Set background color
        legend.get_frame().set_edgecolor('black')      # Set edge color
        
        # Set the color of the legend text
        for text in legend.get_texts():
            text.set_color("darkblue")  # Set text color

        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig(os.path.join(folder_path, 'ContourPlot_Density_case_'+ str(i)+'.png'), dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close()
        







# ###################### DRAWING BASED ON THE DATALOSS VALUE ##############################################################################################################################################
# plot_test_Dataloss_contourplots(input_dim_decoder, u_test, r_test, p_test, num_mixtures, num_gaussians, num_samples_per_mixture, selected_features, positions, det_solutions,  folder_path )

