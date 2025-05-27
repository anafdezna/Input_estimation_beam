#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:12:59 2024

@author: afernandez
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
# Parameters
num_points = 200000
n_dims = 5  # For example, adjust based on your mixture dimensionality
dims_to_plot = list(combinations(range(n_dims), 2))  # All 2D combinations of dimensions



# Parameters for the 3 Gaussians
n_dims = 5  # Dimensionality of each Gaussian
num_components = 3  # Number of Gaussians

# Means for each Gaussian component
means = tf.constant([
    [0.2, 0.3, 0.5, 0.7, 0.9],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.8, 0.2, 0.4, 0.6, 0.1]
], dtype=tf.float32)

# Covariance matrices (diagonal for simplicity)
covariances = tf.constant([
    [0.01, 0.02, 0.02, 0.01, 0.01],
    [0.03, 0.03, 0.02, 0.02, 0.02],
    [0.01, 0.01, 0.01, 0.01, 0.02]
], dtype=tf.float32)

# Construct the individual Gaussian distributions
components = [
    tfd.MultivariateNormalDiag(loc=means[i], scale_diag=tf.sqrt(covariances[i]))
    for i in range(num_components)
]

# Mixing probabilities (weights)
mixing_probs = tf.constant([0.4, 0.4, 0.2], dtype=tf.float32)

# Create the Mixture Distribution
mixture = tfd.Mixture(
    cat=tfd.Categorical(probs=mixing_probs),
    components=components
)

# Draw samples and evaluate probabilities
num_samples = 10000
# Sample data
samples = mixture.sample(num_points).numpy()  # Assuming mixture is a TensorFlow/Keras model
probabilities = mixture.prob(samples).numpy()

# Normalize probabilities for color scaling
norm_probs = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())

# Create pairwise scatter plots
fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=(10, 5 * len(dims_to_plot)))

for ax, (dim1, dim2) in zip(axes, dims_to_plot):
    ax.scatter(samples[:, dim1], samples[:, dim2], c=norm_probs, cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel(f'z{dim1 + 1}')
    ax.set_ylabel(f'z{dim2 + 1}')
    ax.set_title(f'Projection onto dimensions z{dim1 + 1} vs z{dim2 + 1}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Parameters
n_dims = 5  # Dimensionality of the Gaussian mixture
grid_size = 100  # Resolution of the grid for contour plots
num_samples = 10000
samples = mixture.sample(num_samples).numpy()  # Sample from the mixture

# Generate all unique pairs of dimensions
pairs = list(combinations(range(n_dims), 2))

# Loop through each pair of dimensions
for dim1, dim2 in pairs:
    # Create a grid of points over [0, 1] x [0, 1] for the two selected dimensions
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Stack grid points and set the other dimensions to zero (or mean values if preferred)
    grid_points = np.zeros((grid_size**2, n_dims))
    grid_points[:, dim1] = X.ravel()
    grid_points[:, dim2] = Y.ravel()
    
    # Evaluate the mixture's probability density at each grid point
    probabilities = mixture.prob(grid_points).numpy().reshape(grid_size, grid_size)
    
    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, probabilities, levels=50, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title(f'Contour Plot for Dimensions z{dim1+1} vs z{dim2+1}')
    plt.xlabel(f'z{dim1+1}')
    plt.ylabel(f'z{dim2+1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


