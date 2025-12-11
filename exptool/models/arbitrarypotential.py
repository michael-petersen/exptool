"""
Provides functions to compute the density from an arbitrary potential using JAX.
"""

import jax
import jax.numpy as jnp

def calculate_density(potential, gravitational_constant):
    # Define a function to compute the Laplacian of the potential
    def laplacian(potential):
        return jnp.sum(jax.hessian(jnp.sum)(potential))

    # Calculate the Laplacian of the potential
    laplacian_potential = jax.vmap(laplacian)(potential)

    # Calculate the density using Poisson's equation
    density = laplacian_potential / (4 * jnp.pi * gravitational_constant)

    return density
