"""
use jax to take an arbitrary potential and compute the density

# Example usage
potential = ...  # Replace with your potential data
gravitational_constant = 6.67430e-11  # Gravitational constant (adjust as needed)

# Convert potential and gravitational constant to JAX arrays
potential = jnp.array(potential)
gravitational_constant = jnp.array(gravitational_constant)

# Compute the density using JAX
density = calculate_density(potential, gravitational_constant)


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
