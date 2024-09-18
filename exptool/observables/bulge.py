"""
bulge.py
Tools for measuring the bulge mass of a galaxy.
    - measure_bulge_mass: Compute the bulge mass of a galaxy

18 Sep 2024: First introduction

"""
import numpy as np

def measure_bulge_mass(x, y, vx, vy, mass):
    """
    Compute the bulge mass of a galaxy based on particle positions, velocities, and masses.
    
    The function assumes that the bulge is composed of stars with negative tangential velocities
    (i.e., retrograde orbits), based on the assumption that prograde stars are part of the disc.
    It computes the radius that contains 50% of the bulge stars and returns the total bulge mass.

    Parameters:
    - x: Array of particle x-positions.
    - y: Array of particle y-positions.
    - vx: Array of particle velocities in the x-direction.
    - vy: Array of particle velocities in the y-direction.
    - mass: Array of particle masses.

    Returns:
    - bulge_mass: The estimated total bulge mass (double the sum of negative velocity stars' masses).

    TODO:
    - Extend to 3D: Currently, this function works in 2D (x and y plane). If ywe want a more accurate 3D mass measurement, we might want to include the z-coordinate for the radial position or compute velocities more generally in 3D.

    - Handling edge cases: we might want to include additional checks for cases where no retrograde particles are found, or where the tangential velocity is undefined due to division by zero (e.g., at the galaxy center where rpos = 0).

    """
    # Compute the radial distance from the center in the x-y plane
    rpos = np.sqrt(x**2 + y**2)
    
    # Compute the tangential velocity
    vtan = (x * vy - y * vx) / rpos
    
    # Determine the overall rotational direction (positive for prograde, negative for retrograde)
    direction = np.sign(np.nanmean(vtan))
    
    # Align all tangential velocities with the prograde direction
    vtan *= direction
    
    # Identify particles with retrograde orbits (vtan < 0)
    negvel = np.where(vtan < 0)[0]
    
    # Compute the radius containing 50% of the bulge stars (those with negative velocities)
    bulge_radius_50 = np.nanpercentile(rpos[negvel], 50)
    print(f"50% bulge radius: {bulge_radius_50:3.2f} kpc")
    
    # Compute the total bulge mass as double the mass of retrograde stars
    bulge_mass = 2.0 * np.nansum(mass[negvel])
    
    return bulge_mass


