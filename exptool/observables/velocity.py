"""
velocity.py

Various velocity analysis methods

"""

import numpy as np

from ..utils import kde_3d
from ..io import particle



def measured_rotation(infile, comp, rmax=0.06, nsamp=256):
    '''
    Calculate the measured rotation velocity from particle data.

    This function computes the rotational velocity profile based on particle positions 
    and velocities within a specified radial extent (rmax). A 2D Kernel Density Estimation (KDE) 
    is performed on the radial and angular coordinates, both weighted and unweighted by 
    rotational contributions, to derive the rotation profile.

    Parameters
    ----------
    infile : str
        Path to the input file containing particle data.
    comp : str
        Component of the data to be analyzed (e.g., "disk", "bulge").
    rmax : float, optional
        Maximum radial distance to consider for the rotation measurement, in the same units 
        as particle positions. Default is 0.06.
    nsamp : int, optional
        Number of sampling points for the KDE grid along each dimension. Default is 256.

    Returns
    -------
    rbins : ndarray
        Array of radial bin edges from 0 to rmax.
    rotation_profile : ndarray
        Array representing the rotation velocity profile as a function of radius.
    '''

    # Load particle data from file and component type specified
    O = particle.Input(infile, comp=comp)

    # Set the extent and number of KDE samples
    extent = rmax
    nsamp = 256

    # Compute rotation-weighted KDE weight as (x*yvel - y*xvel) / sqrt(x^2 + y^2)
    # This approximates the tangential (rotational) velocity component in 2D
    kde_weight = (O.xpos * O.yvel - O.ypos * O.xvel) / ((O.xpos**2 + O.ypos**2)**0.5)

    # Calculate radial distances and angular coordinates of particles
    rvals = (O.xpos**2 + O.ypos**2)**0.5  # Radial distance from origin
    tvals = np.arctan2(O.ypos, O.xpos)    # Angular position in radians

    # Filter particles within the specified radial limit (rmax)
    w = np.where(abs(rvals) < rmax)[0]

    # Perform KDE on (rvals, tvals) with weights (kde_weight) for rotation-weighted density
    vv = kde_3d.fast_kde_two(
        rvals[w], tvals[w],
        gridsize=(nsamp, nsamp),
        extents=(0., rmax, -np.pi, np.pi),
        nocorrelation=False,
        weights=kde_weight[w]
    )

    # Perform KDE on (rvals, tvals) without weights for unweighted particle density
    tt = kde_3d.fast_kde_two(
        rvals[w], tvals[w],
        gridsize=(nsamp, nsamp),
        extents=(0., rmax, -np.pi, np.pi),
        nocorrelation=False,
        weights=None
    )

    # Generate radial bins from 0 to rmax for the resulting profile
    rbins = np.linspace(0.0, rmax, nsamp)

    # Compute rotation velocity profile by normalizing weighted density by unweighted density
    return rbins, vv / tt
