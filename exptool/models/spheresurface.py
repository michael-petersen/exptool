"""
spheresurface

simply generate points on the surface of a sphere
can be used to test simple prolate/oblate models

04 Mar 2021: introduction


"""
import numpy as np

def generate_sphere_surface(npoints, cartesian=True):
    """Generate evenly spaced points on the surface of a sphere.

    Args:
        npoints (int): The maximum number of points to generate on the sphere's surface.
        cartesian (bool, optional): If True, return (x, y, z) coordinates. If False, return (phi, theta) angles.

    Returns:
        tuple: A tuple containing arrays (or tuples) representing the points on the sphere's surface.

    Note:
        npoints is the maximum because we are enforcing equal spacing; not all npoints values return even spacing.
    """

    # Initialize arrays for points in Cartesian and spherical coordinates
    points = np.zeros((npoints, 3))
    sphpoints = np.zeros((npoints, 2))

    # Calculate angular increments
    a = 4 * np.pi / npoints
    d = np.sqrt(a)
    Mtheta = int(np.floor(np.pi / d))
    dtheta = np.pi / Mtheta
    dphi = a / dtheta

    npoint = 0  # Counter for generated points

    for m in range(Mtheta):
        theta = np.pi * (m + 0.5) / Mtheta
        Mphi = int(np.floor(2 * np.pi * np.sin(theta) / dphi))

        for n in range(Mphi):
            phi = 2 * np.pi * n / Mphi
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            # Store the points in the arrays
            points[npoint] = np.array([x, y, z])
            sphpoints[npoint] = np.array([phi, theta])
            npoint += 1

    if cartesian:
        return points[:npoint], None  # Returning None for spherical points
    else:
        return None, sphpoints[:npoint]
