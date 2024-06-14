'''

plummer (part of exptool.models)
    Implementation of the analytic Plummer models

31 Oct 2017: First construction
09 Jul 2021: Upgrades I
24 Sep 2023: Documentation improvement

'''
# general python imports
import numpy as np

import numpy as np

class Plummer:
    '''
    Plummer class model

    The Plummer model represents a spherical mass distribution with a Plummer potential.
    It is often used to describe the density distribution of some stellar systems like globular clusters.

    Parameters:
    -----------
    rscl : float, optional
        The scale radius of the Plummer model (default is 1.0).

    Methods:
    --------
    get_mass(r)
        Calculate the mass enclosed within a given radius 'r' from the center.

    get_dens(r)
        Calculate the density at a given radius 'r' from the center.

    get_cartesian_forces(x, y, z)
        Calculate the forces in the x, y, and z directions at a given cartesian position (x, y, z).

    get_cartesian_forces_array(arr)
        Calculate the forces in the x, y, and z directions at a given cartesian position stored in an array [x, y, z].

    get_force(r)
        Calculate the gravitational force at a given radius 'r'.

    get_pot(r)
        Calculate the gravitational potential at a given radius 'r'.
    '''
    def __init__(self, rscl=1.0):
        '''
        Initialize a Plummer model with the given scale radius 'rscl'.
        '''
        self.rscl = rscl

    def get_mass(self, r):
        '''
        Calculate the mass enclosed within a given radius 'r' from the center.

        Parameters:
        -----------
        r : float
            The radius at which to calculate the enclosed mass.

        Returns:
        --------
        float
            The enclosed mass at the given radius 'r'.
        '''
        return r * r * np.power((r + self.rscl), -2.)

    def get_dens(self, r):
        '''
        Calculate the density at a given radius 'r' from the center.

        Parameters:
        -----------
        r : float
            The radius at which to calculate the density.

        Returns:
        --------
        float
            The density at the given radius 'r'.
        '''
        return self.rscl * (2. * np.pi * r) ** -1. * (r + self.rscl) ** -3.

    def get_cartesian_forces(self, x, y, z):
        '''
        Calculate the forces in the x, y, and z directions at a given cartesian position (x, y, z).

        Parameters:
        -----------
        x : float
            The x-coordinate of the position.
        y : float
            The y-coordinate of the position.
        z : float
            The z-coordinate of the position.

        Returns:
        --------
        tuple
            A tuple containing the x, y, and z components of the force.
        '''
        r3 = np.sqrt(x * x + y * y + z * z)
        fr = self.get_force(r3)

        fx = x * fr / r3
        fy = y * fr / r3
        fz = z * fr / r3

        return fx, fy, fz

    def get_cartesian_forces_array(self, arr):
        '''
        Calculate the forces in the x, y, and z directions at a given cartesian position stored in an array [x, y, z].

        Parameters:
        -----------
        arr : array-like
            An array containing the x, y, and z coordinates of the position.

        Returns:
        --------
        array
            An array containing the x, y, and z components of the force.
        '''
        x, y, z = arr[0], arr[1], arr[2]
        r3 = np.sqrt(x * x + y * y + z * z)
        fr = self.get_force(r3)

        fx = -x * fr / r3
        fy = -y * fr / r3
        fz = -z * fr / r3

        return np.array([fx, fy, fz])

    def get_force(self, r):
        '''
        Calculate the gravitational force at a given radius 'r'.

        Parameters:
        -----------
        r : float
            The radius at which to calculate the force.

        Returns:
        --------
        float
            The gravitational force at the given radius 'r'.
        '''
        b = (r * r + self.rscl * self.rscl)
        return r / np.power(b, 1.5)

    def get_pot(self, r):
        '''
        Calculate the gravitational potential at a given radius 'r'.

        Parameters:
        -----------
        r : float
            The radius at which to calculate the potential.

        Returns:
        --------
        float
            The gravitational potential at the given radius 'r'.
        '''
        return -1. * ( (r ** 2. + self.rscl ** 2.) ** -0.5)
