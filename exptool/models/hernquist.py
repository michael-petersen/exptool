'''

hernquist (part of exptool.models)
    Implementation of the analytic Hernquist models
    (compare with basis set in exptool.basis.hernquist)


29 Oct 2017  First construction
11 Jul 2021  Much more usable (read:correct) now!
25 Jan 2022  Revise, add force, comment
24 Sep 2023  New docstrings


example
-------------
H = Hernquist(np.linspace(0.001,1.,300))


todo
------------
- allow user to specify either total mass or central density
- how does one construct a R D M P table if they want to from this?


'''

# general python imports
import numpy as np

class Hernquist():
    '''
    The Hernquist (1990) model

    The Hernquist model represents a spherically symmetric mass distribution characterized by a scale radius (rscl) and an overall mass (M).

    Attributes:
        rscl (float): The scale radius of the Hernquist model.
        G (float): The gravitational constant.
        M (float): The overall mass of the model.
        rho0 (float): The central density of the Hernquist model.

    Methods:
        get_rho0():
            Calculate the central density of the Hernquist model.

        get_mass(r):
            Calculate the mass enclosed within a given radius (r).

        get_dens(r):
            Calculate the density at a given radius (r).

        get_pot(r):
            Calculate the gravitational potential at a given radius (r).

        get_dphi_dr(r):
            Calculate the radial force at a given radius (r).

    Example usage:
        H = Hernquist(rscl=1.0, G=0.0000043009125, M=1.0)
        print(H.get_mass(2.0))
    '''

    def __init__(self, rscl=1.0, G=1, M=1):
        """
        Initialize the Hernquist model.

        Args:
            rscl (float, optional): The scale radius (rscl) of the model. Defaults to 1.0.
            G (float, optional): The gravitational constant. Defaults to 1.
            M (float, optional): The overall mass of the model. Defaults to 1.

        Attributes:
            rscl (float): The scale radius of the Hernquist model.
            G (float): The gravitational constant.
            M (float): The overall mass of the model.
            rho0 (float): The central density of the Hernquist model.
        """

        self.rscl = rscl
        self.G = G
        self.M = M
        self.rho0 = self.get_rho0()

    def get_rho0(self):
        """Calculate the central density of the Hernquist model.

        Returns:
            float: The central density in units of mass / (length^3).
        """
        rs = 1000.  # Evaluate at r = 1000 * rscl (reasonable approximation)
        return self.M * (2 * (1 + rs) * (1 + rs)) / (rs * rs) / (4 * np.pi * np.power(self.rscl, 3))

    def get_mass(self, r):
        """Calculate the mass enclosed within a given radius (r).

        Args:
            r (float): The radius at which to compute the enclosed mass.

        Returns:
            float: The enclosed mass in units of mass.
        """
        rs = r / self.rscl
        return 4 * np.pi * self.rho0 * np.power(self.rscl, 3) * (rs * rs) / (2 * (1 + rs) * (1 + rs))

    def get_dens(self, r):
        """Calculate the density at a given radius (r).

        Args:
            r (float): The radius at which to compute the density.

        Returns:
            float: The density in units of mass / (length^3).
        """
        alpha = 1
        beta = 4
        return self.rho0 * (r / self.rscl) ** (-alpha) * (1. + r / self.rscl) ** (-beta + alpha)

    def get_pot(self, r):
        """Calculate the gravitational potential at a given radius (r).

        Args:
            r (float): The radius at which to compute the potential.

        Returns:
            float: The gravitational potential in units of (length^2) * (mass^2) / (time^2).
        """
        return -4 * np.pi * self.G * self.rho0 * self.rscl * self.rscl * ((2 * (1. + r / self.rscl)) ** -1.)

    def get_dphi_dr(self, r):
        """Calculate the radial force at a given radius (r).

        Args:
            r (float): The radius at which to compute the radial force.

        Returns:
            float: The radial force in units of (length * mass) / (time^2).
        """
        return self.G * self.M / (self.rscl + r) ** 2
