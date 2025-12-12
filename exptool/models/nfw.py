'''

nfw (part of exptool.models)
    Implementation of the analytic NFW model


25 Jan 2022  First construction
24 Sep 2023  New docstrings


example
-------------
N = NFW(np.linspace(0.001,1.,300))


todo
------------


'''

# general python imports
import numpy as np

class NFW():
    """Navarro-Frenk-White (1996) dark matter density profile model.

    The NFW (Navarro-Frenk-White) model describes the dark matter density profile of a halo.

    Attributes:
        rscl (float): The scale radius of the NFW profile.
        G (float): The gravitational constant.
        Mvir (float): The virial mass of the model.
        Rvir (float): The virial radius of the model.
        conc (float): The concentration parameter, computed as Rvir / rscl.
        rho0 (float): The central density of the NFW profile.

    Methods:
        get_rho0():
            Calculate the central density of the NFW profile.

        get_mass(r):
            Calculate the mass enclosed within a given radius r.

        get_dens(r):
            Calculate the density at a given radius r.

        get_pot(r):
            Calculate the gravitational potential at a given radius r.

        get_dphi_dr(r):
            Calculate the radial force at a given radius r.
    """

    def __init__(self, rscl=1.0, G=1, Mvir=1, Rvir=1.0):
        """Initialize the NFW dark matter density profile.

        Args:
            rscl (float, optional): The scale radius of the NFW profile. Defaults to 1.0.
            G (float, optional): The gravitational constant. Defaults to 1.
            Mvir (float, optional): The virial mass of the model. Defaults to 1.
            Rvir (float, optional): The virial radius of the model. Defaults to 1.0.
        """

        self.rscl = rscl
        self.G = G
        self.Mvir = Mvir
        self.Rvir = Rvir

        # Calculate the concentration parameter (c = Rvir / rscl)
        self.conc = self.Rvir / self.rscl

        # Calculate the central density rho0
        self.rho0 = self.get_rho0()

    def get_rho0(self):
        """Calculate the central density of the NFW profile.

        Returns:
            float: The central density in units of mass / (length^3).
        """

        rs = self.Rvir / self.rscl
        return self.Mvir / (4 * np.pi * np.power(self.rscl, 3) * (np.log(1 + rs) - rs / (1. + rs)))

    def get_mass(self, r):
        """Calculate the mass enclosed within a given radius r.

        Args:
            r (float): The radius to compute enclosed mass for.

        Returns:
            float: The enclosed mass in units of mass.
        """

        rs = r / self.rscl
        return 4 * np.pi * self.rho0 * np.power(self.rscl, 3) * (np.log(1 + rs) - rs / (1. + rs))

    def get_dens(self, r):
        """Calculate the density at a given radius r.

        Args:
            r (float): The radius to compute density at.

        Returns:
            float: The density in units of mass / (length^3).
        """

        alpha = 1
        beta = 3
        return self.rho0 * (r / self.rscl)**(-alpha) * (1. + r / self.rscl)**(-beta + alpha)

    def get_pot(self, r):
        """Calculate the gravitational potential at a given radius r.

        Args:
            r (float): The radius to compute potential at.

        Returns:
            float: The potential in units of (length^2) * (mass^2) / (time^2).
        """

        return -4 * np.pi * self.G * self.rho0 * self.rscl * self.rscl * (np.log(1. + r / self.rscl) / (r / self.rscl))

    def get_dphi_dr(self, r):
        """Calculate the radial force at a given radius r.

        Args:
            r (float): The radius to compute radial force at.

        Returns:
            float: The radial force in units of length * mass / (time^2).
        """

        prefac = -4 * np.pi * self.G * self.rho0 * self.rscl * self.rscl
        return prefac * self.rscl * (r - (r + self.rscl) * np.log(1 + r / self.rscl)) / (r**2 * (r + self.rscl))
