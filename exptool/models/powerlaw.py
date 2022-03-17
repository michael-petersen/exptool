'''

powerlaw (part of exptool.models)
    Implementation of a power law model with optional exponential cutoff


25 Jan 2022  First construction


example
-------------
P = powerlaw(np.linspace(0.001,1.,300))


todo
------------


'''

# general python imports
import numpy as np

from scipy.special import gamma,gammainc


class powerlaw():
    """A basic power-law density model


    notes
    ----------
    """
    def __init__(self,alpha=-2.,G=1,M=1,Rcut=1.0):
        """

        inputs
        ----------------
        G     : the gravitational constant, in units of (length^3) * (mass) / (time^2)
                 (an astronomical value is 0.0000043009125, in the equivalent (km/s)^2 * kpc / Msun)

        """
        self.rcut = rcut
        self.G    = G
        self.alpha= alpha
        self.M    = M

        # calculate the central density
        self.rho0 = self.get_rho0()

    def get_rho0(self):
        """solve the mass enclosed at 100rcut to recover the central density

        returned in units of density, mass/(length^3)
        """

        r = 100.*self.rscl
        tmp_mass =  2.*np.pi*self.rcut**(3.-self.alpha)*gammainc(1.5-self.alpha/2.,(r/self.rcut)**2.)*gamma(1.5-self.alpha/2.)

        return self.M/tmp_mass

    def get_mass(self,r):
        """galpy power-law mass

        be careful of the scipy gamma incomplete gamma function definition!

        inputs
        ---------------
        r    : radius to compute enclosed mass for
        """
        return 2.*np.pi*self.rho0*self.rcut**(3.-self.alpha)*gammainc(1.5-self.alpha/2.,(r/self.rcut)**2.)*gamma(1.5-self.alpha/2.)

    def get_dens(self,r):
        """galpy power-law density

        inputs
        ---------------
        r    : radius to compute density at
        """
        return 1./r**self.alpha*numpy.exp(-(r/self.rc)**2.)


    def get_pot(self,r):
        """galpy's power-law potential

        inputs
        ---------------
        r    : radius to compute potential at

        returned in units of (length^2)*(mass^2)/(time^2)
        """
        return 2.*np.pi*self.rcut**(3.-self.alpha)/r*(r/self.rc*gamma(1.-self.alpha/2.)*gammainc(1.-self.alpha/2.,(r/self.rcut)**2.)-gamma(1.5-self.alpha/2.)*gammainc(1.5-self.alpha/2.,(r/self.rc)**2.))

    def get_dphi_dr(self,r):
        """power-law radial force, using Newton's theorems based on mass enclosed.

        inputs
        ---------------
        r    : radius to compute radial force at

        returned in units of length * mass / (time^2)
        """

        return -self.mass(r)/r**2.
