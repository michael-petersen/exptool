'''

hernquist (part of exptool.models)
    Implementation of the analytic Hernquist models
    (compare with basis set in exptool.basis.hernquist)


29 Oct 2017  First construction
11 Jul 2021  Much more usable (read:correct) now!
25 Jan 2022  Revise, add force, comment


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
    """The Hernquist (1990) model


    notes
    ----------
    The Hernquist profile has a gravitational radii of 6rscl.
    """
    def __init__(self,rscl=1.0,G=1,M=1):
        """

        inputs
        ----------------
        rscl  : the scale radius, in units of length
        G     : the gravitational constant, in units of (length^3) * (mass) / (time^2)
                 (an astronomical value is 0.0000043009125, in the equivalent (km/s)^2 * kpc / Msun)
        M     : the overall mass of the model, in units of mass

        """
        self.rscl = rscl
        self.G    = G
        self.M    = M
        self.rho0 = self.get_rho0()

    def get_rho0(self):
        """solving BT08, eq. 2.66 at r==1000a

        this is a reasonable approximation because Hernquist is finite in mass.
        exercise to the coder: what fraction of the mass are we excluding?

        returned in units of density, mass/(length^3)
        """
        rs = 1000.
        return self.M*(2*(1+rs)*(1+rs))/(rs*rs)/(4*np.pi*np.power(self.rscl,3))

    def get_mass(self,r):
        """Hernquist mass enclosed: BT08, eq. 2.66

        inputs
        ---------------
        r    : radius to compute enclosed mass for
        """
        rs = r/self.rscl
        return 4*np.pi*self.rho0*np.power(self.rscl,3)*(rs*rs)/(2*(1+rs)*(1+rs))

    def get_dens(self,r):
        """Hernquist density: BT08, eq. 2.64, with alpha=1, beta=4

        inputs
        ---------------
        r    : radius to compute density at
        """
        alpha = 1
        beta  = 4
        return self.rho0 * (r/self.rscl)**(-alpha) * (1. + r/self.rscl)**(-beta+alpha)

    def get_pot(self,r):
        """Hernquist potential: BT08, eq. 2.67

        inputs
        ---------------
        r    : radius to compute potential at

        Note: this is a more complicated generalisation of the Hernquist potential,
        \Phi = -GM/(r+rscl)
        Feel free to inject that formula instead!

        returned in units of (length^2)*(mass^2)/(time^2)
        """
        return -4*np.pi*self.G*self.rho0 *self.rscl*self.rscl * ( (2*(1.+r/self.rscl))**-1.)

    def get_dphi_dr(self,r):
        """Hernquist radial force: differentiate -GM/(r+a) using Wolfram Alpha

        inputs
        ---------------
        r    : radius to compute radial force at

        returned in units of length * mass / (time^2)
        """
        return self.G*self.M/(self.rscl+r)**2
