'''

nfw (part of exptool.models)
    Implementation of the analytic NFW model


25 Jan 2022  First construction


example
-------------
N = NFW(np.linspace(0.001,1.,300))


todo
------------


'''

# general python imports
import numpy as np


class NFW():
    """The Navarro-Frenk-White (1996) model


    notes
    ----------
    """
    def __init__(self,rscl=1.0,G=1,Mvir=1,Rvir=1.0):
        """

        inputs
        ----------------
        rscl  : the scale radius, in units of length
        G     : the gravitational constant, in units of (length^3) * (mass) / (time^2)
                 (an astronomical value is 0.0000043009125, in the equivalent (km/s)^2 * kpc / Msun)
        Mvir  : the virial mass of the model, in units of mass
        rvir  : the virial radius for the model (such that M(rvir)=Mvir), in units of length

        """
        self.rscl = rscl
        self.G    = G
        self.Mvir = Mvir
        self.Rvir = Rvir

        # retrieve the concentration: c = Rvir/Rscl (see BT08 discussion after eq. 2.64)
        self.conc = self.Rvir/self.rscl

        self.rho0 = self.get_rho0()

    def get_rho0(self):
        """solving BT08, eq. 2.66 at r==rvir, set equal to Mvir

        given a virial mass of the NFW model, compute the central density

        returned in units of density, mass/(length^3)
        """
        rs = self.Rvir/self.rscl
        return self.Mvir/(4*np.pi*np.power(self.rscl,3)*(np.log(1+rs) - rs/(1.+rs)))

    def get_mass(self,r):
        """NFW mass enclosed: BT08, eq. 2.66

        inputs
        ---------------
        r    : radius to compute enclosed mass for
        """
        rs = r/self.rscl
        return 4*np.pi*self.rho0*np.power(self.rscl,3)*(np.log(1+rs) - rs/(1.+rs))

    def get_dens(self,r):
        """NFW density: BT08, eq. 2.64, with alpha=1, beta=3

        inputs
        ---------------
        r    : radius to compute density at
        """
        alpha = 1
        beta  = 3
        return self.rho0 * (r/self.rscl)**(-alpha) * (1. + r/self.rscl)**(-beta+alpha)

    def get_pot(self,r):
        """NFW potential: BT08, eq. 2.67

        inputs
        ---------------
        r    : radius to compute potential at

        returned in units of (length^2)*(mass^2)/(time^2)
        """
        return -4*np.pi*self.G*self.rho0 *self.rscl*self.rscl * (np.log(1. + r/self.rscl)/(r/self.rscl))

    def get_dphi_dr(self,r):
        """NFW radial force: differentiate potential above using Wolfram Alpha

        (strip out prefactors not dependent on r from the potential)
        https://www.wolframalpha.com/input/?i=d%2Fdr++%28log%281+%2B+r%2Fs%29%2F%28r%2Fs%29%29

        inputs
        ---------------
        r    : radius to compute radial force at

        returned in units of length * mass / (time^2)
        """
        # or:
        # prefac * (1/(r*(1.+r/self.rscl)) + self.rscl*np.log(1.+r/self.rscl)/(r**2))

        prefac = -4*np.pi*self.G*self.rho0 *self.rscl*self.rscl
        return prefac * self.rscl * (r - (r+self.rscl)*np.log(1+r/self.rscl)) / (r**2 * (r+self.rscl))
