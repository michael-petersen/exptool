###########################################################################################
#
#  hernquist.py
#     Compute analytic Hernquist models
#
# 29 Oct 2017: First construction
# 11 Jul 2021: Much more usable (read:correct) now!
#
# Generalization is (BT 2.64)
#
#                           rho0
#   rho(r) = -------------------------------
#            (r/a)^alpha (1+r/a)^(beta-alpha)
#
# where alpha=1, beta=4 for the Hernquist model
#
# (beta=4 is the dehnen family of models. dehnen models with alpha in [0.6,2] reasonably describe elliptical galaxy centers)
# (alpha=2, beta=4 is Jaffe model)
# (alpha=1, beta=3 is NFW model)
#
# Enclosed mass is then
#
#                                               s^{2-\alpha}
#    M(r) = 4\pi \rho_0 a^3 \int_0^{r/a} ds --------------------
#                                            (1+s)^{\beta-alpha}
#

'''


hernquist (part of exptool.models)
    Implementation of the analytic Hernquist models

#H = Hernquist(np.linspace(0.001,1.,300))

# how does one construct a R D M P table if they want to from this?



'''
from __future__ import absolute_import, division, print_function, unicode_literals



# general python imports
import numpy as np
import math
import time
import sys
import os


class Hernquist():
    """The classic Hernquist model.
    Hernquist has a gravitational radii of 6rscl.
    """
    def __init__(self,rscl=1.0,G=1,M=1):
        self.rscl = rscl
        self.G    = G
        self.M    = M
        self.rho0 = self.get_rho0()
    def get_rho0(self):
        """solving BT08, eq. 2.66 at r==1000a"""
        rs = 1000.
        return self.M*(2*(1+rs)*(1+rs))/(rs*rs)/(4*np.pi*np.power(self.rscl,3))
    def get_mass(self,r):
        """BT08, eq. 2.66"""
        rs = r/self.rscl
        return 4*np.pi*self.rho0*np.power(self.rscl,3)*(rs*rs)/(2*(1+rs)*(1+rs))
    def get_dens(self,r):
        """BT08, eq. 2.64, with alpha=1, beta=4"""
        alpha = 1
        beta  = 4
        return self.rho0 * (r/self.rscl)**(-alpha) * (1. + r/self.rscl)**(-beta+alpha)
    def get_pot(self,r):
        """BT08, eq. 2.67"""
        return -4*np.pi*self.G*self.rho0 *self.rscl*self.rscl * ( (2*(1.+r/self.rscl))**-1.)




