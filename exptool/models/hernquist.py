###########################################################################################
#
#  hernquist.py
#     Compute analytic Hernquist models
#
# 10-29-2017: First construction
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

H = Hernquist(np.linspace(0.001,1.,300))




'''
from __future__ import absolute_import, division, print_function, unicode_literals



# general python imports
import numpy as np
import math
import time
import sys
import os


# exptool imports
from exptool.utils import utils
from exptool.utils import halo_methods
from exptool.io import psp_io







class Hernquist():
    def __init__(self,radii,rscl=1.0):
        self.r = radii
        self.rscl = 1.0
        self.m = Hernquist.get_mass(self)
        self.d = Hernquist.get_dens(self)
        self.p = Hernquist.get_pot(self)
    def get_mass(self):
        return self.r *self.r * np.power((self.r + self.rscl),-2.)
    def get_dens(self):
        return self.rscl * (2.*np.pi*self.r)**-1. * (self.r + self.rscl)**-3.
    def get_pot(self):
        return -1. * ( (self.r+self.rscl)**-1.)




