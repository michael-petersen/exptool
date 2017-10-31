###########################################################################################
#
#  plummer.py
#     Compute analytic Plummer models
#
# 10-31-2017: First construction
#
#
#
'''


plummer (part of exptool.models)
    Implementation of the analytic Plummer models

P = P(np.linspace(0.001,1.,300))


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







class Plummer():
    '''
    Plummer class models

    




    '''
    def __init__(self,radii,rscl=1.0):
        self.r = radii
        self.rscl = 1.0
        self.m = Plummer.get_mass(self)
        self.d = Plummer.get_dens(self)
        self.p = Plummer.get_pot(self)
        
    def get_mass(self):
        return self.r *self.r * np.power((self.r + self.rscl),-2.)
    
    def get_dens(self):
        return self.rscl * (2.*np.pi*self.r)**-1. * (self.r + self.rscl)**-3.
    
    def get_pot(self):
        return -1. * ( (self.r+self.rscl)**-1.)




