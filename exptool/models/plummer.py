###########################################################################################
#
#  plummer.py
#     Compute analytic Plummer models
#
# 31 Oct 2017: First construction
# 09 Jul 2021: Upgrades I
#
#
'''

plummer (part of exptool.models)
    Implementation of the analytic Plummer models

'''
from __future__ import absolute_import, division, print_function, unicode_literals

# general python imports
import numpy as np


class Plummer():
    '''
    Plummer class model

    The Wikipedia page is actually pretty good!

    P = Plummer(rscl=.1)
    print(P.get_pot(1.))

    '''
    def __init__(self,rscl=1.0):
        self.rscl = rscl
        
    def get_mass(self,r):
        return r *r * np.power((r + self.rscl),-2.)
    
    def get_dens(self,r):
        return self.rscl * (2.*np.pi*r)**-1. * (r + self.rscl)**-3.

    def get_cartesian_forces(self,x,y,z):
        r3 = np.sqrt(x*x+y*y+z*z)
        fr = self.get_force(r3)

        fx = x*fr/r3
        fy = y*fr/r3
        fz = z*fr/r3

        return fx,fy,fz

    def get_cartesian_forces_array(self,arr):
        x,y,z = arr[0],arr[1],arr[2]
        r3 = np.sqrt(x*x+y*y+z*z)
        fr = self.get_force(r3)

        fx = -x*fr/r3
        fy = -y*fr/r3
        fz = -z*fr/r3

        return np.array([fx,fy,fz])
    
    def get_force(self,r):
        """B&T eq. 2.44a
        
        G=M=1
        """
        b = (r*r + self.rscl*self.rscl)
        return  r / np.power(b,1.5)
       
    def get_pot(self,r):
        """B&T eq. 2.44a
        
        G=M=1
        """
        return -1. * ( (r**2.+ self.rscl**2.)**-0.5)


