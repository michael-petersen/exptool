###########################################################################################
#
#  laguerre.py
#     Compute cofficients and forces in an analytic Laguerre basis.
#
#     
#
# 04-Feb-21: first introduced.
#
'''     

laguerre.py (part of exptool.basis)
    Implementation of a 2-d laguerre basis



'''

# general python imports
import numpy as np
import time
import sys
import os


# the generalised Laguerre polynomial evaluator...
from scipy.special import eval_genlaguerre


def gamma_n(n,rscl):
    """define the Laguerre normalisation"""
    return (rscl/2.)*np.sqrt(n+1.)

def G_n(R,n,rscl):
    """define the Laguerre basis"""
    return np.exp(-R/rscl)*eval_genlaguerre(n,1,2*R/rscl)/gamma_n(n,rscl)

def n_m(m):
    """deltam0 is 0 for all orders except m=0, when it is 1.
    this is the angular normalisation."""
    deltam0 = 0.
    if m==0: deltam0 = 1.
    return np.power( (deltam0+1)*np.pi/2. , -0.5)



