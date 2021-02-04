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

# for orthogonality checks
from scipy import integrate



def gamma_n(n,rscl):
    """define the Laguerre normalisation. This is the L_2 norm."""
    return (rscl/2.)*np.sqrt(n+1.)

def G_n(R,n,rscl):
    """define the Laguerre basis with weighting"""
    return np.exp(-R/rscl)*eval_genlaguerre(n,1,2*R/rscl)/gamma_n(n,rscl)

def L_n(R,n,rscl):
    """define the Laguerre basis without weighting"""
    return eval_genlaguerre(n,1,2*R/rscl)

def n_m(m):
    """deltam0 is 0 for all orders except m=0, when it is 1.
    this is the angular normalisation."""
    deltam0 = 0.
    if m==0: deltam0 = 1.
    return np.power( (deltam0+1)*np.pi/2. , -0.5)



def laguerre_amplitudes(R,mass,phi,velocity,rscl,m,n):
    """equation 42 of WP2020
    
    force alpha=0 in the eval_genlaguerre call
    
    """
    cosm = n_m(m)*np.cos(m*phi)*velocity
    sinm = n_m(m)*np.sin(m*phi)*velocity
 
    G_j = G_n(R,n,rscl)
    
    return np.sum(mass * G_j * cosm),np.sum(mass * G_j * sinm) 


'''


# the original orthogonality check
f1 = lambda x,n,d: x*G_n(x,n,1.)*G_n(x,n+d,1.)

# orthogonality check with out the measure in the functions. force rscl=1.
f2 = lambda x,n,d: np.exp(-x/1.)*x*L_n(x,n,2.)*L_n(x,n+d,2.)/(n+1)


# first arg is the Laguerre order to check
# second arg is the offset from the Laguerre order. Set as 0 for othogonality tests
print(integrate.quad(f1, 0., np.inf,args=(5,0))[0])
print(integrate.quad(f2, 0., np.inf,args=(2,0))[0])

# prove that offset functions are 0.
print(integrate.quad(f1, 0., np.inf,args=(0,1))[0])






rscl = 3.
morder = 2
for n in range(0,20):
    print(laguerre_amplitudes(rval,np.ones(rval.size),phi,svel,rscl,morder,n))

# resample a Laguerre field

xvals = np.linspace(-13.,13.,200)
xx,yy = np.meshgrid(xvals,xvals)
rr,pp = np.sqrt(xx*xx+yy*yy),np.arctan2(yy,xx)



mmax = 12
nmax = 12

rscl = 3.

ffc = 0
ffs = 0

gvel[np.abs(gvel)>500] = 0.

for m in range(0,mmax):
    for n in range(0,nmax):
        tmpffc,tmpffs = laguerre_amplitudes(rval,np.ones(rval.size),phi,svel,rscl,m,n)

        ffc += (0.5/np.pi)*tmpffc*n_m(m)*np.cos(m*pp)*G_n(rr,n,3.)
        ffs += (0.5/np.pi)*tmpffs*n_m(m)*np.sin(m*pp)*G_n(rr,n,3.)



'''
