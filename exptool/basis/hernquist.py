###########################################################################################
#
#  hernquist.py
#     Compute cofficients and forces in an analytic Hernquist basis.
#
#     
#
# 04 Feb 21: first introduced.
#
'''     

hernquist.py (part of exptool.basis)
    Implementation of Hernquist+Ostriker basis

see 
src/Hernquist.H
src/Hernquist.cc

This code is a mixture of two Hernquist implementations:
1. straight Hernquist+Ostriker (1992) equations
2. Weinberg's numerically-favourable implementation

Much of the Weinberg numerically-favourable version stems from
Clutton-Brock (1973) notation.

Did you know that the scale radius (a) of the Hernquist sphere is related
to the half-mass radius as 
r_{M_1/2} = (1+np.sqrt(2.))*a
? [From Herquist & Ostriker 1992, after equation 2.2]

'''

# no more python2 compatibility; only assuming python3 from now on.
# python2/3 compatibility
#from __future__ import absolute_import, division, print_function, unicode_literals

# general python imports
import numpy as np
import time
import sys
import os


# exptool imports
#from exptool.utils import utils
#from exptool.utils import halo_methods
#from exptool.io import psp_io


# special math imports
from scipy.special import gammaln
from scipy.special import gamma
from scipy.special import gegenbauer
from scipy.special import factorial
from scipy.special import eval_gegenbauer
from scipy.special import sph_harm

# multiprocessing imports
import multiprocessing
import itertools
from multiprocessing import Pool, freeze_support




def dgammln(xx):
    """
    Compute the double precision log gamma function
    
    exputil/dgammln.cc
    
    """

    cof = [76.18009173,-86.50532033,24.01409822,
                        -1.231739516,0.120858003e-2,-0.536382e-5];

    if (xx <= 0.0):
        return 0.0;
    else:
        x=xx-1.0;
        tmp=x+5.5;
        tmp -= (x+0.5)*log(tmp);
        ser=1.0;

        for j in range(0,6):#(j=0;j<=5;j++) {
            x += 1.0;
            ser += cof[j]/x;

        return -tmp+np.log(2.50662827465*ser);



def Knl(n, l):
    """
    return Hernquist K_{nl}
    
    Hernqusit & Ostriker 1992 eq. 2.23
    
    Garavito-Camargo et al. eq. A5
    
    """
    return 0.5*n*(n+4*l+3) + (l+1)*(2*l+1);


def Knl_mat(lmax,nmax,rep=1):
    """
    return Hernquist K_{nl}
    
    Hernqusit & Ostriker 1992 eq. 2.23
    
    Garavito-Camargo et al. eq. A5
    
    """
    
    p = np.zeros([lmax+1,nmax])

    for l in range(0,lmax+1):
        for n in range(0,nmax):
            p[l][n] = 0.5*n*(n+4*l+3) + (l+1)*(2*l+1);

    if rep == 1:
        return p
    else:
        fullp = np.zeros([lmax+1,nmax,rep])
        for i in range(0,rep):
            fullp[:,:,i] = p
        
        return fullp


def Hernquist_norm(n, l):
    """
      return M_PI * knl(n,l) * 
        exp( 
    -log(2.0)*((double)(8*l+4))
    - lgamma((double)(1+n)) - 2.0*lgamma((double)(1.5+2.0*l))
    + lgamma((double)(4*l+n+3))
    )/(double)(2*l+n+1.5);

    """
    #extern double dgammln(double);
    
    return np.pi * Knl(n,l) * np.exp( -np.log(2.0)*((8*l+4)) \
                                     - dgammln((1+n)) \
                                     - 2.0*dgammln((1.5+2.0*l)) \
                                     + dgammln((4*l+n+3)))/(2*l+n+1.5);



def Inl(n,l):
    """
    return the Hernquist normalisation factor
    
    
    
    Hernquist & Ostriker 1992, eq. 2.31
    
    """
    
    return -Knl(n,l) * \
            (4.*np.pi/np.power(2,8*l+6)) * \
            gamma(n + 4*l + 3)/\
            (factorial(n)*(n + 2*l + 1.5) * (gamma(2*l+1.5))**2.)
    
    
    

def get_dpotl(lmax, nmax, r):
    """
    
    
    """
    x = r_to_rq(r);
    dx = d_r_to_rq(r);
    
    fac = 0.25*(1.0 - x*x);
    rfac = 0.5*(1.0 - x);
    drfac = -1.0/(1.0 - x*x);
    
    for l in range(0,lmax+1):#(l=0; l<=lmax; l++) {
        dfac1 = 1.0 + x + 2.0*x*l;
        dfac2 = 4.0*l + 3.0;

        u  = get_ultra(nmax-1, 2.0*l+0.5, x, u[tid]);
        du = get_ultra(nmax-1, 2.0*l+1.5, x, du[tid]);

    for n in range(0,nmax):#(n=0; n<nmax; n++) 
        p[l][n+1] = rfac*u[n];
    
    dp[l][1] = dx*drfac*dfac1*rfac*u[0];
    
    for n in range(1,nmax):#(n=1; n<nmax; n++) 
        dp[l][n+1] = dx*rfac*(drfac*dfac1*u[n] + dfac2*du[n-1]);

    rfac *= fac;

    return p,dp


def get_potl(lmax, nmax, r):
    """
    get the Hernquist potential value 
    
    """
    
    p = np.zeros([lmax+1,nmax])

    x = r_to_rq(r);
    fac = 0.25*(1.0 - x*x);
    rfac = 0.5*(1.0 - x);
  
    for l in range(0,lmax+1):
        u = get_ultra(nmax-1, 2.0*l+1.5, x);
    
        p[l] = rfac*u
        
        # increase the prefactor...
        rfac *= fac;

    return p


def get_dens(lmax, nmax, r):
    
    p = np.zeros([lmax+1,nmax,r.size])
    
    x = r_to_rq(r);
    fac = 0.25*(1.0 - x*x);
    rfac = 0.25*pow(1.0 - x, 5.0)/(1.0 - x*x);
    
    knl = Knl_mat(lmax,nmax,r.size)
    
    for l in range(0,lmax+1):#(l=0; l<=lmax; l++) {
        u = get_ultra(nmax-1, 2.0*l+0.5, x);
        
        p[l] = knl[l]*rfac*u;
            
        rfac *= fac;

    return p
        
    
def get_potl_dens(lmax, nmax, r):
    """
    get Hernquist potential and density
    
    inputs
    ------------
    
    returns
    ------------
    p
    d
    
    
    requires
    ------------
    get_ultra
    
    """

    p = np.zeros([lmax+1,nmax,r.size])
    d = np.zeros([lmax+1,nmax,r.size])

    
    x = r_to_rq(r);
    fac = 0.25*(1.0 - x*x);
    rfacp = 0.5*(1.0 - x);
    rfacd = 0.25*pow(1.0 - x, 5.0)/(1.0 - x*x);
    
    knl = Knl_mat(lmax,nmax,r.size)

        
    for l in range(0,lmax+1):#(l=0; l<=lmax; l++)
        u = get_ultra(nmax-1, 2.0*l+0.5, x);
        
        p[l] = rfacp*u
        d[l] = knl[l]*rfacd*u;

        
        rfacp *= fac;
        rfacd *= fac;

    return p,d



def rq_to_r(rq):
    """
    Convert to reduced coordinate                               
                                                                        
                  r - 1                                                 
           rq =  -------                                                
                  r + 1  
    
    Hernquist & Ostriker 1992 eq. 2.15
    """
    BIG = 1.e30
    if (rq>=1.0):
        return BIG;
    else:
        return (1.0+rq)/(1.0-rq);


def r_to_rq(r):
    """
    reduced coordinate inverse:                                   
                                                        
               (1+rq)
           r = ------
               (1-rq)
    
    Hernquist & Ostriker 1992 eq. 2.16
    """
    return (r-1.0)/(r+1.0);


def d_r_to_rq(r):
    """
    derivative of inverse coordinate transformation
    
    Hernquist & Ostriker 1992 eq. 2.17
    """
    fac = r + 1.0;
    return 2.0/(fac*fac);


def Knl_mat(lmax,nmax,rep=1):
    """
    return Hernquist K_{nl}
    
    Hernqusit & Ostriker 1992 eq. 2.23
    
    Garavito-Camargo et al. eq. A5
    
    """
    
    p = np.zeros([lmax+1,nmax])

    for l in range(0,lmax+1):
        for n in range(0,nmax):
            p[l][n] = 0.5*n*(n+4*l+3) + (l+1)*(2*l+1);

    if rep == 1:
        return p
    else:
        fullp = np.zeros([lmax+1,nmax,rep])
        for i in range(0,rep):
            fullp[:,:,i] = p
        
        return fullp



def r_to_zeta(r):
    return (r-1.)/(r+1.)


def get_potl(lmax, nmax, r):
    """
    get the Hernquist potential value 

    inputs
    lmax : 
    nmax :
    r    :
    
    """
    
    p = np.zeros([lmax+1,nmax,r.size])

    zeta = r_to_zeta(r);
    #fac = 0.25*(1.0 - x*x);
    #rfac = 0.5*(1.0 - x);
  
    for l in range(0,lmax+1):
        lfac = -( r**l )/((1.+r)**(2.*l+1.))
        
        for n in range(0,nmax):
            u = eval_gegenbauer(n, 2.0*l+1.5, zeta);
            p[l][n] = lfac*u

    return p


#rvals = 10.**np.linspace(-4,0.1,1000)
#H = get_potl(6,10,rvals/.08)


def get_potl_martin(lmax, nmax, r):
    """
    get the Hernquist potential value 
    
    """
    
    p = np.zeros([lmax+1,nmax,r.size])

    x = r_to_rq(r);
    fac = 0.25*(1.0 - x*x);
    rfac = 0.5*(1.0 - x);
    
    for l in range(0,lmax+1):
        
        # I'm not sure why this is 0.5...must be the ultraspherical implementation
        u = get_ultra(nmax-1, 2.0*l+0.5, x);

        p[l] = rfac*u
        
        # increase the prefactor...
        rfac *= fac;

    return p


def get_dens_martin(lmax, nmax, r):
    
    p = np.zeros([lmax+1,nmax,r.size])
    
    x = r_to_rq(r);
    fac = 0.25*(1.0 - x*x);
    rfac = 0.25*pow(1.0 - x, 5.0)/(1.0 - x*x);
    
    knl = Knl_mat(lmax,nmax,r.size)
    print(knl.shape)
    
    for l in range(0,lmax+1):#(l=0; l<=lmax; l++) {
        u = get_ultra(nmax-1, 2.0*l+0.5, x);
        
        p[l] = knl[l]*rfac*u;
            
        rfac *= fac;

    return p
        
    
#rvals = 10.**np.linspace(-4,0.,1000)
#H2 = get_dens_martin(6,10,rvals/.06)




def get_ultra(nmax, l, x):
    """
    nmax   : the maximum order
    l      : the harmonic order
    x      : the position to evalute the ultraspherical polynomial
    
    returns
    ------------
    p      : vector of ultraspherical polynomials, 
             recursively found up to order nmax
    
    src/ultrasphere.cc
    
    The ultraspherical, or Gegenbauer polynomials
    See Hernquist & Ostriker 1992
    
    Written as C^{(\alpha)}_n(\zeta). We want \alpha=(2*l+3/2).
    
    
    
    """
    p = np.zeros([nmax+1,x.size])
    p[0] = u2 = 1.0;
    p[1] = u  = 2.0*x*(l+1.0);
    
    for j in range(2,nmax+1): #(j=2; j<=nmax; j++) {
        u1 = u2;
        u2 = u;    
        a = 2.0*x*(l+j)/(j);
        b = -(2.0*l + j)/(j);
        # check that these come out as doubles, please.

        p[j] = u = a*u2 + b*u1;

    return p

