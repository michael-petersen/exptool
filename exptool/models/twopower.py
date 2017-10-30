###########################################################################################
#
#  twopower.py
#     Compute twopower halo models
#
# 10-29-2017: First construction
#
#
#
'''


twopower (part of exptool.models)
    Implementation of Martin Weinberg's twopower.cc



R,D,M,P,DP = twopower.gen_model(200.,0.0025,0.0,rtarget=0.005,alpha=1.,beta=4.,rtrunc=0.01,wtrunc=0.2,rmax=0.1)


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
from exptool.io import psp



def gen_model(concentration,outmass,rcore,logarithmic=True,heggie=False,verbose=False,truncate=True,number=4000,rmin=1e-5,rmax=2.,alpha=1.,beta=2.,W=-0.5,rtrunc=1.25,wtrunc=0.3,rtarget=1.0):
    '''
    gen_twopower


    inputs
    -----------
    concentration
    outmass
    rcore
    logarith


    '''
    rs = 1./concentration
    r =  np.zeros(number)
    d =  np.zeros(number)
    m =  np.zeros(number)
    pw = np.zeros(number)
    p0 = np.zeros(number)
    if (logarithmic): dr = (np.log(rmax) - np.log(rmin))/(number - 1)
    else: dr = (rmax - rmin)/(number - 1)
    for i in range(1,number):
        if (logarithmic):
            r[i] = rmin*np.exp(dr*(i-1))
        else:
            r[i] = rmin + dr*(i-1)
        #
        #
        d[i] = rs**3.*(((r[i]+rcore)**-alpha) * ((r[i]+rs)**-beta))
        #d[i] = (r[i]/rs)**-alpha * (1.+ (r[i]/rs))**-beta
        if (truncate):
            #print('Truncating...')
            d[i] *= 0.5*(1.0 - math.erf((r[i]-rtrunc)/wtrunc))
    m[1] = 0.
    pw[1] = 0.        
    for i in range(2,number):
        m[i] = m[i-1] + 2.0*np.pi*(r[i-1]*r[i-1]*d[i-1] + r[i]*r[i]*d[i])*(r[i] - r[i-1])
        pw[i] = pw[i-1] + 2.0*np.pi*(r[i-1]*d[i-1] + r[i]*d[i])*(r[i] - r[i-1])
    for i in range(1,number): 
        p0[i] = -m[i]/(r[i]+1.0e-10) - (pw[-1] - pw[i])
    W0 = 0.0
    for i in range(1,number): W0 += np.pi*(r[i-1]*r[i-1]*d[i-1]*p0[i-1] + r[i]*r[i]*d[i]*p0[i-1])*(r[i] - r[i-1]);
    print("orig PE = ",W0)
    M0 = m[-1];
    R0 = r[-1];
    #
    # Compute new scaling
    #
    # first grab mass at R=1. for the proper scaling
    #
    M0 = m[ (abs(r-rtarget)).argmin()]
    #
    if (heggie): 
        Beta = (W/W0) * (M0/outmass)
        Gamma = (W/W0)**1.5 * (M0/outmass)**3.5
        #print "! Scaling:  W=",W,"  M=",M
    else:
        Beta = (outmass/M0) * (R0/rmax);
        Gamma = np.sqrt((M0*R0)/(outmass*rmax)) * (R0/rmax);
        #print "! Scaling:  R=",R,"  M=",M
    #if (truncate): print >>f,"!  alpha=",alpha,"  beta=",beta,"  rcore=",rcore,"  rtrunc=",rtrunc,"  wtrunc=",wtrunc
    #else: print >>f,"!  alpha=",alpha,"  beta=",beta,"  rcore=",rcore
    #print >>f,"! 1) = r   2) = rho   3) = M(r)   4) U(r) "
    #print >>f,number
    rfac = Beta**-0.25 * Gamma**-0.5
    dfac = Beta**1.5 * Gamma
    mfac = Beta**0.75 * Gamma**-0.5
    pfac = Beta;
    #for i in range(1,number): print >>f,r[i]*rfac, d[i]*dfac, m[i]*mfac, p0[i]*pfac;
    #
    # Sanity checks
    #
    M0 = 0.0;
    W0 = 0.0;
    for i in range(1,number):
        M0 += 2.0*np.pi*(r[i-1]*r[i-1]*d[i-1] + r[i]*  r[i]*  d[i]   ) * (r[i] - r[i-1]);
        W0 += np.pi*(r[i-1]**2.*d[i-1]*p0[i-1] + r[i]**2. * d[i]*  p0[i-1]) * (r[i] - r[i-1]);
    print("new M0 = " , M0*mfac)
    print("new PE = " , W0*rfac**3.*dfac*pfac)
    print("Rfac = " , rfac)
    print("Dfac = " , dfac)
    print("Mfac = " , mfac)
    print("Pfac = " , pfac)
    R = r[1:-1]*rfac
    D = d[1:-1]*dfac
    M = m[1:-1]*mfac
    P = p0[1:-1]*pfac
    #
    # dPotential
    #
    DP = np.zeros(number-2)
    for i in range(2,number-2): DP[i-1] = ((p0[i]-p0[i-1])*pfac)/((r[i]-r[i-1])*rfac)
    return R,D,M,P,DP



def write_model(outputfile,R,D,M,P):
    '''
    outputfile = '/scratch/mpetersen/Disk025/SLGridBULGE2.model'

    '''
    f = open(outputfile,'w')
    print('! Designed Hernquist bulge model, a=0.003, m=0.0025',file=f)
    print('! twopower call: R,D,M,P,DP = gen_twopower(200.,0.0025,0.0,rtarget=0.005,alpha=1.,beta=4.,rtrunc=0.01,wtrunc=0.2,rmax=0.1)',file=f)
    print(len(R),file=f)
    for i in range(0,len(R)):
        print(R[i],D[i],M[i],P[i],file=f)
    #
    f.close()




