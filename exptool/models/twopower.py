###########################################################################################
#
#  twopower.py
#     Compute twopower halo models
#
# 10-29-2017: First construction
# 04-Feb-2021: revamp
#
#
'''


twopower (part of exptool.models)
    Implementation of Martin Weinberg's twopower.cc



if you'd like a MW-like model, try this:


old usage technique:
R,D,M,P,DP = twopower.gen_model(15.,1.,0.01,rtarget=1.,alpha=1.,beta=2.,rtrunc=30,wtrunc=5,rmax=2.)
twopower.write_model(outputfile,R,D,M,P)


'''

# general python imports
import numpy as np
import math
import time
import sys
import os


def Hernquist(r,a,rc=0,rho0=1.,beta=0):
    """return unscaled Herquist density
    
    r      : the radius to evaluate
    a      : the scale radius
    rc     : the core radius
    rho0   : the central density
    
    
    """
    ra = r/a
    return rho0/((ra+rc)*((1+ra)**3.))

def NFW(r,a,rc,beta=0):
    """return unscaled NFW density"""
    ra = r/a
    return 1./((ra+rc)*((1+ra)**2.))


def powerhalo(r,rs=1.,rc=0.,alpha=1.,beta=1.e-7):
    """return generic twopower law distribution
    
    inputs
    ----------
    r      : (float) radius values
    rs     : (float, default=1.) scale radius 
    rc     : (float, default=0. i.e. no core) core radius
    alpha  : (float, default=1.) inner halo slope
    beta   : (float, default=1.e-7) outer halo slope
    
    returns
    ----------
    densities evaluated at r
    
    notes
    ----------
    different combinations are known distributions.
    alpha=1,beta=2 is NFW
    alpha=1,beta=3 is Hernquist
    alpha=2.5,beta=0 is a typical single power law halo
    
    
    """
    ra = r/rs
    return 1./(((ra+rc)**alpha)*((1+ra)**beta))



def make_twopower_model(func,M,R,rs,alpha=1.,beta=1.e-7,rc=0.,\
                        pfile='',plabel='',\
                        verbose=False,truncate=True,\
                        perc=97.,numr=4000):
    """make a two power distribution
    
    inputs
    -------------
    func
    M
    R
    rs
    alpha
    beta
    rc
    pfile
    plabel
    verbose
    truncate
    perc
    numr
    
    
    """
    
    # set the radius values
    rvals = 10.**np.linspace(-6.,np.log10(2.),numr)
    
    # original flexible version
    rtrunc = np.nanpercentile(rvals,perc)
    
    # hardwired so edge is sharp
    rtrunc = np.nanpercentile(rvals,99.9)
    wtrunc = np.nanpercentile(rvals,99.9)-np.nanpercentile(rvals,perc)
    
    if verbose: 
        print('Truncation settings: rtrunc={0:3.2f},wtrunc={1:3.2f}'.format(rtrunc,wtrunc))


    # query out the density values
    dvals = func(rvals,rs,rc,alpha=alpha,beta=beta)
    
    # apply the truncation
    if truncate:
        dvals *= 0.5*(1.0 - scipy.special.erf((rvals-rtrunc)/wtrunc))
    
    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
    
    # evaluate potential
    for indx in range(0,dvals.size):
        pvals[indx] = -mvals[indx]/(rvals[indx]+1.e-10) - (pwvals[dvals.size-1] - pwvals[indx])
    
    
    # prepare to rescale by potential energy
    W0 = 0.0;
    for indx in range(0,dvals.size): 
        W0 += np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1]*pvals[indx-1] + \
                rvals[indx]*rvals[indx]*dvals[indx]*pvals[indx-1])*(rvals[indx] - rvals[indx-1]);

    if verbose:
        print("orig PE = ",W0 )

    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0:12.10f} {1:15.7f} {2:16.15f} {3:16.14f}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)
    
        f.close()
    
    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


def check_concentration(R,D):
    """
    check the  concentration of a halo
    by finding where the power law is most similar to alpha^-2
    
    return 1./radius, which is the concentration.
    (so find the scale radius by taking 1./concentration)
    
    """
    func = np.log10(R**-2.)-np.log10(D)
    print('Concentration={}'.format(1./R[np.nanargmin(func)]))
    





"""
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

"""


