####################################
#
# Tools to compute Dark Matter Direct Detection Parameters
#
#    MSP 2016
#    Added to exptool 01.24.17
#
#    Built to encompass the calculations that are published in Petersen, Katz, Weinberg 2016 PRD.
#


import numpy as np
from scipy import interpolate

def velocity_pack(Particles,vlsr,pec=[0,0,0],orb=[0,0,0]):
    #
    # MAKE VELOCITIES
    #
    # Available metrics:
    #     Vtot: total velocity
    #     Vrad: radial velocity component
    #     Vcirc: tangential velocity component
    #     Vzrel: total velocity removing the galactic circular velocity
    #     Vtrel: total velocity removing galactic circular velocity AND peculiar velocities
    #     Vmrel: total velocity with Vc, pec, and annual orbital motion
    Vtot = (Particles.xvel*Particles.xvel + Particles.yvel*Particles.yvel + Particles.zvel*Particles.zvel)**0.5
    Vrad = ((Particles.xpos*Particles.xvel)+(Particles.ypos*Particles.yvel))/(Particles.xpos**2.+Particles.ypos**2.)**0.5
    Vcirc = ((Particles.xpos*Particles.yvel)-(Particles.ypos*Particles.xvel))/(Particles.xpos**2.+Particles.ypos**2.)**0.5
    Vrel = Vcirc-vlsr
    #
    # should we assume some token vertical motion?
    Vzrel = (Vrad*Vrad + Vrel*Vrel + Particles.zvel*Particles.zvel)**0.5 # does this need the radial velocities?? Yes, it's cylindrical!
    Vtrel = ( (Vrad-pec[0])**2. + (Vrel-pec[1])**2. + (Particles.zvel-pec[2])**2.)**0.5
    Vmrel = ( (Vrad-pec[0]-orb[0])**2. + (Vrel-pec[1]-orb[1])**2. + (Particles.zvel-pec[2]-orb[2])**2.)**0.5
    return Vtot,Vrad,Vcirc,Vzrel,Vtrel,Vmrel





def calc_geevee(veldist,scalefac,vbins = np.linspace(0.0,4.5,45)):
    vmins = vbins*scalefac
    nMAT = norma_dist(vmins,veldist)
    gvee = np.zeros_like(vmins)
    for i,vthresh in enumerate(vmins):
        goodg =  np.where(vmins > (vthresh))[0]
        gvee[i] = np.sum(nMAT[goodg]*(vbins[1]-vbins[0])/(vbins[goodg]))
    return vmins,nMAT,gvee



#############################################################################################
#
#   Standard Halo Model CALCULATIONS
#
#############################################################################################


def multimax(v1,v2,v3,sigma1,sigma2,sigma3,cen1=0.,cen2=0.,cen3=0.):
    return 4.*np.pi*(v1**2.+v2**2.+v3**2.)*np.exp( -((v1-cen1)**2./(2.*sigma1**2.)) -((v2-cen2)**2./(2.*sigma2**2.)) -((v3-cen3)**2./(2.*sigma3**2.)))



def norma_dist(x,y):
    # only for an evenly spaced distribution
    dx = x[1]-x[0]
    fac = np.sum(dx*y)
    return y/fac


def positize(inputv,inputd):
    # positize a velocity distribution
    sbins = np.linspace(0.,5.,100)
    sbinsb = np.zeros_like(sbins)
    for i in range(0,len(inputd)):
        wb = abs((abs(inputv[i])-sbins)).argmin()
        sbinsb[wb] += inputd[i]
    return sbins,sbinsb


def shm_geevee(vmins,vlsr,scalefac,pec=[0,0,0]):
    #
    # Do the SHM comparison
    #
    # Requires multimax,positize,norma_dist
    #
    vdist = np.linspace(-5.,5.,50000)
    vdisp = vlsr*np.sqrt(1.5)
    mw = multimax(vdist,vdist,vdist,vdisp,vdisp,vdisp,cen1=pec[0]/scalefac,cen2=( (scalefac*vlsr) +pec[1])/scalefac,cen3=pec[2]/scalefac)
    Vv,Aa = positize(vdist,mw)
    Aa = norma_dist(Vv*scalefac,Aa)
    gSee = np.zeros_like(vmins)
    for i,vthresh in enumerate(vmins):
        goodg =  np.where(Vv*scalefac > (vthresh))[0]
        gSee[i] = np.sum(Aa[goodg]*(Vv[1]-Vv[0])/(Vv[goodg]))
    return Vv,Aa,gSee




#############################################################################################
#
#   Mock m sigma plane
#
#############################################################################################


def make_m_sigma(mxs,vmins,gSee,A,Z,Er,rho0=0.3):
    # constants
    mp = 0.9382
    mn = 0.93957
    c2 = 8.88e10 # speed of light, squared
    #
    # could change these if needed
    #
    vminarr = np.zeros_like(mxs)
    for i,mx in enumerate(mxs):
        tmass = (A-Z)*mn + Z*mp
        mu = tmass*mx/(tmass+mx)
        red = tmass*c2/(2.*mu**2.)
        vminarr[i] = (Er*1.e-6*red)**0.5
    #print vminarr
    # SHM value
    g = interpolate.interp1d(vmins,gSee)
    SHMg = np.zeros_like(mxs)
    for i in range(0,len(mxs)):
        #print vminarr[i],np.max(vmins)
        if vminarr[i] < np.max(vmins):
            SHMg[i] = g(vminarr[i])
        else:
            SHMg[i] = np.nan
    sigma = (mn*mxs)/(rho0*SHMg)
    return sigma


