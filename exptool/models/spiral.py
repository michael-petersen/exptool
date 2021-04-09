###########################################################3
#
# 09-Apr-2021: introduction

#
"""
spiral (Cox & Gomez)

Follow notes from Hunt & Bovy (2018) for the demo application

I'm struck that the spirals continue to unwind, whereas in simulations
we see focused tight densities even at large radii. How would one
construct a model to reflect this? Make the arm distribution
`peakier', I guess!

"""
# standard python modules
import numpy as np



# code up Cox & Gomez potential

def return_Kn(n,N,r,alpha):
    """
    Cox & Gomez equation 5
    
    """
    return (n*N)/(r*np.sin(alpha))

def return_betan(n,N,r,alpha,H):
    """
    Cox & Gomez equation 6
    
    """
    Kn = return_Kn(n,N,r,alpha)
    return Kn*H*(1+0.4*Kn*H)

def return_Dn(n,N,r,alpha,H):
    """
    Cox & Gomez equation 7
    
    """
    Kn = return_Kn(n,N,r,alpha)
    return (1 + Kn*H + 0.3*Kn*Kn*H*H)/(1+0.3*Kn*H)
    
    
def return_gamma(r,phi,r0,alpha,phip=0.):
    """
    Cox & Gomez equation 3
    
    phip=0 by default; can change the phi location of the peak if desired
    """
    return N*(phi - phip - np.log(r/r0)/np.tan(alpha))



def phase_pattern(r,phi,z,phip,r0,alpha,H):
    """
    Cox & Gomez equation 4
    """
    #print(r,phi,N,phip,r0,alpha)
    gamma = return_gamma(r,phi,r0,alpha,phip)
    
    # loop to do the n sum
    nsum = 0
    for n0,Cn in enumerate([8./(3*np.pi),0.5,8./(15.*np.pi)]):
        # check out the enumerate function
        # n is the index, in this case, [0,1,2]
        # but we want [1,2,3]
        # so add 1 immediately
        n = n0+1

        nsum += np.cos(n*gamma)
    
    return nsum




def combined_arms_pot(r,phi,z,N,phip,r0,alpha,H):
    """
    sum part of Cox & Gomez equation 8
    """
    #print(r,phi,N,phip,r0,alpha)
    gamma = return_gamma(r,phi,r0,alpha,phip)
    
    # loop to do the n sum
    nsum = 0
    for n0,Cn in enumerate([8./(3*np.pi),0.5,8./(15.*np.pi)]):
        # check out the enumerate function
        # n is the index, in this case, [0,1,2]
        # but we want [1,2,3]
        # so add 1 immediately
        n = n0+1
        Kn = return_Kn(n,N,r,alpha)
        Dn = return_Dn(n,N,r,alpha,H)
        Bn = return_betan(n,N,r,alpha,H)
        
        term1 = (Cn/(Kn*Dn))
        term2 = np.cos(n*gamma)*np.power((1./np.cosh((Kn*z)/(Bn))),Bn)
        
        nsum += term1*term2
    
    return nsum


def spiral_pot(r,phi,z,rho0,r0,Rs,N,alpha,H,phip=0):
    """
    Cox & Gomez equation 8
    
    N     : number of arms
    alpha : pitch angle (in radians?)
    Rs    : radial scale length of drop-off in density amplitude of arms
    rho0  : midplane arm density
    r0    : fiducial radius (peak of arm density)
    H     : scale height of the arm perturbation
    
    set G=1
    """
    G = 1
    
    nsum   = combined_arms_pot(r,phi,z,N,phip,r0,alpha,H)
    expval = np.exp( -(r-r0)/(Rs))
    prefac = -4.*np.pi*G*rho0
    
    return prefac*expval*nsum
    
    
def combined_arms_dens(r,phi,z,N,phip,r0,alpha,H):
    """
    sum part of Cox & Gomez equation 10
    """
    #print(r,phi,N,phip,r0,alpha)
    gamma = return_gamma(r,phi,r0,alpha,phip)
    
    # loop to do the n sum
    nsum = 0
    for n0,Cn in enumerate([8./(3*np.pi),0.5,8./(15.*np.pi)]):
        # check out the enumerate function
        # n is the index, in this case, [0,1,2]
        # but we want [1,2,3]
        # so add 1 immediately
        n = n0+1
        Kn = return_Kn(n,N,r,alpha)
        Dn = return_Dn(n,N,r,alpha,H)
        Bn = return_betan(n,N,r,alpha,H)
        
        term1 = Cn * ( ((Kn*H)/(Dn))*((Bn+1)/(Bn)) )
        term2 = np.cos(n*gamma)*np.power((1./np.cosh((Kn*z)/(Bn))),2+Bn)
        
        nsum += term1*term2
    
    return nsum



    
def spiral_dens(r,phi,z,rho0,r0,Rs,N,alpha,H,phip=0):
    """
    Cox & Gomez equation 10
    
    N     : number of arms
    alpha : pitch angle (in radians?)
    Rs    : radial scale length of drop-off in density amplitude of arms
    rho0  : midplane arm density
    r0    : fiducial radius (peak of arm density)
    H     : scale height of the arm perturbation
    
    """
    
    nsum   = combined_arms_dens(r,phi,z,N,phip,r0,alpha,H)
    expval = np.exp( -(r-r0)/(Rs))
    prefac = rho0
    
    return prefac*expval*nsum


"""

# plotting utilities
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors



# reproduce Figure 1 from Hunt
r = 1.0
phi = np.linspace(0.,2.*np.pi,100)
z = 0.1
N = 2.
phip = 0.
r0 = 1.
alpha = np.deg2rad(12.)
H = 1.

modulation = phase_pattern(r,phi,z,phip,r0,alpha,H)

plt.plot(phi,modulation,color='black')
plt.plot(phi,np.nanmax(modulation)*np.cos(2*phi),color='black',linestyle='dashed')


# reproduce Figure 4
N     = 2
alpha = np.deg2rad(15.)
Rs    = 7. # kpc
rho0  = 1.#m*n0, can set this later
r0    = 8. # kpc
H     = 0.18 # kpc

rvals = np.linspace(5.,11.,100)
zvals = np.linspace(-1.5,1.5,100)
RR,ZZ = np.meshgrid(rvals,zvals)

outpot = spiral_pot(RR,np.deg2rad(45.),ZZ,rho0,r0,Rs,N,alpha,H,phip=0.)


fig = plt.figure()
ax = fig.gca(projection='3d')



# Plot the surface.
surf = ax.plot_surface(RR, ZZ, outpot, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



# reproduce Figure 11

N     = 2
alpha = np.deg2rad(15.)
Rs    = 7. # kpc
rho0  = 1.#m*n0, can set this later
r0    = 8. # kpc
H     = 0.18 # kpc

rvals = np.linspace(3.,22.,100)
zvals = np.linspace(0.,2.*np.pi,100)
RR,PP = np.meshgrid(rvals,zvals)

outpot = spiral_pot(RR,PP,0.,rho0,r0,Rs,N,alpha,H,phip=0.)


fig = plt.figure()
ax = fig.gca(projection='3d')


surf = ax.plot_surface(RR*np.cos(PP), RR*np.sin(PP), outpot, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax.view_init(elev=70.)

"""
