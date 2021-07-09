'''
milky way

04-Feb-2021 MSP

the rudimentary MW potential, consisting of
-NFW (1997) halo
-Miyamoto-Nagai (1975) disc

One could add an analytic bulge,
or a second (or third) disc component to mimic the thick disc
and gaseous disc. I suspect these are very subdominant.

The presence of the LMC potential itself probably has more an effect.
(I also have simple NFW formulae for this, if interesting --
actually, maybe this is really interesting, because it's an obvious structure localised on the sky.)

Sorry for mixing units, I think I straightened them all out -- I normally think in virial.

In these units, the sun is at
R=8.1780  kpc (Abuter et al. [Gravity collaboration], 2019)
z=0.0208  kpc (Bennett & Bovy 2020)

# construct a rudimentary MW potential

astronomicalG  = 0.0000043009125             # gravitational constant, (km/s)^2 * kpc / Msun
disc_mass      = 4.e10                       # solar masses
disc_length    = 3.0                         # in kpc
disc_height    = 0.280                       # in kpc
halo_rho0      = 0.00714*np.power(1000.,3.)  # Msun/pc^3 -> Msun/kpc^3
halo_length    = 20.0                        # in kpc
    

H = NFW(halo_length,halo_rho0,G=astronomicalG)
D = MiyamotoNagai(disc_length,disc_height,M=disc_mass,G=astronomicalG)

rvals = np.linspace(0.,300.,10) # in kpc

plt.figure()
plt.plot(rvals,D.potential(0.,rvals),color='red',linestyle='dashed')
plt.plot(rvals,D.potential(rvals,0.),color='red')
plt.plot(rvals,H.potential(rvals),color='black')
plt.xlabel('radius (kpc)',size=12)
plt.ylabel('$\Phi$ (km/s)$^2$')

plt.figure()
plt.plot(rvals,np.log10(D.density(0.,rvals)),color='red',linestyle='dashed')
plt.plot(rvals,np.log10(D.density(rvals,0.)),color='red')
plt.plot(rvals,np.log10(H.density(rvals)),color='black')
plt.xlabel('radius (kpc)',size=12)
plt.ylabel('$\\rho$ (M$_{\odot}$/kpc$^3$)')





'''

import numpy as np
import matplotlib.pyplot as plt


class NFW(object):
    
    
    def __init__(self,a,rho0,G=1.):
        """
        initialise the disc
        
        inputs
        ----------------
        a    : halo scale length
        rho0 : central density
        G    : gravitational constant
        
        
        """
        
        self.a    = a
        self.rho0 = rho0
        self.G    = G
        
    def density(self,r):
        """return the NFW halo density
        
        h/t BinneyTremaine2008 2.64, p.71
        
        """
        return (self.rho0)/(np.power(r/self.a,1.)*np.power(1+r/self.a,2.))
        

    def potential(self,r):
        """return the potential of an NFW halo

        h/t BinneyTremaine2008 2.67, p.71

        """
        prefac = -4*np.pi*self.G*self.rho0*self.a*self.a
        return prefac * ( (np.log(1+r/self.a))/(r/self.a))
    
    def mass(self,r):
        """return the mass enclosed of an NFW halo
        
        h/t BinneyTremaine2008 2.66, p.71
        
        """
        prefac = 4*np.pi*self.rho0*self.a*self.a*self.a
        return prefac * ( np.log1(1+r/self.a) - (r/self.a)/(1+r/self.a))

    def rforce(self,r):

        return 0.


class MiyamotoNagai(object):
    """an instantiation of a Miyamoto-Nagai disc
    
    The Miyamoto-Nagai (1975) disk associated with the potential of a thin disk is given by
    $$\rho_M(R,z) = \left(\frac{b^2M}{4\pi}\right)\frac{aR^2+(a+3\sqrt{z^2+b^2})(a+\sqrt{z^2+b^2})^2}{\left[R^2+(a+\sqrt{z^2+b^2})^2\right]^{5/2}(z^2+b^2)^{3/2}}
    $$
    
    
    """
    
    def __init__(self,a,b,M=1,G=1):
        """
        initialise the disc
        
        inputs
        ----------------
        a : disc scale length
        b : disc scale height
        M : mass
        G : gravitational constant
        
        
        """
        
        self.a = a
        self.b = b
        self.M = M
        self.G = G

    def potential(self,R,z):
        """return the potential of a miyamoto-nagai disc

        h/t BinneyTremaine2008 2.69a, p.73

        """
        return -(self.G*self.M)/(np.sqrt(R**2. + (self.a+np.sqrt(z*z + self.b*self.b))**2.))


    def density(self,R,z):
        """return the density of a miyamomo-nagai disc

        h/t BinneyTremaine2008 2.69b, p.73

        returns the non-scaled density <- what does this mean?

        """
        zscale = z*z + self.b*self.b
        
        prefac = (self.b*self.b*self.M/(4.*np.pi))
        
        numerator = ( (self.a*R*R) + (self.a+3.*np.sqrt(zscale))\
              * (self.a + np.sqrt(zscale))**2.)
        
        denominator = ( (R*R + (self.a + np.sqrt(zscale))**2.)**(5./2.) * (zscale)**(3./2.) )

        return   prefac*numerator/denominator
    
    def zforce(self,R,z):
        """ compute the vertical force
        
        """
        
        zb = np.sqrt(z*z + self.b*self.b)
        ab = self.a + zb
        dn = np.sqrt(R*R + ab*ab)
        d3 = dn*dn*dn
        
        return -self.M*z*ab/(zb*d3)
                
    def rforce(self,R,z):
        """ compute the radial force
                
        """
        
        zb = np.sqrt(z*z + self.b*self.b)
        ab = self.a + zb
        dn = np.sqrt(R*R + ab*ab)
        d3 = dn*dn*dn
        
        return -self.M*R/d3
    

    


