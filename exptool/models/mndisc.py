"""
mndisc.py

class for Miyamoto-Nagai (1975) disc.

supporting:
MiyamotoNagai



"""
import numpy as np


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
        a
        b
        M
        G
        
        
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

    
    def mass(self,R,z):
        """return the spherical enclosed mass

        """
        rad = np.sqrt(R*R+z*z)
        return rad*-self.potential(R,z)



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
    

    
  
