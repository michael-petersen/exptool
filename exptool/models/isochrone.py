###########################################################3
#
# 06-Apr-2021: introduction

#
"""
isochrone (Henon)


an isochrone model, which I am just starting to understand to utility of.

primarily following Fouvry, Hamilton, Rozier, Pichon (2021), see
equation numbers there.

open questions:
-what does the DF look like for an isochrone model?

"""



class Isochrone():
    """
    the Henon isochrone model
    can .compute_orbit_EL(E,L)
    can .compute_orbit_ap(ra,rp)
    
    """
    
    def __init__(self,bc,M=1,G=1):
        
        self.bc     = bc
        self.M      = M
        self.G      = G
        self.omega0 = _omega0(self.bc,self.M,self.G)
        self.E0     = _Escale(self.bc,self.M,self.G)
        self.L0     = _Lscale(self.bc,self.M,self.G)
        self.rho0   = _rho0(self.bc,self.M,self.G)

    def _omega0(bc,M=1,G=1): 
        """the frequency scale, in line after G4"""
        return np.sqrt(G*M/(bc*bc*bc))

    def _rho0(bc,M=1,G=1): 
        """the central density, eq. 2.50 BT08"""
        return (3*M)/(16*np.pi*bc*bc*bc)
    
    def _Escale(bc,M=1,G=1):
        """return the energy scale, in line after G9"""
        return -G*M/bc
    
    def _Lscale(bc,M=1,G=1):
        """return the action scale, in text after G9"""
        return np.sqrt(G*M*bc)
    
    def potential(self,r):
        """equation G1"""
        return -(self.G*self.M)/(self.bc+np.sqrt(self.bc*self.bc + r*r))
    
    def jmax(self,E):
        """solve G3 for Jr=0

        input in wolfram
        """
        return 0.5*(np.sqrt(2/(-self.E))*self.G*self.M - 2*np.sqrt(2*-self.E)*self.bc)



def etaEL(E,L,bc,G=1,M=1):
    """equation G7, line 1"""
    return 0.5*(1.+(L/(np.sqrt(L*L + 4*G*M*bc))))

def omegaE(E,bc,M=1,G=1):
    """equation G5a"""
    Emin = -G*M/(2*bc)
    return np.power(E/Emin,1.5)

def omega1E(E,L,bc,M=1,G=1):
    """equation G4, computed using E"""
    return omegaE(E,bc,M,G) * omega0(bc,M,G)

def omega2E(E,L,bc,M=1,G=1):
    """equation G6, computed using E"""
    return omegaE(E,bc,M,G)*etaEL(E,L,bc,G,M)*omega0(bc,M,G)
 

def iso_jr(E,L,bc,M=1,G=1):
    """equation G3
    
    the radial action at a given E and L
    """
    return (G*M/np.sqrt(-2*E)) - 0.5*(L+np.sqrt(L*L+4*G*M*bc))



