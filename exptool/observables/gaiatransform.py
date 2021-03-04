"""
principal set of transformations defining RA-Dec to Galactic
coordinates, as per Gaia instructions.

# see https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html

# TODO: find where I did the Cartesian velocities.
# TODO: add the translation to the solar location (rather, away from)

# add EXAMPLES


"""

legacy = False

import numpy as np

if legacy:
    # this is all legacy. compare and then delete if not needed
    def return_gaia_Agprime():
        """return the matrix in eq 3.61, key to transform from ICRS to galactic coordinates"""
        return np.array([[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],
                             [+0.4941094278755837,-0.4448296299600112,+0.7469822444972189],
                             [-0.8676661490190047,-0.1980763734312015,+0.4559837761750669]])

    def return_ricrs(a,d):
        """ eq."""
        return np.array([np.cos(a)*np.cos(d),np.sin(a)*np.cos(d),np.sin(d)]).T

    def return_picrs(a,d):
        """ eq. 3.64, unit vector of increasing alpha"""
        return np.array([-np.sin(a),np.cos(a),0.]).T

    def return_qicrs(a,d):
        """ eq. 3.64, unit vector of increasing delta"""
        return np.array([-np.cos(a)*np.sin(d),-np.sin(a)*np.sin(d),np.cos(d)]).T

    def return_muicrs(a,d,mua,mud):
        """ eq. 3.66, the proper motion vector"""
        p = return_picrs(a,d)
        q = return_qicrs(a,d)
        return np.dot(p,mua) + np.dot(q,mud)


    def return_rgal(l,b):
        """ eq."""
        return np.array([np.cos(l)*np.cos(b),np.sin(l)*np.cos(b),np.sin(b)]).T

    def return_pgal(l,b):
        """ eq. 3.66, unit vector of increasing alpha"""
        return np.array([-np.sin(l),np.cos(l),0.]).T

    def return_qgal(l,b):
        """ eq. 3.66, unit vector of increasing delta"""
        return np.array([-np.cos(l)*np.sin(b),-np.sin(l)*np.sin(b),np.cos(b)]).T

    def return_mugal(l,b,mul,mub):
        """ eq. 3.66, the proper motion vector"""
        p = return_pgal(l,b)
        q = return_qgal(l,b)
        return np.dot(p,mul) + np.dot(q,mub)


    def rotate_velocities(a,d,mua,mud):
        """eq 3.68, """
        mu = return_muicrs(a,d,mua,mud)
        mugal = np.dot(return_gaia_Agprime(),mu) # eq. 3.68
    
        # solve for positions
        ricrs = return_ricrs(a,d)
        rgal = np.dot(return_gaia_Agprime(),ricrs)

        # implement eq 3.63
        ell,b = np.arctan2(rgal[1],rgal[0]),np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))
    
        p = return_pgal(ell,b)
        q = return_qgal(ell,b)
    
        mul = np.dot(p.T,mugal)
        mub = np.dot(q.T,mugal)
        #print(mul,mub)
        return mul,mub



    def rotate_errors(a,d,pmra_e,pmdec_e,pmcorr):
        """rotate covariance error from ra/dec to l/b."""
        ricrs = return_ricrs(a,d)
        picrs = return_picrs(a,d)
        qicrs = return_qicrs(a,d)

        rgal = np.dot(return_gaia_Agprime(),ricrs)

        # implement eq 3.63
        ell = np.arctan2(rgal[1],rgal[0])
        b = np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))

        pgal = return_pgal(ell,b)
        qgal = return_qgal(ell,b)

        pqgal = np.stack((pgal, qgal), axis=-1)
        pqicrs = np.stack((picrs, qicrs), axis=-1)

        cov = np.array([[pmra_e*pmra_e,pmra_e*pmdec_e*pmcorr],[pmra_e*pmdec_e*pmcorr,pmdec_e*pmdec_e]])
        #print(cov)

        G = np.einsum('ab,ac->bc', pqgal,
                          np.einsum('ji,ik->jk', return_gaia_Agprime(), pqicrs))

        cov_to = np.einsum('ba,ac->bc', G,
                               np.einsum('ij,ki->jk', cov, G))
    
        return cov_to


    def rotate_positions(a,d):
        """helper transformation from ra/dec to l/b"""
        ricrs = return_ricrs(a,d)
        rgal = np.dot(return_gaia_Agprime(),ricrs)

        ell = np.arctan2(rgal[1],rgal[0])
        b = np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))
        return ell,b

    def rotate_positions_cartesian(a,d,r):
        """following galactic conventions"""
        ricrs = return_ricrs(a,d)
        rgal = np.dot(return_gaia_Agprime(),ricrs)

        x = rgal[0]*r
        y = rgal[1]*r
        z = rgal[2]*r
        return x,y,z


    def rotate_velocities_cartesian(a,d,r,mua,mud,vlos):
        """following galactic conventions. This is a right-handed system, I think."""
        ell,b   = rotate_positions(a,d)
        mul,mub = rotate_velocities(a,d,mua,mud)
    
        k = 4.74057
        vl,vb = k*mul*r,k*mub*r # transform to km/s
    
        cost,sint = np.cos(b),np.sin(b)
        cosp,sinp = np.cos(ell),np.sin(ell)
    
        xdot = cost*cosp*vlos - sint*cosp*vb - sinp*vl
        ydot = cost*sinp*vlos - sint*sinp*vb + cosp*vl
        zdot = sint     *vlos + cost     *vb
    
        return xdot,ydot,zdot
    



def return_gaia_Agprime():
    """return the matrix in eq 3.61, key to transform from ICRS to galactic coordinates"""
    return np.array([[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],
                     [+0.4941094278755837,-0.4448296299600112,+0.7469822444972189],
                     [-0.8676661490190047,-0.1980763734312015,+0.4559837761750669]])

def return_gaia_Ag():
    """set the Hipparcos computation

    if truly obsessed see https://www.cosmos.esa.int/documents/532822/552851/vol1_all.pdf

    though this has higher precision!!
    """
    return np.array([[-0.0548755604162154,+0.4941094278755837,-0.8676661490190047],
                     [-0.8734370902348850,-0.4448296299600112,-0.1980763734312015],
                     [-0.4838350155487132,+0.7469822444972189,+0.4559837761750669]])

    
def return_ricrs(a,d):
    """ eq. 3.57"""
    return np.array([np.cos(a)*np.cos(d),np.sin(a)*np.cos(d),np.sin(d)])

def return_picrs(a,d):
    """ eq. 3.64, unit vector of increasing alpha"""
    if hasattr(a,'size'):
        return np.array([-np.sin(a),np.cos(a),np.zeros(a.size)])
    else:
        return np.array([-np.sin(a),np.cos(a),0.])

def return_qicrs(a,d):
    """ eq. 3.64, unit vector of increasing delta"""
    return np.array([-np.cos(a)*np.sin(d),-np.sin(a)*np.sin(d),np.cos(d)])

def return_muicrs(a,d,mua,mud):
    """ eq. 3.66, the proper motion vector"""
    p = return_picrs(a,d)
    q = return_qicrs(a,d)
    return p*mua + q*mud


def return_rgal(l,b):
    """ eq. 3.58"""
    return np.array([np.cos(l)*np.cos(b),np.sin(l)*np.cos(b),np.sin(b)])

def return_pgal(l,b):
    """ eq. 3.66, unit vector of increasing alpha"""
    if hasattr(l,'size'):
        return np.array([-np.sin(l),np.cos(l),0.*np.cos(l)])
    else:
        return np.array([-np.sin(l),np.cos(l),0.*np.cos(l)])

def return_qgal(l,b):
    """ eq. 3.66, unit vector of increasing delta"""
    return np.array([-np.cos(l)*np.sin(b),-np.sin(l)*np.sin(b),np.cos(b)])

def return_mugal(l,b,mul,mub):
    """ eq. 3.66, the proper motion vector"""
    p = return_pgal(l,b)
    q = return_qgal(l,b)
    return p*mul + q*mub


def rotate_velocities(a,d,mua,mud):
    """eq 3.68, """
    mu = return_muicrs(a,d,mua,mud)
    mugal = np.dot(return_gaia_Agprime(),mu) # eq. 3.68
    
    # solve for positions
    ricrs = return_ricrs(a,d)
    rgal = np.dot(return_gaia_Agprime(),ricrs)

    # implement eq 3.63
    ell,b = np.arctan2(rgal[1],rgal[0]),np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))
    
    p = return_pgal(ell,b)
    q = return_qgal(ell,b)
    
    mul = np.sum(p*mugal,axis=0)
    mub = np.sum(q*mugal,axis=0)
    #print(mul,mub)
    return mul,mub



def rotate_errors(a,d,pmra_e,pmdec_e,pmcorr):
    ricrs = return_ricrs(a,d)
    picrs = return_picrs(a,d)
    qicrs = return_qicrs(a,d)

    rgal = np.dot(return_gaia_Agprime(),ricrs)

    # implement eq 3.63
    ell = np.arctan2(rgal[1],rgal[0])
    b = np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))

    pgal = return_pgal(ell,b)
    qgal = return_qgal(ell,b)

    pqgal = np.stack((pgal, qgal), axis=-1)
    pqicrs = np.stack((picrs, qicrs), axis=-1)

    cov = np.array([[pmra_e*pmra_e,pmra_e*pmdec_e*pmcorr],[pmra_e*pmdec_e*pmcorr,pmdec_e*pmdec_e]])
    #print(cov)

    G = np.einsum('ab,ac->bc', pqgal,
                      np.einsum('ji,ik->jk', return_gaia_Agprime(), pqicrs))

    cov_to = np.einsum('ba,ac->bc', G,
                           np.einsum('ij,ki->jk', cov, G))
    
    return cov_to


def rotate_to_galactic(a,d,dist):
    """eq 3.68, but built for speed"""
    # solve for positions
    if a.size>1:
        ricrs = return_ricrs(a,d)
        rgal = np.dot(return_gaia_Agprime(),ricrs)
        cpos = dist*rgal        
    else:
        ricrs = return_ricrs(a,d)
        rgal = np.dot(return_gaia_Agprime(),ricrs)
        cpos = np.dot(dist,rgal)        
    return cpos


def rotate_observed(l,b):
    """eq 3.68, THIS DOES NOT WORK FOR b=90 (exactly)!"""
    # solve for positions
    rgal = return_rgal(l,b)
    ricrs = np.dot(return_gaia_Ag(),rgal)
    #d = np.arcsin(ricrs[2])
    #a = np.arccos(ricrs[0]/np.cos(d))
    #print(a,d)

    # implement eq 3.63
    a,d = np.arctan2(ricrs[1],ricrs[0]),np.arctan2(ricrs[2],np.sqrt(ricrs[0]*ricrs[0]+ricrs[1]*ricrs[1]))

    return a,d


def rotate_velocities_observed(l,b,mul,mub):
    """eq 3.68, """
    mu = return_mugal(l,b,mul,mub)
    muicrs = np.dot(return_gaia_Ag(),mu) # eq. 3.68
    
    rgal = return_rgal(l,b)
    ricrs = np.dot(return_gaia_Ag(),rgal)
    a,d = np.arctan2(ricrs[1],ricrs[0]),np.arctan2(ricrs[2],np.sqrt(ricrs[0]*ricrs[0]+ricrs[1]*ricrs[1]))
    
    p = return_picrs(a,d)
    q = return_qicrs(a,d)
    
    mua = np.sum(p*muicrs,axis=0)
    mud = np.sum(q*muicrs,axis=0)
    #print(mul,mub)
    return mua,mud



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on a sphere (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = (np.pi/180.)*lon1, (np.pi/180.)*lat1, (np.pi/180.)*lon2, (np.pi/180.)*lat2

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2. * np.arcsin(np.sqrt(a)) 
    return (180./np.pi)*c
    


def define_transformation_matrix(C1,C2,D1,D2,start=0):
    """define a rotation matrix to go from ICRS to arbitrary rotation
    
    takes two points on the sphere and returns the transformation that 
    puts the points along the latitude=0 plane
    
    default will produce ICRS->GAL?
    
    inputs
    ------------
    C1      :
        the azimuthal coordinate of the first point
    C2      :
        the polar coordinate of the first point
    D1      :
        the azimuthal coordinate of the second point
    D2      :
        the polar coordinate of the second point
    start   : 
        the final rotation angle defining (0,0)
    
    returns
    ------------
    A       : 3x3 array
        the transformation array, such that r' = Ar transforms r->r'
    
    
    """
    # find the pole from the defined plane
    rC    = [np.cos(C1)*np.cos(C2),np.sin(C1)*np.cos(C2),np.sin(C2)]
    rD    = [np.cos(D1)*np.cos(D2),np.sin(D1)*np.cos(D2),np.sin(D2)]
    CD    = np.cross(rC,rD)
    pole1 = np.arctan2(CD[1],CD[0])
    pole2 = np.arctan2(CD[2],np.sqrt(CD[0]*CD[0]+CD[1]*CD[1]))
    
    # see Gaia convention
    phi   = start*np.pi/180.
    theta = (90.-(180./np.pi)*pole2)*np.pi/180.
    psi   = ((180./np.pi)*pole1+90.)*np.pi/180.
    
    # see Wolfram convention
    D     = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
    C     = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
    B     = np.array([[np.cos(psi),np.sin(psi),0],[-np.sin(psi),np.cos(psi),0],[0,0,1]])

    A     = np.dot(D,np.dot(C,B))
    
    return A
    

def rotate_arbitrary(a,d,Aprime):
    """eq 3.68, """
    #mu = return_muicrs(a,d,mua,mud)
    #mugal = np.dot(return_gaia_Agprime(),mu) # eq. 3.68
    
    # solve for positions
    ricrs = return_ricrs(a,d)
    rgal = np.dot(Aprime,ricrs)

    # implement eq 3.63
    ell,b = np.arctan2(rgal[1],rgal[0]),np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))

    return ell,b


def rotate_velocities_arbitrary(a,d,mua,mud,Aprime):
    """eq 3.68, """
    mu = return_muicrs(a,d,mua,mud)
    mugal = np.dot(Aprime,mu) # eq. 3.68
    
    # solve for positions
    ricrs = return_ricrs(a,d)
    rgal = np.dot(Aprime,ricrs)

    # implement eq 3.63
    ell,b = np.arctan2(rgal[1],rgal[0]),np.arctan2(rgal[2],np.sqrt(rgal[0]*rgal[0]+rgal[1]*rgal[1]))
    
    p = return_pgal(ell,b)
    q = return_qgal(ell,b)
    
    mul = np.sum(p*mugal,axis=0)
    mub = np.sum(q*mugal,axis=0)
    #print(mul,mub)
    return mul,mub



