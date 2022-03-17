"""
isothermal

02 Mar 2021  introduction

an isothermal model, useful for some basic perturbation theory calculations

primarily following Weinberg (1985), see equation numbers there

"""
import numpy as np



def isothermal_potential(r,sigma,a1):
    """equation 5, the potential of an isothermal sphere"""
    return 2.*sigma**2.*np.log(r/a1)


def isothermal_df(E,sigma,a1):
    """equation 7, the distribution function for the isothermal sphere"""
    return (1./(4.*np.sqrt(2.)*np.pi**2.5)*(1./(sigma*a1*a1)))*np.exp(-E/sigma**2.)



#
# coordinate translation helpers
#
def reduced_rad(r,a1,E,sigma):
    """define the reduced radius (in text on p. 457)"""
    return (r/a1) * np.exp(E/(2*sigma**2.))

# the inverse
def real_rad(redrad,a1,E,sigma):
    return (redrad*a1) * np.exp(-E/(2*sigma**2.))




def jmax(E,a1,sigma):
    """the maximum angular momentum for a circular orbit in the isothermal potential

    to generalise the find_aps equation below, will need to compute this relationship.


    """
    return np.sqrt(2*sigma)*a1*np.log(0.5*((E/(sigma*sigma))-1))



def find_aps(r,kappain):
    """find the roots of the transcendental equation

    -2*ln(r) = kappa^2/(e*r^2)

    """
    return -2.*np.log(r) - kappain**2./(np.exp(1.)*r**2.)


def find_turning(kappain):
    """
    compute the roots of the transcendental equation (find_aps)

    the obvious root is exp(-0.5),so set as the limit

    this will determine the pericenter
    """

    rperi = brentq(find_aps,0.000001,np.exp(-0.5),args=(kappain))
    rapo = brentq(find_aps,np.exp(-0.5),1.0,args=(kappain))
    return [rperi,rapo]
