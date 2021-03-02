"""
isothermal

02-Mar-2021: introduction

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




