###########################################################################################
#
#  cluttonbrock.py
#     Compute cofficients and forces in an analytic Clutton-Brock (1973) basis.
#
#     
#
# 06-Apr-21: first introduced.
#
'''
See Fouvry, Hamilton, Rozier, Pichon (2021) for equation numbers.

First follow Appendix B, Clutton-Brock basis.

'''


import numpy as np

from scipy.special import eval_gegenbauer
from scipy.special import factorial
from scipy.special import sph_harm


def gegenbauer(rho,n,alpha):
    """make Gegenbauer recursion
    
    eq. B5
    (exactly the same as eval_gegenbauer(n,alpha,rho))
    not stable for anything but alpha=1, it seems.
    """
    if n==0: return 1.
    elif n==1: return 2*alpha*rho
    else:
        cn = np.zeros(n+1)
        cn[0] = 1
        cn[1] = 2*alpha*rho
        for n in range(2,n+1):
            cn[n] = (2*(n+alpha)*rho*cn[n-1] - (n+2*alpha-1)*cn[n-2])/(n+1)
    return cn[n]

def rescaled_rho(r,Rb):
    """rescaled coordinate with scale radius Rb, equation B4
    
    Clutton-Brock (1973): The two-dimensional Hankel-Laguerre functions involve polynomials with this variable.
    """
    rrb = r/Rb
    return (rrb*rrb - 1)/(rrb*rrb + 1)

def Aln(l,n,Rb,G=1):
    denominator2 = 4*(n-1)*(n+2*l+1) + (2*l+1)*(2*l+3)
    return -np.sqrt(G/Rb)*np.power(2,2*l+3)*factorial(l)*\
            np.power(((factorial(n-1)*(n+l))/(factorial(n+2*l)*denominator2)),0.5)

def Bln(l,n,Rb,G=1):
    denominator2 = 4*(n-1)*(n+2*l+1) + (2*l+1)*(2*l+3)
    return (1./(np.sqrt(G)*np.power(Rb,2.5)))*np.power(2,2*l+3)*(1./(4*np.pi))*factorial(l)*\
            np.power(((factorial(n-1)*(n+l)*denominator2)/(factorial(n+2*l))),0.5)


def Uln(r,l,n,Rb):
    rrb = r/Rb
    return Aln(l,n,Rb) * (np.power(rrb,l)/np.power(1.+(rrb*rrb),l+0.5)) * eval_gegenbauer(n-1,l+1,rescaled_rho(r,Rb))
    
def Dln(r,l,n,Rb):
    rrb = r/Rb
    return Bln(l,n,Rb) * (np.power(rrb,l)/np.power(1+(rrb*rrb),l+2.5)) * eval_gegenbauer(n-1,l+1,rescaled_rho(r,Rb))
    


#plt.plot(np.log10(r),Uln(r,0,1,bc)/iso_potential(r,bc))
#plt.plot(np.log10(r),)

#plt.plot(np.log10(r),Dln(r,0,1,2.*bc))



# search for the minimum, I guess?
# little bit of bias at Rb=2bc

# as defined, n=1 is the lowest-order term

