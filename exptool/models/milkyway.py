'''
milky way

04 Feb 2021 First version

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

from . import nfw
from . import hernquist
from . import mndisc


class MilkyWayGala():
    """a duplicate of the modified MilkyWayPotential14 in Adrian Price-Whelan's gala package

    """



class MilkyWayPotential14():
    """a duplicate of MilkyWayPotential14, the famous Bovy (2015) example Milky Way potential


    mvir = 0.8x10^12
    mdisc = 6.8x10^10
    mbulge = 0.5x10^10
    mhalo  = 0.8-0.068-0.05 = 0.682x10^12 msun

    """
    def __init__(self,
                 halo_scale=16.,halo_mass=0.682e12,halo_rvir=245.,
                 disc_scale=3.0,disc_height=0.280,disc_mass=6.8e10,
                 bulge_scale=,bulge_mass,bulge_cutoff,
                 G=0.0000043009125):
        """

        inputs
        ----------------

        G    : gravitational constant, default in astrophysical units


        """

        self.G           = G
        self.halo_scale  = halo_scale
        self.halo_mass   = halo_mass
        self.halo_rvir   = halo_rvir
        self.disc_scale  = disc_scale
        self.disc_height = disc_height
        self.disc_mass   = disc_mass
        self.bulge_scale = bulge_scale
        self.bulge_mass  = bulge_mass
        self.bulge_cutoff= bulge_cutoff

        self.halo = nfw.NFW(self.halo_scale,self.G,self.halo_mass,self.halo_rvir)
        self.disc = mndisc.MiyamotoNagai(self.disc_scale,self.disc_height,M=self.disc_mass,G=self.G)

    def get_mass(self):

        return
