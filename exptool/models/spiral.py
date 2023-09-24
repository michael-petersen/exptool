"""
spiral (Cox & Gomez [year])

09 Apr 2021:  Introduction
24 Sep 2023:  Convert to class, improve documentation

Follow notes from Hunt & Bovy (2018) for the demo application

I'm struck that the spirals continue to unwind, whereas in simulations
we see focused tight densities even at large radii. How would one
construct a model to reflect this? Make the arm distribution
`peakier', I guess!

"""
# standard python modules
import numpy as np

class SpiralPotential:
    def __init__(self, N, alpha, Rs, rho0, r0, H, phip=0):
        self.N = N
        self.alpha = alpha
        self.Rs = Rs
        self.rho0 = rho0
        self.r0 = r0
        self.H = H
        self.phip = phip
        self.G = 1  # Gravitational constant, set to 1 for simplicity

    def _return_Kn(self, n, r, alpha):
        return (n * self.N) / (r * np.sin(alpha))

    def _return_betan(self, n, r, alpha, H):
        Kn = self._return_Kn(n, r, alpha)
        return Kn * H * (1 + 0.4 * Kn * H)

    def _return_Dn(self, n, r, alpha, H):
        Kn = self._return_Kn(n, r, alpha)
        return (1 + Kn * H + 0.3 * Kn * Kn * H * H) / (1 + 0.3 * Kn * H)

    def _return_gamma(self, r, phi):
        return self.N * (phi - self.phip - np.log(r / self.r0) / np.tan(self.alpha))

    def _combined_arms_pot(self, r, phi, z):
        gamma = self._return_gamma(r, phi)

        nsum = 0
        for n in range(1, 4):
            Kn = self._return_Kn(n, r, self.alpha)
            Dn = self._return_Dn(n, r, self.alpha, self.H)
            Bn = self._return_betan(n, r, self.alpha, self.H)

            term1 = (8 / (3 * np.pi)) / (Kn * Dn)
            term2 = np.cos(n * gamma) * (1 / np.cosh((Kn * z) / Bn)) ** Bn

            nsum += term1 * term2

        return nsum

    def potential(self, r, phi, z):
        gamma = self._return_gamma(r, phi)
        expval = np.exp(-(r - self.r0) / self.Rs)
        prefac = -4 * np.pi * self.G * self.rho0

        return prefac * expval * self._combined_arms_pot(r, phi, z)

    def _combined_arms_dens(self, r, phi, z):
        gamma = self._return_gamma(r, phi)

        nsum = 0
        for n in range(1, 4):
            Kn = self._return_Kn(n, r, self.alpha)
            Dn = self._return_Dn(n, r, self.alpha, self.H)
            Bn = self._return_betan(n, r, self.alpha, self.H)

            term1 = (8 / (3 * np.pi)) * ((Kn * self.H) / Dn) * ((Bn + 1) / Bn)
            term2 = np.cos(n * gamma) * (1 / np.cosh((Kn * z) / Bn)) ** (2 + Bn)

            nsum += term1 * term2

        return nsum

    def density(self, r, phi, z):
        gamma = self._return_gamma(r, phi)
        expval = np.exp(-(r - self.r0) / self.Rs)
        prefac = self.rho0

        return prefac * expval * self._combined_arms_dens(r, phi, z)

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
