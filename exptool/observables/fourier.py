"""
A module to compute Fourier moments and velocity-weighted Fourier moments from particle data.

18 Sep 2024: First introduction

"""
import numpy as np


class FourierAnalysis:
    """
    A class to compute Fourier moments and velocity-weighted Fourier moments from particle data.
    
    It can accept either a ParticleAlignment object or phase-space vectors directly.

    Example usage:

    import numpy as np
    import matplotlib.pyplot as plt

    # Alternatively, you can use phase-space vectors directly:
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    vx = np.random.randn(1000)
    vy = np.random.randn(1000)
    mass = np.random.rand(1000)

    # Create the FourierAnalysis object with either particle_data or phase-space vectors
    fourier_analysis = FourierAnalysis(x=x, y=y, vx=vx, vy=vy, mass=mass)

    # Compute Fourier moments
    mod, angle = fourier_analysis.compute_fourier(harmonic=2)

    # Compute velocity-weighted Fourier moments
    vel_mod = fourier_analysis.compute_fourier_vel(harmonic=2)

    # Compute Fourier moments and velocity moments in radial bins
    rtest, fpower, fangle, fvpower = fourier_analysis.fourier_tabulate(bins=50)

    # the simplest test is the ratio of m=4 to m=2 Fourier velocities
    plt.plot(rtest,fvpower[4]/fvpower[2])
    # when this ratio is over 1, it's the end of the bar.

    """

    def __init__(self, particle_data=None, x=None, y=None, vx=None, vy=None, mass=None):
        """
        Initialize the FourierAnalysis object with either a ParticleAlignment object or phase-space vectors.

        Parameters:
        - particle_data: A ParticleAlignment object (optional).
        - x, y: Arrays of particle positions in x and y (optional if particle_data is provided).
        - vx, vy: Arrays of particle velocities in x and y (optional if particle_data is provided).
        - mass: Array of particle masses (optional if particle_data is provided).
        """
        if particle_data is not None:
            self.x = particle_data.x
            self.y = particle_data.y
            self.vx = particle_data.vx
            self.vy = particle_data.vy
            self.mass = particle_data.mass
        else:
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.mass = mass

        # Ensure the phase-space vectors are all provided or inferred from ParticleAlignment
        if self.x is None or self.y is None or self.vx is None or self.vy is None or self.mass is None:
            raise ValueError("Must provide either particle_data or all phase-space vectors (x, y, vx, vy, mass).")
        
    def within_annulus(self, rcentral, rwidth, zheight):
        """
        Determine which particles are within a cylindrical annulus defined by 
        central radius `rcentral`, width `rwidth`, and height `zheight`.

        Parameters:
        - rcentral: Central radius of the annulus.
        - rwidth: Width of the annulus.
        - zheight: Height of the cylindrical region.

        Returns:
        - Boolean array indicating particles within the annulus.
        """
        rval = np.sqrt(self.xpos**2 + self.ypos**2)
        within_annulus = (rval > rcentral - rwidth) & (rval < rcentral + rwidth) & (np.abs(self.zpos) < zheight)
        return within_annulus

    def within_radius(self, radius):
        """
        Determine which particles are within a spherical radius.

        Parameters:
        - radius: Radius to filter particles within.

        Returns:
        - Boolean array indicating particles within the radius.
        """
        r = np.sqrt(self.xpos**2 + self.ypos**2 + self.zpos**2)
        return r < radius
    
    def compute_fourier(self, harmonic=2):
        """
        Compute Fourier moments for the particle system.

        Parameters:
        - harmonic: The harmonic number for the Fourier moment (default is 2).

        Returns:
        - mod: The modulus of the Fourier moment.
        - angle: The phase angle of the Fourier moment.
        """
        phi = np.arctan2(self.y, self.x)
        A = np.nansum(self.mass * np.cos(harmonic * phi)) / np.nansum(self.mass)
        B = np.nansum(self.mass * np.sin(harmonic * phi)) / np.nansum(self.mass)
        mod = np.sqrt(A * A + B * B)
        angle = np.arctan2(B, A) / 2.
        return mod, angle
    
    def compute_fourier_vel(self, harmonic=2):
        """
        Compute velocity-weighted Fourier moments.

        Parameters:
        - harmonic: The harmonic number for the Fourier moment (default is 2).

        Returns:
        - mod: The modulus of the velocity-weighted Fourier moment.
        """
        phi = np.arctan2(self.y, self.x)
        # Compute the tangential velocity
        vel = (self.x * self.vy - self.y * self.vx) / np.sqrt(self.x**2 + self.y**2)
        A = np.nansum(self.mass * vel * np.cos(harmonic * phi)) / np.nansum(self.mass)
        B = np.nansum(self.mass * vel * np.sin(harmonic * phi)) / np.nansum(self.mass)
        mod = np.sqrt(A * A + B * B)
        return mod
    
    def fourier_tabulate(self, bins=75):
        """
        Compute the Fourier moments and velocity-weighted Fourier moments in radial bins.

        Parameters:
        - bins: Number of radial bins to use (default is 75).

        Returns:
        - rtest: The radial bin centres.
        - fpower: The Fourier moments for harmonics 0 to 4.
        - fangle: The Fourier angles for harmonics 0 to 4.
        - fvpower: The velocity-weighted Fourier moments for harmonics 0 to 4.
        """
        rval = np.sqrt(self.x**2 + self.y**2)
        rtest = np.linspace(0., np.nanpercentile(rval, 75), bins)
        dr = rtest[1] - rtest[0]
        rbin = (np.floor(rval / dr)).astype(int)
        
        fpower = np.zeros((5, rtest.size))
        fangle = np.zeros((5, rtest.size))
        fvpower = np.zeros((5, rtest.size))
        
        for ir, rv in enumerate(rtest):
            w = np.where(rbin == ir)[0]
            for h in range(5):
                fpower[h, ir], fangle[h, ir] = self.compute_fourier(harmonic=h)
                fvpower[h, ir] = self.compute_fourier_vel(harmonic=h)
        
        return rtest, fpower, fangle, fvpower

    def compute_fourier_in_annuli(self, rcentral, rwidth, zheight, harmonic=2):
        """
        Compute Fourier moments for particles within a specified annulus.

        Parameters:
        - rcentral: Central radius of the annulus.
        - rwidth: Width of the annulus.
        - zheight: Height of the cylindrical region.
        - harmonic: Harmonic number for Fourier analysis.

        Returns:
        - mod: Amplitude of the Fourier component within the annulus.
        - angle: Angle of the Fourier component within the annulus.
        """
        within = self.within_annulus(rcentral, rwidth, zheight)
        return self.compute_fourier(self.xpos[within], self.ypos[within], self.mass[within], harmonic)

    def compute_fourier_in_radius(self, radius, harmonic=2):
        """
        Compute Fourier moments for particles within a specified spherical radius.

        Parameters:
        - radius: Radius to filter particles within.
        - harmonic: Harmonic number for Fourier analysis.

        Returns:
        - mod: Amplitude of the Fourier component within the radius.
        - angle: Angle of the Fourier component within the radius.
        """
        within = self.within_radius(radius)
        return self.compute_fourier(self.xpos[within], self.ypos[within], self.mass[within], harmonic)


