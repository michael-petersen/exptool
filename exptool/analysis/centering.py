"""
centering.py
Tools for wrangling unruly n-body data.
    - recenter particle data
    - rotate particle data
    - compute density profiles
    - compute Fourier moments
    - compute velocity-weighted Fourier moments
    - compute Fourier moments in radial bins

    
Originally used for NewHorizon bar detection (Reddish et al. 2022)
    
09 Jul 2021: First introduction
18 Sep 2024: Class structure overhaul
"""
import numpy as np

class ParticleAlignment:
    """
    Example usage:

    import numpy as np

    # Sample particle data (replace with your actual data)
    N = 1000  # Number of particles
    x = np.random.randn(N)   # Random x positions
    y = np.random.randn(N)   # Random y positions
    z = np.random.randn(N)   # Random z positions
    vx = np.random.randn(N)  # Random velocities in x
    vy = np.random.randn(N)  # Random velocities in y
    vz = np.random.randn(N)  # Random velocities in z
    mass = np.random.rand(N) # Random masses

    # Create an instance of ParticleAlignment
    particle_system = ParticleAlignment(x, y, z, vx, vy, vz, mass)

    # Recenter positions and velocities within a maximum radius of 10
    x_recentered, y_recentered, z_recentered, vx_recentered, vy_recentered, vz_recentered = particle_system.recentre_pos_and_vel(r_max=10)

    # Rotate particles around a specified axis (e.g., z-axis) by 45 degrees
    axis = np.array([0, 0, 1.])  # Rotate around z-axis
    angle = np.pi / 4  # 45 degrees in radians
    x_rot, y_rot, z_rot = particle_system.rotate(axis, angle)

    # Compute the rotation axis and angle to align the angular momentum with a given vector
    vec = np.array([1., 0, 0])  # Target vector (e.g., x-axis)
    axis, angle = particle_system.compute_rotation_to_vec(vec)

    # Use the shrinking sphere method to recenter and shrink
    w = np.random.rand(N)  # Random weighting factors for the particles
    x_shrunk, y_shrunk, z_shrunk, vx_shrunk, vy_shrunk, vz_shrunk = particle_system.shrinking_sphere(w)

    # Compute the density profile for the system
    R = np.sqrt(x**2 + y**2 + z**2)  # Radial distances
    W = mass  # Weights as the particle masses
    density, enclosed_mass, potential = particle_system.compute_density_profile(R, W)

    # Print results
    print("Recentered positions:", x_recentered, y_recentered, z_recentered)
    print("Rotated positions:", x_rot, y_rot, z_rot)
    print("Rotation axis and angle:", axis, angle)
    print("Density profile:", density)

    
    
    """
    def __init__(self, x, y, z, vx, vy, vz, mass):
        """
        Initialize the ParticleAlignment with particle positions, velocities, and masses.

        Parameters:
        x, y, z : arrays
            Position coordinates of the particles.
        vx, vy, vz : arrays
            Velocity components of the particles.
        mass : array
            Mass of the particles.
        """
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

    def recentre_pos_and_vel(self, r_max):
        """
        Recenter the position and velocity of the particles based on the center of mass 
        and mass-weighted velocity within r_max.

        Parameters:
        r_max : float
            Maximum radius to consider for recentering.

        Returns:
        Recentered positions and velocities.
        """
        mask = (self.x**2 + self.y**2 + self.z**2) < r_max**2
        mass_tot = np.sum(self.mass[mask])
        x_cm = np.sum(self.x[mask]*self.mass[mask])/mass_tot
        y_cm = np.sum(self.y[mask]*self.mass[mask])/mass_tot
        z_cm = np.sum(self.z[mask]*self.mass[mask])/mass_tot
        vx_cm = np.sum(self.vx[mask]*self.mass[mask])/mass_tot
        vy_cm = np.sum(self.vy[mask]*self.mass[mask])/mass_tot
        vz_cm = np.sum(self.vz[mask]*self.mass[mask])/mass_tot
        return self.x - x_cm, self.y - y_cm, self.z - z_cm, self.vx - vx_cm, self.vy - vy_cm, self.vz - vz_cm

    def rotate(self, axis, angle):
        """
        Rotate the particles around a given axis by a specified angle.

        Parameters:
        axis : array
            Axis of rotation.
        angle : float
            Angle of rotation in radians.

        Returns:
        Rotated positions.
        """
        axisx, axisy, axisz = axis

        # Rotation for position vector (x, y, z)
        dot_pos = self.x * axisx + self.y * axisy + self.z * axisz  # Dot product for position
        crossx_pos = axisy * self.z - axisz * self.y  # Cross product for position
        crossy_pos = axisz * self.x - axisx * self.z
        crossz_pos = axisx * self.y - axisy * self.x

        cosa = np.cos(angle)
        sina = np.sin(angle)

        x_rot = self.x * cosa + crossx_pos * sina + axisx * dot_pos * (1 - cosa)
        y_rot = self.y * cosa + crossy_pos * sina + axisy * dot_pos * (1 - cosa)
        z_rot = self.z * cosa + crossz_pos * sina + axisz * dot_pos * (1 - cosa)

        # Rotation for velocity vector (vx, vy, vz)
        dot_vel = self.vx * axisx + self.vy * axisy + self.vz * axisz  # Dot product for velocity
        crossx_vel = axisy * self.vz - axisz * self.vy  # Cross product for velocity
        crossy_vel = axisz * self.vx - axisx * self.vz
        crossz_vel = axisx * self.vy - axisy * self.vx

        vx_rot = self.vx * cosa + crossx_vel * sina + axisx * dot_vel * (1 - cosa)
        vy_rot = self.vy * cosa + crossy_vel * sina + axisy * dot_vel * (1 - cosa)
        vz_rot = self.vz * cosa + crossz_vel * sina + axisz * dot_vel * (1 - cosa)

        return x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot

    def compute_rotation_to_vec(self, vec):
        """
        Compute the rotation axis and angle to align the angular momentum of the system 
        with a given vector.

        Parameters:
        vec : array
            Target vector to align with.

        Returns:
        axis : array
            Rotation axis.
        angle : float
            Angle to rotate.
        """
        Lxtot = np.sum(self.y * self.vz * self.mass - self.z * self.vy * self.mass)
        Lytot = np.sum(self.z * self.vx * self.mass - self.x * self.vz * self.mass)
        Lztot = np.sum(self.x * self.vy * self.mass - self.y * self.vx * self.mass)
        
        L = np.array([Lxtot, Lytot, Lztot])
        L /= np.linalg.norm(L)
        vec /= np.linalg.norm(vec)

        axis = np.cross(L, vec)
        axis /= np.linalg.norm(axis)

        c = np.dot(L, vec)
        angle = np.arccos(np.clip(c, -1, 1))

        return axis, angle

    def shrinking_sphere(self, w, rmin=1., stepsize=0.5, tol=0.001, verbose=0):
        """
        Apply the shrinking sphere method to recentre the particle system.

        Parameters:
        w : array
            Weighting for the particles.
        rmin : float, optional
            Minimum radius to stop shrinking (default is 1).
        stepsize : float, optional
            Fraction by which to shrink the radius at each step (default is 0.5).
        tol : float, optional
            Minimum fraction of particles required in a sphere to continue (default is 0.001).
        verbose : int, optional
            Level of verbosity for output (default is 0).

        Returns:
        Recentered positions and velocities.
        """
        tshiftx = np.nanmedian(np.nansum(w * self.x) / np.nansum(w))
        tshifty = np.nanmedian(np.nansum(w * self.y) / np.nansum(w))
        tshiftz = np.nanmedian(np.nansum(w * self.z) / np.nansum(w))

        self.x -= tshiftx
        self.y -= tshifty
        self.z -= tshiftz

        rval = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        rmax = np.nanmax(rval)
        rmax0 = np.nanmax(rval)

        if verbose:
            print(f'Initial guess: {tshiftx:.0f}, {tshifty:.0f}, {tshiftz:.0f}')

        while rmax > rmin:
            u = np.where(rval < stepsize * rmax)[0]
            if float(u.size) / float(self.x.size) < tol:
                print(f'Too few particles to continue at radius ratio {stepsize * rmax / rmax0}')
                break

            comx = np.nanmedian(np.nansum(w[u] * self.x[u]) / np.nansum(w[u]))
            comy = np.nanmedian(np.nansum(w[u] * self.y[u]) / np.nansum(w[u]))
            comz = np.nanmedian(np.nansum(w[u] * self.z[u]) / np.nansum(w[u]))

            self.x -= comx
            self.y -= comy
            self.z -= comz

            tshiftx += comx
            tshifty += comy
            tshiftz += comz

            rval = np.sqrt(self.x**2 + self.y**2 + self.z**2)
            rmax *= stepsize

        comvx = np.nanmedian(np.nansum(w[u] * self.vx[u]) / np.nansum(w[u]))
        comvy = np.nanmedian(np.nansum(w[u] * self.vy[u]) / np.nansum(w[u]))
        comvz = np.nanmedian(np.nansum(w[u] * self.vz[u]) / np.nansum(w[u]))

        if verbose:
            print(f'Final shift: {tshiftx:.0f}, {tshifty:.0f}, {tshiftz:.0f}')
            print(f'Final velocity shift: {comvx:.0f}, {comvy:.0f}, {comvz:.0f}')

        self.vx -= comvx
        self.vy -= comvy
        self.vz -= comvz

        return self.x, self.y, self.z, self.vx, self.vy, self.vz

    def compute_density_profile(self, R, W, rbins=10.**np.linspace(-3.7, 0.3, 100)):
        """
        Compute the density profile, enclosed mass, and potential for the particle system.

        Parameters:
        R : array
            Radial positions of the particles.
        W : array
            Weights of the particles.
        rbins : array, optional
            Radial bins to compute the profile (default is logarithmic bins from 10^-3.7 to 10^0.3).

        Returns:
        dens : array
            Density profile.
        menc : array
            Enclosed mass profile.
        potp : array
            Potential profile.
        """
        dens = np.zeros(rbins.size)
        menc = np.zeros(rbins.size)
        potp = np.zeros(rbins.size)

        astronomicalG = 0.0000043009125
        rbinstmp = np.concatenate([rbins, [2. * rbins[-1] - rbins[-2]]])

        for indx, val in enumerate(rbinstmp[:-1]):
            w = np.where((R > rbinstmp[indx]) & (R < rbinstmp[indx + 1]))[0]
            wenc = np.where((R < rbinstmp[indx + 1]))[0]
            shellsize = (4 / 3.) * np.pi * (rbinstmp[indx + 1]**3 - rbinstmp[indx]**3)
            dens[indx] = np.nansum(W[w]) / shellsize
            menc[indx] = np.nansum(W[wenc])
            potp[indx] = np.sqrt(astronomicalG * menc[indx] / (rbinstmp[indx + 1]))

        return dens, menc, potp

