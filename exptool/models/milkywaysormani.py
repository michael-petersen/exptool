"""
The inner MW model of Sormani+ 2022
(itself a fit to Portail+ 2017)

23 Sep 2023  Implement
24 Sep 2023  New docstrings


density only at present


# Example usage:
model = Sormani2022MilkyWay()

# quick and dirty 3d grid sampling for mass enclosed
radius = 1.0
xvals = np.linspace(0.001,radius,60)
dx = xvals[1]-xvals[0]
enclosed_mass = 0.0
for x in xvals:
    for y in xvals:
        for z in xvals:
            if np.sqrt(x**2+y**2+z**2)<=radius:
                enclosed_mass += (dx**3)*model.total_density(x,y,z)

enclosed_mass *= 8.0


print(f"Enclosed Mass within radius {radius} kpc: {enclosed_mass} Msun")

# compute the equivalent circular velocity
# vc = sqrt(GM/r)
astroG = 4.3e4 # kpc km^2 s^(-2) Msun^(-1) (e10)
vcirc = np.sqrt(astroG*enclosed_mass/radius)

print(f"Circular velocity at radius {radius} kpc: {vcirc} km/s")


import matplotlib.pyplot as plt

# make a very simple x,y grid
xvals,yvals = np.linspace(-6.,6.,100),np.linspace(-6.,6.,100)
xx,yy = np.meshgrid(xvals,yvals)

plt.figure()

totaldensity = model.total_density(xx,yy,0.8)
print(f"Density at {x} kpc: {totaldensity} Msun")

plt.contourf(xx,yy,np.log10(totaldensity))
plt.colorbar()

plt.show()

"""

import numpy as np
from scipy.integrate import tplquad
#from multiprocessing import Pool

class Sormani2022MilkyWay:
    def __init__(self):
        # Define the parameters for each component: drawn from table 1
        self.params_bar1 = {
            "rho_1": 0.316,
            "alpha": 0.626,
            "x_1": 0.490,
            "y_1": 0.392,
            "z_1": 0.229,
            "c_parallel": 1.991,
            "c_perp": 2.232,
            "m": 0.873,
            "n": 1.940,
            "c": 1.342,
            "x_c": 0.751,
            "y_c": 0.469,
            "r_cut": 4.370
        }

        self.params_bar2 = {
            "rho_i": 0.050,
            "x_i": 5.364,
            "y_i": 0.959,
            "z_i": 0.611,
            "n_i": 3.051,
            "c_perp_i": 0.970,
            "R_i_out": 3.190,
            "R_i_in": 0.558,
            "n_i_out": 16.731,
            "n_i_in": 3.196
        }

        self.params_bar3 = {
            "rho_i": 1743.049,
            "x_i": 0.478,
            "y_i": 0.267,
            "z_i": 0.252,
            "n_i": 0.980,
            "c_perp_i": 1.879,
            "R_i_out": 2.204,
            "R_i_in": 7.607,
            "n_i_out": -27.291,
            "n_i_in": 1.630
        }

        self.params_disc = {
            "Sigma_0": 0.103,
            "z_d": 0.151,
            "R_d": 4.754,
            "R_cut": 4.688,
            "n_d": 1.536,
            "m_d": 0.716
        }

    def total_density(self, x, y, z):
        """
        Calculate the total density as the sum of the bar components and the disc component.

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.

        Returns:
        - float: The calculated total density.
        """
        rho_bar = self.rho_bar_a(x, y, z, **self.params_bar1) + self.rho_bar_i(x, y, z, 2, **self.params_bar2) + self.rho_bar_i(x, y, z, 3, **self.params_bar3)
        return rho_bar + self.rho_disc(np.sqrt(x**2 + y**2), z, **self.params_disc)

    def rho_bar_a(self, x, y, z, rho_1, alpha, x_1, y_1, z_1, c_parallel, c_perp, m, n, c, x_c, y_c, r_cut):
        """
        Calculate the density for bar component 1 (rho_bar_a).
        eq. 2 of Sormani+ 2022

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.
        - rho_1 (float): Density parameter for bar1.
        - alpha (float): Alpha parameter for bar1.
        - x_1 (float): X-coordinate parameter for bar1.
        - y_1 (float): Y-coordinate parameter for bar1.
        - z_1 (float): Z-coordinate parameter for bar1.
        - c_parallel (float): Parallel parameter for bar1.
        - c_perp (float): Perpendicular parameter for bar1.
        - m (float): M parameter for bar1.
        - n (float): N parameter for bar1.
        - c (float): C parameter for bar1.
        - x_c (float): X-coordinate parameter for bar1.
        - y_c (float): Y-coordinate parameter for bar1.
        - r_cut (float): R_cut parameter for bar1.

        Returns:
        - float: The calculated density for bar component 1.
        """
        a_plus, a_minus = self.a_plus_minus(x, y, z, c, x_c, y_c)
        return (
            rho_1 * (1./np.cosh(self.a(x, y, z, x_1, y_1, z_1, c_perp, c_parallel)))
            * (1 + alpha * (np.exp(-a_plus ** n) + np.exp(-a_minus ** n)))
            * np.exp(-(np.sqrt(x**2 + y**2 + z**2) / r_cut) ** 2)
        )

    def rho_bar_i(self, x, y, z, i, rho_i, x_i, y_i, z_i, n_i, c_perp_i, R_i_out, R_i_in, n_i_out, n_i_in):
        """
        Calculate the density for bar components 2 and 3 (rho_bar_i).
        eq. 6 in Sormani+ 2022

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.
        - i (int): Component index (2 or 3) for bar2 or bar3.
        - rho_i (float): Density parameter for bar2 or bar3.
        - x_i (float): X-coordinate parameter for bar2 or bar3.
        - y_i (float): Y-coordinate parameter for bar2 or bar3.
        - z_i (float): Z-coordinate parameter for bar2 or bar3.
        - n_i (float): N parameter for bar2 or bar3.
        - c_perp_i (float): Perpendicular parameter for bar2 or bar3.
        - R_i_out (float): R_outer parameter for bar2 or bar3.
        - R_i_in (float): R_inner parameter for bar2 or bar3.
        - n_i_out (float): N_outer parameter for bar2 or bar3.
        - n_i_in (float): N_inner parameter for bar2 or bar3.

        Returns:
        - float: The calculated density for bar component 2 or 3.
        """

        a_i = self.ai(x, y, x_i, y_i, c_perp_i)
        bigR = np.sqrt(x**2 + y**2)

        return (
            rho_i * np.exp(-a_i ** n_i) * (1./np.cosh(z / z_i)) ** 2
            * np.exp(-(bigR/R_i_out) ** n_i_out)
            * np.exp(-(R_i_in/bigR) ** n_i_in)
        )

    def a(self, x, y, z, x_i, y_i, z_i, c_perp_i, c_parallel_i):
        """
        Calculate 'a' for the density distribution calculation.
        eq. 3 of Sormani+ 2022

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.
        - x_i (float): X-coordinate parameter for the calculation.
        - y_i (float): Y-coordinate parameter for the calculation.
        - z_i (float): Z-coordinate parameter for the calculation.
        - c_perp_i (float): Perpendicular parameter for the calculation.
        - c_parallel_i (float): Parallel parameter for the calculation.

        Returns:
        - float: The calculated 'a' value.
        """
        # Calculate 'a' for the density distribution calculation
        return (((np.abs(x) / x_i) ** c_perp_i + (np.abs(y) / y_i) ** c_perp_i) ** (c_parallel_i / c_perp_i) + (np.abs(z) / z_i) ** c_parallel_i ) ** (1./c_parallel_i)

    def a_plus_minus(self, x, y, z, c, x_c, y_c):
        """
        Calculate 'a_plus' and 'a_minus' for the density distribution calculation.
        eq. 4 from Sormani+ 2022

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - z (float): Z-coordinate.
        - x_c (float): X-coordinate parameter for the calculation.
        - y_c (float): Y-coordinate parameter for the calculation.
        - c_perp (float): Perpendicular parameter for the calculation.

        Returns:
        - tuple: A tuple containing 'a_plus' and 'a_minus' values.
        """

        # Calculate 'a_plus' and 'a_minus' for the density distribution calculation
        a_plus = np.sqrt(((x+c*z) / x_c) ** 2 + (y / y_c) ** 2)
        a_minus = np.sqrt(((x-c*z) / x_c) ** 2 + (y / y_c) ** 2)
        return a_plus, a_minus

    def ai(self, x, y, x_i, y_i, c_perp_i):
        """
        Calculate 'a' for the density distribution calculation.
        eq. 7 of Sormani+ 2022

        Parameters:
        - x (float): X-coordinate.
        - y (float): Y-coordinate.
        - x_i (float): X-coordinate parameter for the calculation.
        - y_i (float): Y-coordinate parameter for the calculation.
        - c_perp_i (float): Perpendicular parameter for the calculation.
        - c_parallel_i (float): Parallel parameter for the calculation.

        Returns:
        - float: The calculated 'a' value.
        """
        # Calculate 'a' for the density distribution calculation
        return (((np.abs(x) / x_i) ** c_perp_i + (np.abs(y) / y_i) ** c_perp_i)) ** (1./c_perp_i)


    def rho_disc(self, R, z, **params_disc):
        """
        Calculate the density distribution for the disc component.
        eq. 9 of Sormani+ 2022

        Parameters:
        - R (float): Cylindrical radius.
        - z (float): Z-coordinate.
        - **params_disc (dict): Dictionary of parameters for the disc component.

        Returns:
        - float: The calculated density for the disc component.
        """
        return (params_disc["Sigma_0"] / (4 * params_disc["z_d"])) * np.exp(-(R / params_disc["R_d"]) ** params_disc["n_d"]) * np.exp(-params_disc["R_cut"] / R) * (1./(np.cosh(np.abs(z) / params_disc["z_d"]) ** params_disc["m_d"]))
