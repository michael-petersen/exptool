"""
mndisc.py

class for Miyamoto-Nagai (1975) disc.

supporting:
MiyamotoNagai

24 Sep 2023  New docstrings


"""
import numpy as np

class MiyamotoNagai(object):
    """Instantiation of a Miyamoto-Nagai disc potential.

    The Miyamoto-Nagai (1975) disk associated with the potential of a thin disk is given by:
    $$\rho_M(R, z) = \left(\frac{b^2M}{4\pi}\right)\frac{aR^2+(a+3\sqrt{z^2+b^2})(a+\sqrt{z^2+b^2})^2}{\left[R^2+(a+\sqrt{z^2+b^2})^2\right]^{5/2}(z^2+b^2)^{3/2}}

    Attributes:
        a (float): The scale parameter for the Miyamoto-Nagai disc.
        b (float): The vertical scale parameter for the disc.
        M (float): The total mass of the disc.
        G (float): The gravitational constant.

    Methods:
        potential(R, z):
            Calculate the gravitational potential of the Miyamoto-Nagai disc.

        mass(R, z):
            Calculate the spherical enclosed mass.

        density(R, z):
            Calculate the density of the Miyamoto-Nagai disc.

        zforce(R, z):
            Compute the vertical force.

        rforce(R, z):
            Compute the radial force.
    """

    def __init__(self, a, b, M=1, G=1):
        """Initialize the Miyamoto-Nagai disc.

        Args:
            a (float): The scale parameter for the Miyamoto-Nagai disc.
            b (float): The vertical scale parameter for the disc.
            M (float, optional): The total mass of the disc. Defaults to 1.
            G (float, optional): The gravitational constant. Defaults to 1.
        """

        self.a = a
        self.b = b
        self.M = M
        self.G = G

    def potential(self, R, z):
        """Calculate the gravitational potential of the Miyamoto-Nagai disc.

        Args:
            R (float): The radial distance from the center.
            z (float): The vertical distance from the plane.

        Returns:
            float: The gravitational potential at the given (R, z) position.
        """

        return -(self.G * self.M) / (np.sqrt(R**2. + (self.a + np.sqrt(z*z + self.b*self.b))**2.))

    def mass(self, R, z):
        """Calculate the spherical enclosed mass.

        Args:
            R (float): The radial distance from the center.
            z (float): The vertical distance from the plane.

        Returns:
            float: The spherical enclosed mass at the given (R, z) position.
        """

        rad = np.sqrt(R*R + z*z)
        return rad * -self.potential(R, z)

    def density(self, R, z):
        """Calculate the density of the Miyamoto-Nagai disc.

        Args:
            R (float): The radial distance from the center.
            z (float): The vertical distance from the plane.

        Returns:
            float: The density at the given (R, z) position.
        """

        zscale = z*z + self.b*self.b
        prefac = (self.b*self.b*self.M / (4. * np.pi))
        numerator = ((self.a * R * R) + (self.a + 3. * np.sqrt(zscale)) * (self.a + np.sqrt(zscale))**2.)
        denominator = ((R*R + (self.a + np.sqrt(zscale))**2.)**(5./2.) * (zscale)**(3./2.))
        return prefac * numerator / denominator

    def zforce(self, R, z):
        """Compute the vertical force.

        Args:
            R (float): The radial distance from the center.
            z (float): The vertical distance from the plane.

        Returns:
            float: The vertical force at the given (R, z) position.
        """

        zb = np.sqrt(z*z + self.b*self.b)
        ab = self.a + zb
        dn = np.sqrt(R*R + ab*ab)
        d3 = dn*dn*dn
        return -self.M * z * ab / (zb * d3)

    def rforce(self, R, z):
        """Compute the radial force.

        Args:
            R (float): The radial distance from the center.
            z (float): The vertical distance from the plane.

        Returns:
            float: The radial force at the given (R, z) position.
        """

        zb = np.sqrt(z*z + self.b*self.b)
        ab = self.a + zb
        dn = np.sqrt(R*R + ab*ab)
        d3 = dn*dn*dn
        return -self.M * R / d3
