###########################################################################################
#
#  logpot.py
#     Initialise a logarithmic potential
#     Useful for following Binney & Tremaine chapter 3
#
# 09 Jul 2021: Introduction
#
#
'''

logpot (part of exptool.models)
    Implementation of a planar logarithmic potential

24 Sep 2023  New docstrings


'''

class LogPot():
    '''
    Planar Logarithmic Potential

    The LogPot class represents a planar logarithmic potential.

    Attributes:
        rscl2 (float): The square of the scaling radius (rscl).
        q (float): The axis ratio (q).
        v02 (float): The square of the characteristic velocity (v0).

    Methods:
        get_pot(x, y):
            Calculate the gravitational potential at a given (x, y) position.

        get_xforce(x, y):
            Calculate the x-component of the gravitational force at a given (x, y) position.

        get_yforce(x, y):
            Calculate the y-component of the gravitational force at a given (x, y) position.

        get_cartesian_forces_array(arr):
            Calculate the gravitational forces as a numpy array [xforce, yforce, 0] at a given (x, y) position.

    Example usage:
        L = LogPot(rscl=0.1, q=0.8, v0=1.0)
        print(L.get_pot(-1, -1))
    '''

    def __init__(self, rscl=1., q=1.0, v0=1.):
        """
        Initialize the LogPot object.

        Args:
            rscl (float, optional): The scaling radius (rscl). Defaults to 1.0.
            q (float, optional): The axis ratio (q). Defaults to 1.0.
            v0 (float, optional): The characteristic velocity (v0). Defaults to 1.0.
        """

        self.rscl2 = rscl * rscl
        self.q = q
        self.v02 = v0 * v0

    def get_pot(self, x, y):
        """Calculate the gravitational potential at a given (x, y) position.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            float: The gravitational potential at the specified position.
        """

        logval = self.rscl2 + x * x + y * y / self.q / self.q
        return 0.5 * self.v02 * np.log(logval)

    def get_xforce(self, x, y):
        """Calculate the x-component of the gravitational force at a given (x, y) position.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            float: The x-component of the gravitational force at the specified position.
        """

        yscl2 = y * y / self.q / self.q
        return -(self.v02 * x) / (self.rscl2 + yscl2 + x * x)

    def get_yforce(self, x, y):
        """Calculate the y-component of the gravitational force at a given (x, y) position.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            float: The y-component of the gravitational force at the specified position.
        """

        return -(self.v02 * y) / (self.q * self.q * (self.rscl2 + x * x) + y * y)

    def get_cartesian_forces_array(self, arr):
        """Calculate the gravitational forces as a numpy array [xforce, yforce, 0] at a given (x, y) position.

        Args:
            arr (numpy.ndarray): A numpy array of shape (2,) representing the (x, y) position.

        Returns:
            numpy.ndarray: A numpy array of shape (3,) representing [xforce, yforce, 0].
        """

        x = arr[0]
        y = arr[1]

        xforce = self.get_xforce(x, y)
        yforce = self.get_yforce(x, y)

        return np.array([xforce, yforce, 0.])
