'''
bsplinebasis.py
Set up bspline basis

31-Mar-2021: First written
04 Nov 2023: revised to improve playground

vision is to follow Vasiliev's technique for deriving radial basis functions using spline representation

This version has edges clamped at 0.


TODO
1. investigate loss functions
2. investigate penalised fitting
3. explore monotone cubic spline

import numpy as np
import matplotlib.pyplot as plt

# Generate random control points for the B-spline curve
np.random.seed(0)
control_points = np.random.rand(10) * 10  # 10 random control points

# Generate knot vector (assuming cubic B-spline with open uniform knots)
num_points = len(control_points)
num_knots = num_points + 4  # 3 for cubic B-spline
knots = np.linspace(0, 10, num_knots)

# Create a B-spline curve object
bspline_curve = BSpline(control_points)

# Evaluate the B-spline curve at various points for plotting
x_values = np.linspace(0, 10, 1000)
y_values = np.array([bspline_curve.evaluate(x, knots) for x in x_values])

# Plot the B-spline curve
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, color='blue', label='B-spline Curve')
plt.scatter(knots[self.degree:-self.degree], control_points, color='red', label='Control Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-spline Curve with Control Points')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the error between B-spline curve and observed data
error = bspline_curve.evaluate_error(observed_data, knots)
print("Error:", error)

# Function to minimize (difference between B-spline curve and points)
def objective_function(control_points):
    curve_values = np.array([bspline(x, t, control_points, k) for x in points])
    error = np.sum((curve_values - points) ** 2)
    return error

# Optimize control points to minimize the difference between B-spline curve and points
from scipy.optimize import minimize
result = minimize(objective_function, control_points, method='BFGS')

# Fitted control points
fitted_control_points = result.x


'''

import numpy as np


class BSpline:
    def __init__(self, control_points, degree=3):
        """
        Initialize a B-spline curve with given control points and degree.

        Parameters:
        - control_points: List of control points defining the shape of the B-spline curve.
        - degree: Degree of the B-spline curve (default is 3 for cubic B-spline).
        """
        self.control_points = control_points
        self.degree = degree

    def _basis_function(self, x, k, i, t):
        """
        Compute the value of a B-spline basis function.

        Parameters:
        - x: Evaluation point.
        - k: Degree of the B-spline basis function.
        - i: Index of the basis function within the B-spline curve.
        - t: Knot vector defining the position and multiplicity of knots.

        Returns:
        - The value of the B-spline basis function at the given point x.
        """
        # Base case: B-spline basis function of degree 0 is 1 within the interval [t[i], t[i+1])
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0

        # Recursive calculation for higher degree basis functions
        c1 = 0.0 if t[i+k] == t[i] else (x - t[i]) / (t[i+k] - t[i]) * B(x, k-1, i, t)
        c2 = 0.0 if t[i+k+1] == t[i+1] else (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)

        return c1 + c2

    def evaluate(self, x, knots):
        """
        Evaluate the B-spline curve at a specific point.

        Parameters:
        - x: Evaluation point.
        - knots: Knot vector.

        Returns:
        - The value of the B-spline curve at the given point x.
        """
        n = len(knots) - self.degree - 1
        assert (n >= self.degree + 1) and (len(self.control_points) >= n), "Invalid number of control points or knots."

        # Calculate the B-spline curve value by summing contributions from control points
        curve_value = sum(self.control_points[i] * self._basis_function(x, self.degree, i, knots) for i in range(n))
        return curve_value

    def evaluate_error(self, observed_data, knots):
        """
        Evaluate the error of the B-spline curve compared to observed data.

        Parameters:
        - observed_data: List or array of observed data points.
        - knots: Knot vector.

        Returns:
        - The sum of squared differences between the B-spline curve and observed data points.
        """
        # Evaluate the B-spline curve at each observed data point
        bspline_values = np.array([self.evaluate(x, knots) for x in observed_data])

        # Calculate the sum of squared differences (error) between B-spline curve and observed data
        error = np.sum((bspline_values - observed_data) ** 2)
        return error