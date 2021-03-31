'''
bsplinebasis.py
Set up bspline basis

31-Mar-2021: First written

vision is to follow Vasiliev's technique for deriving radial basis functions using spline representation

built on scipy's implementation, but may use the naive implementation
as well.

TODO
1. investigate loss functions
2. investigate penalised fitting

'''

import numpy as np

from scipy.interpolate import BSpline


def B(x, k, i, t):
    '''
    the pedagogical example from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

    '''
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))
