####################################################################
#
# Utility routines for various exptool items
#
#    04.01.17: Commented for better understanding!
#
#
####################################################################

'''
utils.py: part of exptool

todo
--------------------
break into different modules.

brainstorming ideas:
1. 1d data treatments (e.g. smoothing/signal processing)
2. 2d data treatments (e.g. kde wrappers)
3. style elements (e.g. printing, plot styles)


What's included:

print_progress : print incremental progress to screen with a tracking bar
resample       : quick wrapper for scipy UnivariateSpline that returns the resampled values

legendre_smooth : THIS NEEDS THE THEORETICAL UNDERPINNING!

normalhist : quick and dirty normalized histogram (to 1)
binnormalhist : normalhist that returns bins
quick_contour : quick and dirty 2-d histogram sampling
argrelextrema : robust method to find local minima/maxima (needs better documentation!)
savitzky_golay : smoothing using a savitzky_golay filter



unwrapped_phases = unwrap_phase(times,phases,max_periods=1000)


TODO
from scipy.signal import argrelextrema
supersedes argrelextema here. Can cut this out of code perhaps?


'''
from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import sys
from scipy.interpolate import UnivariateSpline
    
import numpy as np
from math import factorial

from numpy.polynomial.legendre import legvander,legder
from scipy.linalg import qr,solve_triangular,lu_factor,lu_solve

import matplotlib.pyplot as plt


def print_progress(current_n,total_n,module):
    #
    # print progress with a tracking bar
    #
    last = 0

    if float(current_n+1)==float(total_n): last = 1

    bar = ('=' * int(float(current_n)/total_n * 20.)).ljust(20)
    percent = int(float(current_n)/total_n * 100.)

    if last:
        bar = ('=' * int(20.)).ljust(20)
        percent = int(100)
        print('{0:20s}: [{1:20s}] {2:2d}%'.format(module, bar, percent))
        
    else:
        print('{0:20s}: [{1:20s}] {2:2d}%'.format(module, bar, percent),end='\r')
        sys.stdout.flush()





def resample(x,y,new_x,sord=0):
    #
    # simple wrapper for UnivariateSpline routine that does the resampling.
    #
    # dv = resample(
    #newx = np.linspace(np.min(T),np.max(T),len(T)*impr)
    sX = UnivariateSpline(x,y,s=sord)
    return sX(new_x)





def normalhist(array,nbins,colorst='black'):
    '''
    
    quick and dirty normalized histogram
    
    '''

    bins = np.linspace(np.min(array),np.max(array),nbins)
    binsind  = bins+(0.5*(bins[1]-bins[0]))
    array_out = np.zeros(nbins)

    for i in range(0,len(array)):
        binn = np.floor( (array[i]-bins[0])/(bins[1]-bins[0]) )
        array_out[binn] += 1

    array_out /= (sum(array_out))

    #plt.plot(binsind,array_out,color=colorst,linewidth=2.0,drawstyle='steps-mid')
	#plt.draw()
    

def binnormalhist(array,bins,weights=None):
    '''
    binnormalhist
         return the 1d histogram given bins and a 1d array (plus optional weighting array)
    '''
    
    if weights==None:
        weights = np.ones(len(array))
    
    binsind = bins+(0.5*(bins[1]-bins[0]))
    array_out = np.zeros(len(bins))
    
    for i in range(0,len(array)):
        binn = int(np.floor( (array[i]-bins[0])/(bins[1]-bins[0])))
    
        if ((binn >= 0) & (binn < len(bins)) ):
            array_out[binn] += weights[i]
        
    array_out /= (sum(array_out))

    return array_out



def quick_contour(ARR1,ARR2,weights=None,X=None,Y=None,resolution=25):
    #
    # Quickly bin data (BRUTE FORCE)
    #
    # Depends: numpy import as np
    #
    # Inputs:  X    (index values in first dimension)
    #          Y    (index values in second dimension)
    #          ARR1 (data values in first dimension)
    #          ARR2 (data values in second dimension)
    #          ARR3 (weights of each data point--set as np.ones(len(ARR1)) for equal weighting)
    #
    # Outputs: XX   (mask of first dimension values)
    #          YY   (mask of second dimension values)
    #          OUT  (binned data)
    #
    # Plot with matplotlib.pyplot.contour(XX,YY,OUT)
    #
    try:
        truefalse = X[0]
        truefalse = Y[0]
    except:
        X = np.linspace(np.min(ARR1),np.max(ARR1),resolution)
        Y = np.linspace(np.min(ARR2),np.max(ARR2),resolution)
        
    try:
        truefalse = weights[0]
        ARR3 = weights
        
    except:

        ARR3 = np.ones([len(ARR1)])


    XX,YY = np.meshgrid(X,Y)

    OUT = np.zeros([len(Y),len(X)])

    for i in range(0,len(ARR1)):

        xind = int(np.floor((ARR1[i]-X[0])/(X[1]-X[0])))

        yind = int(np.floor((ARR2[i]-Y[0])/(Y[1]-Y[0])))

        if xind>=0 and xind<len(X) and yind>=0 and yind<len(Y): OUT[yind][xind]+=ARR3[i]

    return XX,YY,OUT




#
# class to implement legedre smoothing
#



class legendre_smooth(object):
    def __init__(self,n,k,a,m):
        self.k2 = 2*k

        # Uniform grid points
        x = np.linspace(-1,1,n)

        # Legendre Polynomials on grid 
        self.V = legvander(x,m-1)

        # Do QR factorization of Vandermonde for least squares 
        self.Q,self.R = qr(self.V,mode='economic')

        I = np.eye(m)
        D = np.zeros((m,m))
        D[:-self.k2,:] = legder(I,self.k2)

        # Legendre modal approximation of differential operator
        self.A = I-a*D

        # Store LU factors for repeated solves   
        self.PLU = lu_factor(self.A[:-self.k2,self.k2:])

    def fit(self,z):

        # Project data onto orthogonal basis 
        Qtz = np.dot(self.Q.T,z)

        # Compute expansion coefficients in Legendre basis
        zhat = solve_triangular(self.R,Qtz,lower=False)

        # Solve differential equation       
        yhat = np.zeros(len(zhat))
        q = np.dot(self.A[:-self.k2,:self.k2],zhat[:self.k2])
        r = zhat[:-self.k2]-q
        yhat[:self.k2] = zhat[:self.k2]
        yhat[self.k2:] = lu_solve(self.PLU,r)
        y = np.dot(self.V,yhat)
        return y





# find these at https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding.py#L176
# might be nice to check out numpy.convolve


#
# the utilities for accurate aps finding
# 

def _boolrelextrema(data, comparator,
                  axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.
    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take 2 numbers as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.  'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'.  See numpy.take
    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.
    See also
    --------
    argrelmax, argrelmin

    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)

    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)

    # this is the legacy python2 version with xrange (keep to avoid breaking)
    try:
        for shift in xrange(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if(~results.any()):
                return results

    # but if python2 breaks, shift to python3 version
    except:
        for shift in range(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if(~results.any()):
                return results
    return results





def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take 2 numbers as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.  'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default is 'clip'.  See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    results = _boolrelextrema(data, comparator,
                              axis, order, mode)
    return np.where(results)




def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
        
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:# ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
        
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    
    return np.convolve( m[::-1], y, mode='valid')



def unwrap_phase(times,phases,max_periods=1000):
    """
    unwrap raw phase information
    
    -the big advantage of this new algorithm is that it can go clockwise or counterclockwise
    (finally)
    
    -assumes phases were calculated using arctan2

    inputs
    -------------
    times
    phases
    max_periods : (int, default=1000) maximum number of periods expected
    
    outputs
    -------------
    unwrapped_phase : (array, same size as phases) the unwrapped phase
    
    """
    # check data validity
    if (times.size != phases.size):
        raise ValueError('unwrap_phases: times and phases must be equal size.')
    
    # check if phase is outside of boundaries
    if ((np.nanmax(phases)>2.*np.pi) | (np.nanmin(phases) < -np.pi)):
        raise ValueError('unwrap_phases: Values are outside of the accepted phase boundaries.')
        
    if np.nanmin(phases) < 0:
        phases += np.pi
        
    if (np.nanmax(phases) - np.nanmin(phases) < np.pi):
        print('unwrap_phases WARNING: Were the phases calculated with arctan2?')
        
    # initialize phase array    
    unwrapped_phase = np.zeros(phases.size)
    unwrapped_phase[0] = phases[0]

    offset_array = np.arange(-max_periods,max_periods,1)

    for t in range(1,times.size):
    
        # generate the blank array
        shifted_array = phases[t]+(2*np.pi*offset_array)
    
        # find the closest in absolute difference
        closest = np.abs(shifted_array - unwrapped_phase[t-1]).argmin()
        #print(closest)
        #print(shifted_array[closest])
        unwrapped_phase[t] = shifted_array[closest]
        
    return unwrapped_phase
    
    



def bilinear_interpolation(x, y, points):
    '''
    basic bilinear interpolation scheme

    '''
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')
    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)



