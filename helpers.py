#
# Random helpers
#

import numpy as np



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
    
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
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



def resample(x,y,new_x,sord=0):
    # dv = resample(
    from scipy.interpolate import UnivariateSpline
    #newx = np.linspace(np.min(T),np.max(T),len(T)*impr)
    sX = UnivariateSpline(x,y,s=sord)
    return sX(new_x)





def quick_contour(ARR1,ARR2,weights=-1,X=0,Y=0,resolution=25):
    #
    # Quickly bin data
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
    if (X==0) | (Y==0):
        X = np.linspace(np.min(ARR1),np.max(ARR1),resolution)
        Y = np.linspace(np.min(ARR2),np.max(ARR2),resolution)
    if weights<0:
        ARR3 = np.ones([len(ARR1)])
	XX,YY = np.meshgrid(X,Y)
	OUT = np.zeros([len(X),len(Y)])
	for i in range(0,len(ARR1)):
		xind = int(np.floor((ARR1[i]-X[0])/(X[1]-X[0])))
		yind = int(np.floor((ARR2[i]-Y[0])/(Y[1]-Y[0])))
		if xind>=0 and xind<len(X) and yind>=0 and yind<len(Y): OUT[yind][xind]+=ARR3[i]
	return XX,YY,OUT



import numpy as np
from numpy.polynomial.legendre import legvander,legder
from scipy.linalg import qr,solve_triangular,lu_factor,lu_solve

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





'''
resx = 10.**np.linspace(-3.,-1.,100)

dv = resample(Pd.rbins,Pd.vcirc,resx)
hv = resample(P.rbins,P.vcirc,resx)

lines = 'solid'
plt.plot(resx,dv,linestyle=lines,color='red')
plt.plot(resx,hv,linestyle=lines,color='gray')
plt.plot(resx,(dv**2.+hv**2.)**0.5,linestyle=lines,color='black')



plt.xlabel('Radius')
plt.ylabel('Vcirc')
'''


"""
A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than 
scipy's original scipy.stats.kde.gaussian_kde()).
-Joe Kington
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.signal

def fast_kde(x, y, gridsize=(200, 200), extents=None, nocorrelation=False, weights=None):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    This function is typically several orders of magnitude faster than 
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
    produces an essentially identical result.

    Input:
        x: The x-coords of the input data points
        y: The y-coords of the input data points
        gridsize: (default: 200x200) A (nx,ny) tuple of the size of the output 
            grid
        extents: (default: extent of input data) A (xmin, xmax, ymin, ymax)
            tuple of the extents of output grid
        nocorrelation: (default: False) If True, the correlation between the
            x and y coords will be ignored when preforming the KDE.
        weights: (default: None) An array of the same shape as x & y that 
            weighs each sample (x_i, y_i) by each value in weights (w_i).
            Defaults to an array of ones the same size as x & y.
    Output:
        A gridded 2D kernel density estimate of the input points. 
    """
    #---- Setup --------------------------------------------------------------
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
    
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2) 

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.sum(kernel, axis=0) / 2.0 
    kernel = np.exp(-kernel) 
    kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.  
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return np.flipud(grid)


