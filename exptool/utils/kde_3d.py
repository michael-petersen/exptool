"""
kde_3d (part of exptool.utils)
      gaussian kernel density estimation in two or three dimensions


#    makes use of sparse matrices

A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than 
scipy's original scipy.stats.kde.gaussian_kde()).
-Joe Kington


( from http://pastebin.com/LNdYCZgw
and http://stackoverflow.com/questions/18921419/implementing-a-2d-fft-based-kernel-density-estimator-in-python-and-comparing-i )

Heavily modified by MSP

TODO:
1. Add kernel with compact support.

"""

import numpy as np
from numpy import fft
import scipy as sp
import scipy.sparse
import scipy.signal



def fast_kde(x, y, z, gridsize=(200, 200, 200), extents=None, nocorrelation=False, weights=None):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 3D histogram of the data.

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
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)

    fft_true = False
    
    if (x.size != y.size) & (x.size != z.size):
        raise ValueError('Input x, y, & z arrays must be the same size!')

    nx, ny, nz = gridsize
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
        zmin, zmax = z.min(), z.max()
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    xyzi = np.vstack((x,y,z)).T
    #xyzi -= [xmin, ymin, zmin]
    #xyzi /= [dx, dy, dz]
    #xyzi = np.floor(xyzi, xyzi).T
    #xyzi = np.floor(xyzi)

    #j = 0
    #print xyzi[j][0],xyzi[j][1],xyzi[j][2]
    
    #grid = np.zeros([nx,ny,nz])
    #for j in range(0,n):
        #grid[xyzi[j][0],xyzi[j][1],xyzi[j][2]] += weights[j]

    #grid, edges = np.histogramdd(xyzi,bins=(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(zmin,zmax,nz)),weights=weights)


    if fft_true: fgrid = np.fft.fftn(grid)
        
    # Next, make a 3D histogram of x, y, z
    # Avoiding np.histogram2d due to excessive memory usage with many points
    #grid = sp.sparse.coo_matrix((weights, xyzi), shape=(nx, ny, nz)).toarray()

    #grid = sp.sparse.coo_matrix(xyzi, shape=(nx, ny, nz)).toarray()

    #grid = xyzi.reshape(nx, ny, nz)

    xyzi -= [xmin, ymin, zmin]
    xyzi /= [dx, dy, dz]
    xyzi = np.floor(xyzi, xyzi).T
    #xyzi = np.floor(xyzi,xyzi).T
    
    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyzi)

    #print cov
    
    # make all off-diagonals zero
    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    #scotts_factor = np.power(n, -1.0 / 6) # For 2D
    scotts_factor = np.power(n, -1.0 / 7) # For 3D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny, kern_nz = np.round(scotts_factor * 3 * np.pi * std_devs)

    if fft_true: kern_nx, kern_ny, kern_nz = nx,ny,nz

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**3.)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    zz = np.arange(kern_nz, dtype=np.float) - kern_nz / 2.0
    xx, yy, zz = np.meshgrid(xx, yy, zz)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))    
    kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.sum(kernel, axis=0) / 3.0 # maybe 2
    kernel = np.exp(-kernel) 
    kernel = kernel.reshape((kern_ny, kern_nx, kern_nz))

    if fft_true: fKer = np.fft.fftn(kernel)

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 3D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    #grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T
    grid = sp.ndimage.filters.convolve(grid, kernel, mode='constant').T

    if fft_true: kde1 = np.fft.fftshift(np.fft.ifftn(fgrid*fKer))/64
    
    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.  
    norm_factor = 2 * np.pi * cov * scotts_factor**3.
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * dz * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    if fft_true:
        return kde1
    else:
        return np.flipud(grid)





def fast_kde_two(x, y, gridsize=(200, 200), extents=None, nocorrelation=False, weights=None, npower=6.,ktype='gaussian'):
    '''
    fast_kde_two
        Performs a gaussian kernel density estimate over a regular grid using a
        convolution of the gaussian kernel with a 2D histogram of the data.

        This function is typically several orders of magnitude faster than 
        scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
        produces an essentially identical result.

    inputs
    ---------------
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
    npower:

    ktype: kernel type to use. Options:
        'gaussian'
        'epanechnikov'
        'linear' : not implemented yet
        

    returns
    --------------
        A gridded 2D kernel density estimate of the input points. 
    '''
    #---- Setup --------------------------------------------------------------
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
   
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    try:
        if len(gridsize)==2:
            nx, ny = gridsize
    except:
        nx = ny = gridsize
        
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
        try:
            if len(extents) == 4:
                xmin, xmax, ymin, ymax = map(float, extents)
        except:
            xmin = ymin = -1.*extents
            xmax = ymax = extents

            
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)


    within_extent = np.where( (x >= xmin) & (x <= xmax) & (y > ymin) & (y < ymax))[0]

    x = x[within_extent]
    y = y[within_extent]

    weights = weights[within_extent]

    # 12.26.17: why did I put these here? They will override the cutouts above?
    #x, y = np.asarray(x), np.asarray(y)
    #x, y = np.squeeze(x), np.squeeze(y)
    

    
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

    #grid, edges = np.histogramdd(xyi,bins=(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny)),weights=weights)


    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / npower) 



    #---- Make the kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(np.abs(cov)))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2) 

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    kernel = np.vstack((xx.flatten(), yy.flatten()))

    if ktype=='linear':
        # still in testing
        raise NotImplementedError()

    elif ktype=='epanechnikov':
        kernel = np.dot(inv_cov, kernel) * kernel 
        kernel = np.abs(1. - np.sum(kernel, axis=0))

    else:
        # implement gaussian as catchall
        if ktype != 'gaussian':
            print('kde_3d.fast_kde_two: falling back to gaussian kernel')
        # Then evaluate the gaussian function on the kernel grid
        kernel = np.dot(inv_cov, kernel) * kernel 
        kernel = np.sum(kernel, axis=0) / 2.0 
        kernel = np.exp(-kernel) 


    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))


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



def total_kde_two(x, y, gridsize=128, extents=1., nocorrelation=False, npower=6.,surfacedensity=False,ktype='gaussian',**kwargs):
    '''
    total_kde_two
         quick wrapper to return x and y grids to go along with the kernel densities



    inputs
    ---------------------



    returns
    --------------------




    '''

    # gridsize can be a tuple
    try:
        if len(gridsize)==2:
            nx, ny = gridsize
    except:
        nx = ny = gridsize
        
    
    if 'weights' in kwargs.keys():
        weights = kwargs['weights']
        
    else:
        weights = None
    
    if 'opt_third' in kwargs.keys():
        if 'opt_third_constraint' in kwargs.keys():
        
            w = np.where( kwargs['opt_third'] < kwargs['opt_third_constraint'])[0]

        else: print('kde_3d.total_kde_two: opt_third_constraint required to use opt_third.')

        x = x[w]
        y = y[w]

        if 'weights' in kwargs.keys():
            weights = weights[w]

    #
    # only set to return square, evenly space grids currently 08-26-16
    # 
    KDEArray = fast_kde_two(x, y, gridsize=gridsize, extents=extents, nocorrelation=nocorrelation, weights=weights, npower=npower,ktype=ktype)

    try:
        xbins = np.linspace(-1.*extents,extents,gridsize)
        xx,yy = np.meshgrid( xbins,xbins)

    except:
        # if extents tuple is passed
        xbins = np.linspace(extents[0],extents[1],gridsize)
        ybins = np.linspace(extents[2],extents[3],gridsize)
        xx,yy = np.meshgrid(xbins,ybins)
    
    effvolume = ((xbins[1]-xbins[0])*(xbins[1]-xbins[0]))#*(2.*zlim))

    if surfacedensity:
        KDEArray /= effvolume

    return xx,yy,KDEArray
    



      
'''
# for that pesky np.sqrt, if desired
    
    with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')


#http://stackoverflow.com/questions/29347987/why-cant-i-suppress-numpy-warnings
'''
