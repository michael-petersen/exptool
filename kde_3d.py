"""
A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than 
scipy's original scipy.stats.kde.gaussian_kde()).
-Joe Kington


( from http://pastebin.com/LNdYCZgw
and http://stackoverflow.com/questions/18921419/implementing-a-2d-fft-based-kernel-density-estimator-in-python-and-comparing-i )

import psp_io
import kde_3d

O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='star')

tt = kde_3d.fast_kde(O.xpos,O.ypos,O.zpos, gridsize=(64,64,64), extents=[-0.05,0.05,-0.05,0.05,-0.005,0.005], nocorrelation=False, weights=None)

XX,YY = np.meshgrid(np.linspace(-0.05,0.05,63),np.linspace(-0.05,0.05,63))

plt.figure(0)
plt.contourf(XX,YY,np.log10(np.rot90(np.sum(tt,axis=0),1)),36)
plt.axis([-0.04,0.04,-0.04,0.04])
plt.text(-0.03,0.03,'Fiducial, T=2.0')
plt.xticks([-0.03,0.0,0.03])
plt.yticks([-0.03,0.0,0.03])
plt.savefig('/Users/mpetersen/Desktop/041916_2.png')




plt.figure(1)
O = psp_io.Input('/Users/mpetersen/Research/NBody/OUT.run013p.01000',comp='star')

tt = kde_3d.fast_kde(O.xpos,O.ypos,O.zpos, gridsize=(64,64,64), extents=[-0.05,0.05,-0.05,0.05,-0.005,0.005], nocorrelation=False, weights=None)

plt.contourf(XX,YY,np.log10(np.sum(tt,axis=0)),36)
plt.axis([-0.04,0.04,-0.04,0.04])
plt.text(-0.03,0.03,'StellarDisk, T=2.0')
plt.xticks([-0.03,0.0,0.03])
plt.yticks([-0.03,0.0,0.03])
plt.savefig('/Users/mpetersen/Desktop/041916_2.png')




plt.figure(2)
O = psp_io.Input('/Users/mpetersen/Research/NBody/OUT.run013p.01000',comp='darkdisk')

tt = kde_3d.fast_kde(O.xpos,O.ypos,O.zpos, gridsize=(64,64,64), extents=[-0.05,0.05,-0.05,0.05,-0.005,0.005], nocorrelation=False, weights=None)

plt.contourf(XX,YY,np.log10(np.sum(tt,axis=0)),36)
plt.axis([-0.04,0.04,-0.04,0.04])
plt.text(-0.03,0.03,'DarkDisk, T=2.0')
plt.xticks([-0.03,0.0,0.03])
plt.yticks([-0.03,0.0,0.03])
plt.savefig('/Users/mpetersen/Desktop/041916_3.png')




plt.figure(3)
O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.00300',comp='star')

tt = kde_3d.fast_kde(O.xpos,O.ypos,O.zpos, gridsize=(64,64,64), extents=[-0.05,0.05,-0.05,0.05,-0.005,0.005], nocorrelation=False, weights=None)

plt.contourf(XX,YY,np.log10(np.sum(tt,axis=0)),36)
plt.axis([-0.04,0.04,-0.04,0.04])
plt.text(-0.03,0.03,'Fiducial, T=0.6')
plt.xticks([-0.03,0.0,0.03])
plt.yticks([-0.03,0.0,0.03])
plt.savefig('/Users/mpetersen/Desktop/041916_4.png')




plt.contourf(np.sum(tt,axis=2))

xmin,xmax = np.min(O.xpos),np.max(O.xpos)
ymin,ymax = np.min(O.ypos),np.max(O.ypos)
zmin,zmax = np.min(O.zpos),np.max(O.zpos)

nx,ny,nz = (128, 128, 128)

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)

xyzi = np.vstack((O.xpos,O.ypos,O.zpos)).T
#xyzi -= [xmin, ymin, zmin]
#xyzi /= [dx, dy, dz]
#xyzi = np.floor(xyzi)


H, edges = np.histogramdd(xyzi,bins=(np.linspace(-0.05,0.05,64),np.linspace(-0.05,0.05,64),np.linspace(-0.005,0.005,64)))


def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'

import numpy as np
from numpy import fft
import scipy as sp
import scipy.sparse
import scipy.signal

def fast_kde(x, y, z, gridsize=(200, 200, 200), extents=None, nocorrelation=False, weights=None):
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

    grid, edges = np.histogramdd(xyzi,bins=(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(zmin,zmax,nz)))


    if fft_true: fgrid = np.fft.fftn(grid)
        
    # Next, make a 3D histogram of x & y
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

    print cov
    
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
