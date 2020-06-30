
# basic imports
from __future__ import print_function
import numpy as np
from numpy.linalg import eig, inv


# plotting elements
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12



# processing elements
from scipy.interpolate import UnivariateSpline
import scipy.ndimage.filters as filters
from skimage.measure import find_contours


# exptool imports
from .io import psp_io
from .utils import kde_3d
from .observables import transform
from .analysis import pattern


# location of the files to become a movie: satellite edition

indir = '/proj/mpetersen/Disk044/'
runtag = 'run044g2j'
outputdir = '/proj/mpetersen/MWmovies/movie6/'
comptime=50


class Movie():
    """
    wrapper class for a movie

    """
    def __init__():
        """specify the input directory

        """
        pass


    
def build_colorbars(indir,runtag,comptime=0,\
                        gridsize=300,\
                        face_extents=0.15,\
                        edge_extents=0.02,\
                        ktype='gaussian',\
                        npower=5.,\
                        cres=84,\
                        velocity=False):
    """


    """
    
    # build a test case output for scaline the colorbar

    # ideally, adaptively pick the late time snapshot

    PSPDump = psp_io.Input(indir+'OUT.'+runtag+'.{0:05d}'.format(comptime),'star')

    kdeX1,kdeY1,kdePOSXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                       gridsize=gridsize,\
                                       extents=face_extents,\
                                       weights=PSPDump.mass,\
                                       ktype=ktype,npower=npower)

    if velocity:
        kdeXv1,kdeYv1,kdePOSvXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                       gridsize=gridsize,\
                                       extents=face_extents,\
                                       weights=PSPDump.mass*PSPDump.yvel,\
                                       ktype=ktype,npower=npower)

        # make the velocity field:
        velfield = kdePOSvXY/kdePOSXY

        # blank regions of low S/N: set to zero velocity? or nan
        threshhold = 1.*np.min(PSPDump.mass)
        velfield[kdePOSXY<=threshhold] = 0.

        flatvelfield = velfield.reshape(-1,)

        # define the face-on velocity range
        vrange = np.linspace(np.nanmin(velfield),np.nanmax(velfield),cres)


    # check minimum and maximum
    # Also do an edge-on view
    crange = np.linspace(np.log10(0.99*np.min(PSPDump.mass)),np.log10(np.max(kdePOSXY)),cres)

    # edge view for colorbar
    zgridsize = 100
    edge_extents = 0.03


    kdeX1,kdeY1,kdePOSXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                       gridsize=gridsize,\
                                       extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                       weights=PSPDump.mass,\
                                       ktype=ktype,npower=npower)

    if velocity:
        kdeXv1,kdeYv1,kdePOSvXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                       gridsize=gridsize,\
                                       extents=face_extents,\
                                       weights=PSPDump.mass*PSPDump.yvel,\
                                       ktype=ktype,npower=npower)



cres=84
crange_edge = np.linspace(np.log10(0.99*np.min(PSPDump.mass)),np.log10(np.max(kdePOSXY)),cres)


velfield = kdePOSvXY/kdePOSXY
# blank regions of low S/N: set to zero velocity? or nan
threshhold = 1.*np.min(PSPDump.mass)
velfield[kdePOSXY<threshhold] = 0.

flatvelfield = velfield.reshape(-1,)
#plt.plot(flatvelfield[flatvelfield.argsort()],color='red')





# diagnostic printing for the scalebar?






