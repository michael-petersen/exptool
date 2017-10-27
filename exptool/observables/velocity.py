

import numpy as np

from exptool.utils import kde_3d
from exptool.io import psp_io



def measured_rotation(infile,comp,rmax=0.06,nsamp=256):
    '''
    velocity.measured_rotation
        measure rotation velocity from particles

    inputs
    ---------------
    infile
    comp
    rmax
    nsamp

    outputs
    --------------


    '''
    O = psp_io.Input(infile,comp=comp)
    extent = rmax
    nsamp = 256
    kde_weight = (O.xpos*O.yvel - O.ypos*O.xvel)/( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5)
    #
    rvals = ( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5)
    tvals = np.arctan2( O.ypos,O.xpos)
    w = np.where( (abs(rvals) < rmax))[0]
    vv = kde_3d.fast_kde_two(rvals[w],tvals[w], gridsize=(nsamp,nsamp), extents=(0.,rmax,-np.pi,np.pi), nocorrelation=False, weights=kde_weight[w])
    tt = kde_3d.fast_kde_two(rvals[w],tvals[w], gridsize=(nsamp,nsamp), extents=(0.,rmax,-np.pi,np.pi), nocorrelation=False, weights=None)
    rbins = np.linspace(0.0,rmax,nsamp)
    return rbins,vv/tt


