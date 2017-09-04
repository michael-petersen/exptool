
'''

transform.py : part of exptool

to view a galaxy on the sky in some way (LOS usefulness)
                      
'''

# standard libraries
import numpy as np
import time

# exptool libraries
from exptool.io import psp_io


def rotate_points(PSPDump,xrotation,yrotation,zrotation):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump
    xrotation   : rotation into/out of page, in degrees
    yrotation
    zrotation



    returns
    ------------------



    '''
    
    radfac = np.pi/180.
    
    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation
    
    # construct the rotation matrix
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.dot(Rx,np.dot(Ry,Rz))
    
    # structure the points for rotation

    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
    
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()
    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    tmp = np.dot(vpts.T,Rmatrix)
    PSPOut.xvel = tmp[:,0]
    PSPOut.yvel = tmp[:,1]
    PSPOut.zvel = tmp[:,2]
    #
    
    return PSPOut

