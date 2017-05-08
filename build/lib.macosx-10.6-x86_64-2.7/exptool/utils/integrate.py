

#
# take a field instance and do an integration
#

import exptool.orbits.orbit



import numpy as np
import time

verbose = True

'''
WISHLIST

1. new integrators
2. new timestep criteria


'''


def transform(xarray,yarray,thetas):
    ''' counterclockwise planar transformation'''
    new_xpos = np.cos(thetas)*xarray - np.sin(thetas)*yarray
    new_ypos = np.sin(thetas)*xarray + np.cos(thetas)*yarray
    return new_xpos,new_ypos


def clock_transform(xarray,yarray,thetas):
    ''' clockwise planar transformation '''
    new_xpos = np.cos(thetas)*xarray + np.sin(thetas)*yarray
    new_ypos = -1.*np.sin(thetas)*xarray + np.cos(thetas)*yarray
    return new_xpos,new_ypos





def leapfrog_integrate(FieldInstance,nint,dt,initpos,initvel,rotfreq=0.,no_odd=False,halo_l=-1,halo_n=-1,disk_m=-1,disk_n=-1):
    '''


    inputs
    --------------
    FieldInstance:
         must have
                .set_field_parameters
                .return_forces_cart

    nint            : (int)    number of integration steps
    dt              : (float)  step size
    initpos         : (vector) [ x0, y0, z0]
    initvel         : (vector) [vx0,vy0,vz0]
    rotfreq         : (float)  rotation frequency of rotating, in radians/time
    no_odd          : (bool)   if True, restriction to m=0,2,4,6,...
    halo_l          : (int)    if >=0, limit number of azimuthal terms in halo
    halo_n          : (int)    if >=0, limit number of radial terms in halo
    disk_m          : (int)    if >=0, limit number of azimuthal terms in disk
    disk_n          : (int)    if >=0, limit number of radial terms in disk

    outputs
    --------------
    OrbitDictionary:
          similar to orbit.py dictionaries so that the same tools can be used to plot (plus Fx, Fy, Fz)

    '''
    #
    # set Field parameters
    FieldInstance.set_field_parameters(no_odd=no_odd,halo_l=halo_l,halo_n=halo_n,disk_m=disk_m,disk_n=disk_n)
    t0 = time.time()
    times = np.arange(0,nint,1)*dt
    barpos = 2.*np.pi*rotfreq*times
    xarray = np.zeros(nint)
    yarray = np.zeros(nint)
    zarray = np.zeros(nint)
    vxarray = np.zeros(nint)
    vyarray = np.zeros(nint)
    vzarray = np.zeros(nint)
    force_xarray = np.zeros(nint)
    force_yarray = np.zeros(nint)
    force_zarray = np.zeros(nint)
    potential = np.zeros(nint)
    #
    xarray[0] = initpos[0]
    yarray[0] = initpos[1]
    zarray[0] = initpos[2]
    #
    vxarray[0] = initvel[0]
    vyarray[0] = initvel[1]
    vzarray[0] = initvel[2]
    #
    pote = np.zeros(nint)
    #
    #
    # now step forward one, using leapfrog (drift-kick-drift) integrator?
    #    https://en.wikipedia.org/wiki/Leapfrog_integration
    #
    step = 1
    dfx,hfx,dfy,hfy,dfz,hfz,dp,hp = FieldInstance.return_forces_cart(xarray[step-1],yarray[step-1],zarray[step-1],rotpos=barpos[step-1])
    force_xarray[step-1] = dfx+hfx
    force_yarray[step-1] = dfy+hfy
    force_zarray[step-1] = dfz+hfz
    pote[step-1]         = dp+hp
    #
    #
    for step in range(1,nint):
        #print step 
        xarray[step]   = xarray[step-1]   + (vxarray[step-1]*dt  )  + (0.5*force_xarray[step-1]  * (dt**2.))
        yarray[step]   = yarray[step-1]   + (vyarray[step-1]*dt  )  + (0.5*force_yarray[step-1]  * (dt**2.))
        zarray[step]   = zarray[step-1]   + (vzarray[step-1]*dt  )  + (0.5*force_zarray[step-1]  * (dt**2.))
        #print rarray[step]*np.cos(phiarray[step]),rarray[step]*np.sin(phiarray[step]),zarray[step]
        #
        dfx,hfx,dfy,hfy,dfz,hfz,dp,hp = FieldInstance.return_forces_cart(xarray[step],yarray[step],zarray[step],rotpos=barpos[step])
        force_xarray[step] = dfx+hfx
        force_yarray[step] = dfy+hfy
        force_zarray[step] = dfz+hfz
        pote[step]         = dp+hp
        #
        vxarray[step]   = vxarray[step-1]   + (0.5*(force_xarray[step-1]+force_xarray[step])    *dt)
        vyarray[step]   = vyarray[step-1]   + (0.5*(force_yarray[step-1]+force_yarray[step])    *dt)
        vzarray[step]   = vzarray[step-1]   + (0.5*(force_zarray[step-1]+force_zarray[step])    *dt)
    if verbose:
        print '%4.3f seconds to integrate.' %(time.time()-t0)
    # put into dictionary form
    OrbitDictionary = orbit.Orbits()
    OrbitDictionary['T'] = times
    OrbitDictionary['X'] = xarray
    OrbitDictionary['Y'] = yarray
    OrbitDictionary['Z'] = zarray
    OrbitDictionary['VX'] = vxarray
    OrbitDictionary['VY'] = vyarray
    OrbitDictionary['VZ'] = vzarray
    OrbitDictionary['FX'] = force_xarray
    OrbitDictionary['FY'] = force_yarray
    OrbitDictionary['FZ'] = force_zarray
    OrbitDictionary['P']  = pote
    
    
    if np.min(barpos) < 0.:
        OrbitDictionary['TX'],OrbitDictionary['TY'] = transform(OrbitDictionary['X'],OrbitDictionary['Y'],barpos)
        OrbitDictionary['VTX'],OrbitDictionary['VTY'] = transform(OrbitDictionary['VX'],OrbitDictionary['VY'],barpos)
    else:
        OrbitDictionary['TX'],OrbitDictionary['TY'] = clock_transform(OrbitDictionary['X'],OrbitDictionary['Y'],barpos)
        OrbitDictionary['VTX'],OrbitDictionary['VTY'] = clock_transform(OrbitDictionary['VX'],OrbitDictionary['VY'],barpos)
        
    return OrbitDictionary



#
# helper definitions to initialize orbits
# 


def gen_init_step(xpos,vtan,z0=0.0,zvel0=0.):
    '''
    initialize an orbit at (planar) apocenter

    inputs
    -----------
    xpos   : (float) radius (x) position
    vtan   : (float) tangential (y) velocity
    z0     : (float) initial z position (default=0.)
    zvel   : (float) initial z velocity (default=0.)

    outputs
    -----------
    two 3d arrays for input in integrator above

    '''

    return [xpos,0.0,z0],[0.0,vtan,zvel0]



