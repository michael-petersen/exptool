
'''
integrate.py: part of exptool
         integration technique(s)

#
# take a field instance and do an orbit integration
#



TODO:


1. new integrators



'''

import sys

# can these be removed?
sys.path.append('/home/filion/martinsims/exptools/exptool/exptool/orbits')
sys.path.append('/home/filion/martinsims/exptools/exptool/exptool/potential')

# exptool imports
from ..orbits import orbit
from ..basis import potential
from ..analysis import pattern
from ..io import particle
# standard imports
import numpy as np
import time


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




def leapfrog_integrate(FieldInstance,nint,dt,initpos,initvel,\
                       rotfreq=0.,no_odd=False,\
                       halo_l=-1,halo_n=-1,disk_m=-1,disk_n=-1,verbose=0,force=False,ap_max=1000,apse=False):
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
    verbose         : (bool)   if True, report timing diagnostics
    force           : (bool)   if Tre, return force fields ('FX','FY','FZ')
    ap_max          : (int)    maximum number of apsides to integrate through
    apse            : (bool)

    outputs
    --------------
    OrbitDictionary:
          similar to orbit.py dictionaries so that the same tools can be used to plot (plus Fx, Fy, Fz)

    '''
    #
    # set Field parameters
    FieldInstance.set_field_parameters(no_odd=no_odd,halo_l=halo_l,halo_n=halo_n,disk_m=disk_m,disk_n=disk_n)

    # start the timer
    t0 = time.time()

    times = np.arange(0,nint,1)*dt

    # this needs a guard for mismatched size at some point
    barpos = 2.*np.pi*rotfreq*times

    # initialize blank arrays
    xarray = np.zeros(nint); yarray = np.zeros(nint); zarray = np.zeros(nint)

    vxarray = np.zeros(nint); vyarray = np.zeros(nint); vzarray = np.zeros(nint)

    force_xarray = np.zeros(nint); force_yarray = np.zeros(nint); force_zarray = np.zeros(nint)

    pote = np.zeros(nint)

    # initialize beginning values
    xarray[0] = initpos[0]; yarray[0] = initpos[1]; zarray[0] = initpos[2]

    vxarray[0] = initvel[0]; vyarray[0] = initvel[1]; vzarray[0] = initvel[2]

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

    # set up integration loop
    n_aps = 0
    while (n_aps < ap_max) & (step < nint):

        # advance positions
        xarray[step]   = xarray[step-1]   + (vxarray[step-1]*dt  )  + (0.5*force_xarray[step-1]  * (dt**2.))
        yarray[step]   = yarray[step-1]   + (vyarray[step-1]*dt  )  + (0.5*force_yarray[step-1]  * (dt**2.))
        zarray[step]   = zarray[step-1]   + (vzarray[step-1]*dt  )  + (0.5*force_zarray[step-1]  * (dt**2.))

        # calculate new forces
        dfx,hfx,dfy,hfy,dfz,hfz,dp,hp = FieldInstance.return_forces_cart(xarray[step],yarray[step],zarray[step],rotpos=barpos[step])
        force_xarray[step] = dfx+hfx
        force_yarray[step] = dfy+hfy
        force_zarray[step] = dfz+hfz
        pote[step]         = dp+hp

        # advance velocities
        vxarray[step]   = vxarray[step-1]   + (0.5*(force_xarray[step-1]+force_xarray[step])    *dt)
        vyarray[step]   = vyarray[step-1]   + (0.5*(force_yarray[step-1]+force_yarray[step])    *dt)
        vzarray[step]   = vzarray[step-1]   + (0.5*(force_zarray[step-1]+force_zarray[step])    *dt)

        # check for completion of aps criteria
        if apse:
            if step > 1:
                rstep0 = (xarray[step-2]*xarray[step-2] + yarray[step-2]*yarray[step-2])
                rstep1 = (xarray[step-1]*xarray[step-1] + yarray[step-1]*yarray[step-1])
                rstep2 = (xarray[step]*xarray[step] + yarray[step]*yarray[step])

                if (rstep1 > rstep0) & (rstep1 > rstep2):
                    n_aps += 1

        step += 1

    if verbose:
        print('{0:4.3f} seconds to integrate.'.format(time.time()-t0))

    # put into dictionary form
    OrbitDictionary = orbit.Orbits()
    OrbitDictionary['T'] = times[0:step]
    OrbitDictionary['X'] = xarray[0:step]
    OrbitDictionary['Y'] = yarray[0:step]
    OrbitDictionary['Z'] = zarray[0:step]
    OrbitDictionary['VX'] = vxarray[0:step]
    OrbitDictionary['VY'] = vyarray[0:step]
    OrbitDictionary['VZ'] = vzarray[0:step]
    OrbitDictionary['P']  = pote[0:step]
    barpos = barpos[0:step]


    if force:
        OrbitDictionary['FX'] = force_xarray[0:step]
        OrbitDictionary['FY'] = force_yarray[0:step]
        OrbitDictionary['FZ'] = force_zarray[0:step]


    # do transformations, adaptively finding which way bar is rotating
    if np.min(barpos) < 0.:

        OrbitDictionary['TX'],OrbitDictionary['TY'] = transform(OrbitDictionary['X'],OrbitDictionary['Y'],barpos)
        OrbitDictionary['VTX'],OrbitDictionary['VTY'] = transform(OrbitDictionary['VX'],OrbitDictionary['VY'],barpos)

    else:

        OrbitDictionary['TX'],OrbitDictionary['TY'] = clock_transform(OrbitDictionary['X'],OrbitDictionary['Y'],barpos)
        OrbitDictionary['VTX'],OrbitDictionary['VTY'] = clock_transform(OrbitDictionary['VX'],OrbitDictionary['VY'],barpos)

    return OrbitDictionary





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



def compute_timestep(FieldInstance,start_pos,start_vel,dyn_res=100.,verbose=False):
    '''
    compute_timestep:
       calculate a best timestep for an orbit using EXP criteria
       (see multistep.cc)

        dtd = eps* rscale/v_i    -- char. drift time scale
        dtv = eps* min(v_i/a_i)  -- char. force time scale
        dta = eps* phi/(v * a)   -- char. work time scale
        dtA = eps* sqrt(phi/a^2) -- char. "escape" time scale


    inputs
    --------------
    FieldInstance  :
    start_pos      :
    start_vel      :
    dyn_res        :  minimum number of substeps to resolve (default: 100., likely overkill)
    #
    returns
    --------------
    dt
    '''
    #
    eps = 1.e-10
    #
    vtot = np.sum(np.array(start_vel)**2.)**0.5  + eps
    rtot = np.sum(np.array(start_pos)**2.)**0.5
    #
    #
    dfx,hfx,dfy,hfy,dfz,hfz,dp,hp = FieldInstance.return_forces_cart(start_pos[0],start_pos[1],start_pos[2])
    #
    dtr = np.sum(np.array(start_pos)*np.array([(dfx+hfx),(dfy+hfy),(dfz*hfz)]))
    #
    atot = ((dfx+hfx)**2. + (dfy+hfy)**2. + (dfz*hfz)**2.0)**0.5 + eps
    ptot = np.abs(dp+hp)
    #
    T = {}
    #
    T['dtd'] = 1.0/np.sqrt(vtot+eps);
    T['dtv'] = np.sqrt(vtot/(atot+eps));
    T['dta'] = ptot/(np.abs(dtr)+eps);
    T['dtA'] = np.sqrt(ptot/(atot*atot+eps));
    #
    # now, grab the smallest and figure out how many sub-timesteps are needed.
    #
    time_criteria = [T[x] for x in T.keys()]
    if verbose:
        print('The chosen time criteria is {0:s}, dt={1:6.5f}'.format(T.keys()[np.argmin(time_criteria)],np.min(time_criteria)/dyn_res))
    #
    #
    #
    #
    dt = np.min(time_criteria)/dyn_res
    #
    return dt




#
# ###################################################################################
#
#   multistepping block


import itertools
from multiprocessing import Pool, freeze_support
import multiprocessing




def do_integrate_multi(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,\
                   verbose=0,nprocs=-1,threedee=False, zs=None, vzs=None, launch='x'):
    '''
    do_integrate_multi:
       multiprocessor integration of orbits

    '''

    if nprocs <= 0:
        nprocs = multiprocessing.cpu_count()

    subrads = redistribute_arrays(rads,nprocs)

    if (verbose > 0):
        print('Beginning integration, using {} processors.'.format(nprocs))

    t1 = time.time()
    freeze_support()
    #
    
    if threedee == True:
        if (zs is None) or (vzs is None):
            print('ERROR: 3D orbit specified, but no z or vz values passed!')
        else:
            out_arrays = multi_compute_integration_3D(subrads,nprocs,vels,F,\
                   nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,\
                    zs, vzs,
                   verbose=verbose, launch=launch)
            orbit_array = re_form_orbit_arrays_3D(out_arrays)
    else:
        out_arrays = multi_compute_integration(subrads,nprocs,vels,F,\
                   nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,\
                   verbose=verbose, launch=launch)
        orbit_array = re_form_orbit_arrays(out_arrays)
    #
    # out_arrays have [rads.size,vels.size,4,nint]
    #
    print('Total integration calculation took {0:3.2f} seconds, or {1:3.2f} seconds per orbit.'.format(time.time()-t1,(time.time()-t1)/float(rads.size*vels.size)))
    #
    
    #
    #print_orbit_array(outfile,orbit_array)
    #
    return orbit_array




def multi_compute_integration(subrads,nprocs,vels,F,\
                   nint,dt,rotfreq,no_odd,halo_l,disk_m,\
                   dyn_res,ap_max,\
                   launch, verbose=5):
    #
    pool = Pool(nprocs)
    #
    a_args = [subrads[i] for i in range(0,nprocs)]
    second_arg = vels
    third_arg = F
    fourth_arg = nint
    fifth_arg = dt
    sixth_arg = rotfreq
    seventh_arg = no_odd
    eighth_arg = halo_l
    ninth_arg = disk_m
    tenth_arg = dyn_res
    eleventh_arg = ap_max
    #
    #integrate_grid(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m)
    #
    twelvth_arg = [0 for i in range(0,nprocs)]
    twelvth_arg[0] = verbose
    #
    if launch == 'x':
        out_vals = pool.map(integrate_grid_star, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),twelvth_arg))
    elif launch=='y':
        out_vals = pool.map(integrate_grid_star_launchy, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),twelvth_arg))

    #pool.imap(integrate_grid_star, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),twelvth_arg),5)
    #
    # clean up to exit
    pool.close()
    pool.join()
    #
    #print(list(out_vals))
    return out_vals




def integrate_grid_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return integrate_grid(*a_b)

def integrate_grid_star_launchy(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return integrate_grid_launchy(*a_b)

def integrate_grid_star_3D(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return integrate_grid_3D(*a_b)

def integrate_grid_star_3D_launchy(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return integrate_grid_3D_launchy(*a_b)   



'''
def integrate_grid(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,verbose):
    


    #treat dt as the maximum value as a guard

    #0: TX
    #1: TY
    #2: VX
    #3: VY
    #4: T


    #
    Oarray = np.zeros([rads.size,vels.size,5,nint])
    #
    for irad,rad in enumerate(rads):
        #print 'Radius is ',rad
        #
        for ivel,vel in enumerate(vels):
            #start_pos,start_vel = integrate.gen_init_step(rad,vel,z0=0.000)#,zvel0=0.2)
            start_pos = [rad,0.,0.]
            start_vel = [0.,vel,0.]
            #
            dtime = np.max([compute_timestep(F,start_pos,start_vel,dyn_res=dyn_res,verbose=False),dt])

            Orbit = leapfrog_integrate(F,nint,dtime,start_pos,start_vel,rotfreq=rotfreq,no_odd=no_odd,halo_l=halo_l,disk_m=disk_m,verbose=verbose,ap_max=ap_max)
            #print_orbit(f,O)
            #
            # bring orbits up to uniform length
            Oarray[irad,ivel,0] = np.concatenate((Orbit['TX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,1] = np.concatenate((Orbit['TY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,2] = np.concatenate((Orbit['VX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,3] = np.concatenate((Orbit['VY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,4] = np.concatenate((Orbit['T'],np.zeros(nint-Orbit['T'].size)))
    #
    #
    return Oarray
'''


#
def redistribute_arrays(rads,divisions):
    '''
    given a grid of initial conditions (1d array rads) and a number of divisions (int divisions)


    '''
    #
    subrads = [ [] for i in range(0,divisions) ]
    #
    rads_per_proc = int(np.floor(rads.size/divisions))
    first_partition = int(rads.size - rads_per_proc*(divisions-1))
    #
    print('Each processor has {} radii.'.format((rads_per_proc)))#, first_partition
    #
    for i in range(0,divisions):
        #
        if (i>0):
            #
            subrads[i] = rads[((i-1)*rads_per_proc)+first_partition : ((i)*rads_per_proc)+first_partition]
        #
        else:
            subrads[i] = rads[0 : first_partition]
    #
    return subrads





def re_form_orbit_arrays(array):
    #
    print(array)
    norb_master = 0
    for processor in range(0,len(array)): norb_master += array[processor].shape[0]
    #
    #
    # now initialize new blank array
    net_array = np.zeros([norb_master,array[0].shape[1],array[0].shape[2],array[0].shape[3]],dtype='f4')
    #
    start_index = 0
    for processor in range(0,len(array)): 
        #
        end_index = start_index + array[processor].shape[0]
        #
        net_array[start_index:end_index] = array[processor]
        start_index = end_index
    #

    return net_array

def re_form_orbit_arrays_3D(array):
    #
    norb_master = 0
    for processor in range(0,len(array)): norb_master += array[processor].shape[0]
    #
    #
    # now initialize new blank array
    net_array = np.zeros([norb_master,array[0].shape[1],array[0].shape[2],array[0].shape[3],\
        array[0].shape[4],array[0].shape[5]],dtype='f4')
    #
    start_index = 0
    for processor in range(0,len(array)): 
        #
        end_index = start_index + array[processor].shape[0]
        #
        net_array[start_index:end_index] = array[processor]
        start_index = end_index
    #
    return net_array




def print_orbit_array(f,OrbitArray):
    for rad in range(0,OrbitArray.shape[0]):
        for vel in range(0,OrbitArray.shape[1]):
            # find non-zero values
            nsteps = OrbitArray.shape[-1]
            #
            print(nsteps,OrbitArray[rad,vel,0,0], #R_0
                        OrbitArray[rad,vel,5,0], #V_0
                        OrbitArray[rad,vel,-1,1], #dT
                        end=' ',file=f)
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,0,x],end=' ',file=f) #x
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,1,x],end=' ',file=f) #y
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,2,x],end=' ',file=f) #tx
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,3,x],end=' ',file=f) #ty
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,4,x],end=' ',file=f) #vx
            for x in range(0,nsteps):
                print(OrbitArray[rad,vel,5,x],end=' ',file=f) #vy
            print('',file=f) # end the line



def print_orbit_array_3D(f,OrbitArray):
    for rad in range(0,OrbitArray.shape[0]):
        for vel in range(0,OrbitArray.shape[1]):
            for z in range(0,OrbitArray.shape[2]):
                for vz in range(0,OrbitArray.shape[3]):
                    # find non-zero values
                    nsteps = OrbitArray.shape[-1]
                    #
                    print(nsteps,OrbitArray[rad,vel,z, vz, 0,0], #R_0
                                OrbitArray[rad,vel,z, vz,6,0], #V_0
                                OrbitArray[rad,vel,z, vz,2,0],#Z_0
                                OrbitArray[rad,vel,z, vz,7,0],#VZ_0
                                OrbitArray[rad,vel,z, vz,-1,1],#dT
                                end=' ',file=f)
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,0,x],end=' ',file=f) #x
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,1,x],end=' ',file=f) #y
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,2,x],end=' ',file=f) #z
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,3,x],end=' ',file=f) #tx
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,4,x],end=' ',file=f) #ty
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,5,x],end=' ',file=f) #vx
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,6,x],end=' ',file=f) #vy
                    for x in range(0,nsteps):
                        print(OrbitArray[rad,vel,z, vz,7,x],end=' ',file=f) #vz
                    print('',file=f) # end the line

def run_time(simulation_directory,simulation_name,\
                 eof_file,sph_file,model_file,\
                 intime,\
                 rads,vels,\
                 nint,dt,no_odd,halo_l,max_m,dyn_res,ap_max,\
                 verbose,nprocs=-1,omegap=-1.,orbitfile='',transform=True,fileprefix='OUT', launch='x'):
    '''
    run_time
          execute all necessary steps to run an integration grid


    inputs
    ------------------
    simulation_directory
    simulation_name
    eof_file
    sph_file
    model_file
    intime
    rads
    vels
    nint
    dt
    no_odd
    halo_l
    max_m
    dyn_res
    ap_max
    verbose
    nprocs=-1
    omegap=-1.
    orbitfile=''
    transform=True


    returns
    ------------------

    '''

    if verbose:
        print('exptool.integrate.run_time: in directory {}, run {} at output {}, with transform={}'.format(simulation_directory,simulation_name,intime,transform))

    F,patt,rotfreq = potential.get_fields(simulation_directory,simulation_name,intime,eof_file,sph_file,model_file,transform=transform,fileprefix=fileprefix)



    # use a supplied pattern speed if given
    if omegap >= 0.:
        patt = omegap

    rotfreq = -1.*abs(patt/(2.*np.pi))

    OrbitArray = do_integrate_multi(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,max_m,dyn_res,ap_max,verbose=verbose,nprocs=nprocs, launch=launch)


    if orbitfile != '':
        f = open(orbitfile,'w')

    else:
        f = open(simulation_directory+'omap_'+str(intime)+'.txt','w')

    print_orbit_array(f,OrbitArray)

    f.close()


def run_time_mod(simulation_directory,simulation_name,\
                 eof_file,sph_file,model_file,\
                 intime,\
                 rads,vels,\
                 nint,dt,no_odd,halo_l,max_m,dyn_res,ap_max,\
                 verbose,nprocs=-1,omegap=-1.,orbitfile='',transform=False,
                 save_field=True, field_file_name='',field_file=None, bar_file='',fileprefix='OUT',
                 threedee=False, zs = None, vzs = None, launch='x'):
    '''
    run_time
          execute all necessary steps to run an integration grid


    inputs
    ------------------
    simulation_directory
    simulation_name
    eof_file
    sph_file
    model_file
    intime
    rads
    vels
    nint
    dt
    no_odd
    halo_l
    max_m
    dyn_res
    ap_max
    verbose
    nprocs=-1
    omegap=-1.
    orbitfile=''
    transform=True


    returns
    ------------------

    '''

    if verbose:
        print('exptool.integrate.run_time: in directory {}, run {} at output {}, with transform={}'.format(simulation_directory,simulation_name,intime,transform))

    if transform == True:
        if bar_file == '':
            print('error - no bar file supplied for transform')
            return

    if field_file == None:
        F,patt,rotfreq = potential.get_fields(simulation_directory,simulation_name,intime,eof_file,sph_file,model_file,transform=transform, bar_file=bar_file,fileprefix=fileprefix)
        if save_field == True:
            if str(field_file_name) != '':
                F.save_field(str(field_file_name))
            else:
                print('saving field file with default name (field_file) to local directory')
                F.save_field('field_file')

    if field_file != None:
        print('field file supplied! File name:')
        print(field_file_name)
        F = potential.restore_field(str(field_file_name))


        if transform:
            BarInstance = pattern.BarDetermine()
            BarInstance.read_bar(bar_file)

            # reset the derivative
            BarInstance.frequency_and_derivative(spline_derivative=2)

            # put in modern psp reader format

            infile = simulation_directory+fileprefix+'.'+simulation_name+'.{0:05d}'.format(intime)
            PSPDump = particle.Input(infile, comp='star')

            patt = pattern.find_barpattern(PSPDump.time,BarInstance,smth_order=None)

            rotfreq = patt/(2.*np.pi)

        else:
            patt = 0.
            rotfreq = 0.
    # use a supplied pattern speed if given
    if omegap >= 0.:
        patt = omegap

    rotfreq = -1.*abs(patt/(2.*np.pi))

    if threedee == False:
        print('launching from ', launch, ' axis')
        OrbitArray = do_integrate_multi(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,max_m,dyn_res,ap_max,verbose=verbose,nprocs=nprocs, launch=launch)
        if orbitfile != '':
            f = open(orbitfile,'w')

        else:
            print('Saving 2D orbit')
            f = open(simulation_directory+'omap_'+str(intime)+'.txt','w')

        print_orbit_array(f,OrbitArray)

        f.close()
    else:
        if (zs is None) or (vzs is None):
            print('ERROR: 3D orbit specified, but no z or vz values passed!')
        else:
            OrbitArray = do_integrate_multi(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,max_m,dyn_res,ap_max,\
                    verbose=verbose,nprocs=nprocs, threedee=threedee, zs=zs, vzs=vzs, launch=launch)


        if orbitfile != '':
            f = open(orbitfile,'w')

        else:
            print('Saving 3D orbit')
            f = open(simulation_directory+'omap_3D_'+str(intime)+'.txt','w')

        print_orbit_array_3D(f,OrbitArray)

        f.close()

def integrate_grid_3D(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max, zs, vzs, verbose):
    '''    
    if threedee:
        0: X
        1: Y
        2: Z
        3: TX
        4: TY
        5: VX
        6: VY
        7: VZ
        8: T
        '''
    if (zs is None) or (vzs is None):
        print('ERROR: 3D orbit specified, but no z or vz values passed!')
    else:
        Oarray = np.zeros([rads.size, vels.size, zs.size, vzs.size, 9, nint])
        #
        for irad,rad in enumerate(rads):
            #print 'Radius is ',rad
            #
            for ivel,vel in enumerate(vels):
                for iz, z in enumerate(zs):
                    for ivz, vz in enumerate(vzs):
                        start_pos = [rad,0.,z]
                        start_vel = [0.,vel,vz]
                        #
                        dtime = np.max([compute_timestep(F,start_pos,start_vel,dyn_res=dyn_res,verbose=False),dt])

                        Orbit = leapfrog_integrate(F,nint,dtime,start_pos,start_vel,rotfreq=rotfreq,no_odd=no_odd,halo_l=halo_l,disk_m=disk_m,verbose=verbose,ap_max=ap_max)
                        #print_orbit(f,O)
                        #
                        # bring orbits up to uniform length
                        Oarray[irad,ivel, iz, ivz, 0] = np.concatenate((Orbit['X'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,1] = np.concatenate((Orbit['Y'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,2] = np.concatenate((Orbit['Z'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel, iz, ivz, 3] = np.concatenate((Orbit['TX'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,4] = np.concatenate((Orbit['TY'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,5] = np.concatenate((Orbit['VX'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,6] = np.concatenate((Orbit['VY'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,7] = np.concatenate((Orbit['VZ'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,8] = np.concatenate((Orbit['T'],np.zeros(nint-Orbit['T'].size)))
    return Oarray

def integrate_grid_3D_launchy(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max, zs, vzs, verbose):
    '''    
    if threedee:
        0: X
        1: Y
        2: Z
        3: TX
        4: TY
        5: VX
        6: VY
        7: VZ
        8: T
        '''
    if (zs is None) or (vzs is None):
        print('ERROR: 3D orbit specified, but no z or vz values passed!')
    else:
        Oarray = np.zeros([rads.size, vels.size, zs.size, vzs.size, 9, nint])
        #
        for irad,rad in enumerate(rads):
            #print 'Radius is ',rad
            #
            for ivel,vel in enumerate(vels):
                for iz, z in enumerate(zs):
                    for ivz, vz in enumerate(vzs):
                        start_pos = [0.,rad,z] #launching from y-axis
                        start_vel = [0.,vel,vz]
                        #
                        dtime = np.max([compute_timestep(F,start_pos,start_vel,dyn_res=dyn_res,verbose=False),dt])

                        Orbit = leapfrog_integrate(F,nint,dtime,start_pos,start_vel,rotfreq=rotfreq,no_odd=no_odd,halo_l=halo_l,disk_m=disk_m,verbose=verbose,ap_max=ap_max)
                        #print_orbit(f,O)
                        #
                        # bring orbits up to uniform length
                        Oarray[irad,ivel, iz, ivz, 0] = np.concatenate((Orbit['X'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,1] = np.concatenate((Orbit['Y'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,2] = np.concatenate((Orbit['Z'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel, iz, ivz, 3] = np.concatenate((Orbit['TX'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,4] = np.concatenate((Orbit['TY'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,5] = np.concatenate((Orbit['VX'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,6] = np.concatenate((Orbit['VY'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,7] = np.concatenate((Orbit['VZ'],np.zeros(nint-Orbit['T'].size)))
                        Oarray[irad,ivel,iz, ivz,8] = np.concatenate((Orbit['T'],np.zeros(nint-Orbit['T'].size)))
    return Oarray

def integrate_grid(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,verbose):
    '''
    treat dt as the maximum value as a guard
        0: X
        1: Y
        2: TX
        3: TY
        4: VX
        5: VY
        6: T

    '''


    Oarray = np.zeros([rads.size,vels.size,7,nint])
    #
    for irad,rad in enumerate(rads):
        #print 'Radius is ',rad
        #
        for ivel,vel in enumerate(vels):
            start_pos = [rad,0.,0.]
            start_vel = [0.,vel,0.]
            #
            dtime = np.max([compute_timestep(F,start_pos,start_vel,dyn_res=dyn_res,verbose=False),dt])

            Orbit = leapfrog_integrate(F,nint,dtime,start_pos,start_vel,rotfreq=rotfreq,no_odd=no_odd,halo_l=halo_l,disk_m=disk_m,verbose=verbose,ap_max=ap_max)
            #print_orbit(f,O)
            #
            # bring orbits up to uniform length
            Oarray[irad,ivel,0] = np.concatenate((Orbit['X'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,1] = np.concatenate((Orbit['Y'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,2] = np.concatenate((Orbit['TX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,3] = np.concatenate((Orbit['TY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,4] = np.concatenate((Orbit['VX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,5] = np.concatenate((Orbit['VY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,6] = np.concatenate((Orbit['T'],np.zeros(nint-Orbit['T'].size)))        
    return Oarray

def integrate_grid_launchy(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m,dyn_res,ap_max,verbose):
    '''
    treat dt as the maximum value as a guard
        0: X
        1: Y
        2: TX
        3: TY
        4: VX
        5: VY
        6: T

    '''


    Oarray = np.zeros([rads.size,vels.size,7,nint])
    #
    for irad,rad in enumerate(rads):
        #print 'Radius is ',rad
        #
        for ivel,vel in enumerate(vels):
            start_pos = [0.,rad,0.] #launching from y-axis
            start_vel = [0.,vel,0.]
            #
            dtime = np.max([compute_timestep(F,start_pos,start_vel,dyn_res=dyn_res,verbose=False),dt])

            Orbit = leapfrog_integrate(F,nint,dtime,start_pos,start_vel,rotfreq=rotfreq,no_odd=no_odd,halo_l=halo_l,disk_m=disk_m,verbose=verbose,ap_max=ap_max)
            #print_orbit(f,O)
            #
            # bring orbits up to uniform length
            Oarray[irad,ivel,0] = np.concatenate((Orbit['X'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,1] = np.concatenate((Orbit['Y'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,2] = np.concatenate((Orbit['TX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,3] = np.concatenate((Orbit['TY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,4] = np.concatenate((Orbit['VX'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,5] = np.concatenate((Orbit['VY'],np.zeros(nint-Orbit['T'].size)))
            Oarray[irad,ivel,6] = np.concatenate((Orbit['T'],np.zeros(nint-Orbit['T'].size)))        
    return Oarray

def multi_compute_integration_3D(subrads,nprocs,vels,F,\
                   nint,dt,rotfreq,no_odd,halo_l,disk_m,\
                   dyn_res,ap_max,\
                   zs, vzs,launch,
                   verbose=5):
    #
    pool = Pool(nprocs)
    #
    a_args = [subrads[i] for i in range(0,nprocs)]
    second_arg = vels
    third_arg = F
    fourth_arg = nint
    fifth_arg = dt
    sixth_arg = rotfreq
    seventh_arg = no_odd
    eighth_arg = halo_l
    ninth_arg = disk_m
    tenth_arg = dyn_res
    eleventh_arg = ap_max
    twelvth_arg = zs
    thirteenth_arg = vzs
    #
    #integrate_grid(rads,vels,F,nint,dt,rotfreq,no_odd,halo_l,disk_m)
    #
    fourteenth_arg = [0 for i in range(0,nprocs)]
    fourteenth_arg[0] = verbose
    #
    if launch=='x':
        out_vals = pool.map(integrate_grid_star_3D, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                        itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),\
                        itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),\
                        itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),itertools.repeat(twelvth_arg),\
                            itertools.repeat(thirteenth_arg), fourteenth_arg))
    elif launch=='y':
        out_vals = pool.map(integrate_grid_star_3D_launchy, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                        itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),\
                        itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),\
                        itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),itertools.repeat(twelvth_arg),\
                            itertools.repeat(thirteenth_arg), fourteenth_arg))
    #
    # clean up to exit
    pool.close()
    pool.join()
    return out_vals

# follows file format of integrator output
def read_integrations(infile):
    f = open(infile,'r')
    D = {}
    D['R_0'] = {}
    D['V_0'] = {}
    D['dT'] = {}
    D['X'] = {}
    D['Y'] = {}
    D['TX'] = {}
    D['TY'] = {}
    D['VX'] = {}
    D['VY'] = {}
    #
    linenum = 0
    for line in f:
        d = [float(q) for q in line.split()]
        npoints = int(d[0])
        D['R_0'][linenum] = d[1]
        D['V_0'][linenum] = d[2]
        D['dT'][linenum] = d[3]
        D['X'][linenum] = d[4:npoints+4]
        D['Y'][linenum] = d[npoints+4:(2*npoints)+4]
        D['TX'][linenum] = d[(2*npoints)+4:(3*npoints)+4]
        D['TY'][linenum] = d[(3*npoints)+4:(4*npoints)+4]
        D['VX'][linenum] = d[(4*npoints)+4:(5*npoints)+4]
        D['VY'][linenum] = d[(5*npoints)+4:(6*npoints)+4]
        linenum += 1
    #
    f.close()
    return D

def read_integrations_3D(infile):
    f = open(infile,'r')
    D = {}
    D['R_0'] = {}
    D['V_0'] = {}
    D['Z_0'] = {}
    D['VZ_0'] = {}
    D['dT'] = {}
    D['X'] = {}
    D['Y'] = {}
    D['Z'] = {}
    D['TX'] = {}
    D['TY'] = {}
    D['VX'] = {}
    D['VY'] = {}
    D['VZ'] = {}
    #
    linenum = 0
    for line in f:
        d = [float(q) for q in line.split()]
        npoints = int(d[0])
        D['R_0'][linenum] = d[1]
        D['V_0'][linenum] = d[2]
        D['Z_0'][linenum] = d[3]
        D['VZ_0'][linenum] = d[4]
        D['dT'][linenum] = d[5]
        D['X'][linenum] = d[6:npoints+6]
        D['Y'][linenum] = d[npoints+6:(2*npoints)+6]
        D['Z'][linenum] = d[(2*npoints)+6:(3*npoints)+6]
        D['TX'][linenum] = d[(3*npoints)+6:(4*npoints)+6]
        D['TY'][linenum] = d[(4*npoints)+6:(5*npoints)+6]
        D['VX'][linenum] = d[(5*npoints)+6:(6*npoints)+6]
        D['VY'][linenum] = d[(6*npoints)+6:(7*npoints)+6]
        D['VZ'][linenum] = d[(7*npoints)+6:(8*npoints)+6]
        linenum += 1
    #
    f.close()
    return D

