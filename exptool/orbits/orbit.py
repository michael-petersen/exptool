"""
#
# orbit.py
#
#    part of exptool: orbit input/output from simulations, pure data processing convenience.
#
#    11-12-2016 formalized class structure, provided example usage.
#    11-19-2016 documented definitions; considered for generic upgrade.
#    11-20-2016 do generic upgrade to make a single utility
#    04-07-2017 update orbit mapping to use memory mapping capability (2x speedup!). dangerous if PSP structure changes between dumps (unlikely)
#
#    TODO:
        eliminate Python2 print formatting
        make indexing work
#
#    WISHLIST:
#       orbit plotting routines
#       may want saving capability, though 1000 dumps takes 6s, so perhaps no problem
#
"""

# exptool imports
import numpy as np

from ..io import particle
from ..analysis import trapping
from ..utils import utils
from ..utils import kde_3d

# standard imports
from scipy.interpolate import UnivariateSpline
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

'''
# Quick Start Demo:

# which exp-numbered files to read in
tarr = np.arange(0,100,1,dtype='int')

# read in from files and return a dictionary
O = orbit.Orbits()
O.map_orbits('.','runtag',tarr,comp='disc',dictionary=True, norb=10,verbose=1)

'''


class Orbits(dict):

    def __init__(self):#,simulation_directory,runtag):

        # check to see if an outfile already exists?

        pass

    def map_orbits(self,simulation_directory,runtag,time_array,norb=1,comp='star',verbose=0,fileprefix='OUT',**kwargs):
        '''
        make_orbit_map

        : slice across the grain for PSPDumps to track individual particles

        Parameters:
        ----------
        simulation_directory: string, filename
            leading directory structure

        runtag: string
            name of the simulation

        time_array:  integer array
            array of integers that correspond to simulation files to be queried

        norb: integer
            number of orbits to return

        comp: string, component name
            name of simulation component to retrieve orbits from


        verbose: integer
            verbose keyword to be passed to particle

        **kwargs:
            'orblist' : integer array of orbit indices to be returned
            'dictionary' : boolean True/False to return a dictionary

        Returns:
        --------
        None

        -or-

        Orbits : OrbitDictionary-like instance
            see class OrbitDictionary below.


        '''

        infile_template = simulation_directory+'/'+fileprefix+'.'+runtag+'.'


        # check to see if an orbit list has been set
        if 'orblist' in kwargs:
            orbvals = kwargs['orblist'] # this needs to be passed as an integer array

            # right now, we need to pull the maximum orbit number in order to make sure we get all requested orbits
            # a natural improvement would be developing a memmap mask
            norb = np.max(orbvals)+1

            print('orbit.map_orbit: N_orbits accepted {} orbits'.format(len(orbvals)))
        else:
            orbvals = np.arange(0,norb,1,dtype='i')


        # check initial file for memmap construction
        #O = psp_io.Input(infile_template+'%05i' %time_array[0])
        O = particle.Input(infile_template+'{0:05d}'.format(time_array[0]))

        # make the holding arrays
        self['id'] = np.zeros( [time_array.size,orbvals.size],dtype='i8')
        self['m']  = np.zeros( [time_array.size,orbvals.size])
        self['x']  = np.zeros( [time_array.size,orbvals.size])
        self['y']  = np.zeros( [time_array.size,orbvals.size])
        self['z']  = np.zeros( [time_array.size,orbvals.size])
        self['vx'] = np.zeros( [time_array.size,orbvals.size])
        self['vy'] = np.zeros( [time_array.size,orbvals.size])
        self['vz'] = np.zeros( [time_array.size,orbvals.size])
        self['p']  = np.zeros( [time_array.size,orbvals.size])

        times = []
        bad_times = []
        prev_time = -1.

        # start the loop over files

        for indx,val in enumerate(time_array):

            if (verbose > 0) & (val < np.max(time_array)): utils.print_progress(val,np.max(time_array),'orbit.map_orbit')

            infile = infile_template+'{0:05d}'.format(val)

            O = particle.Input(infile, comp=comp) #need to give comp - not sure if we want to hard-code in that this is always for stars?

            # this is the EXP template for files with indexing.
            dtype = [('id', '<i8'), ('m', '<f4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'), ('p', '<f4')]

            # sift through times to make sure always increasing
            # this is a block for old restart simulations: it would probably be better to take an input list of all valid simulation output files
            if (indx > 0):

                # check if time is increasing
                if (O.time <= prev_time):

                    # if not, discard file
                    print('orbit.map_orbit: Bad file number {}, removing'.format(val))
                    bad_times.append(indx)


                else:
                    times.append(O.time)
                    tmp = np.memmap(infile,dtype=dtype,shape=(1,norb),offset=int(O.header[comp]['data_start']),mode='r',order='F')

                    self['id'][indx] = tmp['id'][0]
                    self['m'][indx]  = tmp['m'][0]
                    self['x'][indx]  = tmp['x'][0]
                    self['y'][indx]  = tmp['y'][0]
                    self['z'][indx]  = tmp['z'][0]
                    self['vx'][indx] = tmp['vx'][0]
                    self['vy'][indx] = tmp['vy'][0]
                    self['vz'][indx] = tmp['vz'][0]
                    self['p'][indx]  = tmp['p'][0]

            else:
                times.append(O.time)
                tmp = np.memmap(infile,dtype=dtype,shape=(1,norb),offset=int(O.header[comp]['data_start']),mode='r',order='F')

                self['id'][indx] = tmp['id'][0]
                self['m'][indx]  = tmp['m'][0]
                self['x'][indx]  = tmp['x'][0]
                self['y'][indx]  = tmp['y'][0]
                self['z'][indx]  = tmp['z'][0]
                self['vx'][indx] = tmp['vx'][0]
                self['vy'][indx] = tmp['vy'][0]
                self['vz'][indx] = tmp['vz'][0]
                self['p'][indx]  = tmp['p'][0]

            prev_time = O.time

        times = np.array(times)

        #
        # populate the time dictionary
        self['T'] = times


    def compute_quantities(self):
        '''
        add energy and angular momentum to an orbit instance

        *needs an error check for orbit instance

        '''

        v2 = self['VX']*self['VX'] + self['VY']*self['VY'] + self['VZ']*self['VZ']
        self['E'] = v2 + self['P']

        self['LZ'] = self['X']*self['VY'] - self['Y']*self['VX']


    def polar_coordinates(self):
        '''
        add polar coordinates to an orbit instance

        *needs an error check for orbit instance

        '''

        self['Rp'] = np.power(self['X']*self['X'] + self['Y']*self['Y'],0.5)
        self['Phi'] = np.arctan2(self['Y'],self['X'])

        try:
            self['Phib'] = np.arctan2(self['TY'],self['TX'])
        except:
            pass



    def resample_orbit_map(self,impr=4,sord=0,transform=False,**kwargs):
        '''
        return a resampled orbit map

        inputs
        -----------------------------




        outputs
        -----------------------------


        TODO
        -----------------------------
        what's the best way to extend this to multiple orbits?
        what about adding velocity?

        transform via
        bar=BarInstance

        '''
        old_dict = copy.copy(self)

        newT = np.linspace(np.min(old_dict['T']),np.max(old_dict['T']),len(old_dict['T'])*impr)

        # initialize a new dictionary
        self['T'] = newT
        self['M'] = old_dict['M']

        self['X']  = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['Y']  = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['Z']  = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['VX'] = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['VY'] = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['VZ'] = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')
        self['P']  = np.zeros([self['T'].size,old_dict['M'].size],dtype='f4')



        for orbit in range(0,len(old_dict['M'])):
            sX = UnivariateSpline(old_dict['T'],old_dict['X'][:,orbit],s=sord)
            sY = UnivariateSpline(old_dict['T'],old_dict['Y'][:,orbit],s=sord)
            sZ = UnivariateSpline(old_dict['T'],old_dict['Z'][:,orbit],s=sord)
            sVX = UnivariateSpline(old_dict['T'],old_dict['VX'][:,orbit],s=sord)
            sVY = UnivariateSpline(old_dict['T'],old_dict['VY'][:,orbit],s=sord)
            sVZ = UnivariateSpline(old_dict['T'],old_dict['VZ'][:,orbit],s=sord)
            sP = UnivariateSpline(old_dict['T'],old_dict['P'][:,orbit],s=sord)

            self['X'][:,orbit] = sX(newT)
            self['Y'][:,orbit] = sY(newT)
            self['Z'][:,orbit] = sZ(newT)
            self['VX'][:,orbit] = sVX(newT)
            self['VY'][:,orbit] = sVY(newT)
            self['VZ'][:,orbit] = sVZ(newT)
            self['P'][:,orbit] = sP(newT)

        if transform:
            try:
                BarInstance = kwargs['bar']


                bar_positions = pattern.find_barangle(self['T'],BarInstance)

                # make a tiled version for fast computation
                manybar = np.tile(bar_positions,(self['M'].shape[0],1)).T

                # transform positions: these are hard-wired in one direction: bad!
                self['TX'] = self['X']*np.cos(manybar) - self['Y']*np.sin(manybar)
                self['TY'] = self['X']*np.sin(manybar) + self['Y']*np.cos(manybar)

                # transform velocities
                self['VTX'] = self['VX']*np.cos(manybar) - self['VY']*np.sin(manybar)
                self['VTY'] = self['VX']*np.sin(manybar) + self['VY']*np.cos(manybar)




            except:
                print('orbit.resample_orbit: bar file reading failed. Input using bar keyword.')



def write_obj_skeleton(outfile, Orbit, lo=0, hi=10000, prefac=100.):
    """
    output in a format for Blender to read (.obj)

    """
    if hi > Orbit['T'].shape[0]:
        hi = Orbit['T'].shape[0]

    npts = hi - lo

    with open(outfile, 'w') as f:
        f.write('o ORBIT1\n')
        for indx in range(0, npts):
            f.write('v {} {} {}\n'.format(prefac*Orbit['TX'][lo+indx], prefac*Orbit['TY'][lo+indx], prefac*Orbit['Z'][lo+indx]))

        for indx in range(1, npts):
            f.write('l {} {}\n'.format(indx, indx+1))

        # this connects the last point to the first
        f.write('l {} {}\n'.format(npts, 1))



def make_orbit_density(OrbitInstance,orbit=None,window=[0,10000],replot=False,scalelength=0.01,colorplot=True,nsamp=56,transform=True,rescale=True):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    OrbitInstance
        -with transformation and polar coordinates setup. will add forcing for this at some point


    '''

    lo = window[0]
    hi = window[1]

    if (lo) > OrbitInstance['T'].size:
        print('orbit.make_orbit_density: invalid lower time boundary. resizing...')
        lo = 0

    # check size boundaries
    if (hi+1) > OrbitInstance['T'].size: hi = OrbitInstance['T'].size - 1


    if transform:
        try:
            x_coord_tmp = OrbitInstance['TX']
            y_coord_tmp = OrbitInstance['TY']

        except:
            print('orbit.make_orbit_density: transformation must be defined in orbit dictionary.')

    else:
        x_coord_tmp = OrbitInstance['X']
        y_coord_tmp = OrbitInstance['Y']

    z_coord_tmp = OrbitInstance['Z']

    # check if orbit instance is multidimensional
    try:
        orbit += 1
        orbit -= 1
        x_coord = x_coord_tmp[lo:hi,orbit]
        y_coord = y_coord_tmp[lo:hi,orbit]
        z_coord = z_coord_tmp[lo:hi,orbit]

    except:
        x_coord = x_coord_tmp[lo:hi]
        y_coord = y_coord_tmp[lo:hi]
        z_coord = z_coord_tmp[lo:hi]


    # undo scaling
    scalefac = 1./scalelength


    if not replot:
        fig = plt.figure(figsize=(6.46,4.53),dpi=100)
    else:
        plt.clf()
        fig = plt.gcf()

    #
    #want to re-scale the extent to make a more intelligent boundary
    #
    extentx_in = 1.2*np.max(abs(x_coord))
    extenty_in = 1.2*np.max(abs(y_coord))
    extentz_in = 1.2*np.max(abs(z_coord))
    #
    xbins = np.linspace(-extentx_in,extentx_in,nsamp)
    ybins = np.linspace(-extenty_in,extenty_in,nsamp)
    xx,yy = np.meshgrid( xbins,ybins)
    zbins = np.linspace(-extentz_in,extentz_in,nsamp)
    xxz,zz = np.meshgrid( xbins,zbins)



    # test the kde waters
    try:
        tt = kde_3d.fast_kde_two(x_coord,y_coord,\
                             gridsize=(nsamp,nsamp), extents=(-extentx_in,extentx_in,-extenty_in,extenty_in),\
                             nocorrelation=True, weights=None)
        tz = kde_3d.fast_kde_two(x_coord,z_coord,\
                             gridsize=(nsamp,nsamp), extents=(-extentx_in,extentx_in,-extentz_in,extentz_in),\
                             nocorrelation=True, weights=None)
    except:
        tt = tz = np.zeros([nsamp,nsamp])

    # set up the axes
    ax1 = fig.add_axes([0.15,0.55,0.25,0.35])
    ax2 = fig.add_axes([0.52,0.55,0.25,0.35])
    ax3 = fig.add_axes([0.80, 0.55, 0.03, 0.35])
    ax4 = fig.add_axes([0.15,0.15,0.25,0.35])
    ax5 = fig.add_axes([0.52,0.15,0.25,0.35])
    if colorplot: ax6 = fig.add_axes([0.80, 0.15, 0.03, 0.35])

    #


    _ = ax1.contourf(scalefac*xx,scalefac*yy,np.flipud(tt/np.sum(tt)),cmap=cm.Greys)
    _ = ax2.contourf(scalefac*xxz,scalefac*zz,np.flipud(tz/np.sum(tz)),cmap=cm.Greys)

    _ = ax2.set_ylabel('Z [R$_d$]')
    _ = ax5.set_ylabel('Z [R$_d$]')

    if transform:
        _ = ax1.set_ylabel('Y$_{\\rm bar}$ [R$_d$]')
        _ = ax4.set_ylabel('Y$_{\\rm bar}$ [R$_d$]')
        _ = ax4.set_xlabel('X$_{\\rm bar}$ [R$_d$]')
        _ = ax5.set_xlabel('X$_{\\rm bar}$ [R$_d$]')

    else:
        _ = ax1.set_ylabel('Y [R$_d$]')
        _ = ax4.set_ylabel('Y [R$_d$]')
        _ = ax4.set_xlabel('X [R$_d$]')
        _ = ax5.set_xlabel('X [R$_d$]')

    _ = ax1.set_xticklabels(())
    _ = ax2.set_xticklabels(())

    if colorplot:
        loT = OrbitInstance['T'][lo]
        hiT = OrbitInstance['T'][hi]
        dT  = (hiT-loT)/float(hi-lo)
        spacing = 5

        for indx in range(1,(hi-lo)+1,spacing):
            _ = ax4.plot(scalefac*x_coord[indx:indx+spacing+1],scalefac*y_coord[indx:indx+spacing+1],color=cm.gnuplot(indx/float(hi-lo),1.),lw=0.5)
            _ = ax5.plot(scalefac*x_coord[indx:indx+spacing+1],scalefac*z_coord[indx:indx+spacing+1],color=cm.gnuplot(indx/float(hi-lo),1.),lw=0.5)


    else:
        _ = ax4.plot(scalefac*x_coord,scalefac*y_coord,color='black',lw=0.5)
        _ = ax5.plot(scalefac*x_coord,scalefac*z_coord,color='black',lw=0.5)

    # double all window sizes?
    pfac = 1.
    pfacz = 1.

    if rescale:
        # allow for rescaling of the plots?
        #   e.g. don't use this if making a library

        if np.max([np.max(x_coord),np.max(y_coord)]) < 0.75*scalelength:
            pfac = 0.5

        if np.min([np.max(x_coord),np.max(y_coord)]) > 1.5*scalelength:
            pfac = 2.

        if np.min([np.max(x_coord),np.max(y_coord)]) > 2.5*scalelength:
            pfac = 4.

        if np.max(z_coord) > 0.7*scalelength:
            pfacz = 1.5

        if np.max(z_coord) < 0.33*scalelength:
            pfacz = 0.5



    _ = ax1.axis([-2.*pfac,2.*pfac,-2.*pfac,2.*pfac])
    _ = ax4.axis([-2.*pfac,2.*pfac,-2.*pfac,2.*pfac])

    _ = ax2.axis([-2.*pfac,2.*pfac,-0.8*pfacz,0.8*pfacz])
    _ = ax5.axis([-2.*pfac,2.*pfac,-0.8*pfacz,0.8*pfacz])

    xy_lims = [str(int(np.round(-2.*pfac,0))),str(int(np.round(-1.*pfac,0))),str(int(np.round(1.*pfac,0))),str(int(np.round(2.*pfac,0)))]
    xz_lims = [str(np.round(-0.8*pfacz,1)),str(np.round(-0.4*pfacz,1)),str(np.round(0.4*pfacz,1)),str(np.round(0.8*pfacz,1))]

    _ = ax4.set_xticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax4.set_yticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax1.set_yticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax2.set_yticklabels([xz_lims[0],'',xz_lims[1],'','0','',xz_lims[2],'',xz_lims[3]],size=12)
    _ = ax5.set_xticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax5.set_yticklabels([xz_lims[0],'',xz_lims[1],'','0','',xz_lims[2],'',xz_lims[3]],size=12)

    cmap = mpl.cm.Greys; norm = mpl.colors.Normalize(vmin=0., vmax=1.)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,norm=norm)
    _ = cb1.set_label('Relative Frequency',size=10)
    _ = cb1.set_ticks([0.,0.25,0.5,0.75,1.])

    if colorplot:
        cmap = mpl.cm.gnuplot; norm = mpl.colors.Normalize(vmin=loT, vmax=hiT)
        cb1 = mpl.colorbar.ColorbarBase(ax6, cmap=cmap,norm=norm)
        _ = cb1.set_label('System Time',size=10)
        _ = cb1.set_ticks([np.round(loT,2),np.round( 0.33*(hiT-loT) + loT,2),np.round( 0.66*(hiT-loT) + loT,2),np.round( 0.98*(hiT-loT) + loT,2)])









def make_ensemble_density(OrbitInstance,ensemble,window=[0,10000],replot=False,scalelength=0.01,nsamp=56,transform=True,rescale=True,logscale=True,ncol=18):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    OrbitInstance
        -with transformation and polar coordinates setup. will add forcing for this at some point


    '''

    lo = window[0]
    hi = window[1]

    if (lo) > OrbitInstance['T'].size:
        print('orbit.make_orbit_density: invalid lower time boundary. resizing...')
        lo = 0

    # check size boundaries
    if (hi+1) > OrbitInstance['T'].size: hi = OrbitInstance['T'].size - 1


    if transform:
        try:
            x_coord = OrbitInstance['TX'][lo:hi,:].flatten()
            y_coord = OrbitInstance['TY'][lo:hi,:].flatten()

        except:
            print('orbit.make_orbit_density: transformation must be defined in orbit dictionary.')

    else:
        x_coord = OrbitInstance['X'][lo:hi,:].flatten()
        y_coord = OrbitInstance['Y'][lo:hi,:].flatten()

    z_coord = OrbitInstance['Z'][lo:hi,:].flatten()

    # undo scaling
    scalefac = 1./scalelength


    if not replot:
        fig = plt.figure(figsize=(7.32,  2.86),dpi=100)
    else:
        plt.clf()
        fig = plt.gcf()

    #
    #want to re-scale the extent to make a more intelligent boundary
    #
    extentx_in = 1.2*np.max(abs(x_coord))
    extenty_in = 1.2*np.max(abs(y_coord))
    extentz_in = 1.2*np.max(abs(z_coord))
    #
    xbins = np.linspace(-extentx_in,extentx_in,nsamp)
    ybins = np.linspace(-extenty_in,extenty_in,nsamp)
    xx,yy = np.meshgrid( xbins,ybins)
    zbins = np.linspace(-extentz_in,extentz_in,nsamp)
    xxz,zz = np.meshgrid( xbins,zbins)



    # test the kde waters
    try:
        tt = kde_3d.fast_kde_two(x_coord,y_coord,\
                             gridsize=(nsamp,nsamp), extents=(-extentx_in,extentx_in,-extenty_in,extenty_in),\
                             nocorrelation=True, weights=None)
        tz = kde_3d.fast_kde_two(x_coord,z_coord,\
                             gridsize=(nsamp,nsamp), extents=(-extentx_in,extentx_in,-extentz_in,extentz_in),\
                             nocorrelation=True, weights=None)
    except:
        tt = tz = np.zeros([nsamp,nsamp])

    tt += 1.
    tz += 1.

    # set up the axes
    ax1 = fig.add_axes([0.15,0.23,0.25,0.65])
    ax2 = fig.add_axes([0.52,0.23,0.25,0.65])
    ax3 = fig.add_axes([0.80, 0.23, 0.03, 0.65])

    #

    if logscale:
        _ = ax1.contourf(scalefac*xx,scalefac*yy,np.log10(np.flipud(tt/np.sum(tt))),ncol,cmap=cm.Greys)
        _ = ax2.contourf(scalefac*xxz,scalefac*zz,np.log10(np.flipud(tz/np.sum(tz))),ncol,cmap=cm.Greys)

    else:
        _ = ax1.contourf(scalefac*xx,scalefac*yy,np.flipud(tt/np.sum(tt)),ncol,cmap=cm.Greys)
        _ = ax2.contourf(scalefac*xxz,scalefac*zz,np.flipud(tz/np.sum(tz)),ncol,cmap=cm.Greys)

    _ = ax2.set_ylabel('Z [R$_d$]')

    if transform:
        _ = ax1.set_ylabel('Y$_{\\rm bar}$ [R$_d$]')
        _ = ax1.set_xlabel('X$_{\\rm bar}$ [R$_d$]')
        _ = ax2.set_xlabel('X$_{\\rm bar}$ [R$_d$]')


    else:
        _ = ax1.set_ylabel('Y [R$_d$]')
        _ = ax1.set_xlabel('X [R$_d$]')
        _ = ax2.set_xlabel('X [R$_d$]')


    # double all window sizes?
    pfac = 1.
    pfacz = 1.

    if rescale:
        # allow for rescaling of the plots?
        #   e.g. don't use this if making a library

        if np.max([np.max(x_coord),np.max(y_coord)]) < 0.75*scalelength:
            pfac = 0.5

        if np.min([np.max(x_coord),np.max(y_coord)]) > 1.5*scalelength:
            pfac = 2.

        if np.min([np.max(x_coord),np.max(y_coord)]) > 2.5*scalelength:
            pfac = 4.

        if np.max(z_coord) > 0.7*scalelength:
            pfacz = 1.5

        if np.max(z_coord) < 0.33*scalelength:
            pfacz = 0.5



    _ = ax1.axis([-2.*pfac,2.*pfac,-2.*pfac,2.*pfac])

    _ = ax2.axis([-2.*pfac,2.*pfac,-0.8*pfacz,0.8*pfacz])

    xy_lims = [str(int(np.round(-2.*pfac,0))),str(int(np.round(-1.*pfac,0))),str(int(np.round(1.*pfac,0))),str(int(np.round(2.*pfac,0)))]
    xz_lims = [str(np.round(-0.8*pfacz,1)),str(np.round(-0.4*pfacz,1)),str(np.round(0.4*pfacz,1)),str(np.round(0.8*pfacz,1))]

    _ = ax1.set_xticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax1.set_yticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax2.set_xticklabels([xy_lims[0],'',xy_lims[1],'','0','',xy_lims[2],'',xy_lims[3]],size=12)
    _ = ax2.set_yticklabels([xz_lims[0],'',xz_lims[1],'','0','',xz_lims[2],'',xz_lims[3]],size=12)

    cmap = mpl.cm.Greys

    norm = mpl.colors.Normalize(vmin=0., vmax=1.)

    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,norm=norm)

    if logscale:
        _ = cb1.set_ticks([0.,0.25,0.5,0.75,1.])
        _ = cb1.set_ticklabels(['-8','-6','-4','-2','0'])
        _ = cb1.set_label('log Relative Frequency',size=10)

    else:
        _ = cb1.set_ticks([0.,0.25,0.5,0.75,1.])
        _ = cb1.set_label('Relative Frequency',size=10)
