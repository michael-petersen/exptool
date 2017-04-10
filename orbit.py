#
# orbit.py
#
#    part of exptool: orbit input/output from simulations, pure data processing convenience.
#
#    11.12.16 formalized class structure, provided example usage.
#    11.19.16 documented definitions; considered for generic upgrade.
#    11.20.16 do generic upgrade to make a single utility
#    04.07.17 update orbit mapping to use memory mapping capability (2x speedup!). dangerous if PSP structure changes between dumps (unlikely)
#
#    WISHLIST:
#       orbit plotting routines
#       may want saving capability, though 1000 dumps takes 6s, so perhaps no problem
#

# future compatibility: let's make this Python 3 setup
from __future__ import print_function

# exptool imports
import numpy as np
import psp_io
import trapping
import utils
import kde_3d

# standard imports
from scipy.interpolate import UnivariateSpline
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

'''
# Quick Start Demo:

# which exp-numbered files to read in
tarr = np.arange(0,12,1,dtype='int')

# read in from files and return a dictionary
Orbits = orbit.map_orbits('/path/to/outfile.dat','/simulation/directory','runtag',tarr,comp='dark',dictionary=True, norb=10)


# Orbits is a dictionary with several quantities (see initialize_orbit_dictionary below)

'''

def initialize_orbit_dictionary():
    '''
    make the dictionary that handles orbits
    '''

    OrbitDictionary = {}
    OrbitDictionary['T'] = None
    OrbitDictionary['X'] = None
    OrbitDictionary['Y'] = None
    OrbitDictionary['Z'] = None
    OrbitDictionary['VX'] = None
    OrbitDictionary['VY'] = None
    OrbitDictionary['VZ'] = None
    OrbitDictionary['P'] = None
    OrbitDictionary['M'] = None

    return OrbitDictionary
    
    


class Orbits(dict):

    def __init__(self):#,simulation_directory,runtag):

        # check to see if an outfile already exists

        pass



    def map_orbits(self,simulation_directory,runtag,time_array,norb=1,comp='star',verbose=0,**kwargs):
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
            verbose keyword to be passed to psp_io

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

        infile_template = simulation_directory+'/OUT.'+runtag+'.'


        # check to see if an orbit list has been set
        if 'orblist' in kwargs:
            orbvals = kwargs['orblist'] # this needs to be passed as an integer array
            norb = np.max(orbvals)+1
            print('orbit.map_orbit: N_orbits accepted {} orbits'.format(len(orbvals)))
        else:
            orbvals = np.arange(0,norb,1,dtype='i')


        # check initial file for memmap construction
        O = psp_io.Input(infile_template+'%05i' %time_array[0])

        id_comp = np.where(np.array(O.comp_titles) == comp)[0]

        # make the holding array
        out_arr = np.zeros( [time_array.size,8,orbvals.size]) # 8 is only valid if no extra parameters in the psp file (update later)


        times = []
        bad_times = []
        prev_time = -1.

        # start the loop over files
        for indx,val in enumerate(time_array):

            if (verbose > 0) & (val < np.max(time_array)): utils.print_progress(val,np.max(time_array),'orbit.map_orbit')

            infile = infile_template+'%05i' %val

            O = psp_io.Input(infile)

            # sift through times to make sure always increasing
            if (indx > 0):

                # check if time is increasing
                if (O.time <= prev_time):

                    # if not, discard file
                    print('orbit.map_orbit: Bad file number {}, removing'.format(val))
                    bad_times.append(indx)


                else:
                    times.append(O.time)
                    tmp = np.memmap(infile,dtype='f',shape=(8,norb),offset=int(O.comp_pos_data[id_comp]),mode='r',order='f')

                    out_arr[indx] = tmp[:,orbvals]
                    
            else:
                times.append(O.time)
                tmp = np.memmap(infile,dtype='f',shape=(8,norb),offset=int(O.comp_pos_data[id_comp]),mode='r',order='f')
                out_arr[indx] = tmp[:,orbvals]

            prev_time = O.time

        times = np.array(times)
        
        out_arr = out_arr[0:times.size,:,:]

        #
        # populate the dictionary
        self['T'] = times
        self['M'] = out_arr[0,0,:]
        self['X'] = out_arr[:,1,:]
        self['Y'] = out_arr[:,2,:]
        self['Z'] = out_arr[:,3,:]
        self['VX'] = out_arr[:,4,:]
        self['VY'] = out_arr[:,5,:]
        self['VZ'] = out_arr[:,6,:]
        self['P'] = out_arr[:,7,:]


    def compute_quantities(self):

        v2 = self['VX']*self['VX'] + self['VY']*self['VY'] + self['VZ']*self['VZ']
        self['E'] = v2 + self['P']

        self['LZ'] = self['X']*self['VY'] - self['Y']*self['VX']


    def polar_coordinates(self):

        self['Rp'] = np.power(self['X']*self['X'] + self['Y']*self['Y'],0.5)
        self['Phi'] = np.arctan2(self['Y'],self['X'])

        try:
            self['Phib'] = np.arctan2(self['TY'],self['TX'])
        except:
            pass

        

    def resample_orbit_map(self,impr=4,sord=0,transform=False,**kwargs):
        '''
        return a resampled orbit map

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


                bar_positions = trapping.find_barangle(self['T'],BarInstance)

                # make a tiled version for fast computation
                manybar = np.tile(bar_positions,(self['M'].shape[0],1)).T

                # transform positions
                self['TX'] = self['X']*np.cos(manybar) - self['Y']*np.sin(manybar)
                self['TY'] = -self['X']*np.sin(manybar) - self['Y']*np.cos(manybar)

                # transform velocities
                self['VTX'] = self['VX']*np.cos(manybar) - self['VY']*np.sin(manybar)
                self['VTY'] = -self['VX']*np.sin(manybar) - self['VY']*np.cos(manybar)




            except:
                print('orbit.resample_orbit: bar file reading failed. Input using bar keyword.')







def find_orbit_frequencies(T,R,PHI,Z,window=[0,10000]):
    '''
    calculate the peak of the orbit frequency plot

    much testing/theoretical work to be done here (perhaps see the seminal papers?)

    what do we want the windowing to look like?

    '''

    if window[1] == 10000:
        window[1] = R.shape[0]

    # get frequency values
    freq = np.fft.fftfreq(T[window[0]:window[1]].shape[-1],d=(T[1]-T[0]))
    
    sp_r = np.fft.fft(  R[window[0]:window[1]])
    sp_t = np.fft.fft(PHI[window[0]:window[1]])
    sp_z = np.fft.fft(  Z[window[0]:window[1]])

    # why does sp_r have a zero frequency peak??
    sp_r[0] = 0.0

    OmegaR = abs(freq[np.argmax(((sp_r.real**2.+sp_r.imag**2.)**0.5))])
    OmegaT = abs(freq[np.argmax(((sp_t.real**2.+sp_t.imag**2.)**0.5))])
    OmegaZ = abs(freq[np.argmax(((sp_z.real**2.+sp_z.imag**2.)**0.5))])

    
    return OmegaR,OmegaT,OmegaZ


                


                

def find_orbit_map_frequencies(OrbitInstance,window=[0,10000]):
    '''
    calculate the peak of the orbit frequency plot

    much testing/theoretical work to be done here (perhaps see the seminal papers?)

    what do we want the windowing to look like?

    '''

    try:
        x = OrbitInstance['Rp']
    except:
        print('orbit.find_orbit_frequencies: must have polar_coordinates.')
        OrbitInstance.polar_coordinates()

    if window[1] == 10000:
        window[1] = OrbitInstance['Phi'].shape[0]

    # get frequency values
    freq = np.fft.fftfreq(OrbitInstance['T'][window[0]:window[1]].shape[-1],d=(OrbitInstance['T'][1]-OrbitInstance['T'][0]))
    
    sp_r = np.fft.fft(OrbitInstance['Rp'][window[0]:window[1]],axis=0)
    sp_t = np.fft.fft(OrbitInstance['Phi'][window[0]:window[1]],axis=0)
    sp_z = np.fft.fft(OrbitInstance['Z'][window[0]:window[1]],axis=0)

    # why does sp_r have a zero frequency peak??
    sp_r[0,:] = np.zeros(OrbitInstance['Phi'].shape[1])

    OmegaR = abs(freq[np.argmax(((sp_r.real**2.+sp_r.imag**2.)**0.5),axis=0)])
    OmegaT = abs(freq[np.argmax(((sp_t.real**2.+sp_t.imag**2.)**0.5),axis=0)])
    OmegaZ = abs(freq[np.argmax(((sp_z.real**2.+sp_z.imag**2.)**0.5),axis=0)])

    
    return OmegaR,OmegaT,OmegaZ


                

def make_orbit_density(OrbitInstance,orbit,window=[0,10000],replot=False,scalelength=0.01,colorplot=True,nsamp=56):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    OrbitInstance
        -with transformation and polar coordinates setup. will add forcing for this at some point


    '''

    lo = window[0]
    hi = window[1]

    if (hi+1) > OrbitInstance['T'].size: hi = OrbitInstance['T'].size - 1

    scalefac = 1./scalelength

    #scalefac = 100.

    if not replot:
        fig = plt.figure(figsize=(6.46,4.53),dpi=100)
    else:
        plt.clf()
        fig = plt.gcf()

    # 
    #want to re-scale the extent to make a more intelligent boundary
    #
    extentx_in = 1.2*np.max(abs(OrbitInstance['TX'][lo:hi,orbit]))
    extenty_in = 1.2*np.max(abs(OrbitInstance['TY'][lo:hi,orbit]))
    extentz_in = 1.2*np.max(abs(OrbitInstance['Z'][lo:hi,orbit]))
    #
    xbins = np.linspace(-extentx_in,extentx_in,nsamp)
    ybins = np.linspace(-extenty_in,extenty_in,nsamp)
    xx,yy = np.meshgrid( xbins,ybins)
    zbins = np.linspace(-extentz_in,extentz_in,nsamp)
    xxz,zz = np.meshgrid( xbins,zbins)

    # test the kde waters
    try:
        tt = kde_3d.fast_kde_two(OrbitInstance['TX'][lo:hi,orbit],OrbitInstance['TY'][lo:hi,orbit],\
                             gridsize=(nsamp,nsamp), extents=(-extentx_in,extentx_in,-extenty_in,extenty_in),\
                             nocorrelation=True, weights=None)
        tz = kde_3d.fast_kde_two(OrbitInstance['TX'][lo:hi,orbit],OrbitInstance['Z'][lo:hi,orbit],\
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
    
    _ = ax1.set_ylabel('Y$_{\\rm bar}$ [R$_d$]')
    _ = ax2.set_ylabel('Z [R$_d$]')
    _ = ax4.set_ylabel('Y$_{\\rm bar}$ [R$_d$]')
    _ = ax4.set_xlabel('X$_{\\rm bar}$ [R$_d$]')
    _ = ax5.set_ylabel('Z [R$_d$]')
    _ = ax5.set_xlabel('X$_{\\rm bar}$ [R$_d$]')
    
    _ = ax1.set_xticklabels(())
    _ = ax2.set_xticklabels(())

    if colorplot:
        loT = OrbitInstance['T'][lo]
        hiT = OrbitInstance['T'][hi]
        dT  = (hiT-loT)/float(hi-lo)
        spacing = 5
        
        for indx in range(1,(hi-lo)+1,spacing):
            _ = ax4.plot(scalefac*OrbitInstance['TX'][lo+indx:lo+indx+spacing+1,orbit],scalefac*OrbitInstance['TY'][lo+indx:lo+indx+spacing+1,orbit],color=cm.gnuplot(indx/float(hi-lo),1.),lw=0.5)
            _ = ax5.plot(scalefac*OrbitInstance['TX'][lo+indx:lo+indx+spacing+1,orbit],scalefac*OrbitInstance['Z'][lo+indx:lo+indx+spacing+1,orbit],color=cm.gnuplot(indx/float(hi-lo),1.),lw=0.5)

        
    else:
        _ = ax4.plot(scalefac*OrbitInstance['TX'][lo:hi,orbit],scalefac*OrbitInstance['TY'][lo:hi,orbit],color='black',lw=0.5)
        _ = ax5.plot(scalefac*OrbitInstance['TX'][lo:hi,orbit],scalefac*OrbitInstance['Z'][lo:hi,orbit],color='black',lw=0.5)


    # any interest in the start point?
    #_ = ax4.scatter(scalefac*OrbitInstance['TX'][lo,orbit],scalefac*OrbitInstance['TY'][lo,orbit],color='red',s=3.)
    _ = ax1.axis([-2,2,-2,2])
    _ = ax4.axis([-2,2,-2,2])
    
    _ = ax2.axis([-2,2,-0.8,0.8])
    _ = ax5.axis([-2,2,-0.8,0.8])
    
    _ = ax4.set_xticklabels(['-2','','-1','','0','','1','','2'])
    _ = ax4.set_yticklabels(['-2','','-1','','0','','1','','2'])
    _ = ax1.set_yticklabels(['-2','','-1','','0','','1','','2'])
    _ = ax2.set_yticklabels(['-.8','','-.4','','0','','.4','','.8'])
    _ = ax5.set_xticklabels(['-2','','-1','','0','','1','','2'])
    _ = ax5.set_yticklabels(['-.8','','-.4','','0','','.4','','.8'])
    
    cmap = mpl.cm.Greys; norm = mpl.colors.Normalize(vmin=0., vmax=1.)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,norm=norm)
    _ = cb1.set_label('Relative Frequency',size=10)
    _ = cb1.set_ticks([0.,0.25,0.5,0.75,1.])

    if colorplot:
        cmap = mpl.cm.gnuplot; norm = mpl.colors.Normalize(vmin=loT, vmax=hiT)
        cb1 = mpl.colorbar.ColorbarBase(ax6, cmap=cmap,norm=norm)
        _ = cb1.set_label('System Time',size=10)
        _ = cb1.set_ticks([np.round(loT,2),np.round( 0.33*(hiT-loT) + loT,2),np.round( 0.66*(hiT-loT) + loT,2),np.round( 0.99*(hiT-loT) + loT,2)])



