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

# standard imports
from scipy.interpolate import UnivariateSpline
import copy

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





def find_orbit_frequencies(OrbitInstance,window=[0,10000]):
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


                

def make_orbit_density(infile):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    infile: string


    '''

    pass

