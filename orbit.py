#
# orbit.py
#
#    part of exptool: orbit input/output from simulations, pure data processing convenience.
#
#    11.12.16 formalized class structure, provided example usage.
#    11.19.16 documented definitions; considered for generic upgrade.
#    11.20.16 do generic upgrade to make a single utility
#                 TODO? consider adding a utility conditioned on memory mapping for individual orbits to speed up process.
#                    not clear if needed, reading back in is very fast.
#

import numpy as np
import psp_io

'''
# Quick Start Demo:

# which exp-numbered files to read in
tarr = np.arange(0,12,1,dtype='int')

# read in from files and return a dictionary
Orbits = orbit.map_orbits('/path/to/outfile.dat','/path/to/exp/files',tarr,comp='dark',dictionary=True, norb=10)

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
    
    



def map_orbits(outfile,infile_template,time_array,norb=1,comp='star',verbose=0,**kwargs):
    '''
    make_orbit_map

    : slice across the grain for PSPDumps to track individual particles

    Parameters:
    ----------
    outfile: string, filename
        where to save the mapping to a file, even if just a temporary filename

    infile_template: string, filename
        leading directory structure and simulation name, point at EXP outputs

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

    if 'dictionary' in kwargs:
        return_dictionary = kwargs['dictionary'] # this needs to be passed as an integer array

    #
    # this writes to file because it is a lot to carry around
    f = open(outfile,'wb')
    #

    # check to see if an orbit list has been set
    if 'orblist' in kwargs:
        orbvals = kwargs['orblist'] # this needs to be passed as an integer array
        norb = np.max(orbvals)+1
        print 'N_orbits accepted: ',len(orbvals)
    else:
        orbvals = np.arange(0,norb,1,dtype='i')


    #
    # print self-describing header to file
    np.array([len(time_array),len(orbvals)],dtype=np.int).tofile(f)
    #

    # get time array from snapshots
    times = []
    for indx,val in enumerate(time_array):
        O = psp_io.Input(infile_template+'%05i' %time_array[val],nout=1,comp=comp)
        times.append(O.ctime)

    # print to file
    np.array(times,dtype=np.float).tofile(f)

    # get mass array from snapshot
    #    only accepts a 0 file to get masses, could eventually be problematic.
    O = psp_io.Input(infile_template+'00000',nout=norb,comp=comp)
    masses = O.mass[orbvals]

    # print to file
    np.array(masses,dtype=np.float).tofile(f)

    # loop through files and extract orbits
    for indx,val in enumerate(time_array):

        O = psp_io.Input(infile_template+'%05i' %time_array[val],nout=norb,comp=comp,verbose=verbose)

        if verbose > 0: print O.ctime

        for star in orbvals:
            np.array([O.xpos[star],O.ypos[star],O.zpos[star],O.xvel[star],O.yvel[star],O.zvel[star]],dtype=np.float).tofile(f)

    f.close()

    if return_dictionary:
        Orbits = read_orbit_map(outfile)

        return Orbits
    


def read_orbit_map(infile):
    '''
    Reads in orbit map file.

    inputs
    ------
    infile: string
        name of the file printed above


    outputs
    ------
    Orbits: dictionary, OrbitDictionary class
        returns an OrbitDictionary class object
        
    '''

    # open file
    f = open(infile,'rb')

    # read header 
    [ntimes,norb] = np.fromfile(f, dtype=np.int,count=2)

    # read times and masses 
    times = np.fromfile(f,dtype=np.float,count=ntimes)
    mass = np.fromfile(f,dtype=np.float,count=norb)

    #print ntimes,norb

    orb = np.memmap(infile,offset=(16 + 8*ntimes + 8*norb),dtype=np.float,shape=(ntimes,norb,6))

    Orbits = initialize_orbit_dictionary()

    Orbits['T'] = times
    Orbits['M'] = mass

    Orbits['X'] = orb[:,:,0]
    Orbits['Y'] = orb[:,:,1]
    Orbits['Z'] = orb[:,:,2]
    Orbits['VX'] = orb[:,:,3]
    Orbits['VY'] = orb[:,:,4]
    Orbits['VZ'] = orb[:,:,5]

    return Orbits



def make_orbit_density(infile):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    infile: string


    '''

    pass
    
