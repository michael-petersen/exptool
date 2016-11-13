#
# orbit.py
#
#    part of exptool
#
#    11.12.16 formalized class structure, provided example usage.
#
'''
ORBIT.py deals with slices the PSP outputs 'against the grain' to collect the details on individual orbits.



'''


import numpy as np
import psp_io
import helpers

'''
#USAGE DEMO:


tarr = np.arange(0,12,1,dtype='int')
orbit.make_orbit_map('/path/to/outfile.dat','/path/to/exp/files',tarr,comp='dark')

times,mass,orb = orbit.read_orbit_map('/path/to/outfile.dat')

'''

'''
class TrackOrbits(object):

    #
    # do I really want this to be a class structure? this might make the definitions too clunky
    #

    def __init__(self):

        pass 
'''

def make_orbit_map(outfile,infile_template,time_array,norb=1,comp='star',verbose=0,**kwargs):

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


    

def read_orbit_map(infile):

    # open file
    f = open(infile,'rb')

    # read header 
    [ntimes,norb] = np.fromfile(f, dtype=np.int,count=2)

    # read times and masses 
    times = np.fromfile(f,dtype=np.float,count=ntimes)
    mass = np.fromfile(f,dtype=np.float,count=norb)

    #print ntimes,norb

    orb = np.memmap(infile,offset=(16 + 8*ntimes + 8*norb),dtype=np.float,shape=(ntimes,norb,6))

    return times,mass,orb


