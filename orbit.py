#
# orbit.py
#
#    part of exptool
#
'''
ORBIT.py deals with slices the PSP outputs 'against the grain' to collect the details on individual orbits.



'''


import numpy as np
import psp_io

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
    # print self-describer to file
    np.array([len(time_array),len(orbvals)],dtype=np.int).tofile(f)
    #
    times = []
    for indx,val in enumerate(time_array):
        O = psp_io.Input(infile_template+'%05i' %time_array[val],nout=1,comp=comp)
        times.append(O.ctime)
    np.array(times,dtype=np.float).tofile(f)
    O = psp_io.Input(infile_template+'00000',nout=norb,comp=comp)
    masses = O.mass[orbvals]
    np.array(masses,dtype=np.float).tofile(f)
    for indx,val in enumerate(time_array):
        # make this self-describing       
        O = psp_io.Input(infile_template+'%05i' %time_array[val],nout=norb,comp=comp,verbose=verbose)
        print O.ctime
        for star in orbvals:
            np.array([O.xpos[star],O.ypos[star],O.zpos[star],O.xvel[star],O.yvel[star],O.zvel[star]],dtype=np.float).tofile(f)
    f.close()


tarr = np.arange(0,1200,1,dtype='int')
make_orbit_map('/scratch/mpetersen/Disk001/orbit_map_dark.dat','/scratch/mpetersen/Disk001/OUT.run001.',tarr,1,'dark',orblist=bar_members)

    
def read_orbit_map(infile):
    f = open(infile,'rb')
    [ntimes,norb] = np.fromfile(f, dtype=np.int,count=2)
    times = np.fromfile(f,dtype=np.float,count=ntimes)
    mass = np.fromfile(f,dtype=np.float,count=norb)
    print ntimes,norb
    orb = np.memmap(infile,offset=(16 + 8*ntimes + 8*norb),dtype=np.float,shape=(ntimes,norb,6))
    return times,mass,orb

# format is orb[time,orbit,quantity]

times,mass,orb = read_orbit_map('/scratch/mpetersen/Disk001/orbit_map_dark.dat')


# steps: identify period at two scalelengths, then convolve all over that window
#

#
# THESE ARE ONLY FOR VISUALIZATION--NOT SCIENCE PRODUCTS
# 
from scipy.interpolate import UnivariateSpline

BarFunction = UnivariateSpline(BarInstance.bar_time,-1.*BarInstance.bar_pos,s=0)



def interpo_orbit(T,X,Y,Z,BarFunction,impr=4,sord=0):
    newT = np.linspace(np.min(T),np.max(T),len(T)*impr)
    sX = UnivariateSpline(T,X,s=sord)
    sY = UnivariateSpline(T,Y,s=sord)
    sZ = UnivariateSpline(T,Z,s=sord)
    # yes do bar transform
    sTX = sX(newT)*np.cos(BarFunction(newT)) - sY(newT)*np.sin(BarFunction(newT))
    sTY = sX(newT)*np.sin(BarFunction(newT)) + sY(newT)*np.cos(BarFunction(newT))
    # needs the transformed velocities
    return newT,sX(newT),sY(newT),sZ(newT),sTX,sTY


o = 80
etime=900
ltime=1100
T,X,Y,Z,sX,sY = interpo_orbit(times[etime:ltime],orb[etime:ltime,o,0],orb[etime:ltime,o,1],orb[etime:ltime,o,2],BarFunction,impr=4,sord=0)

R = (X*X + Y*Y)**0.5
raps = argrelextrema(R, np.greater)

fig = plt.figure(0)
ax = fig.add_subplot(121)
ax.plot(sX,sY,lw=1.)
ax.scatter(sX[raps],sY[raps],color='red',s=20.)

ax2 = fig.add_subplot(122)
ax2.plot(sX,Z,lw=1.)



# orbit 2 scatters off the bar potential; doesn't seem to be trapped, how can it be discriminated?
# try to use r clustering, or variance in x_bar values of the points

# get rid of orbits more massive than a disk particle?
#  use 1.e-8 as a low threshold and look at just massive particles
# 2   35   55   58   60   65   68   79   80   96   97   99  100  110  113

def interpo_orbit_vel(T,X,Y,Z,VX,VY,VZ,TB,PB,impr=4,sord=0):
    newT = np.linspace(np.min(T),np.max(T),len(T)*impr)
    sX = UnivariateSpline(T,X,s=sord)
    sY = UnivariateSpline(T,Y,s=sord)
    sZ = UnivariateSpline(T,Z,s=sord)
    sVX = UnivariateSpline(T,VX,s=sord)
    sVY = UnivariateSpline(T,VY,s=sord)
    sVZ = UnivariateSpline(T,VZ,s=sord)
    # yes do bar transform
    z = np.polyfit(TB[100:-1],PB[100:-1], 3)
    barpos = np.poly1d(z)
    sTX = sX(newT)*np.cos(barpos(newT)) - sY(newT)*np.sin(barpos(newT))
    sTY = sX(newT)*np.sin(barpos(newT)) + sY(newT)*np.cos(barpos(newT))
    # needs the transformed velocities
    return newT,sX(newT),sY(newT),sZ(newT),sVX(newT),sVY(newT),sVZ(newT),sTX,sTY




class Orbit():

    def __init__():

        pass




    
