
#  08-29-16: added maximum radius capabilities to bar_fourier_compute

#  10-25-16: some redundancies noticed (bar_fourier_compute) and should be unified

'''


pattern.py (part of exptool)
     tools to find patterns in the global simulation outputs








BarTransform
BarDetermine


TODO:
-Filtering algorithms for bar determination (e.g. look at better time-series algorithms)


BASIC USAGE:

# to transform a PSP output to have the bar on the X axis
PSPTransform = pattern.BarTransform(PSPInput)

# to read in an existing bar file
BarInstance = pattern.BarDetermine()
BarInstance.read_bar(bar_file)

'''


from __future__ import absolute_import, division, print_function, unicode_literals



# general imports
import time
import numpy as np
import datetime
import os
from scipy import interpolate
from scipy.interpolate import UnivariateSpline


# exptool imports
from exptool.io import psp_io
from exptool.utils import kmeans
from exptool.utils import utils




   


class BarTransform():
    '''
    BarTransform : class to do the work to calculate the bar position and transform particles

    on it's own, BarTransform will reset the particles to be in the bar frame (planar transformation)

    inputs
    -----------------------
    ParticleInstanceIn                 : the input PSP instance
    bar_angle                          : (default=None) the known bar angle
    rel_bar_angle                      : (default=0.) the desired rotation angle relative to the bar major axis, counterclockwise (known or computed)
    minr                               : (default=0.) the MINIMUM radius of particles to use to compute the bar angle
    maxr                               : (default=1.) the MAXIMUM radius of particles to use in compute the bar angle

    outputs
    -----------------------
    None
        (ParticleInstanceIn will be modified to be in the planar bar transformation)


    helper routines
    -----------------------
    calculate_transform_and_return     : overwrite the input PSP instance to have the raw positions be transformed
    bar_fourier_compute                : use m=2 fourier to transform to the bar frame

    '''

    def __init__(self,ParticleInstanceIn,bar_angle=None,rel_bar_angle=0.,minr=0.,maxr=1.):

        '''
        see documentation above

        '''

        self.ParticleInstanceIn = ParticleInstanceIn

        self.bar_angle = bar_angle

        
        if self.bar_angle == None:
            self.bar_angle = -1.*BarTransform.bar_fourier_compute(self,self.ParticleInstanceIn.xpos,self.ParticleInstanceIn.ypos,maxr=maxr)

        # do an arbitary rotation of the particles relative to the bar?
        self.bar_angle += rel_bar_angle

        self.calculate_transform_and_return()
        

    def calculate_transform_and_return(self):
        '''
        calculate_transform_and_return
             do the modification of the input PSP instance to be in the bar frame.

        inputs
        ----------------------------
        self (BarTransform)


        '''

        
        transformed_x = self.ParticleInstanceIn.xpos*np.cos(self.bar_angle) - self.ParticleInstanceIn.ypos*np.sin(self.bar_angle)
        transformed_y = self.ParticleInstanceIn.xpos*np.sin(self.bar_angle) + self.ParticleInstanceIn.ypos*np.cos(self.bar_angle)

        transformed_vx = self.ParticleInstanceIn.xvel*np.cos(self.bar_angle) - self.ParticleInstanceIn.yvel*np.sin(self.bar_angle)
        transformed_vy = self.ParticleInstanceIn.xvel*np.sin(self.bar_angle) + self.ParticleInstanceIn.yvel*np.cos(self.bar_angle)


        self.xpos = transformed_x
        self.ypos = transformed_y
        self.zpos = np.copy(self.ParticleInstanceIn.zpos) # interesting. needs to be a copy for later operations to work!

        self.xvel = transformed_vx
        self.yvel = transformed_vy
        self.zvel = np.copy(self.ParticleInstanceIn.zvel)

        self.mass = self.ParticleInstanceIn.mass
        self.pote = self.ParticleInstanceIn.pote

        self.time = self.ParticleInstanceIn.time
        self.infile = self.ParticleInstanceIn.infile
        self.comp = self.ParticleInstanceIn.comp

    
    def bar_fourier_compute(self,posx,posy,minr=0.,maxr=1.):
        '''
        
        use x and y positions to compute the m=2 power, and find phase angle
        
        TODO:
            generalize to transform to any azimuthal order?

        '''
        w = np.where( ( (posx*posx + posy*posy)**0.5 > minr ) & ((posx*posx + posy*posy)**0.5 < maxr ))[0]
        
        aval = np.sum( np.cos( 2.*np.arctan2(posy[w],posx[w]) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy[w],posx[w]) ) )

        return np.arctan2(bval,aval)/2.



    


class BarDetermine():
    '''
    #
    # class to find the bar
    #

    '''

    def __init__(self,**kwargs):

        if 'file' in kwargs:
            try:
                # check to see if bar file has already been created
                self.read_bar(kwargs['file'])
                print('pattern.BarDetermine: BarInstance sucessfully read.')
            
            except:
                print('pattern.BarDetermine: no compatible bar file found.')
                

        return None
    
    def track_bar(self,filelist,verbose=0,maxr=1.,apse=False):

        self.slist = filelist
        self.verbose = verbose
        self.maxr = maxr

        if apse:
            BarDetermine.cycle_files_aps(self)

        else:
            BarDetermine.cycle_files(self)

        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

    def parse_list(self):
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print('BarDetermine.parse_list: Accepted {0:d} files.'.format(len(self.SLIST)))

    def cycle_files(self):

        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.time = np.zeros(len(self.SLIST))
        self.pos = np.zeros(len(self.SLIST))

        for i in range(0,len(self.SLIST)):
                O = psp_io.Input(self.SLIST[i],comp='star',verbose=self.verbose)
                self.time[i] = O.time
                self.pos[i] = BarDetermine.bar_fourier_compute(self,O.xpos,O.ypos,maxr=self.maxr)


        if self.verbose >= 2:
                print('Computed {0:d} steps in {1:3.2f} minutes, for an average of {2:3.2f} seconds per step.'.format( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) ))

    def cycle_files_aps(self,threedee=False,nout=100000):
        '''
        go through files, but use only the aps positions to determine the bar position. Useful if strong secondary patterns exist in m=2 power.


        EXAMPLE
        A = pattern.BarDetermine()
        A.track_bar(filelist,apse=True,maxr=0.02)


        '''

        # eventually this could be flexible!
        comp='star'
        
        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.time = np.zeros(len(self.SLIST))
        self.pos = np.zeros(len(self.SLIST))

        for i in range(1,len(self.SLIST)-1):

            # open three files to compare
            Oa = psp_io.Input(self.SLIST[i-1],comp=comp,nout=nout,verbose=0)
            Ob = psp_io.Input(self.SLIST[i],comp=comp,nout=nout,verbose=self.verbose)
            Oc = psp_io.Input(self.SLIST[i+1],comp=comp,nout=nout,verbose=0)

            # compute 2d radial positions
            if threedee:
                Oa.R = (Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos + Oa.zpos*Oa.zpos)**0.5
                Ob.R = (Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos + Ob.zpos*Ob.zpos)**0.5
                Oc.R = (Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos + Oc.zpos*Oc.zpos)**0.5

            else:
                Oa.R = (Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos)**0.5
                Ob.R = (Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos)**0.5
                Oc.R = (Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos)**0.5
                
            # use logic to find aps
            aps = np.logical_and( Ob.R > Oa.R, Ob.R > Oc.R )

            
            xposlist = Ob.xpos[aps]
            yposlist = Ob.ypos[aps]
                
            self.time[i] = Ob.time
            self.pos[i] = BarDetermine.bar_fourier_compute(self,xposlist,yposlist,maxr=self.maxr)


        if self.verbose >= 2:
                print('Computed {0:d} steps in {1:3.2f} minutes, for an average of {2:3.2f} seconds per step.'.format( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) ))



    def bar_doctor_print(self):

        #
        # wrap the bar file
        #
        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

        BarDetermine.print_bar(self,outfile)

        

    def unwrap_bar_position(self,jbuffer=-1.,smooth=False,reverse=False,adjust=np.pi):
    

        #
        # modify the bar position to smooth and unwrap
        #
        jnum = 0
        jset = np.zeros_like(self.pos)

        
        for i in range(1,len(self.pos)):

            if reverse:
                if (self.pos[i]-self.pos[i-1]) > -1.*jbuffer:   jnum -= 1

            else:
                if (self.pos[i]-self.pos[i-1]) < jbuffer:   jnum += 1

            jset[i] = jnum

        unwrapped_pos = self.pos + jset*adjust

        if (smooth):
            unwrapped_pos = helpers.savitzky_golay(unwrapped_pos,7,3)

        # to unwrap on twopi, simply do:
        #B.bar_upos%(2.*np.pi)

        self.pos = unwrapped_pos

        #
        # this implementation is not particularly robust, could revisit in future

    def frequency_and_derivative(self,smth_order=None,fft_order=None,spline_derivative=None,verbose=0):

        

        if (smth_order or fft_order):
            
            if (verbose):
                
                print('Cannot assure proper functionality of both order smoothing and low pass filtering.')

        self.deriv = np.zeros_like(self.pos)
        for i in range(1,len(self.pos)):
            self.deriv[i] = (self.pos[i]-self.pos[i-1])/(self.time[i]-self.time[i-1])

            
        if (smth_order):
            smth_params = np.polyfit(self.time, self.deriv, smth_order)
            pos_func = np.poly1d(smth_params)
            self.deriv = pos_func(self.time)

        if (fft_order):
            self.deriv = self.deriv

        if (spline_derivative):

            # hard set as a cubic spline,
            #    number is a smoothing factor between knots, see scipy.UnivariateSpline
            #
            #    recommended: 7 for dt=0.002 spacing
            
            spl = UnivariateSpline(self.time, self.pos, k=3, s=spline_derivative)
            self.deriv = (spl.derivative())(self.time)

            self.dderiv = np.zeros_like(self.deriv)
            #
            # can also do a second deriv
            for indx,timeval in enumerate(self.time):
                
                self.dderiv[indx] = spl.derivatives(timeval)[2]
                
            
            
    def bar_fourier_compute(self,posx,posy,maxr=1.):

        #
        # use x and y positions tom compute the m=2 power, and find phase angle
        #
        w = np.where( (posx*posx + posy*posy)**0.5 < maxr )[0]
        
        aval = np.sum( np.cos( 2.*np.arctan2(posy[w],posx[w]) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy[w],posx[w]) ) )

        return np.arctan2(bval,aval)/2.



    def print_bar(self,outfile):

        #
        # print the barfile to file
        #

        # this will be broken in python 3 compatibility
        
        f = open(outfile,'w')

        for i in range(0,len(self.time)):
            print(self.time[i],self.pos[i],self.deriv[i],file=f)

        f.close()

        return None
 
    def place_ellipse(self):

        return None

    def read_bar(self,infile):

        #
        # read a printed bar file
        #

        f = open(infile)

        time = []
        pos = []
        deriv = []
        for line in f:
            q = [float(d) for d in line.split()]
            time.append(q[0])
            pos.append(q[1])
            try:
                deriv.append(q[2])
            except:
                pass

        self.time = np.array(time)
        self.pos = np.array(pos)
        self.deriv = np.array(deriv)

        if len(self.deriv < 1):

            BarDetermine.frequency_and_derivative(self)



            




def compute_bar_lag(ParticleInstance,rcut=0.01,verbose=0):
    '''
    #
    # simple fourier method to calculate where the particles are in relation to the bar
    #
    '''
    R = (ParticleInstance.xpos*ParticleInstance.xpos + ParticleInstance.ypos*ParticleInstance.ypos)**0.5
    TH = np.arctan2(ParticleInstance.ypos,ParticleInstance.xpos)
    loR = np.where( R < rcut)[0]
    A2 = np.sum(ParticleInstance.mass[loR] * np.cos(2.*TH[loR]))
    B2 = np.sum(ParticleInstance.mass[loR] * np.sin(2.*TH[loR]))
    bar_angle = 0.5*np.arctan2(B2,A2)
    
    if (verbose):
        print('Position angle is {0:4.3f} . . .'.format(bar_angle))
        
    #
    # two steps:
    #   1. rotate theta so that the bar is aligned at 0,2pi
    #   2. fold onto 0,pi to compute the lag
    #
    tTH = (TH - bar_angle + np.pi/2.) % np.pi  # compute lag with bar at pi/2
    #
    # verification plot
    #plt.scatter( R[0:10000]*np.cos(tTH[0:10000]-np.pi/2.),R[0:10000]*np.sin(tTH[0:10000]-np.pi/2.),color='black',s=0.5)
    return tTH - np.pi/2. # retransform to bar at 0



    

def find_barangle(time,BarInstance,interpolate=True):
    '''
    #
    # use a bar instance to match the output time to a bar position
    #
    #    can take arrays!
    #
    #    but feels like it only goes one direction?
    #
    '''
    sord = 0 # should this be a variable?
    #
    if (interpolate):
        bar_func = UnivariateSpline(BarInstance.time,-BarInstance.pos,s=sord)
    #
    try:
        indx_barpos = np.zeros([len(time)])
        for indx,timeval in enumerate(time):
        #
            if (interpolate):
                indx_barpos[indx] = bar_func(timeval)
            #
            #
            else:
                indx_barpos[indx] = -BarInstance.pos[ abs(timeval-BarInstance.time).argmin()]
        #
    except:
        if (interpolate):
            indx_barpos = bar_func(time)
    #
        else:
            indx_barpos = -BarInstance.pos[ abs(time-BarInstance.time).argmin()]
    #    
    return indx_barpos



def find_barpattern(intime,BarInstance,smth_order=2):
    '''
    #
    # use a bar instance to match the output time to a bar pattern speed
    #
    #    simple differencing--may want to be careful with this.
    #    needs a guard for the end points
    #
    '''
    
    # grab the derivative at whatever smoothing order
    BarInstance.frequency_and_derivative(smth_order=smth_order)
    
    try:
        
        barpattern = np.zeros([len(intime)])
        
        for indx,timeval in enumerate(intime):

            best_time = abs(timeval-BarInstance.time).argmin()
            
            barpattern[indx] = BarInstance.deriv[best_time]
            
    except:

        best_time = abs(intime-BarInstance.time).argmin()
        
        barpattern = BarInstance.deriv[best_time]
        
    return barpattern



 
