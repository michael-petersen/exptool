#
#
#
import numpy as np
import time
import helpers


'''
import psp_io
import potential
import helpers

import matplotlib.pyplot as plt
import numpy as np


Od = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.01000',comp='star',verbose=2)
EK = potential.EnergyKappa(Od)

Od0 = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00000',comp='star',verbose=2)
EK0 = potential.EnergyKappa(Od0)

def kappa_change():
rbins = np.linspace(0.,0.05,40)
Od.R = (Od.xpos*Od.xpos + Od.ypos*Od.ypos + Od.zpos*Od.zpos)**0.5
Rindx = np.digitize(Od.R,rbins)

tenb = np.zeros_like(rbins)
fiftyb = np.zeros_like(rbins)
ninetyb = np.zeros_like(rbins)
for i,val in enumerate(rbins):
     kapparr = EK.K[np.where(Rindx==i)[0]]
     if len(kapparr)>0:
          tenb[i] = np.percentile(kapparr,25)
          fiftyb[i] = np.percentile(kapparr,50)
          ninetyb[i] = np.percentile(kapparr,75)


Od = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='star',verbose=2)


# this whole exercise takes about 30 seconds
Od = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.01000',comp='dark',verbose=2)
EK = potential.EnergyKappa(Od)



LZ = Od.xpos*Od.yvel - Od.ypos*Od.xvel
A = potential.EnergyKappa.sum_ek_values(EK,np.ones(len(Od.xpos)))


B = potential.EnergyKappa.sum_ek_values(EK,LZ)
print np.max(B/(A+1.))


Od = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00800',comp='star',verbose=2)
EK2 = potential.EnergyKappa(Od)

LZ2 = Od.xpos*Od.yvel - Od.ypos*Od.xvel

potential.EnergyKappa.ek_grid(EK2,set_ebins=False,ebins_in=EK.Energy)
C = potential.EnergyKappa.sum_ek_values(EK2,np.ones(len(Od.xpos)))
D = potential.EnergyKappa.sum_ek_values(EK2,LZ2)
print np.max(D/(C+1.))


# make a SN mask?
SNmask_row,SNmask_cols = np.where( (C>10))





O = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00000',comp='dark',verbose=2)
P = potential.Potential(O)



Od = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00000',comp='star',verbose=2,nout=10000)
Pd = potential.Potential(Od)

# resample halo to be on disk terms
Hv = helpers.resample(P.rbins,P.vcirc,Pd.rbins)

F = np.max(Pd.vcirc)/np.max((Hv**2.+Pd.vcirc**2.)**0.5)
print 'The submaximality parameter is %3.3f' %F

plt.ion()
plt.figure(1)
plt.plot(Pd.rbins,Hv,'-.',color='gray')
plt.plot(Pd.rbins,Pd.vcirc,'--',color='gray')
plt.plot(Pd.rbins,(Hv**2.+Pd.vcirc**2.)**0.5,color='black')
plt.axis([0.0,0.1,0.0,1.7])
plt.xlabel('Radius',size=20)
plt.ylabel('V$_{\\rm circ}$',size=20)



O = psp_io.Input('/Volumes/SIMSET/OUT.run074a.01000',comp='dark',verbose=2)
P = potential.Potential(O)

Od = psp_io.Input('/Volumes/SIMSET/OUT.run074a.01000',comp='star',verbose=2)
Pd = potential.Potential(Od)


plt.figure(1)
plt.plot(P.rbins,P.vcirc,color='gray')
plt.plot(Pd.rbins,Pd.vcirc,color='gray')
plt.plot(Pd.rbins,(P.vcirc**2.+Pd.vcirc**2.)**0.5,color='black')

xrange = np.linspace(-0.03,0.03,45)
xx,yy,out = quick_contour(xrange,xrange,Od.xpos,Od.ypos,np.ones(len(Od.xpos)))


plt.contourf(xx,yy,np.log10(out),24)

'''

class PArray(object):
    TIME = None
    MASS = None
    XPOS = None
    YPOS = None
    ZPOS = None
    XVEL = None
    YVEL = None
    ZVEL = None
    POTE = None



class Potential():

    #
    # Format of the particle of array (ParticleArray) must be:
    #   ParticleArray.MASS
    #                .XPOS
    #                .YPOS
    #                .ZPOS
    #                .XVEL
    #                .YVEL
    #                .ZVEL
    #                .POTE
    #

    def __init__(self,ParticleArray,nbins=200):


        self.PA = PArray()

        # check to see if this is single or multi-timestep
        try:
            # what is the time input here?
            self.PA.MASS = ParticleArray.MASS
            self.PA.XPOS = ParticleArray.XPOS
            self.PA.YPOS = ParticleArray.YPOS
            self.PA.ZPOS = ParticleArray.ZPOS
            self.PA.XVEL = ParticleArray.XVEL
            self.PA.YVEL = ParticleArray.YVEL
            self.PA.ZVEL = ParticleArray.ZVEL
            self.PA.POTE = ParticleArray.POTE
            self.multitime = True
        except:
            self.PA.TIME = ParticleArray.ctime
            self.PA.MASS = ParticleArray.mass
            self.PA.XPOS = ParticleArray.xpos
            self.PA.YPOS = ParticleArray.ypos
            self.PA.ZPOS = ParticleArray.zpos
            self.PA.XVEL = ParticleArray.xvel
            self.PA.YVEL = ParticleArray.yvel
            self.PA.ZVEL = ParticleArray.zvel
            self.PA.POTE = ParticleArray.pote
            self.multitime = False

        self.nbins = nbins

        Potential.m_zero(self)

        Potential.circular_velocity(self)

    def m_zero(self):

        #
        # calculate the axisymmetric component
        #
        self.RPOS = ( self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS + self.PA.ZPOS*self.PA.ZPOS)**0.5

        
        #
        # calculate internal rbins (default log-spaced, no intent to change presently)
        #
        #
        log_rbins = np.linspace( np.log10(np.min(self.RPOS)),np.log10(np.max(self.RPOS)),self.nbins)

        #    minimum r should be a percentile choice?
        minpad = 0.3 # percentile value to exclude at small radii
        min_rval = np.percentile(self.RPOS,minpad)
        log_rbins = np.linspace( np.log10(min_rval),np.log10(np.max(self.RPOS)),self.nbins)


        #    alternately, could set a minimum floor for the rbins based on a fraction of the scaleheight
        log_rbins = np.linspace( -3.3,np.log10(np.max(self.RPOS)),self.nbins)

        # convert bins to non-log
        rbins_int = 10.**log_rbins
        

        # dr for each bin FOR VOLUME (add extra bin at end)
        drbins = np.ediff1d( (rbins_int**3.))#,to_end=(self.rbins[-1]**3.-self.rbins[-2]**3.))
        volume = (4./3.)*np.pi*drbins

        #self.r_bin_index = np.digitize(self.RPOS,self.rbins)

        #
        # calculate the mass per rbin (last bin is half-open, as per np.histogram)
        #
        # hist, bin_edges
        self.mass_dist,rbins_int2 = np.histogram(self.RPOS,bins=rbins_int,weights=self.PA.MASS)
        self.rbins = rbins_int2[0:-1]
        self.dens_dist = self.mass_dist/volume


    def m_potential(self):

        #
        # calculate the potential given the calculated density distribution
        #

        # smooth the potential distribution
        #    NOTE! This is being output in log now
        self.smth_dens = helpers.savitzky_golay(np.log10(self.dens_dist),5,3)

        # compute errors
        self.smth_err = self.smth_dens - np.log10(self.dens_dist)

        #
        # compute density integral (finite differencing technique)
        #

        # THIS IS BROKEN AND VERY ANNOYING BECAUSE OF IT
        #               0->R             rho(r)     *        dr
        self.idens = np.cumsum((10.**self.smth_dens)*np.ediff1d(self.rbins,to_end=(self.rbins[-1]-self.rbins[-2])))

    def circular_velocity(self):
        
        self.vcirc = (np.cumsum(self.mass_dist) / self.rbins)**0.5
        
        



class EnergyKappa():

    #
    # class to look at energy-kappa mapping
    #
    def __init__(self,ParticleArray,nbins=200,map_file=None):


        self.PA = PArray()

        # check to see if this is single or multi-timestep
        try:
            self.PA.MASS = ParticleArray.MASS
            self.PA.XPOS = ParticleArray.XPOS
            self.PA.YPOS = ParticleArray.YPOS
            self.PA.ZPOS = ParticleArray.ZPOS
            self.PA.XVEL = ParticleArray.XVEL
            self.PA.YVEL = ParticleArray.YVEL
            self.PA.ZVEL = ParticleArray.ZVEL
            self.PA.POTE = ParticleArray.POTE
            self.multitime = True
        except:
            self.PA.TIME = ParticleArray.ctime
            self.PA.MASS = ParticleArray.mass
            self.PA.XPOS = ParticleArray.xpos
            self.PA.YPOS = ParticleArray.ypos
            self.PA.ZPOS = ParticleArray.zpos
            self.PA.XVEL = ParticleArray.xvel
            self.PA.YVEL = ParticleArray.yvel
            self.PA.ZVEL = ParticleArray.zvel
            self.PA.POTE = ParticleArray.pote
            self.multitime = False

        self.nbins = nbins

        

        EnergyKappa.map_ekappa(self)

        if map_file:
            EnergyKappa.output_map(self,map_file)


    def map_ekappa(self,percen=99.9,eres=80):

        # enables plotting of
        # self.
        #      Energy    :    the index
        #      Kappa     :    LZ/LZ_max for each orbit
        #      E         :    energy for each orbit
        #      maxLZ     :    relation to Energy for a circular (planar) orbit
        #      maxL      :    relation to Energy for a spherical orbit
        #      maxR      :    relation to Energy for radius in spherical bins

        # squared velocity
        V2 = (self.PA.XVEL*self.PA.XVEL + self.PA.YVEL*self.PA.YVEL + self.PA.ZVEL*self.PA.ZVEL)

        # angular momentum evaluation
        LX = self.PA.YPOS*self.PA.ZVEL - self.PA.ZPOS*self.PA.YVEL
        LY = self.PA.ZPOS*self.PA.XVEL - self.PA.XPOS*self.PA.ZVEL
        LZ = self.PA.XPOS*self.PA.YVEL - self.PA.YPOS*self.PA.XVEL
        L = (LX*LX + LY*LY + LZ*LZ)**0.5

        # total energy (to be made accessible)
        self.E = 0.5*V2 + self.PA.POTE

        #
        # should think about 2d vs 3d utility
        #
        #R2 = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS)**0.5
        R = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS + self.PA.ZPOS*self.PA.ZPOS)**0.5

        # partition particles into E bins
        self.ebins = np.linspace(0.999*np.min(self.E),1.001*np.max(self.E),eres)
        eindx = np.digitize(self.E,self.ebins)

        # allocate arrays
        self.maxLz = np.zeros_like(self.ebins)
        self.maxR = np.zeros_like(self.ebins)
        self.maxL = np.zeros_like(self.ebins)
        self.circR = np.zeros_like(self.ebins)

        
        for i,energy in enumerate(self.ebins):
            yese = np.where( eindx==i+1)[0]
            
            if len(yese) > 1:
                # reduce operations for speed
                #maxLx[i] = np.percentile(LX[yese],percen)
                #maxLy[i] = np.percentile(LY[yese],percen)
                self.maxLz[i] = np.percentile(LZ[yese],percen) #np.max(LZ[yese])

                # take median of top 100 Lz for guiding center radius
                #   (that is, radius of a circular orbit)
                lzarg = yese[LZ[yese].argsort()]
                self.circR[i] = np.median( R[lzarg[-100:-1]] )
                
                self.maxR[i] = np.percentile(R[yese],percen)
                self.maxL[i] = np.percentile(L[yese],percen)
                
            else: # guard for empty bins
                #maxLx[i] = maxLx[i-1]
                #maxLy[i] = maxLy[i-1]
                self.maxLz[i] = self.maxLz[i-1]
                self.maxR[i] = self.maxR[i-1]
                self.maxL[i] = self.maxL[i-1]
                self.circR[i] = self.circR[i-1]

        # smooth discontinuities from bin choice       
        smthLz = helpers.savitzky_golay(self.maxLz,7,3) # could figure out an adaptive smooth?
        smthL = helpers.savitzky_golay(self.maxL,7,3)

        # return energy and kappa for all orbits
        self.K = LZ/smthLz[eindx-1]



    def clear_output_map_file(self):
        return None

    def output_map(self,file):

        #
        # helper class to 
        #
        f = open(file,'w+')
        print >>f,PA.TIME,len(self.ebins),self.ebins,self.maxLz,self.maxL,self.maxR,self.circR

        f.close()


    def ek_grid(self,eres=80,kres=80,set_ebins=True,ebins_in=None):

        self.Kbins = np.linspace(-1.,1.,kres)
        self.Ebins = self.Energy

        if not (set_ebins):
            self.Ebins = ebins_in

        self.Eindx = np.digitize(self.E,self.Ebins)
        self.Kindx = np.digitize(self.Kappa,self.Kbins)


    def sum_ek_values(self,sumval):

        # sumval is an input of the same lengths as self.Eindx


            
        # has ek_grid already been run?
        #     if not, run it.
        try:
            x = self.Kindx[0]
        except:
            print 'Making Grid...'
            EnergyKappa.ek_grid(self)

        if len(sumval)!=len(self.Eindx):
            print 'Input array must have values for all particles.'
            #break

        ebmax = len(self.Ebins)
        kbmax = len(self.Kbins)

        ARR = np.zeros([ebmax,kbmax])

        for i in range(0,len(self.Eindx)):
            if (self.Eindx[i] < ebmax) and (self.Kindx[i] < kbmax):
                ARR[self.Eindx[i],self.Kindx[i]] += sumval[i]

        return ARR
