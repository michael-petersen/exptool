#############################################
#
#  potential.py
#    An exptool utility to handle energy and kappa calculations
#
#    MSP 10.1.2015
#
#

# standard libraries
import numpy as np
import time

# exptool classes
import utils

# for interpolation
from scipy.interpolate import UnivariateSpline



# use psp_io.particle_holder instead (01.17.17)
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
            self.PA.TIME = ParticleArray.time
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
    def __init__(self,ParticleArray,nbins=200,map_file=None,eres=80,percen=99.5,spline_order=3):


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
            self.PA.TIME = ParticleArray.time
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

        

        EnergyKappa.map_ekappa(self,percen=percen,eres=eres,spline_order=spline_order)

        if map_file:
            EnergyKappa.output_map(self,map_file)


    def map_ekappa(self,percen=99.9,eres=80,twodee=False,spline_order=3,smethod='sg'):

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
        self.Energy = 0.5*V2 + self.PA.POTE

        #
        # should think about 2d vs 3d utility
        #
        if twodee:
            R = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS)**0.5
                
        else:
            R = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS + self.PA.ZPOS*self.PA.ZPOS)**0.5

        # partition particles into Energy bins
        self.Ebins = np.linspace(0.999*np.min(self.Energy),np.min([-2.5,1.001*np.max(self.Energy)]),eres)
        eindx = np.digitize(self.Energy,self.Ebins)

        # allocate arrays
        self.maxLz = np.zeros_like(self.Ebins)
        self.maxR = np.zeros_like(self.Ebins)
        self.maxL = np.zeros_like(self.Ebins)
        self.circR = np.zeros_like(self.Ebins)

        
        for i,energy in enumerate(self.Ebins):
            energy_range = np.where( eindx==i+1)[0]
            
            if len(energy_range) > 1:
                # reduce operations for speed
                #maxLx[i] = np.percentile(LX[yese],percen)
                #maxLy[i] = np.percentile(LY[yese],percen)
                self.maxLz[i] = np.percentile(LZ[energy_range],percen) #np.max(LZ[yese])

                # take median of top 100 Lz for guiding center radius
                #   (that is, radius of a circular orbit)
                lzarg = energy_range[LZ[energy_range].argsort()]
                self.circR[i] = np.median( R[lzarg[-100:-1]] )
                
                self.maxR[i] = np.percentile(R[energy_range],percen)
                self.maxL[i] = np.percentile(L[energy_range],percen)
                
            else: # guard for empty bins
                #maxLx[i] = maxLx[i-1]
                #maxLy[i] = maxLy[i-1]
                self.maxLz[i] = self.maxLz[i-1]
                self.maxR[i] = self.maxR[i-1]
                self.maxL[i] = self.maxL[i-1]
                self.circR[i] = self.circR[i-1]

        # smooth discontinuities from bin choice
        if smethod == 'sg':
            smthLz = helpers.savitzky_golay(self.maxLz,7,3) # could figure out an adaptive smooth?
            smthL = helpers.savitzky_golay(self.maxL,7,3)
            smthR = helpers.savitzky_golay(self.circR,7,3)
        else:
            smthLzf = UnivariateSpline(self.Ebins,self.maxLz,k=spline_order)
            smthLf  = UnivariateSpline(self.Ebins,self.maxL,k=spline_order)
            smthRf  = UnivariateSpline(self.Ebins,self.circR,k=spline_order)
            smthLz = smthLzf(self.Ebins)
            smthL  = smthLf(self.Ebins)
            smthR  = smthRf(self.Ebins)

        
        # return energy and kappa for all orbits
        if smethod == 'sg':
            self.Kappa = LZ/smthLz[eindx-1]
            self.Beta = L/smthL[eindx-1]
            self.cR = smthR[eindx-1]
        else:
            self.Kappa = LZ/smthLzf(self.Energy)
            self.Beta = L/smthLf(self.Energy)
            self.cR = smthRf(self.Energy)
        self.LZ = LZ
        self.L = L

    def clear_output_map_file(self):
        return None

    def output_map(self,file):

        #
        # helper class to 
        #
        f = open(file,'w+')
        print >>f,PA.TIME,len(self.Ebins),self.Ebins,self.maxLz,self.maxL,self.maxR,self.circR

        f.close()


    def ek_grid(self,eres=80,kres=80,set_ebins=True,ebins_in=None):

        self.Kbins = np.linspace(-1.,1.,kres)
        self.Ebins = self.Ebins

        if not (set_ebins):
            self.Ebins = ebins_in

        self.Eindx = np.digitize(self.Energy,self.Ebins)
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

        self.EKarray = np.zeros([ebmax,kbmax])
        self.Egrid,self.Kgrid = np.meshgrid(self.Ebins,self.Kbins)

        for i in range(0,len(self.Eindx)):
            if (self.Eindx[i] < ebmax) and (self.Kindx[i] < kbmax):
                self.EKarray[self.Eindx[i],self.Kindx[i]] += sumval[i]


