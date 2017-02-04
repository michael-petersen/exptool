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
import psp_io
import trapping
import eof
import spheresl
import halo_methods

# for interpolation
from scipy.interpolate import UnivariateSpline



class Fields():
    '''
    class to accumulate (all) particles from a dump and return the field quantities

    UNFORTUNATELY, this only works on halo+disk systems right now. Should offload the ability to plug in multiple components.

    '''
    def __init__(self,infile,eof_file,sph_file,model_file,nhalo=1000000,transform=False,no_odd=False,verbose=1):

        self.infile = infile
        self.eof_file = eof_file
        self.sph_file = sph_file
        self.model_file = model_file
        self.nhalo = nhalo
        self.transform = transform
        self.no_odd = no_odd

        self.verbose = verbose

        # do the total coefficient calculation?
        #Fields.total_coefficients(self)

    def total_coefficients(self):


        # read in the files
        #
        PSPDumpDisk = psp_io.Input(self.infile,comp='star')
        if self.transform:
            PSPDumpDiskTransformed = trapping.BarTransform(PSPDumpDisk)
            
            if self.verbose > 1:
                print 'potential.Fields.total_coefficients: Using bar_angle %4.3f' %PSPDumpDiskTransformed.bar_angle
                
        else:
            PSPDumpDiskTransformed = PSPDumpDisk
            

        PSPDumpHaloT = psp_io.Input(self.infile,comp='dark')

        PSPDumpHalo = psp_io.Input(self.infile,comp='dark',nout=self.nhalo)

        self.halofac = float(PSPDumpHaloT.nbodies)/float(PSPDumpHalo.nbodies)
        
        if self.transform:
            PSPDumpHaloTransformed = trapping.BarTransform(PSPDumpHalo,bar_angle=PSPDumpDiskTransformed.bar_angle)
            
        else:
            PSPDumpHaloTransformed = PSPDumpHalo
            
        #
        # compute coefficients
        #
        self.EOF = eof.compute_coefficients(PSPDumpDiskTransformed,self.eof_file,verbose=self.verbose,no_odd=self.no_odd)

        
        self.SL = spheresl.compute_coefficients(PSPDumpHaloTransformed,self.sph_file,self.model_file,verbose=self.verbose,no_odd=self.no_odd)

    def prep_tables(self):

        try:
            x = self.EOF.eof_file

        except:
            print 'potential.Fields.prep_tables: must first call total_coefficients.'

        # build disk tables
        self.potC,self.rforceC,self.zforceC,self.densC,\
          self.potS,self.rforceS,self.zforceS,self.densS \
          = eof.parse_eof(self.EOF.eof_file)
          
        self.rmindisk,self.rmaxdisk,self.numx,self.numy,self.mmax,self.norder,self.ascale,self.hscale,self.cmapdisk,self.densdisk \
          = eof.eof_params(self.EOF.eof_file)
          
        self.XMIN,self.XMAX,self.dX,self.YMIN,self.YMAX,self.dY \
          = eof.set_table_params(RMAX=self.rmaxdisk,RMIN=self.rmindisk,ASCALE=self.ascale,HSCALE=self.hscale,NUMX=self.numx,NUMY=self.numy,CMAP=self.cmapdisk)

        # build halo tables
        self.lmaxhalo,self.nmaxhalo,self.numrhalo,self.cmaphalo,\
          self.rminhalo,self.rmaxhalo,self.scalehalo,self.ltablehalo,self.evtablehalo,self.eftablehalo \
          = halo_methods.read_cached_table(self.SL.sph_file)
          
        self.xihalo,self.rarrhalo,self.p0halo,self.d0halo \
          = halo_methods.init_table(self.SL.model_file,self.numrhalo,self.rminhalo,self.rmaxhalo,cmap=self.cmaphalo,scale=self.scalehalo)  


    def return_density(self,xval,yval,zval):
        '''
        definition to return the density for the monopole term and total separately, and for the two components

        wrapped elsewhere to some end

        '''
        
        try:
            x = self.EOF.eof_file
            y = self.potC

        except:
            print 'potential.Fields.return_density: must first call total_coefficients and prep_tables.'

        r2val = (xval*xval + yval*yval)**0.5  + 1.e-10
        r3val = (r2val*r2val + zval*zval)**0.5  + 1.e-10
        costh = zval/r3val
        phival = np.arctan2(yval,xval)

            
        # disk evaluation call
        diskp0,diskp,diskfr,diskfp,diskfz,diskden0,diskden1 = eof.accumulated_eval(r2val, zval, phival,\
                                              self.EOF.cos, self.EOF.sin,\
                                              self.potC, self.rforceC, self.zforceC, self.densC,\
                                              self.potS, self.rforceS, self.zforceS, self.densS,\
                                              rmin=self.XMIN,dR=self.dX,zmin=self.YMIN,dZ=self.dY,numx=self.numx,numy=self.numy,fac = 1.0,\
                                              MMAX=self.mmax,NMAX=self.norder,\
                                              ASCALE=self.ascale,HSCALE=self.hscale,CMAP=self.cmapdisk,no_odd=self.no_odd)
        #
        # halo evaluation call
        haloden0,haloden1,halopot0,halopot1,halopotr,halopott,halopotp = spheresl.all_eval(r3val, costh, phival,\
                                                                                           self.halofac*self.SL.expcoef,\
                                                                                           self.xihalo,self.p0halo,self.d0halo,self.cmaphalo,self.scalehalo,\
                                                                                           self.lmaxhalo,self.nmaxhalo,\
                                                                                           self.evtablehalo,self.eftablehalo,no_odd=self.no_odd)

        return haloden0,haloden1,diskden0,diskden1


    def density_calculate(self,rvals=np.linspace(0.,0.1,100)):
        '''
        routine to compute in-plane major axis density values for looking at profile change over time

        '''

        # cheap to just do this here and set everything up as a just-in-case
        Fields.prep_tables(self)

        if not self.densdisk:
            print 'Fields.density_calculate: no density terms in basis!'
            return None

        halodens_mono = np.zeros_like(rvals)
        diskdens_mono = np.zeros_like(rvals)
        halodens_total = np.zeros_like(rvals)
        diskdens_total = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            halodens_mono[indx],halodens_total[indx],diskdens_mono[indx],diskdens_total[indx] = Fields.return_density(self,rval,0.0,0.0)

        self.rvals = rvals
        self.halodens_mono = halodens_mono
        self.diskdens_mono = diskdens_mono
        self.halodens_total = halodens_total
        self.diskdens_total = diskdens_total


    def set_field_parameters(self,no_odd=False,halomonopole=False,diskmonopole=False,truncate_disk_n=1000):
        '''
        in preparation for other definitions, specify


        '''
        
        self.no_odd = no_odd
        self.halomonopole = halomonopole
        self.diskmonopole = diskmonopole
        self.truncate_disk_n = truncate_disk_n


    def return_forces_cart(self,xval,yval,zval,rotpos=0.0):
        
        # to be dealt with elsewhere
        # ,no_odd=False,halomonopole=False,diskmonopole=False,truncate_disk_n=1000

        try:
            x = self.no_odd

        except:
            print 'Fields.return_forces_cart: applying default potential parameters.'
            Fields.set_field_parameters(self)


        r2val = (xval*xval + yval*yval)**0.5  + 1.e-10
        r3val = (r2val*r2val + zval*zval)**0.5  + 1.e-10
        costh = zval/r3val
        phival = np.arctan2(yval,xval)

        # use only halo monopole?
        if self.halomonopole:
            use_l = 0
        else:
            use_l = self.lmaxhalo
            
        #
        if self.diskmonopole:
            use_m = 0
        else:
            use_m = self.mmax
            
        #
        if self.truncate_disk_n < self.norder:
            use_n = self.truncate_disk_n
        else:
            use_n = self.norder
            
        #
        # disk force call
        diskfr,diskfp,diskfz,diskp,diskp0 = eof.accumulated_forces(r2val, zval, phival + rotpos, \
                                                      self.EOF.cos[:,0:use_n], self.EOF.sin[:,0:use_n], \
                                                      self.potC[:,0:use_n,:,:], self.rforceC[:,0:use_n,:,:], self.zforceC[:,0:use_n,:,:],\
                                                      self.potS[:,0:use_n,:,:], self.rforceS[:,0:use_n,:,:], self.zforceS[:,0:use_n,:,:],\
                                                      rmin=self.XMIN,dR=self.dX,zmin=self.YMIN,dZ=self.dY,numx=self.numx,numy=self.numy,fac = 1.0,\
                                                      #MMAX=mmax,NMAX=norder,\
                                                      MMAX=use_m,NMAX=use_n,\
                                                      ASCALE=self.ascale,HSCALE=self.hscale,CMAP=self.cmapdisk,no_odd=self.no_odd)
        #
        # halo force call
        halofr,haloft,halofp,halop,halop0 = spheresl.force_eval(r3val, costh, phival + rotpos, \
                                                   self.halofac*self.SL.expcoef,\
                                                   self.xihalo,self.p0halo,self.d0halo,self.cmaphalo,self.scalehalo,\
                                                   #lmaxhalo,nmaxhalo,\
                                                   use_l,self.nmaxhalo,\
                                                   self.evtablehalo,self.eftablehalo,no_odd=self.no_odd)
                                                   
        # recommended guards against bizarre phi forces

        # do we need any other guards?
        if r3val < np.min(self.xihalo):
            halofp = 0.
            diskfp = 0.
        
        fxdisk = (diskfr*(xval/r2val) - diskfp*(yval/(r2val*r2val)) )
        fxhalo = -1.* ( halofr*(xval/r3val) - haloft*(xval*zval/(r3val*r3val*r3val))) + halofp*(yval/(r2val*r2val))
        
        fydisk = (diskfr*(yval/r2val) + diskfp*(xval/(r2val*r2val)) )
        fyhalo = -1.* ( halofr*(yval/r3val) - haloft*(yval*zval/(r3val*r3val*r3val))) - halofp*(xval/(r2val*r2val))
        
        fzdisk = diskfz
        fzhalo = -1.* ( halofr*(zval/r3val) + haloft*( (r2val*r2val)/(r3val*r3val*r3val)) )

        
        return fxdisk,fxhalo,fydisk,fyhalo,fzdisk,fzhalo,diskp,halop

    
    def rotation_curve(self,rvals=np.linspace(0.,0.1,100)):

        disk_force = np.zeros_like(rvals)
        halo_force = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            disk_force[indx],halo_force[indx],a,b,c,d,e,f = Fields.return_forces_cart(self,rval,0.0,0.0)

        self.rvals = rvals
        self.disk_rotation = (rvals*abs(disk_force))**0.5
        self.halo_rotation = (rvals*abs(halo_force))**0.5
        self.total_rotation = (rvals*(abs(halo_force)+abs(disk_force)))**0.5


    
    def compute_axis_potential(self,rvals=np.linspace(0.,0.1,100)):
        '''
        returns the potential along the major axis

        '''

        disk_pot = np.zeros_like(rvals)
        halo_pot = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            a,b,c,d,e,f,disk_pot[indx],halo_pot[indx] = Fields.return_forces_cart(self,rval,0.0,0.0)

        self.rvals = rvals
        self.disk_pot = disk_pot
        self.halo_pot = halo_pot
        self.total_pot = disk_pot+halo_pot






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


