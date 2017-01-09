#
# globalquantities.py
#
#    part of exptool: extract global quantities from simulations
#
#    11.20.16 formalize Fourier analysis steps
#    12.11.16 cleanup and provide restart support
#
#    TODO:
#      add global angular momentum counters
#


import time
import os
import numpy as np
import psp_io


class GQuantities():

    #
    # basic quantities from simulation
    #

    # INPUTS:
    #     simfile                                        : single file to analyze (or file with list of files when multi=True)
    #

    # DESIRED QUANTITIES
    #   tracking LZ through time: this needs to be cleaned up to make more sense.
    #     total and differential (the latter being very slow)

    
    def __init__(self,simfile=None,track_lz=False,track_diff_lz=False,track_fourier=False,comp=None,verbose=0,rbins=None,multi=False,resample=0):

        #
        # ability to instantiate and exit
        if simfile == None:
            print 'globalquantities.__init__: beginning...'
            return None

    
        self.slist = simfile
        self.verbose = verbose
        self.zfourier = False

        # add an option to make a single file readable (3.5.16 MSP)
        if (multi==True):
            GQuantities.parse_list(self,resample=resample)
            
        else:
            # this is the single file input case
            self.SLIST = np.array([simfile])

            O = psp_io.Input(self.SLIST[0],validate=True)

            try:
                if O.ntot > 0:
                    if verbose >= 1: print 'globalquantities.__init__: Accepted single input file.'

            except:
                print 'globalquantities.__init__: multi=False, but simfile is not a valid PSP file. Exiting.'
            

        self.rbins = rbins
        self.comp = comp
            

        if (track_lz) and (comp):
            if self.verbose >= 1:
                print 'Tracking Global Angular momentum...'
            GQuantities.track_angular_momentum(self,comp)

        if (track_diff_lz) and (comp):
            if self.verbose >= 1:
                print 'Tracking Radial Differential Angular momentum...'
            GQuantities.track_diff_angular_momentum(self,comp,rbins)

        if (track_fourier) and (comp):
            if self.verbose >= 1:
                print('globalquantities.__init__: tracking fourier decompositions...')
            GQuantities.compute_fourier(self,mmax=6)

    def parse_list(self,resample=0):
        #
        # make the list of files to operate on. if reample is >1, will select every (resample) files.
        #

        
        f = open(self.slist)
        s_list = []

        filenum = 0
        
        for line in f:
            
            if resample:
                if (filenum % resample) == 0:
                    d = [q for q in line.split()]
                    s_list.append(d[0])
                    
            else: # not resampling        
                d = [q for q in line.split()]
                s_list.append(d[0])

            filenum += 1

            
        f.close()

        self.SLIST = np.array(s_list)

        # add a block to check that files are existing files are being specified properly (MSP 3.5.16)
        try:
            
            f = open(self.SLIST[0])
            
            if self.verbose >= 1:
                print 'Accepted %i files.' %len(self.SLIST)
                
        except:
            print 'First file is not a valid filename. Perhaps the infile is not an array of files?'

        

    def track_angular_momentum(self,comp,dfreq=50):
        #
        # follow the total angular momentum in a component
        #

        tinit = time.time()
        self.TLZ = np.zeros(len(self.SLIST))

        
        #
        # step through different files    
        for i,file in enumerate(self.SLIST):
                
            O = psp_io.Input(file,comp=comp,verbose=0)

            self.TLZ[i] = np.sum(O.xpos*O.yvel - O.ypos*O.xpos)

            
            if (i % dfreq == 0) and (self.verbose >= 2):
                print '(%i%%, %3.2f)...' %(i/len(self.SLIST),time.time()-tinit),
                


    def track_diff_angular_momentum(self,comp,rbins,dfreq=50):

        tinit = time.time()
        self.TLZ = np.zeros([len(rbins),len(self.SLIST)])

        
        #
        # probably want rbins to be reportable
        #
        #
        # step through different files    
        for i,file in enumerate(self.SLIST):

                
            O = psp_io.Input(file,comp=comp,verbose=0)

            R = (O.xpos*O.xpos + O.ypos*O.ypos + O.zpos*O.zpos)**0.5

            rindx = np.digitize(R,rbins)

            for j in range(0,len(rbins)):
                w = np.where(rindx == j)[0]
                self.TLZ[j,i] = np.sum(O.xpos[w]*O.yvel[w] - O.ypos[w]*O.xpos[w])

            
            if (i % dfreq == 0) and (self.verbose >= 2):
                print '%3.2f%%, %3.2f seconds elapsed\r' %(float(i/len(self.SLIST)),time.time()-tinit),
                


    def compute_discrete_frequency_fourier(self,mmax=6,window='gaussian'):
        '''
        following Roskar+12 (and Press+92 therein), use a fourier discrete time series to attempt to find multiple frequencies

        '''
        
        #
        # output format of the files is:
        #
        # Q.aval[m_val,time_val,rbin_val]
        # 

        try:
            x = self.aval[0,0,0]
        except:
            print 'globalquantities.compute_discrete_frequency_fourier : globalquantities.fourier has not yet been called.'


        frequency_sampling = (2.*np.pi * k * m )/ ( S * dT )

            
        try:
            comp = self.comp
        except:
            print 'globalquantities.compute_fourier: No component specified...trying star.'
            comp = 'star'
            try:
                O = psp_io.Input(self.SLIST[0],comp='star',verbose=0)
            except:
                print 'globalquantities.compute_fourier: No star...trying dark.'
                comp = 'dark'


        self.aval = np.zeros([mmax,len(self.SLIST),len(rbins)])
        self.bval = np.zeros([mmax,len(self.SLIST),len(rbins)])
        self.time = np.zeros([len(self.SLIST)])
        
        for i,file in enumerate(self.SLIST):

                
            O = psp_io.Input(file,comp=comp,verbose=0)
        
            r_dig = np.digitize( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5,rbins,right=True)

            for indx,r in enumerate(rbins):
                yes = np.where( r_dig-1 == indx)[0]
                self.aval[0,i,indx] = np.sum(O.mass[yes])
                for m in range(1,mmax):
                    self.aval[m,i,indx] = np.sum(O.mass[yes] * np.cos(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))
                    self.bval[m,i,indx] = np.sum(O.mass[yes] * np.sin(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))

            self.time[i] = O.time





    def compute_fourier(self,mmax=6):

        #
        # output format of the files is:
        #
        # Q.aval[m_val,time_val,rbin_val]
        # 

        try:
            rbins = self.rbins
            dr = rbins[1] - rbins[0]
        except:
            print 'Rbins must be set in order (or have multiple values) to proceed. Appying default...'
            self.rbins = np.linspace(0.,1.,100)
            rbins = self.rbins

            
        try:
            comp = self.comp
        except:
            print 'globalquantities.compute_fourier: No component specified...trying star.'
            comp = 'star'
            try:
                O = psp_io.Input(self.SLIST[0],comp='star',verbose=0)
            except:
                print 'globalquantities.compute_fourier: No star...trying dark.'
                comp = 'dark'


        self.aval = np.zeros([mmax,len(self.SLIST),len(rbins)])
        self.bval = np.zeros([mmax,len(self.SLIST),len(rbins)])
        self.time = np.zeros([len(self.SLIST)])
        
        for i,file in enumerate(self.SLIST):

            if self.verbose > 1: print 'globalquantities.compute_fourier: working on ',file
                
            O = psp_io.Input(file,comp=comp,verbose=0)
        
            r_dig = np.digitize( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5,rbins,right=True)

            for indx,r in enumerate(rbins):
                yes = np.where( r_dig-1 == indx)[0]
                self.aval[0,i,indx] = np.sum(O.mass[yes])
                for m in range(1,mmax):
                    self.aval[m,i,indx] = np.sum(O.mass[yes] * np.cos(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))
                    self.bval[m,i,indx] = np.sum(O.mass[yes] * np.sin(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))

            self.time[i] = O.time


    def write_fourier(self,outfile):

        f = open(outfile,'wb')

        np.array([self.aval.shape[0],self.aval.shape[1],self.aval.shape[2]],dtype='i4').tofile(f)

        if self.zfourier:
            np.array(1.,dtype='i4').tofile(f)
        else:
            np.array(0.,dtype='i4').tofile(f)
        
        np.array(self.comp,dtype='S12').tofile(f)
        np.array(self.time,dtype='f4').tofile(f)
        np.array(self.rbins,dtype='f4').tofile(f)
        np.array(self.aval.reshape(-1,),dtype='f4').tofile(f)
        np.array(self.bval.reshape(-1,),dtype='f4').tofile(f)
        

        if self.zfourier:
            np.array(self.azval.reshape(-1,),dtype='f4').tofile(f)
            np.array(self.bzval.reshape(-1,),dtype='f4').tofile(f)
            np.array(self.anorm.reshape(-1,),dtype='f4').tofile(f)

        f.close()

    def read_fourier(self,infile):

        f = open(infile,'rb')

        [mmax,tnum,rnum] = np.fromfile(f,dtype='i4',count=3)
        [self.zfourier] = np.fromfile(f,dtype='i4',count=1)
        [self.comp] = np.fromfile(f,dtype='S12',count=1)
        self.time = np.fromfile(f,dtype='f4',count=tnum)
        self.rbins = np.fromfile(f,dtype='f4',count=rnum)
        atmp = np.fromfile(f,dtype='f4',count=mmax*tnum*rnum)
        btmp = np.fromfile(f,dtype='f4',count=mmax*tnum*rnum)

        self.aval = atmp.reshape([mmax,tnum,rnum])
        self.bval = btmp.reshape([mmax,tnum,rnum])

        if self.zfourier:
            aztmp = np.fromfile(f,dtype='f4',count=tnum*rnum)
            bztmp = np.fromfile(f,dtype='f4',count=tnum*rnum)
            anormtmp = np.fromfile(f,dtype='f4',count=tnum*rnum)
            self.azval = aztmp.reshape(tnum,rnum)
            self.bzval = bztmp.reshape(tnum,rnum)
            self.anorm = anormtmp.reshape(tnum,rnum)
        
        f.close()


    def compute_z_fourier(self):

        #
        # output format of the files is:
        #
        # Q.azval[rbin_val,time_val]
        #
        # no m order because it is only considering the m=2 case

        try:
            rbins = self.rbins
            dr = rbins[1] - rbins[0]
        except:
            print 'Rbins must be set in order (or have multiple values) to proceed. Appying default...'
            self.rbins = np.linspace(0.,1.,100)
            rbins = self.rbins

            
        try:
            comp = self.comp
        except:
            print 'No component specified...trying star.'
            comp = 'star'
            try:
                O = psp_io.Input(self.SLIST[0],comp='star',verbose=0)
            except:
                print 'No star...trying dark.'
                comp = 'dark'


        self.azval = np.zeros([len(self.SLIST),len(rbins)])
        self.bzval = np.zeros([len(self.SLIST),len(rbins)])
        self.anorm = np.zeros([len(self.SLIST),len(rbins)])
        
        for i,file in enumerate(self.SLIST):

            if self.verbose > 1: print 'globalquantities.compute_z_fourier: working on ',file
                
            O = psp_io.Input(file,comp=comp,verbose=0)
        
            r_dig = np.digitize( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5,rbins,right=True)

            for indx,r in enumerate(rbins):
                yes = np.where( r_dig-1 == indx)[0]
                self.azval[i,indx] = np.sum(O.mass[yes] * O.zpos[yes] * np.cos(2.*np.arctan2(O.ypos[yes],O.xpos[yes])))
                self.bzval[i,indx] = np.sum(O.mass[yes] * O.zpos[yes] * np.sin(2.*np.arctan2(O.ypos[yes],O.xpos[yes])))
                self.anorm[i,indx] = np.sum(O.mass[yes])

        self.zfourier = True



        


def frac_power(Q,frac=0.5):
    '''
    given a fourier instance, return the radial value where some fraction of the maximum power is realized.

    inputs
    ------
    Q    :   numpy array, [m_order,time index, r index]
    frac :   (optional) float, fraction of the maximum Fourier power to probe.

    returns
    ------
    blen :   the length of the bar as measured by the fractional power
    
    '''
    a2power = (Q.aval[2,:,:]**2. + Q.bval[2,0,:]**2.)/Q.aval[0,:,:]**2.

    k = a2power.argmax()
    blen = 0.
    
    while a2power[k] > frac*np.max(a2power):
        blen = Q.rbins[k]
        k+=1
    return blen



def phase_change(Q,deg=20.):
    '''
    using a Fourier instance, find the radius where the phase angle shifts by deg degrees from the phase angle at the maximum

    '''
    Q.ph = np.arctan(-Q.bval[2,0,:]/Q.aval[2,0,:])
    lowbin = int(np.floor(0.1*len(Q.ph)))
    
    # take average of first 10% of bins
    pavg = np.mean(Q.ph[0:lowbin])
    k = 0


    # or should this just be the phase angle at the maximum?
    #k = a2power.argmax()
    #pavg = Q.ph[k]

    
    pval = Q.ph[k]
    blen = 0.
    
    while abs(pval-pavg) < (deg/180.)*np.pi:
        blen = Q.rbins[k]
        k+=1
        pval = Q.ph[k]
        
    return blen


