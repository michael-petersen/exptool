#
# globalquantities.py
#
#    part of exptool: extract global quantities from simulations
#
#    11.20.16 formalize Fourier analysis steps
#

import time
import os
import numpy as np
import psp_io


'''
import globalquantities

#Q = globalquantities.GQuantities('/scratch/mpetersen/Disk064a/testlist.dat',comp='star',track_diff_lz=True,verbose=2,rbins=np.linspace(0.,0.08,200))


rbins=np.linspace(0.,0.08,200)
Q = globalquantities.GQuantities('/scratch/mpetersen/Disk074a/filelist.dat',comp='star',rbins=rbins,verbose=2)

# Remember (in the original version) that the infile in the call below is a data file that is a list of files.
# That is, a text file where the first (only) line is the file you want to open.
#Q = globalquantities.GQuantities('/Users/mpetersen/Research/NBody/Disk064a/filelist.dat',comp='star',rbins=rbins,verbose=2)

# This is the revised handler, which will take single files.
Q = globalquantities.GQuantities('/Volumes/SIMSET/OUT.run064a.00800',comp='star',rbins=rbins,verbose=2)

Q = globalquantities.GQuantities('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='star',rbins=rbins,verbose=2)


globalquantities.GQuantities.compute_fourier(Q)

# This is one type of plot you could make. What does it show? (see other attached image from Athanassoula2013)

qscale = (Q.aval[2,0,:]**2. + Q.bval[2,0,:]**2.)/Q.aval[0,0,:]**2.
plt.plot(300.*rbins,qscale/np.max(qscale))


#
#
# where the fourier phase deviates by 20 degrees
# compute fourier phase
Q.ph = np.arctan(-Q.bval[2,0,:]/Q.aval[2,0,:])

blenf = frac_power(Q)
blenp = phase_change(Q)


fourmax = np.zeros(1000)
blenf = np.zeros(1000)
blenp = np.zeros(1000)
pmax = np.zeros(1000)
k = 0
for j in range(210,1000,20):
     filename = '/scratch/mpetersen/Disk074a/OUT.run074a.%05i' %j
     print filename
     rbins=np.linspace(0.,0.04,100)
     Q = globalquantities.GQuantities(filename,comp='star',rbins=rbins,verbose=2,multi=False)
     globalquantities.GQuantities.compute_fourier(Q)
     barst = (Q.aval[2,0,:]**2. + Q.bval[2,0,:]**2.)**0.5/Q.aval[0,0,:]
     fourmax[k] = rbins[barst.argmax()]
     pmax[k] = np.max(barst)
     blenf[k] = globalquantities.frac_power(Q)
     blenp[k] = globalquantities.phase_change(Q)
     k+=1


fourmax = fourmax[0:k]
blenf = blenf[0:k]
blenp = blenp[0:k]
pmax = pmax[0:k]


'''


#
#
# and some fraction of the max power
def frac_power(Q,frac=0.5):
    '''
    given a

    inputs
    ------
    Q: numpy array, [m_order,
    
    
    '''
    a2power = (Q.aval[2,0,:]**2. + Q.bval[2,0,:]**2.)/Q.aval[0,0,:]**2.
    k = a2power.argmax()
    blen = 0.
    while a2power[k] > frac*np.max(a2power):
        blen = Q.rbins[k]
        k+=1
    return blen



def phase_change(Q,deg=20.):
    Q.ph = np.arctan(-Q.bval[2,0,:]/Q.aval[2,0,:])
    lowbin = int(np.floor(0.1*len(Q.ph)))
    # take average of first 10% of bins
    pavg = np.mean(Q.ph[0:lowbin])
    k = 0
    pval = Q.ph[0]
    blen = 0.
    while abs(pval-pavg) < (deg/180.)*np.pi:
        blen = Q.rbins[k]
        k+=1
        pval = Q.ph[k]
    return blen




class OQuantities():

    #
    # class to add key orbit quantities
    #

    def __init__(self,infile,comp):

        self.infile = infile
        self.comp = comp

        OQuantities.maker(self)



    def maker(self):

       
        O = psp_io.Input(self.infile,self.comp)

        O.Lz = O.xpos*O.yvel - O.ypos*O.xpos
        O.E = 0.5*(O.xvel*O.xvel + O.yvel*O.yvel + O.zvel*O.zvel) + O.pote
        #
        # do we have a bar pattern to plug in?
        #
        #O.EJ = O.E - pattern*O.Lz




class GQuantities():

    #
    # basic quantities from simulation
    #

    # INPUTS:
    #     simfile                                        : single file to analyze (or file with list of files when multi=True)
    #

    # DESIRED QUANTITIES
    #   tracking LZ through time
    #     total and differential (the latter being very slow)

    
    def __init__(self,simfile,track_lz=False,track_diff_lz=False,comp=None,verbose=0,rbins=None,multi=False):

        self.slist = simfile
        self.verbose = verbose

        # add an option to make a single file readable (3.5.16 MSP)
        if (multi==True):
            GQuantities.parse_list(self)
            
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
        
            

    def parse_list(self):

        
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

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

            self.time[i] = O.ctime





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

                
            O = psp_io.Input(file,comp=comp,verbose=0)
        
            r_dig = np.digitize( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5,rbins,right=True)

            for indx,r in enumerate(rbins):
                yes = np.where( r_dig-1 == indx)[0]
                self.aval[0,i,indx] = np.sum(O.mass[yes])
                for m in range(1,mmax):
                    self.aval[m,i,indx] = np.sum(O.mass[yes] * np.cos(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))
                    self.bval[m,i,indx] = np.sum(O.mass[yes] * np.sin(float(m)*np.arctan2(O.ypos[yes],O.xpos[yes])))

            self.time[i] = O.ctime




    def compute_z_fourier(self):

        #
        # output format of the files is:
        #
        # Q.aval[rbin_val,time_val]
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
            print 'No component specified...trying star.'
            comp = 'star'
            try:
                O = psp_io.Input(self.SLIST[0],comp='star',verbose=0)
            except:
                print 'No star...trying dark.'
                comp = 'dark'


        self.azval = np.zeros([len(rbins),len(self.SLIST)])
        self.bzval = np.zeros([len(rbins),len(self.SLIST)])
        
        for i,file in enumerate(self.SLIST):

                
            O = psp_io.Input(file,comp=comp,verbose=0)
        
            r_dig = np.digitize( (O.xpos*O.xpos + O.ypos*O.ypos)**0.5,rbins,right=True)

            for indx,r in enumerate(rbins):
                yes = np.where( r_dig-1 == indx)[0]
                self.azval[indx,i] = np.sum(O.mass[yes] * O.zpos[yes] * np.cos(2.*np.arctan2(O.ypos[yes],O.xpos[yes])))
                self.bzval[indx,i] = np.sum(O.mass[yes] * O.zpos[yes] * np.sin(2.*np.arctan2(O.ypos[yes],O.xpos[yes])))
                self.anorm[indx,i] = np.sum(O.mass[yes])


