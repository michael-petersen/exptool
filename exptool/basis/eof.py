
# 08-17-16: bug found in accumulate() where call to get_pot() didn't pass MMAX,NMAX
# 08-19-16: cmap consistency added
# 08-26-16: print_progress and verbosity structure added
# TODO (08-26-16): break out c modules

# 08-29-16: added density consistency, still to be fixed in some places

'''
USAGE EXAMPLE


#
# in order to get force fields from an output dump and eof cache file:
#   1) read in cachefile, setting potC and potS in particular
#   2) determine global parameters for the cachefile and set those to be passed to all future functions
#   3) accumulate particles to find coefficients
#

'''

# general definitions
import struct
import numpy as np
import os
import time
import sys
import itertools
import multiprocessing
from collections import OrderedDict
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# exptool definitions
from exptool.utils import utils
from exptool.io import psp_io

from exptool.basis._accumulate_c import r_to_xi,xi_to_r

#
# tools to read in the eof cache and corresponding details
#
def eof_params(file,verbose=0):
    '''
    eof_params

    inputs
    --------
    file    :   (string) input cache filename
    verbose :   (bool)   print eof parameters?

    returns
    --------
    rmin    :   (float) minimum radius for eof expansion
    rmax    :   (float) maximum radius for eof expansion
    numx    :   (int)   number of points in radial direction
    numy    :   (int)   number of points in the vertical direction
    mmax    :   (int)   number of azimuthal orders
    norder  :   (int)   number of radial orders
    ascale  :   (float) scalelength for radial scaling
    hscale  :   (float) scaleheight for vertical scaling
    cmap    :   (bool)  use mapping?
    dens    :   (bool)  contains density terms?


    '''
    f = open(file,'rb')
    #
    # read the header
    #
    a = np.fromfile(f, dtype=np.uint32,count=7)
    mmax = a[0]
    numx = a[1]
    numy = a[2]
    nmax = a[3]
    norder = a[4]
    dens = a[5]
    cmap = a[6]
    #
    # second header piece
    #
    a = np.fromfile(f, dtype='<f8',count=6)
    rmin = a[0]
    rmax = a[1]
    ascale = a[2]
    hscale = a[3]
    cylmass = a[4]
    tnow = a[5]

    if (verbose):

        print 'eof.eof_params: The parameters for this EOF file are:'
        print 'RMIN=%5.4f,RMAX=%5.4f' %(rmin,rmax)
        print 'MMAX=%i' %mmax
        print 'NORDER=%i' %norder
        print 'NMAX=%i' %nmax
        print 'NUMX,NUMY=%i,%i' %(numx,numy)
        print 'DENS,CMAP=%i,%i' %(dens,cmap)
        print 'ASCALE,HSCALE=%5.4f,%5.4f' %(ascale,hscale)
        print 'CYLMASS=%5.4f' %cylmass
        print 'TNOW=%5.4f' %tnow


    f.close()
    
    return rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens




def parse_eof(file):
    '''
    parse_eof

    inputs
    --------
    file    :   (string) input cache filename

    returns
    --------
    potC    :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) cosine potential terms
    rforcec :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) cosine radial force terms
    zforcec :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) cosine vertical force terms
    densc   :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) cosine density terms
    potS    :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) sine potential terms
    rforces :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) sine radial force terms
    zforces :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) sine vertical force terms
    denss   :   (matrix, (mmax+1) x (norder) x (numx+1) x (numy+1) ) sine density terms


    '''

    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(file,verbose=0)
    #
    # open the eof_file
    f = open(file,'rb')

    # skip past the header info, already grabbed it
    f.seek(76)
    
    #
    # initialize blank arrays
    #
    potC = np.zeros([mmax+1,norder,numx+1,numy+1])
    rforcec = np.zeros([mmax+1,norder,numx+1,numy+1])
    zforcec = np.zeros([mmax+1,norder,numx+1,numy+1])
    densc = np.zeros([mmax+1,norder,numx+1,numy+1])
    potS = np.zeros([mmax+1,norder,numx+1,numy+1])
    rforces = np.zeros([mmax+1,norder,numx+1,numy+1])
    zforces = np.zeros([mmax+1,norder,numx+1,numy+1])
    denss = np.zeros([mmax+1,norder,numx+1,numy+1])
    #
    #
    for i in range(0,mmax+1):
        
        for j in range(0,norder):

            for k in range(0,numx+1):
                potC[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            for k in range(0,numx+1):
                rforcec[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            for k in range(0,numx+1):
                zforcec[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            if (dens==1):
                for k in range(0,numx+1):
                    densc[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                    
    for i in range(1,mmax+1): # no zero order m for sine terms
        
        for j in range(0,norder):
            
            for k in range(0,numx+1):
                potS[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            for k in range(0,numx+1):
                rforces[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            for k in range(0,numx+1):
                zforces[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
                
            if (dens==1):
                for k in range(0,numx+1):
                    denss[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)


    f.close()
    
    return potC,rforcec,zforcec,densc,potS,rforces,zforces,denss



def set_table_params(RMAX=20.0,RMIN=0.001,ASCALE=0.01,HSCALE=0.001,NUMX=128,NUMY=64,CMAP=0):
    '''
    set_table_params
        calculate scaled boundary values for the parameter table

    inputs
    -------


    returns
    -------
    

    '''
    M_SQRT1_2 = np.sqrt(0.5)
    Rtable  = M_SQRT1_2 * RMAX
    
    # check cmap, but if cmap=0, r_to_xi = r
    #
    # otherwise, r = (r/ASCALE - 1.0)/(r/ASCALE + 1.0);

    # calculate radial scalings
    XMIN    = r_to_xi(RMIN*ASCALE,CMAP,ASCALE);
    XMAX    = r_to_xi(Rtable*ASCALE,CMAP,ASCALE);
    dX      = (XMAX - XMIN)/NUMX;

    # calculate vertical scalings
    YMIN    = z_to_y(-Rtable*ASCALE,HSCALE);
    YMAX    = z_to_y( Rtable*ASCALE,HSCALE);
    dY      = (YMAX - YMIN)/NUMY;
        
    return XMIN,XMAX,dX,YMIN,YMAX,dY



#
# mapping definitions
#
def z_to_y(z,hscale):
    '''
    return mapping of vertical position to Y-dimension table scaling.

    '''
    return z/(abs(z)+1.e-10)*np.arcsinh(abs(z/hscale))


def y_to_z(y,hscale):
    '''
    return mapping of Y-dimension table scaling to vertical position
    
    '''
    return hscale*np.sinh(y)


#
# particle accumulation definitions
#
def return_bins(r,z,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,ASCALE=0.01,HSCALE=0.001,CMAP=0):
    #
    # routine to return the integer bin numbers based on dimension mapping
    # 
    X = (r_to_xi(r,CMAP,ASCALE) - rmin)/dR
    Y = (z_to_y(z,hscale=HSCALE) - zmin)/dZ
    ix = int( np.floor((r_to_xi(r,CMAP,ASCALE) - rmin)/dR) )
    iy = int( np.floor((z_to_y(z,hscale=HSCALE) - zmin)/dZ) )
    #
    # check the boundaries and set guards
    if ix < 0:
        ix = 0
        X = 0
    if ix >= numx:
        ix = numx - 1
        X = numx - 1
    if iy < 0:
        iy = 0
        Y = 0
    if iy >= numy:
        iy = numy - 1
        Y = numy - 1
    return X,Y,ix,iy



def get_pot(r,z,cos_array,sin_array,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18,ASCALE=0.01,HSCALE=0.001,CMAP=0):
    #
    # returns potential fields for C and S to calculate weightings during accumulation
    #
    #
    # find the corresponding bins
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
    #
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    #
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;
    #
    Vc = np.zeros([MMAX+1,NMAX])
    Vs = np.zeros([MMAX+1,NMAX])
    
    Vc = fac * ( cos_array[:,:,ix,iy] * c00 + cos_array[:,:,ix+1,iy] * c10 + cos_array[:,:,ix,iy+1] * c01 + cos_array[:,:,ix+1,iy+1] * c11 )
    #
    Vs = fac * ( sin_array[:,:,ix,iy] * c00 + sin_array[:,:,ix+1,iy] * c10 + sin_array[:,:,ix,iy+1] * c01 + sin_array[:,:,ix+1,iy+1] * c11 );
    #for mm in range(0,MMAX+1):
        #
        #Vc[mm] = fac * ( cos_array[mm,:,ix,iy] * c00 + cos_array[mm,:,ix+1,iy] * c10 + cos_array[mm,:,ix,iy+1] * c01 + cos_array[mm,:,ix+1,iy+1] * c11 )
        #
        #if (mm>0):
            #Vs[mm] = fac * ( sin_array[mm,:,ix,iy] * c00 + sin_array[mm,:,ix+1,iy] * c10 + sin_array[mm,:,ix,iy+1] * c01 + sin_array[mm,:,ix+1,iy+1] * c11 );
    return Vc,Vs


def get_pot_single_m(r,z,cos_array,sin_array,MORDER,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,NMAX=18,ASCALE=0.01,HSCALE=0.001,CMAP=0):
    #
    # returns potential fields for single C and S order
    #
    #
    # find the corresponding bins
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
    
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;
    
    Vc = np.zeros([1,NMAX])
    Vs = np.zeros([1,NMAX])
    
    Vc = fac * ( cos_array[MORDER,:,ix,iy] * c00 + cos_array[MORDER,:,ix+1,iy] * c10 + \
                         cos_array[MORDER,:,ix,iy+1] * c01 + cos_array[MORDER,:,ix+1,iy+1] * c11 )
    
    if (MORDER>0):
        Vs = fac * ( sin_array[MORDER,:,ix,iy] * c00 + sin_array[MORDER,:,ix+1,iy] * c10 + \
                             sin_array[MORDER,:,ix,iy+1] * c01 + sin_array[MORDER,:,ix+1,iy+1] * c11 );
    return Vc,Vs



def accumulate(ParticleInstance,potC,potS,MMAX,NMAX,XMIN,dX,YMIN,dY,NUMX,NUMY,ASCALE,HSCALE,CMAP,verbose=0,no_odd=False):
    #
    # take all the particles and stuff them into the basis
    #
    norm = -4.*np.pi
    #
    # set up particles
    #
    norb = len(ParticleInstance.mass)
    #
    # set up accumulation arrays
    #
    accum_cos = np.zeros([MMAX+1,NMAX])
    accum_sin = np.zeros([MMAX+1,NMAX])
    #
    for n in range(0,norb):
        
        if (verbose > 0) & ( ((float(n)+1.) % 1000. == 0.0) | (n==0)): utils.print_progress(n,norb,'eof.accumulate')

        # calculate cylindrical coordinates
        r = (ParticleInstance.xpos[n]**2. + ParticleInstance.ypos[n]**2. + 1.e-10)**0.5
        phi = np.arctan2(ParticleInstance.ypos[n],ParticleInstance.xpos[n])
        
        vc,vs = get_pot(r, ParticleInstance.zpos[n], potC,potS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=NUMX,numy=NUMY,fac=1.0,MMAX=MMAX,NMAX=NMAX,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
        
        # skip odd terms?
        #if ((mm % 2) != 0) & (no_odd):
        #    continue
            
            #
        morder = np.tile(np.arange(0.,MMAX+1.,1.),(NMAX,1)).T
        mcos = np.cos(phi*morder)
        msin = np.sin(phi*morder)

        # make a mask to only do the even terms?
        if (no_odd):
            mask = abs(np.cos((np.pi/2.)*morder))

        else:
            mask = np.zeros_like(morder) + 1.

        accum_cos += (norm * ParticleInstance.mass[n] * mcos * vc)
        
        accum_sin += (norm * ParticleInstance.mass[n] * msin * vs)
                             
    return accum_cos,accum_sin





def show_basis(eof_file,plot=False,sine=False):
    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
    rmin,rmax,numx,numy,MMAX,norder,ascale,hscale,cmap,dens = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)

    xvals = xi_to_r(np.array([XMIN + i*dX for i in range(0,numx+1)]),cmap,ascale)
    zvals =  y_to_z(np.array([YMIN + i*dY for i in range(0,numy+1)]),hscale)

    print('eof.show_basis: plotting %i azimuthal orders and %i radial orders...'%(MMAX,norder) )

    xgrid,zgrid = np.meshgrid(xvals,zvals)

    if sine:
        width=2
    else:
        width=1
    
    if plot:

        plt.subplots_adjust(hspace=0.001)
        
        for mm in range(0,MMAX+1):
            fig = plt.figure()

            for nn in range(0,norder):
                if mm > 0: ax = fig.add_subplot(norder,width,width*(nn)+1)
                else: ax = fig.add_subplot(norder,1,nn+1)

                ax.contourf(xgrid,zgrid,potC[mm,nn,:,:].T,cmap=cm.gnuplot)
                ax.axis([0.0,0.08,-0.03,0.03])

                if nn < (norder-1): ax.set_xticklabels(())

                if (mm > 0) & (sine):
                    ax2 = fig.add_subplot(norder,2,2*(nn)+2)

                    ax2.contourf(xgrid,zgrid,potS[mm,nn,:,:].T,cmap=cm.gnuplot)
                    
                    ax2.text(np.max(xgrid),0.,'N=%i' %nn)




def map_basis(eof_file):
    '''
    return memory maps for modification of basis functions.

    --------------

    be careful! this will allow for overwriting of the basis functions. it is smart to make a copy first (which this does)

    --------------
    for example, to zero (actually must be epsilon) specific functions, do:

    EPS = 1.e-10
    # zero out the orders that are vertically asymmetric
    for i in range(0,4):  # 4 if dens, 3 if not
          mC[0, 9,i,:,:] = np.zeros([numx+1,numy+1]) + EPS

    mC.flush() # <--- this locks in the changes to the file.

    '''
    copyfile(eof_file, eof_file+'.original')
    
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof.eof_params(eof_file)

    if (dens):
        mC = np.memmap(eof_file, dtype=np.float64, offset=76, shape=(mmax+1,norder,4,numx+1,numy+1))
        mS = np.memmap(eof_file, dtype=np.float64, offset=76+(8*4*(mmax+1)*norder*(numx+1)*(numy+1)), shape=(mmax,norder,4,numx+1,numy+1))

    else:
        mC = np.memmap(eof_file, dtype=np.float64, offset=76, shape=(mmax+1,norder,3,numx+1,numy+1))
        mS = np.memmap(eof_file, dtype=np.float64, offset=76+(8*3*(mmax+1)*norder*(numx+1)*(numy+1)), shape=(mmax,norder,3,numx+1,numy+1))


    return mC,mS


def accumulated_eval_table(r, z, phi, accum_cos, accum_sin, eof_file, m1=0,m2=1000):
    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
    rmin,rmax,numx,numy,MMAX,norder,ascale,hscale,cmap,dens = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)
    if M2 == -1: M2 = MMAX+1
    fr = 0.0;
    fz = 0.0;
    fp = 0.0;
    p = 0.0;
    p0 = 0.0;
    d = 0.0;
    #
    # compute mappings
    #
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,ASCALE=ascale,HSCALE=hscale,CMAP=cmap)
    #
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    #
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;
    #
    for mm in range(0,MMAX+1):
        if (mm > m2) | (mm < m1): continue
        ccos = np.cos(phi*mm);
        ssin = np.sin(phi*mm);
        #
        fac = accum_cos[mm] * ccos;
        p += np.sum(fac * (potC[mm,:,ix,iy]*c00 + potC[mm,:,ix+1,iy  ]*c10 + potC[mm,:,ix,iy+1]*c01 + potC[mm,:,ix+1,iy+1]*c11));
        fr += np.sum(fac * (rforceC[mm,:,ix,iy] * c00 + rforceC[mm,:,ix+1,iy  ] * c10 + rforceC[mm,:,ix,iy+1] * c01 + rforceC[mm,:,ix+1,iy+1] * c11));
        fz += np.sum(fac * ( zforceC[mm,:,ix,iy] * c00 + zforceC[mm,:,ix+1,iy  ] * c10 + zforceC[mm,:,ix,iy+1] * c01 + zforceC[mm,:,ix+1,iy+1] * c11 ));
        d += np.sum(fac * (densC[mm,:,ix,iy]*c00 + densC[mm,:,ix+1,iy  ]*c10 + densC[mm,:,ix,iy+1]*c01 + densC[mm,:,ix+1,iy+1]*c11));
            #
        fac = accum_cos[mm] * ssin;
            #
        fp += np.sum(fac * mm * ( potC[mm,:,ix,iy] * c00 + potC[mm,:,ix+1,iy] * c10 + potC[mm,:,ix,iy+1] * c01 + potC[mm,:,ix+1,iy+1] * c11 ));
            #
        if (mm > 0):
                #
            fac = accum_sin[mm] * ssin;
                #
            p += np.sum(fac * (potS[mm,:,ix,iy]*c00 + potS[mm,:,ix+1,iy  ]*c10 + potS[mm,:,ix,iy+1]*c01 + potS[mm,:,ix+1,iy+1]*c11));
            fr += np.sum(fac * (rforceS[mm,:,ix,iy] * c00 + rforceS[mm,:,ix+1,iy  ] * c10 + rforceS[mm,:,ix,iy+1] * c01 + rforceS[mm,:,ix+1,iy+1] * c11));
            fz += np.sum(fac * ( zforceS[mm,:,ix,iy] * c00 + zforceS[mm,:,ix+1,iy  ] * c10 + zforceS[mm,:,ix,iy+1] * c01 + zforceS[mm,:,ix+1,iy+1] * c11 ));
            d += np.sum(fac * ( densS[mm,:,ix,iy] * c00 + densS[mm,:,ix+1,iy  ] * c10 + densS[mm,:,ix,iy+1] * c01 + densS[mm,:,ix+1,iy+1] * c11 ));
            fac = -accum_sin[mm] * ccos;
            fp += np.sum(fac * mm * ( potS[mm,:,ix,iy  ] * c00 + potS[mm,:,ix+1,iy  ] * c10 + potS[mm,:,ix,iy+1] * c01 + potS[mm,:,ix+1,iy+1] * c11 ))
                #
        if (mm==0): p0 = p;
    return p0,p,fr,fp,fz,d





def force_eval(r, z, phi, \
                       accum_cos, accum_sin, \
                       potC, rforceC, zforceC,\
                       potS, rforceS, zforceS,\
                       rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,\
                       MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0,no_odd=False,perturb=False):
    '''
    accumulated_forces: just like accumulated_eval, except only with forces

    inputs
    --------
    r,z,phi : positions in cylindrical coordinates
    accum_cos :
    accum_sin : 
    potC, rforceC, zforceC : cosine terms for the potential, radial force, and vertical force
    potS, rforceS, zforceS :   sine terms for the potential, radial force, and vertical force

    outputs
    --------
    fr    :   radial force (two-dimensional)
    fz    :   vertical force
    fp    :   azimuthal force
    p     :   potential
    p0    :   monopole potential
    
    '''

    # reduce the array sizes to the specified sizes
    accum_cos = accum_cos[0:MMAX+1,0:NMAX]
    accum_sin = accum_sin[0:MMAX+1,0:NMAX]

    
    potC    =    potC[0:MMAX+1,0:NMAX,:,:]
    rforceC = rforceC[0:MMAX+1,0:NMAX,:,:]
    zforceC = zforceC[0:MMAX+1,0:NMAX,:,:]
    potS    =    potS[0:MMAX+1,0:NMAX,:,:]
    rforceS = rforceS[0:MMAX+1,0:NMAX,:,:]
    zforceS = zforceS[0:MMAX+1,0:NMAX,:,:]

    #
    # compute mappings
    #
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
    #
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    #
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;

    # make numpy array for the sine and cosine terms
    morder = np.tile(np.arange(1.,MMAX+1.,1.),(NMAX,1)).T
    ccos = np.cos(phi*morder)
    ssin = np.sin(phi*morder)

    # make a mask to only do the even terms?
    if (no_odd):
        mask = abs(np.cos((np.pi/2.)*morder))

    else:
        mask = np.zeros_like(morder) + 1.

    #
    # modified 04-19-17 to be perturbation based.
    #
    
    fac  = accum_cos[1:,:] * ccos;
    p0   = np.sum(   accum_cos[0] *  (   potC[0,:,ix,iy] * c00 +    potC[0,:,ix+1,iy  ] * c10 +    potC[0,:,ix,iy+1] * c01 +    potC[0,:,ix+1,iy+1] * c11 ));
    p    = np.sum(mask * fac * (   potC[1:,:,ix,iy] * c00 +    potC[1:,:,ix+1,iy  ] * c10 +    potC[1:,:,ix,iy+1] * c01 +    potC[1:,:,ix+1,iy+1] * c11 ));
    fr   = np.sum(mask * fac * (rforceC[1:,:,ix,iy] * c00 + rforceC[1:,:,ix+1,iy  ] * c10 + rforceC[1:,:,ix,iy+1] * c01 + rforceC[1:,:,ix+1,iy+1] * c11 ));
    fz   = np.sum(mask * fac * (zforceC[1:,:,ix,iy] * c00 + zforceC[1:,:,ix+1,iy  ] * c10 + zforceC[1:,:,ix,iy+1] * c01 + zforceC[1:,:,ix+1,iy+1] * c11 ));

    fr0  = np.sum(accum_cos[0] * (rforceC[0,:,ix,iy] * c00 + rforceC[0,:,ix+1,iy  ] * c10 + rforceC[0,:,ix,iy+1] * c01 + rforceC[0,:,ix+1,iy+1] * c11 ));
    fz0  = np.sum(accum_cos[0] * (zforceC[0,:,ix,iy] * c00 + zforceC[0,:,ix+1,iy  ] * c10 + zforceC[0,:,ix,iy+1] * c01 + zforceC[0,:,ix+1,iy+1] * c11 ));

    # switch factor for azimuthal force
    fac = accum_cos[1:,:] * ssin;
    fp  = np.sum(mask * fac * morder * ( potC[1:,:,ix,iy] * c00 + potC[1:,:,ix+1,iy] * c10 + potC[1:,:,ix,iy+1] * c01 + potC[1:,:,ix+1,iy+1] * c11 ));

    # do sine terms
    fac = accum_sin[1:,:] * ssin;
                
    p  += np.sum(mask * fac * (   potS[1:,:,ix,iy] * c00 +    potS[1:,:,ix+1,iy  ] * c10 +    potS[1:,:,ix,iy+1] * c01 +    potS[1:,:,ix+1,iy+1] * c11 ));
    fr += np.sum(mask * fac * (rforceS[1:,:,ix,iy] * c00 + rforceS[1:,:,ix+1,iy  ] * c10 + rforceS[1:,:,ix,iy+1] * c01 + rforceS[1:,:,ix+1,iy+1] * c11 ));
    fz += np.sum(mask * fac * (zforceS[1:,:,ix,iy] * c00 + zforceS[1:,:,ix+1,iy  ] * c10 + zforceS[1:,:,ix,iy+1] * c01 + zforceS[1:,:,ix+1,iy+1] * c11 ));

    # switch factor for azimuthal force
    fac = -accum_sin[1:,:] * ccos;
    fp += np.sum(mask * fac * morder * ( potS[1:,:,ix,iy  ] * c00 + potS[1:,:,ix+1,iy  ] * c10 + potS[1:,:,ix,iy+1] * c01 + potS[1:,:,ix+1,iy+1] * c11 ))
                
    if perturb:
        return fr,fp,fz,p,p0,fr0,fz0
    
    else:
        return (fr+fr0),fp,(fz+fz0),p,p0

    #return (fr+fr0),fp,(fz+fz0),p,p0

def accumulated_eval(r, z, phi, accum_cos, accum_sin, potC, rforceC, zforceC, densC, potS, rforceS, zforceS, densS, rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0,no_odd=False):#, 	double &p0, double& p, double& fr, double& fz, double &fp)
    fr = 0.0;
    fz = 0.0;
    fp = 0.0;
    p = 0.0;
    p0 = 0.0;
    d = 0.0;
    d0 = 0.0;
    #
    # compute mappings
    #
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
    #
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    #
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;
    #
    for mm in range(0,MMAX+1):

        if ((mm % 2) != 0) & (no_odd):
            continue
        
        ccos = np.cos(phi*mm);
        ssin = np.sin(phi*mm);
        #
        fac = accum_cos[mm] * ccos;
        p += np.sum(fac * (potC[mm,:,ix,iy]*c00 + potC[mm,:,ix+1,iy  ]*c10 + potC[mm,:,ix,iy+1]*c01 + potC[mm,:,ix+1,iy+1]*c11));
        fr += np.sum(fac * (rforceC[mm,:,ix,iy] * c00 + rforceC[mm,:,ix+1,iy  ] * c10 + rforceC[mm,:,ix,iy+1] * c01 + rforceC[mm,:,ix+1,iy+1] * c11));
        fz += np.sum(fac * ( zforceC[mm,:,ix,iy] * c00 + zforceC[mm,:,ix+1,iy  ] * c10 + zforceC[mm,:,ix,iy+1] * c01 + zforceC[mm,:,ix+1,iy+1] * c11 ));
        d += np.sum(fac * (densC[mm,:,ix,iy]*c00 + densC[mm,:,ix+1,iy  ]*c10 + densC[mm,:,ix,iy+1]*c01 + densC[mm,:,ix+1,iy+1]*c11));
            #
        fac = accum_cos[mm] * ssin;
            #
        fp += np.sum(fac * mm * ( potC[mm,:,ix,iy] * c00 + potC[mm,:,ix+1,iy] * c10 + potC[mm,:,ix,iy+1] * c01 + potC[mm,:,ix+1,iy+1] * c11 ));
            #
        if (mm > 0):
                #
            fac = accum_sin[mm] * ssin;
                #
            p += np.sum(fac * (potS[mm,:,ix,iy]*c00 + potS[mm,:,ix+1,iy  ]*c10 + potS[mm,:,ix,iy+1]*c01 + potS[mm,:,ix+1,iy+1]*c11));
            fr += np.sum(fac * (rforceS[mm,:,ix,iy] * c00 + rforceS[mm,:,ix+1,iy  ] * c10 + rforceS[mm,:,ix,iy+1] * c01 + rforceS[mm,:,ix+1,iy+1] * c11));
            fz += np.sum(fac * ( zforceS[mm,:,ix,iy] * c00 + zforceS[mm,:,ix+1,iy  ] * c10 + zforceS[mm,:,ix,iy+1] * c01 + zforceS[mm,:,ix+1,iy+1] * c11 ));
            d += np.sum(fac * ( densS[mm,:,ix,iy] * c00 + densS[mm,:,ix+1,iy  ] * c10 + densS[mm,:,ix,iy+1] * c01 + densS[mm,:,ix+1,iy+1] * c11 ));
            fac = -accum_sin[mm] * ccos;
            fp += np.sum(fac * mm * ( potS[mm,:,ix,iy  ] * c00 + potS[mm,:,ix+1,iy  ] * c10 + potS[mm,:,ix,iy+1] * c01 + potS[mm,:,ix+1,iy+1] * c11 ))
                #
        if (mm==0):
            p0 = p;
            d0 = d;
    return p0,p,fr,fp,fz,d0,d





def accumulated_eval_contributions(r, z, phi, accum_cos, accum_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0):
    #
    # this accumulation method returns the value for the entire [M,N] matrix
    fr = np.zeros([2,MMAX+1,NMAX])
    fz = np.zeros([2,MMAX+1,NMAX])
    fp = np.zeros([2,MMAX+1,NMAX])
    p = np.zeros([2,MMAX+1,NMAX])
    #
    #rr = np.sqrt(r*r + z*z);
    X,Y,ix,iy = return_bins(r,z,rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
    #
    delx0 = ix + 1.0 - X;
    dely0 = iy + 1.0 - Y;
    delx1 = X - ix;
    dely1 = Y - iy;
    #
    c00 = delx0*dely0;
    c10 = delx1*dely0;
    c01 = delx0*dely1;
    c11 = delx1*dely1;
    #
    for mm in range(0,MMAX+1):
        ccos = np.cos(phi*mm);
        ssin = np.sin(phi*mm);
        #
        fac = accum_cos[mm] * ccos;
        p[0,mm] = (fac * (potC[mm,:,ix,iy]*c00 + potC[mm,:,ix+1,iy  ]*c10 + potC[mm,:,ix,iy+1]*c01 + potC[mm,:,ix+1,iy+1]*c11));
        fr[0,mm] = (fac * (rforceC[mm,:,ix,iy] * c00 + rforceC[mm,:,ix+1,iy  ] * c10 + rforceC[mm,:,ix,iy+1] * c01 + rforceC[mm,:,ix+1,iy+1] * c11));
        fz[0,mm] = (fac * ( zforceC[mm,:,ix,iy] * c00 + zforceC[mm,:,ix+1,iy  ] * c10 + zforceC[mm,:,ix,iy+1] * c01 + zforceC[mm,:,ix+1,iy+1] * c11 ));
            #
        fac = accum_cos[mm] * ssin;
            #
        fp[0,mm] = np.sum(fac * mm * ( potC[mm,:,ix,iy] * c00 + potC[mm,:,ix+1,iy] * c10 + potC[mm,:,ix,iy+1] * c01 + potC[mm,:,ix+1,iy+1] * c11 ));
            #
        if (mm > 0):
                #
            fac = accum_sin[mm] * ssin;
                #
            p[1,mm] += (fac * (potS[mm,:,ix,iy]*c00 + potS[mm,:,ix+1,iy  ]*c10 + potS[mm,:,ix,iy+1]*c01 + potS[mm,:,ix+1,iy+1]*c11));
            fr[1,mm] += (fac * (rforceS[mm,:,ix,iy] * c00 + rforceS[mm,:,ix+1,iy  ] * c10 + rforceS[mm,:,ix,iy+1] * c01 + rforceS[mm,:,ix+1,iy+1] * c11));
            fz[1,mm] += (fac * ( zforceS[mm,:,ix,iy] * c00 + zforceS[mm,:,ix+1,iy  ] * c10 + zforceS[mm,:,ix,iy+1] * c01 + zforceS[mm,:,ix+1,iy+1] * c11 ));
            fac = -accum_sin[mm] * ccos;
            fp[1,mm] += (fac * mm * ( potS[mm,:,ix,iy  ] * c00 + potS[mm,:,ix+1,iy  ] * c10 + potS[mm,:,ix,iy+1] * c01 + potS[mm,:,ix+1,iy+1] * c11 ))
                #
    return p,fr,fp,fz




#(holding,nprocs,a_cos,a_sin,potC,rforceC, zforceC,potS,rforceS,zforceS,XMIN,dX,YMIN,dY,numx,numy, mmax,norder)


#(Particles 1 , accum_cos2 , accum_sin 3, potC 4, rforceC 5, zforceC 6, potS 7, rforceS 8 , zforceS 9,rmin=0 10,dR=0 11,zmin=0 12,dZ=0 13,numx=0 14,numy=0 15,MMAX=6 16,NMAX=18)

def accumulated_eval_particles(Particles, accum_cos, accum_sin, \
                               potC=0, rforceC=0, zforceC=0, potS=0, rforceS=0, zforceS=0,\
                               rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0,m1=0,m2=1000,verbose=0,eof_file='',density=False):
    '''
    accumulated_eval_particles
        -takes a set of particles with standard PSP attributes (see psp_io documentation) and returns potential and forces.
        -wrapped to be the single-processor version

    inputs
    -------------
    Particles  :  psp_io-style input particles
    accum_cos  :  cosine component of the coefficients
    accum_sin  :  sine   component of the coefficients

    eof_file   :  if defined, will set all parameters in the block below

    (eof_file parameters)
    potC       :  cosine potential functions
    rforceC    :  cosine radial force functions
    zforceC    :  cosine vertical force functions
    potS       :  sine potential functions
    rforceS    :  sine radial force functions
    zforceS    :  sine vertical force functions
    rmin
    dR
    zmin
    dZ
    numx
    numy
    MMAX
    NMAX
    ASCALE
    HSCALE
    CMAP

    
    m1         :  minimum azimuthal order to include
    m2         :  maximum azimuthal order to include
    verbose    :  verbosity (1=print progress)

    outputs
    --------------
    p0         :    monopole potential   (m = 0)
    p          :    perturbing potential (m > 0)
    fr         :    radial force
    fp         :    azimuthal force
    fz         :    vertical force
    R          :    planar radius (x,y,z=0)

    '''
    dens = 0
    # add support for just using an eof_file input
    if eof_file != '':
        potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
        rmin,rmax,numx,numy,MMAX,NMAX,ASCALE,HSCALE,CMAP,dens = eof_params(eof_file)
        # overwrite rmin/zmin with the scaled values
        rmin,rmax,dR,zmin,zmax,dZ = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ASCALE,HSCALE=HSCALE,NUMX=numx,NUMY=numy,CMAP=CMAP)

    # check to see if density works and flag is properly set
    if (dens == 0) & (density == True):
        print('eof.accumulated_eval_particles: cannot compute density (functions not specified). moving on without...')
        density = False
    
    #
    #
    #
    norb = len(Particles.xpos)
    fr = np.zeros(norb);
    fz = np.zeros(norb);
    fp = np.zeros(norb)
    p = np.zeros(norb)
    p0 = np.zeros(norb)
    if density:
        d = np.zeros(norb)
        d0 = np.zeros(norb)
    #
    RR = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos + Particles.zpos*Particles.zpos+1.e-10)**0.5
    PHI = np.arctan2(Particles.ypos,Particles.xpos)
    R = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos + 1.e-10)**0.5
    #
    # cycle particles
    for part in range(0,norb):
        
        if (verbose > 0) & ( ((float(part)+1.) % 1000. == 0.0) | (part==0)): utils.print_progress(part,norb,'eof.accumulated_eval_particles')
            
        phi = PHI[part]
        X,Y,ix,iy = return_bins(R[part],Particles.zpos[part],rmin=rmin,dR=dR,zmin=zmin,dZ=dZ,numx=numx,numy=numy,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP)
        #
        delx0 = ix + 1.0 - X;
        dely0 = iy + 1.0 - Y;
        delx1 = X - ix;
        dely1 = Y - iy;
        #
        c00 = delx0*dely0;
        c10 = delx1*dely0;
        c01 = delx0*dely1;
        c11 = delx1*dely1;
        #

        for mm in range(0,MMAX+1):
            
            if (mm > m2) | (mm < m1):
                continue
            
            ccos = np.cos(phi*mm);
            ssin = np.sin(phi*mm);
            
            
            fac = accum_cos[mm] * ccos;
            p[part]  += np.sum(fac * (   potC[mm,:,ix,iy] * c00 +    potC[mm,:,ix+1,iy  ] * c10 +    potC[mm,:,ix,iy+1] * c01 +    potC[mm,:,ix+1,iy+1] * c11))
            fr[part] += np.sum(fac * (rforceC[mm,:,ix,iy] * c00 + rforceC[mm,:,ix+1,iy  ] * c10 + rforceC[mm,:,ix,iy+1] * c01 + rforceC[mm,:,ix+1,iy+1] * c11))
            fz[part] += np.sum(fac * (zforceC[mm,:,ix,iy] * c00 + zforceC[mm,:,ix+1,iy  ] * c10 + zforceC[mm,:,ix,iy+1] * c01 + zforceC[mm,:,ix+1,iy+1] * c11))

            if density: d[part] += np.sum(fac * (   densC[mm,:,ix,iy] * c00 +    densC[mm,:,ix+1,iy  ] * c10 +    densC[mm,:,ix,iy+1] * c01 +    densC[mm,:,ix+1,iy+1] * c11))
                                              
            fac = accum_cos[mm] * ssin;
            
            fp[part] += np.sum(fac * mm * ( potC[mm,:,ix,iy] * c00 + potC[mm,:,ix+1,iy] * c10 + potC[mm,:,ix,iy+1] * c01 + potC[mm,:,ix+1,iy+1] * c11 ));



            # accumulate sine terms
            if (mm > 0):
                #
                fac = accum_sin[mm] * ssin;
                #
                p[part] += np.sum(fac * (potS[mm,:,ix,iy]*c00 + potS[mm,:,ix+1,iy  ]*c10 + potS[mm,:,ix,iy+1]*c01 + potS[mm,:,ix+1,iy+1]*c11));
                fr[part] += np.sum(fac * (rforceS[mm,:,ix,iy] * c00 + rforceS[mm,:,ix+1,iy  ] * c10 + rforceS[mm,:,ix,iy+1] * c01 + rforceS[mm,:,ix+1,iy+1] * c11));
                fz[part] += np.sum(fac * ( zforceS[mm,:,ix,iy] * c00 + zforceS[mm,:,ix+1,iy  ] * c10 + zforceS[mm,:,ix,iy+1] * c01 + zforceS[mm,:,ix+1,iy+1] * c11 ));
                if density: d[part] += np.sum(fac * (densS[mm,:,ix,iy]*c00 + densS[mm,:,ix+1,iy  ]*c10 + densS[mm,:,ix,iy+1]*c01 + densS[mm,:,ix+1,iy+1]*c11));

                fac = -accum_sin[mm] * ccos;
                fp[part] += np.sum(fac * mm * ( potS[mm,:,ix,iy  ] * c00 + potS[mm,:,ix+1,iy  ] * c10 + potS[mm,:,ix,iy+1] * c01 + potS[mm,:,ix+1,iy+1] * c11 ))
                #

                
            if (mm==0):
                
                p0[part] = p[part]
                
                # reset for perturbing potential
                p[part]  = 0.

                if density:
                    d0[part] = d[part]
                    d[part]  = 0.

    if density:
        return p0,p,d0,d,fr,fp,fz,R
    
    else:
        return p0,p,fr,fp,fz,R




    

############################################################################################
#
# DO ALL
#
############################################################################################


def compute_coefficients(PSPInput,eof_file,verbose=1,no_odd=False):

    EOF_Out = EOF_Object()
    EOF_Out.time = PSPInput.time
    EOF_Out.dump = PSPInput.infile
    EOF_Out.comp = PSPInput.comp
    EOF_Out.nbodies = PSPInput.mass.size
    EOF_Out.eof_file = eof_file
    
    nprocs = multiprocessing.cpu_count()
    
    if verbose > 1:
        rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(eof_file,verbose=1)

    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)

    EOF_Out.mmax = mmax # don't forget +1 for array size
    EOF_Out.nmax = norder
    
    if nprocs > 1:
        a_cos,a_sin = make_coefficients_multi(PSPInput,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=verbose,no_odd=no_odd)
        
    else:
        # do the accumulation call, not implemented yet
        print 'eof.compute_coefficients: This definition has not yet been generalized to take a single processor.'
        a_cos = 0
        a_sin = 0

    EOF_Out.cos = a_cos
    EOF_Out.sin = a_sin
        
    return EOF_Out


def compute_forces(PSPInput,EOF_Object,verbose=1,nprocs=-1,m1=0,m2=1000):
    '''
    compute_forces
        main wrapper for computing EOF forces



    '''
    if nprocs == -1:
        nprocs = multiprocessing.cpu_count()
    
    if verbose > 1:
        eof_quickread(EOF_Object.eof_file)

    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(EOF_Object.eof_file)
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(EOF_Object.eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)

    if nprocs > 1:
        p0,p,fr,fp,fz,r = find_forces_multi(PSPInput,nprocs,EOF_Object.cos,EOF_Object.sin,potC,rforceC, zforceC,potS,rforceS,zforceS,XMIN,dX,YMIN,dY,numx,numy, mmax,norder,ascale,hscale,cmap,m1=m1,m2=m2,verbose=verbose)

    else:
        p0,p,fr,fp,fz,r = accumulated_eval_particles(PSPInput, EOF_Object.cos, EOF_Object.sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder,ASCALE=ascale,HSCALE=hscale,CMAP=cmap,m1=m1,m2=m2,verbose=verbose)
         

    return p0,p,fr,fp,fz,r



#
# MULTIPROCESSING BLOCK
#

# for multiprocessing help see
# http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments



#
# set up particle structure for multiprocessing distribution
#
def redistribute_particles(ParticleInstance,divisions):
    npart = np.zeros(divisions)
    holders = [psp_io.particle_holder() for x in range(0,divisions)]
    average_part = int(np.floor(len(ParticleInstance.xpos)/divisions))
    first_partition = len(ParticleInstance.xpos) - average_part*(divisions-1)
    #print average_part, first_partition
    low_particle = 0
    for i in range(0,divisions):
        end_particle = low_particle+average_part
        if i==0: end_particle = low_particle+first_partition
        #print low_particle,end_particle
        holders[i].xpos = ParticleInstance.xpos[low_particle:end_particle]
        holders[i].ypos = ParticleInstance.ypos[low_particle:end_particle]
        holders[i].zpos = ParticleInstance.zpos[low_particle:end_particle]
        holders[i].mass = ParticleInstance.mass[low_particle:end_particle]
        low_particle = end_particle
    return holders




def accumulate_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return accumulate(*a_b)


def multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=0,no_odd=False):
    pool = multiprocessing.Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = potC
    third_arg = potS
    fourth_arg = mmax
    fifth_arg = norder
    sixth_arg = XMIN
    seventh_arg = dX
    eighth_arg = YMIN
    ninth_arg = dY
    tenth_arg = numx
    eleventh_arg = numy
    twelvth_arg = ascale
    thirteenth_arg = hscale
    fourteenth_arg = cmap
    fifteenth_arg = [ 0 for i in range(0,nprocs)]
    fifteenth_arg[0] = verbose
    sixteenth_arg = no_odd
    a_coeffs = pool.map(accumulate_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                                                                itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),\
                                                                itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),\
                                                                itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),itertools.repeat(twelvth_arg),\
                                                                itertools.repeat(thirteenth_arg),itertools.repeat(fourteenth_arg),fifteenth_arg,\
                                                                itertools.repeat(sixteenth_arg) \
                                                                ))
    pool.close()
    pool.join()                                                        
    return a_coeffs



def make_coefficients_multi(ParticleInstance,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=0,no_odd=False):
    '''
    make_coefficients_multi

    master process to distribute particles for accumulation


    '''
    holding = redistribute_particles(ParticleInstance,nprocs)
    

    if (verbose):
        print 'eof.make_coefficients_multi: %i processors, %i particles each.' %(nprocs,len(holding[0].mass))

    # start timer    
    t1 = time.time()
    multiprocessing.freeze_support()
    
    a_coeffs = multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=verbose,no_odd=no_odd)
    
    if (verbose):
        print 'eof.make_coefficients_multi: Accumulation took %3.2f seconds, or %4.2f microseconds per orbit.' \
          %(time.time()-t1, 1.e6*(time.time()-t1)/len(ParticleInstance.mass))

    # sum over processes
    scoefs = np.sum(np.array(a_coeffs),axis=0)
    
    a_cos = scoefs[0]
    a_sin = scoefs[1]
    
    return a_cos,a_sin




def accumulated_eval_particles_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return accumulated_eval_particles(*a_b)


def multi_accumulated_eval(holding,nprocs,a_cos,a_sin,potC,rforceC, zforceC,potS,rforceS,zforceS,XMIN,dX,YMIN,dY,numx,numy, mmax,norder,ascale,hscale,cmap,m1,m2,verbose=0):
    pool = multiprocessing.Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = a_cos
    third_arg = a_sin
    fourth_arg = potC
    fifth_arg = rforceC
    sixth_arg = zforceC
    seventh_arg = potS
    eighth_arg = rforceS
    ninth_arg = zforceS
    tenth_arg = XMIN
    eleventh_arg = dX
    twelvth_arg = YMIN
    thirteenth_arg = dY
    fourteenth_arg = numx
    fifteenth_arg = numy
    sixteenth_arg = mmax
    seventeenth_arg = norder
    eighteenth_arg = ascale
    nineteenth_arg = hscale
    twentieth_arg = cmap
    twentyfirst_arg = m1
    twentysecond_arg = m2
    twentythird_arg = [0 for i in range(0,nprocs)]
    twentythird_arg[0] = verbose
    a_vals = pool.map(accumulated_eval_particles_star,\
                         itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                         itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),\
                         itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),\
                         itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),itertools.repeat(twelvth_arg),\
                         itertools.repeat(thirteenth_arg),itertools.repeat(fourteenth_arg),itertools.repeat(fifteenth_arg),\
                         itertools.repeat(sixteenth_arg),itertools.repeat(seventeenth_arg),itertools.repeat(eighteenth_arg),\
                         itertools.repeat(nineteenth_arg),itertools.repeat(twentieth_arg),itertools.repeat(twentyfirst_arg),itertools.repeat(twentysecond_arg),\
                         twentythird_arg))
    pool.close()
    pool.join()
    return a_vals 



def find_forces_multi(ParticleInstance,nprocs,a_cos,a_sin,potC,rforceC, zforceC,potS,rforceS,zforceS,XMIN,dX,YMIN,dY,numx,numy, mmax,norder,ascale,hscale,cmap,m1=0,m2=1000,verbose=0):
    holding = redistribute_particles(ParticleInstance,nprocs)
    t1 = time.time()
    multiprocessing.freeze_support()
    a_vals = multi_accumulated_eval(holding,nprocs,a_cos,a_sin,potC,rforceC, zforceC,potS,rforceS,zforceS,XMIN,dX,YMIN,dY,numx,numy, mmax,norder,ascale,hscale,cmap,m1=0,m2=1000,verbose=verbose)
    if (verbose): print 'eof.find_forces_multi: Force Evaluation took %3.2f seconds, or %4.2f microseconds per orbit.' %(time.time()-t1, 1.e6*(time.time()-t1)/len(ParticleInstance.mass))
    # accumulate over processes
    p0,p,fr,fp,fz,r = mix_outputs(np.array(a_vals))
    return p0,p,fr,fp,fz,r


#
# helper class for making torques
# 
def mix_outputs(MultiOutput):
    n_instances = len(MultiOutput)
    n_part = 0
    for i in range(0,n_instances):
        n_part += len(MultiOutput[i][0])
    full_p0 = np.zeros(n_part)
    full_p = np.zeros(n_part)
    full_fr = np.zeros(n_part)
    full_fp = np.zeros(n_part)
    full_fz = np.zeros(n_part)
    full_r = np.zeros(n_part)
    #
    #
    first_part = 0
    for i in range(0,n_instances):
        n_instance_part = len(MultiOutput[i][0])
        full_p0[first_part:first_part+n_instance_part] = MultiOutput[i][0]
        full_p [first_part:first_part+n_instance_part] = MultiOutput[i][1]
        full_fr[first_part:first_part+n_instance_part] = MultiOutput[i][2]
        full_fp[first_part:first_part+n_instance_part] = MultiOutput[i][3]
        full_fz[first_part:first_part+n_instance_part] = MultiOutput[i][4]
        full_r [first_part:first_part+n_instance_part] = MultiOutput[i][5]
        first_part += n_instance_part
    return full_p0,full_p,full_fr,full_fp,full_fz,full_r




    
#
# some little helper functions: to be broken out
#


def radial_slice(rvals,a_cos, a_sin,eof_file,z=0.0,phi=0.0):
    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)
    p  = np.zeros_like(rvals)
    fr = np.zeros_like(rvals)
    fp = np.zeros_like(rvals)
    fz = np.zeros_like(rvals)
    d  = np.zeros_like(rvals)
    for i in range(0,len(rvals)):
         p0,p_tmp,fr_tmp,fp_tmp,fz_tmp,d_tmp = accumulated_eval(rvals[i], z, phi, a_cos, a_sin, potC, rforceC, zforceC, densC, potS, rforceS, zforceS, densS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder,ASCALE=ascale,HSCALE=hscale,CMAP=cmap)
         p[i]  =   p_tmp
         fr[i] = -fr_tmp
         fp[i] =  fp_tmp
         fz[i] =  fz_tmp
         d[i]  =   d_tmp
    return p,fr,fp,fz,d



########################################################################################
#
# the tools to save eof coefficient files
#

# make an eof object to carry around interesting bits of data
class EOF_Object(object):
    time = None
    dump = None
    comp = None
    nbodies = None
    mmax = None
    nmax = None
    eof_file = None
    cos  = None  # the cosine coefficient array
    sin  = None  # the sine coefficient array



# and a dictionary for storing multiple EOF_Objects?
#    based on times
    



def eof_coefficients_to_file(f,EOF_Object):

    np.array([EOF_Object.time],dtype='f4').tofile(f)
    np.array([EOF_Object.dump],dtype='S100').tofile(f)
    np.array([EOF_Object.comp],dtype='S8').tofile(f)
    np.array([EOF_Object.nbodies],dtype='i4').tofile(f)
    np.array([EOF_Object.eof_file],dtype='S100').tofile(f)
    # 4+100+8+100 = 216 bytes to here
    
    np.array([EOF_Object.mmax,EOF_Object.nmax],dtype='i4').tofile(f)
    # 4x2 = 8 bytes
    
    np.array(EOF_Object.cos.reshape(-1,),dtype='f8').tofile(f)
    np.array(EOF_Object.sin.reshape(-1,),dtype='f8').tofile(f)
    # 8 bytes X 2 arrays x (m+1) x n = 16(m+1)n bytes to end of array
    

# wrap the coefficients to file
def save_eof_coefficients(outfile,EOF_Object,verbose=0):

    # check to see if file exists
    try:
        f = open(outfile,'rb+')
        f.close()
        
    except:
        f = open(outfile,'wb')
        np.array([0],dtype='i4').tofile(f)
        f.close()
        
    
    f = open(outfile,'rb+')

    ndumps = np.memmap(outfile,dtype='i4',shape=(1))
    ndumps += 1
    ndumps.flush() # update the lead value

    if verbose: print 'eof.save_eof_coefficients: coefficient file currently has %i dumps.' %ndumps

    # seek to the correct position
    # EOF_Object must have the same size as previous dumps...
    f.seek(4 + (ndumps-1)*(16*(EOF_Object.mmax+1)*(EOF_Object.nmax)+224) )

    eof_coefficients_to_file(f,EOF_Object)

    f.close()


def restore_eof_coefficients(infile):

    try:
        f = open(infile,'rb')
    except:
        print 'eof.restore_eof_coefficients: no infile of that name exists.'

    f.seek(0)
    [ndumps] = np.fromfile(f,dtype='i4',count=1)
    
    f.seek(4)

    #
    # this is a little sloppy right now, but it works
    #
    EOF_Dict = OrderedDict()

    for step in range(0,ndumps):

        try:
            EOF_Out = extract_eof_coefficients(f)

            EOF_Dict[EOF_Out.time] = EOF_Out

        except:
            pass

    f.close()

    return EOF_Out,EOF_Dict


    
def extract_eof_coefficients(f):
    # operates on an open file
    EOF_Obj = EOF_Object()


    [EOF_Obj.time] = np.fromfile(f,dtype='f4',count=1)
    [EOF_Obj.dump] = np.fromfile(f,dtype='S100',count=1)
    [EOF_Obj.comp] = np.fromfile(f,dtype='S8',count=1)
    [EOF_Obj.nbodies] = np.fromfile(f,dtype='i4',count=1)
    [EOF_Obj.eof_file] = np.fromfile(f,dtype='S100',count=1)
    
    [EOF_Obj.mmax,EOF_Obj.nmax] = np.fromfile(f,dtype='i4',count=2)
    cosine_flat = np.fromfile(f,dtype='f8',count=(EOF_Obj.mmax+1)*EOF_Obj.nmax)
    sine_flat = np.fromfile(f,dtype='f8',count=(EOF_Obj.mmax+1)*EOF_Obj.nmax)

    EOF_Obj.cos = cosine_flat.reshape([(EOF_Obj.mmax+1),EOF_Obj.nmax])
    EOF_Obj.sin = sine_flat.reshape([(EOF_Obj.mmax+1),EOF_Obj.nmax])
    
    return EOF_Obj


##########################################################################################
#
# add ability to parse PSP files for setup in accumulation
#
#


def parse_components(simulation_directory,simulation_name,output_number):

    # set up a dictionary to hold the details
    ComponentDetails = {}

    PSP = psp_io.Input(simulation_directory+'OUT.'+simulation_name+'.%05i' %output_number,validate=True)

    for comp_num in range(0,PSP.ncomp):

        # find components that have cylindrical matches
        if PSP.comp_expansions[comp_num] == 'cylinder':

            # set up a dictionary based on the component name
            ComponentDetails[PSP.comp_titles[comp_num]] = {}

            # population dictionary with desirables
            ComponentDetails[PSP.comp_titles[comp_num]]['nbodies'] = PSP.comp_nbodies[comp_num]

            # break basis string for eof_file
            broken_basis = PSP.comp_basis[comp_num].split(',')
            broken_basis = [v.strip() for v in broken_basis] # rip out spaces
            basis_dict = {}
            for value in broken_basis:  basis_dict[value.split('=')[0]] = value.split('=')[1]

            # ^^^
            # ideally this will be populated with defaults as well so that all values used are known

            try:
                ComponentDetails[PSP.comp_titles[comp_num]]['eof_file'] = simulation_directory+basis_dict['eof_file']
                
            except:
                print('eof.parse_components: Component %s has no EOF file specified (setting None).' %PSP.comp_titles[comp_num])
                ComponentDetails[PSP.comp_titles[comp_num]]['eof_file'] = None

    return ComponentDetails


#
# visualizing routines
#

def make_eof_wake(EOFObj,exclude=False,orders=None,m1=0,m2=1000,xline = np.linspace(-0.03,0.03,75),zaspect=1.,zoffset=0.,coord='Y',axis=False):
    '''
    make_eof_wake: evaluate a simple grid of points along an axis

    inputs
    ---------
    EOFObj: 





    '''
    #     now a simple grid
    #
    # this will always be square in resolution--could think how to change this?
    zline = xline*zaspect
    xgrid,ygrid = np.meshgrid(xline,zline)


    if axis:
        zline = np.array([0.])
        xgrid = xline[np.where(xline>=0.)[0]]
        xline = xgrid
        ygrid = np.array([0.])
    
    #
    P = psp_io.particle_holder()
    P.xpos = xgrid.reshape(-1,)

    # set the secondary coordinate
    if coord=='Y':
        P.ypos = ygrid.reshape(-1,)
        P.zpos = np.zeros(xline.shape[0]*zline.shape[0]) + zoffset

    if coord=='Z':
        P.ypos = np.zeros(xline.shape[0]*zline.shape[0]) + zoffset
        P.zpos = ygrid.reshape(-1,)
        
    P.mass = np.zeros(xline.shape[0]*zline.shape[0]) # mass doesn't matter for evaluations, just get field values
    #
    #
    cos_coefs_in = np.copy(EOFObj.cos)
    sin_coefs_in = np.copy(EOFObj.sin)
    #
    if exclude:
        for i in orders:
            cos_coefs_in[i] = np.zeros(EOFObj.nmax)
            sin_coefs_in[i] = np.zeros(EOFObj.nmax)
    #
    p0,p,d0,d,fr,fp,fz,R = accumulated_eval_particles(P, cos_coefs_in, sin_coefs_in,m1=m1,m2=m2,eof_file=EOFObj.eof_file,density=True)
    #
    #
    wake = {}
    wake['X'] = xgrid
    wake['Y'] = ygrid

    if zline.shape[0] > 1:
        wake['P'] = p.reshape([xline.shape[0],zline.shape[0]])
        wake['D'] = d.reshape([xline.shape[0],zline.shape[0]])
        wake['fR'] = fr.reshape([xline.shape[0],zline.shape[0]])
        wake['R'] = R.reshape([xline.shape[0],zline.shape[0]])
        wake['fP'] = fp.reshape([xline.shape[0],zline.shape[0]])
        wake['fZ'] = fz.reshape([xline.shape[0],zline.shape[0]])

    else:
        wake['P'] = p
        wake['D'] = d
        wake['fR'] = fr
        wake['R'] = R
        wake['fP'] = fp
        wake['fZ'] = fz

        
    return wake



#
# add eof visualizers
#


def reorganize_eof_dict(EOFDict):
    # reorganize
    dipole_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    quadrupole_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    #
    octopole_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    four_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    five_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    six_sum = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    #
    monopole_tally = np.zeros([np.array(EOFDict.keys()).shape[0],12])
    monopole_sum = np.zeros(np.array(EOFDict.keys()).shape[0])
    time_order = np.zeros(np.array(EOFDict.keys()).shape[0])
    #
    num = 0
    for keyval in EOFDict.keys():
        for j in range(0,12):
            dipole_sum[num,j] = EOFDict[keyval].cos[1,j]**2. + EOFDict[keyval].sin[1,j]**2.
            quadrupole_sum[num,j] = EOFDict[keyval].cos[2,j]**2. + EOFDict[keyval].sin[2,j]**2.
            octopole_sum[num,j] = EOFDict[keyval].cos[3,j]**2. + EOFDict[keyval].sin[3,j]**2.
            four_sum[num,j] = EOFDict[keyval].cos[4,j]**2. + EOFDict[keyval].sin[4,j]**2.
            five_sum[num,j] = EOFDict[keyval].cos[5,j]**2. + EOFDict[keyval].sin[5,j]**2.
            six_sum[num,j] = EOFDict[keyval].cos[6,j]**2. + EOFDict[keyval].sin[6,j]**2.
            monopole_tally[num,j] = EOFDict[keyval].cos[0,j]**2.
        time_order[num] = EOFDict[keyval].time
        monopole_sum[num] = np.sum(EOFDict[keyval].cos[0]**2.)
        num += 1
    # assemble into dictionary
    CDict = {}
    CDict['time']   = time_order[time_order.argsort()]
    CDict['total0'] = monopole_tally[time_order.argsort(),:]
    CDict['total1'] = dipole_sum[time_order.argsort(),:]
    CDict['total2'] = quadrupole_sum[time_order.argsort(),:]
    CDict['total3'] = octopole_sum[time_order.argsort(),:]
    CDict['total4'] = four_sum[time_order.argsort(),:]
    CDict['total5'] = five_sum[time_order.argsort(),:]
    CDict['total6'] = six_sum[time_order.argsort(),:]
    CDict['sum0'] = np.sum(monopole_tally,axis=1)[time_order.argsort()]
    CDict['sum1'] = np.sum(dipole_sum,axis=1)[time_order.argsort()]
    CDict['sum2'] = np.sum(quadrupole_sum,axis=1)[time_order.argsort()]
    CDict['sum3'] = np.sum(octopole_sum,axis=1)[time_order.argsort()]
    CDict['sum4'] = np.sum(four_sum,axis=1)[time_order.argsort()]
    CDict['sum5'] = np.sum(five_sum,axis=1)[time_order.argsort()]
    CDict['sum6'] = np.sum(six_sum,axis=1)[time_order.argsort()]
    return CDict



def calculate_eof_phase(EOFDict):
    '''
    working phase calculations


    '''
    mmax=EOFDict[np.array(EOFDict.keys())[0]].mmax
    nmax=EOFDict[np.array(EOFDict.keys())[0]].nmax
    #
    phases = np.zeros([mmax+1,np.array(EOFDict.keys()).shape[0],nmax])
    netphases = np.zeros([mmax+1,np.array(EOFDict.keys()).shape[0]])
    time_order = np.zeros(np.array(EOFDict.keys()).shape[0])
    #
    num = 0
    for keyval in EOFDict.keys():
        for mterm in range(1,mmax+1):
            for nterm in range(0,nmax):
                phases[mterm,num,nterm] = np.arctan2(EOFDict[keyval].sin[mterm,nterm],EOFDict[keyval].cos[mterm,nterm])
            #
            netphases[mterm,num] = np.arctan2(np.sum(EOFDict[keyval].sin[mterm,:]),np.sum(EOFDict[keyval].cos[mterm,:]))
        time_order[num] = EOFDict[keyval].time
        num += 1
    #
    DC = {}
    DC['time'] = time_order[time_order.argsort()]
    #
    keys = ['phase'+str(int(x)) for x in range(1,mmax+1)]
    #
    for indx,key in enumerate(keys):
        DC[key] = phases[indx+1,time_order.argsort(),:]
        DC['net'+key] = netphases[indx+1,time_order.argsort()]
    #
    skeys = ['speed'+str(int(x)) for x in range(1,mmax+1)]
    #
    for indx,skey in enumerate(skeys):

        # if desired, could put in blocks for unreasonable values here?
        #goodphase = np.where( DC[keys[indx]][:,nterm] )
        
        DC[skey] = np.zeros([np.array(EOFDict.keys()).shape[0],nmax])
        for nterm in range(0,nmax):
            DC[skey][:,nterm] = np.ediff1d(utils.savitzky_golay(utils.unwrap_phase(DC[keys[indx]][:,nterm],tol=-1.5*np.pi,clock=False),101,1),to_begin=0.)/np.ediff1d(DC['time'],to_begin=100.)
        DC['net'+skey] = np.ediff1d(utils.savitzky_golay(utils.unwrap_phase(DC['net'+keys[indx]],tol=-1.5*np.pi,clock=False),101,1),to_begin=0.)/np.ediff1d(DC['time'],to_begin=100.)
    #
    return DC


