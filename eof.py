
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

# exptool definitions
import utils


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

        print 'The parameters for this EOF file are:'
        print 'MMAX=%i' %mmax
        print 'NORDER=%i' %norder
        print 'NMAX=%i' %nmax
        print 'NUMX,NUMY=%i,%i' %(numx,numy)
        print 'DENS,CMAP=%i,%i' %(dens,cmap)
        print 'ASCALE,HSCALE=%5.4f,%5.4f' %(ascale,hscale)
        print 'CYLMASS=%5.4f' %cylmass
        print 'TNOW=%5.4f' %tnow
        
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
    f = open(file,'rb')
    #
    #
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(file,verbose=0)
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
                    
    return potC,rforcec,zforcec,densc,potS,rforces,zforces,denss



def set_table_params(rmin=0.001,rmax=1.,ascale=0.01,hscale=0.001,numx=128,numy=64,CMAP=0):
    '''
    set_table_params
        calculate scaled boundary values for the parameter table

    inputs
    -------


    returns
    -------
    

    '''
    Rtable  = np.sqrt(0.5) * rmax
    
    # check cmap, but if cmap=0, r_to_xi = r
    #
    # otherwise, r = (r/ASCALE - 1.0)/(r/ASCALE + 1.0);

    # calculate radial scalings
    XMIN    = r_to_xi(rmin*ascale,ascale,CMAP);
    XMAX    = r_to_xi(Rtable*ascale,ascale=ascale,cmap=CMAP);
    dX      = (XMAX - XMIN)/numx;

    # calculate vertical scalings
    YMIN    = z_to_y(-Rtable*ascale,hscale=hscale);
    YMAX    = z_to_y( Rtable*ascale,hscale=hscale);
    dY      = (YMAX - YMIN)/numy;
        
    return XMIN,XMAX,dX,YMIN,YMAX,dY



#
# mapping definitions
#
def z_to_y(z,hscale):
    return z/(abs(z)+1.e-10)*np.arcsinh(abs(z/hscale))

def r_to_xi(r,ascale,cmap=0):
    if (cmap):
        return (r/ascale - 1.0)/(r/ascale + 1.0);
    else:
        return r



#
# particle accumulation definitions
#
def return_bins(r,z,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,ASCALE=0.01,HSCALE=0.001,CMAP=0):
    #
    # routine to return the integer bin numbers based on dimension mapping
    # 
    X = (r_to_xi(r,ascale=ASCALE,cmap=CMAP) - rmin)/dR
    Y = (z_to_y(z,hscale=HSCALE) - zmin)/dZ
    ix = int( np.floor((r_to_xi(r,ascale=ASCALE,cmap=CMAP) - rmin)/dR) )
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
    for mm in range(0,MMAX+1):
        #
        Vc[mm] = fac * ( cos_array[mm,:,ix,iy] * c00 + cos_array[mm,:,ix+1,iy] * c10 + cos_array[mm,:,ix,iy+1] * c01 + cos_array[mm,:,ix+1,iy+1] * c11 )
        #
        if (mm>0):
            Vs[mm] = fac * ( sin_array[mm,:,ix,iy] * c00 + sin_array[mm,:,ix+1,iy] * c10 + sin_array[mm,:,ix,iy+1] * c01 + sin_array[mm,:,ix+1,iy+1] * c11 );
    return Vc,Vs



# BROKEN
def get_pot_single_m(r,z,cos_array,sin_array,MORDER,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,NMAX=18):#Matrix& Vc, Matrix& Vs, double r, double z)
    #
    # returns potential fields for C and S to calculate weightings during accumulation
    #
    #
    # define boundaries of the interpolation scheme
    #
    #
    # find the corresponding bins
    X = (r - rmin)/dR
    Y = (z_to_y(z) - zmin)/dZ
    ix = int( np.floor((r - rmin)/dR) )
    iy = int( np.floor((z_to_y(z) - zmin)/dZ) )
    #print X,Y
    #
    # check the boundaries and set guards
    if X < 0: X = 0
    if X > numx: X = numx - 1
    if Y < 0: Y = 0
    if Y > numy: Y = numy - 1
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
    Vc = np.zeros([NMAX])
    Vs = np.zeros([NMAX])
    Vc = fac * ( cos_array[MORDER][:][X  ][Y  ] * c00 + cos_array[MORDER][:][X+1][Y  ] * c10 + cos_array[MORDER][:][X  ][Y+1] * c01 + cos_array[MORDER][:][X+1][Y+1] * c11 )
    if (mm>0):
        Vs = fac * ( sin_array[MORDER][:][X  ][Y  ] * c00 + sin_array[MORDER][:][X+1][Y] * c10 + sin_array[MORDER][:][X  ][Y+1] * c01 + sin_array[MORDER][:][X+1][Y+1] * c11 );
    return Vc,Vs



# BROKEN
def get_pot_single_term(r,z,cos_array,sin_array,mm,nn,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0):#Matrix& Vc, Matrix& Vs, double r, double z)
    #
    # returns potential fields for C and S to calculate weightings during accumulation
    #
    #
    # define boundaries of the interpolation scheme
    #
    #
    # find the corresponding bins
    X = (r - rmin)/dR
    Y = (z_to_y(z) - zmin)/dZ
    ix = int( np.floor((r - rmin)/dR) )
    iy = int( np.floor((z_to_y(z) - zmin)/dZ) )
    print X,Y
    #
    # check the boundaries and set guards
    if X < 0: X = 0
    if X > numx: X = numx - 1
    if Y < 0: Y = 0
    if Y > numy: Y = numy - 1
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
    #print mm,nn
    Vc = fac * ( cos_array[mm][nn][X  ][Y  ] * c00 + cos_array[mm][nn][X+1][Y  ] * c10 + cos_array[mm][nn][X  ][Y+1] * c01 + cos_array[mm][nn][X+1][Y+1] * c11 )
    Vs = 0.0
    if (mm>0):
        Vs = fac * ( sin_array[mm][nn][X  ][Y  ] * c00 + sin_array[mm][nn][X+1][Y] * c10 + sin_array[mm][nn][X  ][Y+1] * c01 + sin_array[mm][nn][X+1][Y+1] * c11 );
    return Vc,Vs



def accumulate(ParticleInstance,potC,potS,MMAX,NMAX,XMIN,dX,YMIN,dY,NUMX,NUMY,ASCALE,HSCALE,CMAP,verbose=0):
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
        #
        r = (ParticleInstance.xpos[n]**2. + ParticleInstance.ypos[n]**2.)**0.5
        phi = np.arctan2(ParticleInstance.ypos[n],ParticleInstance.xpos[n])
        vc,vs = get_pot(r, ParticleInstance.zpos[n], potC,potS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=NUMX,numy=NUMY,fac=1.0,MMAX=MMAX,NMAX=NMAX,ASCALE=ASCALE,HSCALE=HSCALE,CMAP=CMAP);
        for mm in range(0,MMAX+1):
            #
            #
            mcos = np.cos(phi*mm);
            accum_cos[mm] += norm * ParticleInstance.mass[n] * mcos * vc[mm] 
            #
            #
            if (mm>0):
                msin = np.sin(phi*mm);
                accum_sin[mm] += norm * ParticleInstance.mass[n] * msin * vs[mm]               
    return accum_cos,accum_sin



# BROKEN
def accumulate_single_m(ParticleInstance,potC,potS,MORDER,NMAX,XMIN,dX,YMIN,dY,NUMX,NUMY):
    #
    # take all the particles and stuff them into the basis
    #
    nparticles = 0
    cylmass = 0.0
    norm = -4.*np.pi
    SELECT=False
    #
    # set up particles
    #
    norb = len(ParticleInstance.mass)
    accum_cos = np.zeros([NMAX])
    accum_sin = np.zeros([NMAX])
    mcos = np.cos(phi*MORDER);
    msin = np.sin(phi*MORDER);
    #
    for n in range(0,norb):
        if (n % 5000)==0: print 'Particle %i/%i' %(n,norb)
        #rr = sqrt(r*r+z*z);
        r = (ParticleInstance.xpos[n]**2. + ParticleInstance.ypos[n]**2.)**0.5
        phi = np.arctan2(ParticleInstance.ypos[n],ParticleInstance.xpos[n])
        #if (rr/ASCALE>Rtable) return;
        nparticles += 1
        cylmass += ParticleInstance.mass[n]
        vc,vs = get_pot_single_m(r, ParticleInstance.zpos[n], potC,potS,MORDER,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=NUMX,numy=NUMY,NMAX=NMAX);
        #
        #
        accum_cos += norm * ParticleInstance.mass[n] * mcos * vc[MORDER] # a matrix-oriented accumulation scheme
        if (MORDER>0):
            accum_sin += norm * ParticleInstance.mass[n] * msin * vs[MORDER]               
    return accum_cos,accum_sin







def accumulated_eval_table(r, z, phi, accum_cos, accum_sin, eof_file, m1=0,m2=1000):#, 	double &p0, double& p, double& fr, double& fz, double &fp)
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





    

def accumulated_eval(r, z, phi, accum_cos, accum_sin, potC, rforceC, zforceC, densC, potS, rforceS, zforceS, densS, rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0):#, 	double &p0, double& p, double& fr, double& fz, double &fp)
    fr = 0.0;
    fz = 0.0;
    fp = 0.0;
    p = 0.0;
    p0 = 0.0;
    d = 0.0;
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

def accumulated_eval_particles(Particles, accum_cos, accum_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,MMAX=6,NMAX=18,ASCALE=0.0,HSCALE=0.0,CMAP=0,m1=0,m2=1000,verbose=0):#, 	double &p0, double& p, double& fr, double& fz, double &fp)
    #
    #
    #
    norb = len(Particles.xpos)
    fr = np.zeros(norb);
    fz = np.zeros(norb);
    fp = np.zeros(norb)
    p = np.zeros(norb)
    p0 = np.zeros(norb)
    #
    RR = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos + Particles.zpos*Particles.zpos)**0.5
    PHI = np.arctan2(Particles.ypos,Particles.xpos)
    R = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos)**0.5
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
        #double ccos, ssin=0.0, fac;
        for mm in range(0,MMAX+1): #(int mm=0; mm<=MMAX; mm++) {
            if (mm > m2) | (mm < m1): continue
            ccos = np.cos(phi*mm);
            ssin = np.sin(phi*mm);
            #
            #for n in range(0,NMAX): #(int n=0; n<rank3; n++) {  
            fac = accum_cos[mm] * ccos;
            p[part] += np.sum(fac * (potC[mm,:,ix,iy]*c00 + potC[mm,:,ix+1,iy  ]*c10 + potC[mm,:,ix,iy+1]*c01 + potC[mm,:,ix+1,iy+1]*c11));
            fr[part] += np.sum(fac * (rforceC[mm,:,ix,iy] * c00 + rforceC[mm,:,ix+1,iy  ] * c10 + rforceC[mm,:,ix,iy+1] * c01 + rforceC[mm,:,ix+1,iy+1] * c11));
            fz[part] += np.sum(fac * ( zforceC[mm,:,ix,iy] * c00 + zforceC[mm,:,ix+1,iy  ] * c10 + zforceC[mm,:,ix,iy+1] * c01 + zforceC[mm,:,ix+1,iy+1] * c11 ));
            #
            fac = accum_cos[mm] * ssin;
            #
            fp[part] += np.sum(fac * mm * ( potC[mm,:,ix,iy] * c00 + potC[mm,:,ix+1,iy] * c10 + potC[mm,:,ix,iy+1] * c01 + potC[mm,:,ix+1,iy+1] * c11 ));
            #
            if (mm > 0):
                #
                fac = accum_sin[mm] * ssin;
                #
                p[part] += np.sum(fac * (potS[mm,:,ix,iy]*c00 + potS[mm,:,ix+1,iy  ]*c10 + potS[mm,:,ix,iy+1]*c01 + potS[mm,:,ix+1,iy+1]*c11));
                fr[part] += np.sum(fac * (rforceS[mm,:,ix,iy] * c00 + rforceS[mm,:,ix+1,iy  ] * c10 + rforceS[mm,:,ix,iy+1] * c01 + rforceS[mm,:,ix+1,iy+1] * c11));
                fz[part] += np.sum(fac * ( zforceS[mm,:,ix,iy] * c00 + zforceS[mm,:,ix+1,iy  ] * c10 + zforceS[mm,:,ix,iy+1] * c01 + zforceS[mm,:,ix+1,iy+1] * c11 ));
                fac = -accum_sin[mm] * ccos;
                fp[part] += np.sum(fac * mm * ( potS[mm,:,ix,iy  ] * c00 + potS[mm,:,ix+1,iy  ] * c10 + potS[mm,:,ix,iy+1] * c01 + potS[mm,:,ix+1,iy+1] * c11 ))
                #
            if (mm==0): p0[part] = p[part];
    return p0,p,fr,fp,fz,R



############################################################################################
#
# DO ALL
#
############################################################################################


def compute_coefficients(PSPInput,eof_file,verbose=1):

    EOF_Out = EOF_Object()
    EOF_Out.time = PSPInput.time
    EOF_Out.dump = PSPInput.infile
    EOF_Out.comp = PSPInput.comp
    EOF_Out.nbodies = PSPInput.mass.size
    EOF_Out.eof_file = eof_file
    
    nprocs = multiprocessing.cpu_count()
    
    if verbose > 1:
        eof_quickread(eof_file)

    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = parse_eof(eof_file)
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale,cmap,dens = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)

    EOF_Out.mmax = mmax # don't forget +1 for array size
    EOF_Out.nmax = norder
    
    if nprocs > 1:
        a_cos,a_sin = make_coefficients_multi(PSPInput,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=verbose)
        
    else:
        # do the accumulation call, not implemented yet
        print 'eof.compute_coefficients: This definition has not yet been generalized to take a single processor.'
        a_cos = 0
        a_sin = 0

    EOF_Out.cos = a_cos
    EOF_Out.sin = a_sin
        
    return EOF_Out


def compute_forces(PSPInput,EOF_Object,verbose=1,nprocs=-1,m1=0,m2=1000):
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
# extremely minimalist particle holder.
#
class particle_holder(object):
    xpos = None
    ypos = None
    zpos = None
    mass = None


#
# set up particle structure for multiprocessing distribution
#
def redistribute_particles(ParticleInstance,divisions):
    npart = np.zeros(divisions)
    holders = [particle_holder() for x in range(0,divisions)]
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


def multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=0):
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
    a_coeffs = pool.map(accumulate_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                                                                itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),\
                                                                itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),\
                                                                itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg),itertools.repeat(twelvth_arg),\
                                                                itertools.repeat(thirteenth_arg),itertools.repeat(fourteenth_arg),fifteenth_arg\
                                                                ))
    pool.close()
    pool.join()                                                        
    return a_coeffs



def make_coefficients_multi(ParticleInstance,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=0):
    holding = redistribute_particles(ParticleInstance,nprocs)
    if (verbose): print 'eof.make_coefficients_multi: %i processors, %i particles each.' %(nprocs,len(holding[0].mass))
    t1 = time.time()
    multiprocessing.freeze_support()
    a_coeffs = multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy,ascale,hscale,cmap,verbose=verbose)
    if (verbose): print 'eof.make_coefficients_multi: Accumulation took %3.2f seconds, or %4.2f microseconds per orbit.' %(time.time()-t1, 1.e6*(time.time()-t1)/len(ParticleInstance.mass))
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
        full_p[first_part:first_part+n_instance_part] = MultiOutput[i][1]
        full_fr[first_part:first_part+n_instance_part] = MultiOutput[i][2]
        full_fp[first_part:first_part+n_instance_part] = MultiOutput[i][3]
        full_fz[first_part:first_part+n_instance_part] = MultiOutput[i][4]
        full_r[first_part:first_part+n_instance_part] = MultiOutput[i][5]
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
        
        EOF_Out = extract_eof_coefficients(f)

        EOF_Dict[EOF_Out.time] = EOF_Out


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



'''
# p,fr,fp,fz,d = eof.radial_slice(rvals,a_cos, a_sin,eof_file,z=0.0,phi=0.0)

dfdr = np.ediff1d((fr),to_end=0.0)/(rvals[1]-rvals[0])

omega_r = ( 3.* (fr/rvals) + dfdr)**0.5
omega_phi = (fr/rvals)**0.5

plt.plot(rvals,omega_phi)
plt.plot(rvals,omega_phi + 0.5*omega_r)


'''
