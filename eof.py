####################################
#
# Python EOF tools
#
#    MSP 4.10.16
#    Added to exptool 5.15.16
#
import time
import numpy as np
#import os

# for multiprocessing
import itertools
from multiprocessing import Pool, freeze_support

# technique for multiprocessing arguments drawn from
#     http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments





'''
USAGE EXAMPLE


#
# in order to get force fields from an output dump and eof cache file:
#   1) read in cachefile, setting potC and potS in particular
#   2) determine global parameters for the cachefile and set those to be passed to all future functions
#   3) accumulate particles to find coefficients
#   4) use coefficients to determine potential
#


import psp_io
import time
import eof


#
# an example of using the rotation curve
def rotation_curve_contribution(rvals,a_cos, a_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder,z=0.0,phi=0.0):
    pvals = np.zeros_like(rvals)
    for i in range(0,len(rvals)):
         p0,p,fr,fp,fz =eof.accumulated_eval(rvals[i], z, phi, a_cos, a_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder)
         pvals[i] = (rvals[i]*-fr)**0.5
    return pvals




# set up the eof file block
#
eof_file = '/scratch/mpetersen/Disk014/.eof.cache.file'
potC,rforceC,zforceC,potS,rforceS,zforceS = eof.parse_eof(eof_file)
rmin,rmax,numx,numy,mmax,norder,ascale,hscale = eof.eof_params(eof_file)
XMIN,XMAX,dX,YMIN,YMAX,dY = eof.set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy)



# accumulate the coefficients
#
nprocs=16
eof_file = '/scratch/mpetersen/Disk014/.eof.cache.file1'
SDisk = psp_io.Input('/scratch/mpetersen/Disk014/OUT.run014e.00000',comp='star')
a_cos_stellar,a_sin_stellar = eof.make_coefficients_multi(SDisk,eof_file,nprocs)




a_cos_stellar,a_sin_stellar = eof.make_coefficients_multi(SDisk,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy)


# benchmark times: 10 min/16 processors/1M particles
#Accumulation took 584.08 seconds, or 0.58 milliseconds per orbit. (100 processes, 1M particles)
#Accumulation took 585.99 seconds, or 0.59 milliseconds per orbit.(64 processes, 1M particles) ... something is not scaling ideally

rvals = np.linspace(0.,0.1,100)
stellardisk_curve3 = rotation_curve_contribution(rvals,a_cos_stellar, a_sin_stellar, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder)
    
plt.plot(rvals,stellardisk_curve2,color='red',linestyle='dashed')


# same stanza for the dark disk
eof_file = '/scratch/mpetersen/Disk013/.eof.cache.file2'
potC,rforceC,zforceC,potS,rforceS,zforceS = EmpOrth9thd_calculate.parse_eof(eof_file)
rmin,rmax,numx,numy,mmax,norder,ascale,hscale = EmpOrth9thd_calculate.eof_params(eof_file)
XMIN,XMAX,dX,YMIN,YMAX,dY = EmpOrth9thd_calculate.set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy)


DDisk = psp_io.Input('/scratch/mpetersen/Disk013/OUT.run013p.01000',comp='darkdisk')
a_cos_darkdisk,a_sin_darkdisk = EmpOrth9thd_calculate.make_coefficients_multi(DDisk,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy)

darkdisk_curve2 = rotation_curve_contribution(rvals,a_cos_darkdisk, a_sin_darkdisk, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder)
    
plt.plot(rvals,darkdisk_curve2,linestyle='dashed',color='blue')


plt.plot(rvals,((rvals*potr_array)+stellardisk_curve**2. + darkdisk_curve**2.)**0.5)

plt.plot(rvals,((rvals*potr_array)+stellardisk_curve**2.)**0.5,color='gray')



#O = psp_io.Input('/Users/mpetersen/Research/NBody/OUT.run013p.01000',comp='star',nout=50000)


# divide up particles



disk_curve = rotation_curve_contribution(rvals,a_cos, a_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy,MMAX=mmax,NMAX=norder)
    


#a_cos,a_sin = accumulate(O,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy)


t1 = time.time()
a_cos,a_sin = accumulate(O,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy)

print 'Accumulation took %3.2f seconds, or %3.2f milliseconds per orbit.' %(time.time()-t1, 1.e3*(time.time()-t1)/len(O.mass))


# now can get the potential value at any position (should also figure out how to evaluate forces)


#Vc, Vs = get_pot(0.01,0.005,potC,potS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=numx,numy=numy)




'''




def accumulate_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return accumulate(*a_b)

def multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy):
    pool = Pool(nprocs)
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
    a_coeffs = pool.map(accumulate_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),itertools.repeat(seventh_arg),itertools.repeat(eighth_arg),itertools.repeat(ninth_arg),itertools.repeat(tenth_arg),itertools.repeat(eleventh_arg)))
    return a_coeffs




def make_coefficients_multi(O,eof_file,nprocs):
    potC,rforceC,zforceC,potS,rforceS,zforceS = parse_eof(eof_file)
    rmin,rmax,numx,numy,mmax,norder,ascale,hscale = eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy)
    holding = redistribute_particles(O,nprocs)
    t1 = time.time()
    freeze_support()
    a_coeffs = multi_accumulate(holding,nprocs,potC,potS,mmax,norder,XMIN,dX,YMIN,dY,numx,numy)
    print 'Accumulation took %3.2f seconds, or %3.2f milliseconds per orbit.' %(time.time()-t1, 1.e3*(time.time()-t1)/len(O.mass))
    # sum over processes
    scoefs = np.sum(np.array(a_coeffs),axis=0)
    a_cos = scoefs[0]
    a_sin = scoefs[1]
    return a_cos,a_sin








def eof_params(file):
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
    return rmin,rmax,numx,numy,mmax,norder,ascale,hscale



def eof_quickread(file):
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
    print 'The parameters for this EOF file are:'
    print 'MMAX=%i' %mmax
    print 'NORDER=%i' %norder
    print 'NMAX=%i' %nmax
    print 'NUMX,NUMY=%i,%i' %(numx,numy)
    print 'DENS,CMAP=%i,%i' %(dens,cmap)
    print 'ASCALE,HSCALE=%5.4f,%5.4f' %(ascale,hscale)
    print 'CYLMASS=%5.4f' %cylmass
    print 'TNOW=%5.4f' %tnow




def parse_eof(file):
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
    #
    # set up the matrices
    #
    potC = np.zeros([mmax+1,norder,numx+1,numy+1])
    rforcec = np.zeros([mmax+1,norder,numx+1,numy+1])
    zforcec = np.zeros([mmax+1,norder,numx+1,numy+1])
    densc = np.zeros([mmax+1,norder,numx+1,numy+1])
    potS = np.zeros([mmax+1,norder,numx+1,numy+1])
    rforces = np.zeros([mmax+1,norder,numx+1,numy+1])
    zforces = np.zeros([mmax+1,norder,numx+1,numy+1])
    denss = np.zeros([mmax+1,norder,numx+1,numy+1])
    for i in range(0,mmax+1): # I think the padding needs to be here? test.
        for j in range(0,norder):
            #
            # loops for different levels go here
            #
            for k in range(0,numx+1):
                potC[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
            for k in range(0,numx+1):
                rforcec[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
            for k in range(0,numx+1):
                zforcec[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
            if (dens==1):
                for k in range(0,numx+1):
                    densc[i,j,k,:] = np.fromfile(f,dtype='<f8',count=numy+1)
    for i in range(1,mmax+1): # no zero order m here
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
    return potC,rforcec,zforcec,potS,rforces,zforces





def z_to_y(z,hscale=0.001):
    return z/(abs(z)+1.e-10)*np.arcsinh(abs(z/hscale))


def set_table_params(RMAX=20.0,RMIN=0.001,ASCALE=0.01,HSCALE=0.001,NUMX=128,NUMY=64):
    M_SQRT1_2 = np.sqrt(0.5)
    Rtable  = M_SQRT1_2 * RMAX
    # check cmap, but if cmap=0, r_to_xi = r
    #
    # otherwise, r = (r/ASCALE - 1.0)/(r/ASCALE + 1.0);
    XMIN    = RMIN*ASCALE #r_to_xi(RMIN*ASCALE);
    XMAX    = Rtable*ASCALE #r_to_xi(Rtable*ASCALE);
    dX      = (XMAX - XMIN)/NUMX;    
    YMIN    = z_to_y(-Rtable*ASCALE,hscale=HSCALE);
    YMAX    = z_to_y( Rtable*ASCALE,hscale=HSCALE);
    dY      = (YMAX - YMIN)/NUMY;
    return XMIN,XMAX,dX,YMIN,YMAX,dY





def get_pot(r,z,cos_array,sin_array,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18):#Matrix& Vc, Matrix& Vs, double r, double z)
    #
    # returns potential fields for C and S to calculate weightings during accumulation
    #
    #
    # define boundaries of the interpolation scheme
    #
    #
    # find the corresponding bins
    NMAX=NORDER=18
    MMAX=6
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
    Vc = np.zeros([MMAX+1,NMAX])
    Vs = np.zeros([MMAX+1,NMAX])
    for mm in range(0,MMAX+1): #(int mm=0; mm<=MMAX; mm++):
        for nn in range(0,NMAX):#(int n=0; n<rank3; n++):
            #print mm,nn
            Vc[mm][nn] = fac * ( cos_array[mm][nn][X  ][Y  ] * c00 + cos_array[mm][nn][X+1][Y  ] * c10 + cos_array[mm][nn][X  ][Y+1] * c01 + cos_array[mm][nn][X+1][Y+1] * c11 )
            if (mm>0):
                Vs[mm][nn] = fac * ( sin_array[mm][nn][X  ][Y  ] * c00 + sin_array[mm][nn][X+1][Y] * c10 + sin_array[mm][nn][X  ][Y+1] * c01 + sin_array[mm][nn][X+1][Y+1] * c11 );
    return Vc,Vs



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


class particle_holder(object):
    xpos = None
    ypos = None
    zpos = None
    mass = None


def redistribute_particles(ParticleInstance,divisions):
    npart = np.zeros(divisions)
    holders = [particle_holder() for x in range(0,divisions)]
    average_part = int(np.floor(len(ParticleInstance.xpos)/divisions))
    first_partition = len(ParticleInstance.xpos) - average_part*(divisions-1)
    print average_part, first_partition
    low_particle = 0
    for i in range(0,divisions):
        end_particle = low_particle+average_part
        if i==0: end_particle = low_particle+first_partition
        print low_particle,end_particle
        holders[i].xpos = ParticleInstance.xpos[low_particle:end_particle]
        holders[i].ypos = ParticleInstance.ypos[low_particle:end_particle]
        holders[i].zpos = ParticleInstance.zpos[low_particle:end_particle]
        holders[i].mass = ParticleInstance.mass[low_particle:end_particle]
        low_particle = end_particle
    return holders


    
 

def accumulate(ParticleInstance,potC,potS,MMAX,NMAX,XMIN,dX,YMIN,dY,NUMX,NUMY):
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
    #
    # set up accumulation arrays
    #
    #MMAX=6
    #NMAX=18
    accum_cos = np.zeros([MMAX+1,NMAX])
    accum_cos2 = np.zeros([MMAX+1,NMAX])
    accum_sin = np.zeros([MMAX+1,NMAX])
    accum_sin2 = np.zeros([MMAX+1,NMAX])
    #
    for n in range(0,norb):
        if (n % 5000)==0: print 'Particle %i/%i' %(n,norb)
        #rr = sqrt(r*r+z*z);
        r = (ParticleInstance.xpos[n]**2. + ParticleInstance.ypos[n]**2.)**0.5
        phi = np.arctan2(ParticleInstance.ypos[n],ParticleInstance.xpos[n])
        #if (rr/ASCALE>Rtable) return;
        nparticles += 1
        cylmass += ParticleInstance.mass[n]
        vc,vs = get_pot(r, ParticleInstance.zpos[n], potC,potS,rmin=XMIN,dR=dX,zmin=YMIN,dZ=dY,numx=NUMX,numy=NUMY);
        for mm in range(0,MMAX+1): #(mm=0; mm<=MMAX; mm++) {
            mcos = np.cos(phi*mm);
            msin = np.sin(phi*mm);
            for nn in range(0,NMAX): #(int nn=0; nn<rank3; nn++) 
                accum_cos[mm][nn] += norm * ParticleInstance.mass[n] * mcos * vc[mm][nn];
            if (SELECT):
                for nn in range(0,NMAX): #(int nn=0; nn<rank3; nn++) 
                    accum_cos2[mm][nn] += (norm * ParticleInstance.mass[n] * mcos * vc[mm][nn])*(norm * ParticleInstance.mass[n] * mcos * vc[mm][nn])
            if (mm>0):
                for nn in range(0,NMAX): #(int nn=0; nn<rank3; nn++) 
                    accum_sin[mm][nn] += norm * ParticleInstance.mass[n] * msin * vs[mm][nn];                 
                if (SELECT):
                    for nn in range(0,NMAX): #(int nn=0; nn<rank3; nn++) 
                        accum_sin2[mm][nn] += (norm * ParticleInstance.mass[n] * msin * vs[mm][nn])*(norm * ParticleInstance.mass[n] * msin * vs[mm][nn])
    return accum_cos,accum_sin




def accumulated_eval(r, z, phi, accum_cos, accum_sin, potC, rforceC, zforceC, potS, rforceS, zforceS,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,fac = 1.0,MMAX=6,NMAX=18):#, 	double &p0, double& p, double& fr, double& fz, double &fp)
    '''
    {
    if (!coefs_made_all()) {
    if (VFLAG>3)
      cerr << "Process " << myid << ": in EmpCylSL::accumlated_eval, "
	   << "calling make_coefficients()" << endl;
    make_coefficients();
    }
    '''
    fr = 0.0;
    fz = 0.0;
    fp = 0.0;
    p = 0.0;
    p0 = 0.0;
    #
    rr = np.sqrt(r*r + z*z);
    #if (rr/ASCALE>Rtable) return;
    #double X = (r_to_xi(r) - XMIN)/dX;
    #double Y = (z_to_y(z)  - YMIN)/dY;
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
    #double ccos, ssin=0.0, fac;
    for mm in range(0,MMAX+1): #(int mm=0; mm<=MMAX; mm++) {
        ccos = np.cos(phi*mm);
        ssin = np.sin(phi*mm);
        #
        for n in range(0,NMAX): #(int n=0; n<rank3; n++) {  
            fac = accum_cos[mm][n] * ccos;
            p += fac * (potC[mm][n][ix  ][iy  ]*c00 + potC[mm][n][ix+1][iy  ]*c10 + potC[mm][n][ix  ][iy+1]*c01 + potC[mm][n][ix+1][iy+1]*c11);
            #
            fr += fac * (rforceC[mm][n][ix  ][iy  ] * c00 + rforceC[mm][n][ix+1][iy  ] * c10 + rforceC[mm][n][ix  ][iy+1] * c01 + rforceC[mm][n][ix+1][iy+1] * c11);
            #
            fz += fac * ( zforceC[mm][n][ix  ][iy  ] * c00 + zforceC[mm][n][ix+1][iy  ] * c10 + zforceC[mm][n][ix  ][iy+1] * c01 + zforceC[mm][n][ix+1][iy+1] * c11 );
            #
            fac = accum_cos[mm][n] * ssin;
            #
            fp += fac * mm * ( potC[mm][n][ix  ][iy  ] * c00 + potC[mm][n][ix+1][iy  ] * c10 + potC[mm][n][ix  ][iy+1] * c01 + potC[mm][n][ix+1][iy+1] * c11 );
            #
            if (mm > 0):
                #
                fac = accum_sin[mm][n] * ssin;
                #
                p += fac * ( potS[mm][n][ix  ][iy  ] * c00 + potS[mm][n][ix+1][iy  ] * c10 + potS[mm][n][ix  ][iy+1] * c01 + potS[mm][n][ix+1][iy+1] * c11 );
                #
                fr += fac * ( rforceS[mm][n][ix  ][iy  ] * c00 + rforceS[mm][n][ix+1][iy  ] * c10 + rforceS[mm][n][ix  ][iy+1] * c01 + rforceS[mm][n][ix+1][iy+1] * c11 );
                #
                fz += fac * ( zforceS[mm][n][ix  ][iy  ] * c00 + zforceS[mm][n][ix+1][iy  ] * c10 + zforceS[mm][n][ix  ][iy+1] * c01 + zforceS[mm][n][ix+1][iy+1] * c11 );
                #
                fac = -accum_sin[mm][n] * ccos;
                fp += fac * mm * ( potS[mm][n][ix  ][iy  ] * c00 + potS[mm][n][ix+1][iy  ] * c10 + potS[mm][n][ix  ][iy+1] * c01 + potS[mm][n][ix+1][iy+1] * c11 )
                #
        if (mm==0): p0 = p;
    return p0,p,fr,fp,fz



def make_sl(RMIN,RMAX,massR,densR,logarithmic=True):
    number = 10000;
    r =  vector<double>(number);
    d =  vector<double>(number);
    m =  vector<double>(number);
    p =  vector<double>(number);
    vector<double> mm(number);
    vector<double> pw(number);
	#// ------------------------------------------
	#// Make radial, density and mass array
    #// ------------------------------------------
    double dr;
    if (logarithmic):
        dr = (log(RMAX) - log(RMIN))/(number - 1);
    else:
        dr = (RMAX - RMIN)/(number - 1);
    for i in range(0,number+1): #(int i=0; i<number; i++) {
        if (logarithmic):
            r[i] = RMIN*exp(dr*i);
        else:
            r[i] = RMIN + dr*i;
        m[i] = massR(r[i]);
        d[i] = densR(r[i]);
    mm[0] = 0.0;
    pw[0] = 0.0;
    for i in range(1,number+1):#(int i=1; i<number; i++) {
        mm[i] = mm[i-1] + 2.0*M_PI*(r[i-1]*r[i-1]*d[i-1] + r[i]*r[i]*d[i])    *(r[i] - r[i-1]);
        pw[i] = pw[i-1] + 2.0*M_PI*(r[i-1]*d[i-1] + r[i]*d[i])                *(r[i] - r[i-1]);
    for i in range(0,number+1):#(int i=0; i<number; i++) 
        p[i] = -mm[i]/(r[i]+1.0e-10) - (pw[number-1] - pw[i]);
    #if (VFLAG & 1) {
    #ostringstream outf;
    #outf << "test_adddisk_sl." << myid;
    #ofstream out(outf.str().c_str());
    #for (int i=0; i<number; i++) {
    #  out 
	#<< setw(15) << r[i] 
	#<< setw(15) << d[i] 
	#<< setw(15) << m[i] 
	#<< setw(15) << p[i] 
	#<< setw(15) << mm[i] 
	#<< endl;
    #}
    #out.close();
    return r,d,m,p

