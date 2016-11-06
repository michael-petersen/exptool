

# 08-30-16: print_progress and verbosity keys added



'''

sph_file = '/scratch/mpetersen/Disk064a/.slgrid_sph_cache'


sph_file = '/Users/mpetersen/Desktop/.slgrid_sph_064a'


sph_file = '/scratch/mpetersen/Disk013/.slgrid_sph_cache'
model_file = '/scratch/mpetersen/Disk013/SLGridSph.model'


#x = 0.1
#mat = get_halo_dens(x, 0, 20, evtable, eftable, xi, d0, cmap=0, scale=1.0)

rvals = np.linspace(0.,1.,100)
densval = np.zeros_like(rvals)

for i in range(0,len(rvals)):
    densval[i] = np.sum(get_halo_dens(rvals[i], 0, 1, evtable, eftable, xi, d0, cmap=0, scale=1.0))




forcemat = read_sl.get_halo_force(0.1, 6, 12, evtable, eftable, xi, p0, cmap=0, scale=1.0)




rindx = np.linspace(0.,1.,100)
rpot = np.zeros_like(rindx)

for i in range(0,100): rpot[i] = read_sl.get_halo_force(rindx[i], 6, 12, evtable, eftable, xi, p0, cmap=0, scale=1.0)[0,3]


xi,r,p0,d0 = halo_methods.init_table(model_file,numr,rmin,rmax,cmap=0,scale=1.0)

pot_tmp = get_halo_pot(0.01, 0, 1, evtable, eftable, xi, p0, cmap=0, scale=1.0)

    
# FORMATS
#evtable[L,N]
#eftable[L,N,NUMR]


import time
t1 = time.time()
#O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.00000',comp='dark',nout=50000)

O = psp_io.Input('/scratch/mpetersen/Disk013/OUT.run013p.00000',comp='dark',nout=50000)

halo_coefs = compute_coefficients(O,sph_file,model_file)

print 'This took %3.2f seconds' %(time.time()-t1)


# the format is
#halo_coefs[Mterms,Nterms]
# where Mterms maps weirdly in linear space
# 0: l=m=0
# 1: l=1,m=0
# 2: l=1,m=-1
# they start at the squares, then go 1:cos,sin, 2:cos,sin, etc



# nah dog, do this multi-style


import read_sl

import psp_io
O = psp_io.Input('/scratch/mpetersen/Disk013/OUT.run013p.00000',comp='dark',nout=1000000)



sph_file = '/scratch/mpetersen/Disk013/.slgrid_sph_cache'
model_file = '/scratch/mpetersen/Disk013/SLGridSph.model'

t1 = time.time()

nprocs=16
#sph_file = '/scratch/mpetersen/Disk013/SLGridSph.model'
scoeffs2,a_coeffs2 = read_sl.make_coefficients_multi(O,nprocs,sph_file,model_file)

print 'Accumulation took %3.2f seconds, or %3.2f milliseconds per orbit.' %(time.time()-t1, 1.e3*(time.time()-t1)/len(O.mass))


rvals = np.linspace(0.0001,0.1,100)
potr_array_lo = np.zeros_like(rvals)+100.
potr_array_hi = np.zeros_like(rvals)

for i in range(0,len(rvals)):
  for j in np.linspace(-np.pi,np.pi,36):
     #print j
     den0,den1,pot0,pot1,potr,pott,potp = read_sl.all_eval_legacy(rvals[i],0.0,j,scoeffs2,sph_file,model_file)
     #print potr
     potr_array_lo[i] = np.min([ abs(rvals[i]*potr)**0.5,potr_array_lo[i]])
     potr_array_hi[i] = np.max([ abs(rvals[i]*potr)**0.5,potr_array_hi[i]])
  print 'Radius=%4.3f Minimum: %3.2f, Maximum %3.2f' %(rvals[i],potr_array_lo[i],potr_array_hi[i])


plt.plot(rvals,potr_array_lo,color='black')
plt.plot(rvals,potr_array_hi,color='black')


rvals = np.linspace(0.,0.4,100)
pvals = np.zeros_like(rvals)
pvals1 = np.zeros_like(rvals)
dvals = np.zeros_like(rvals)
dvals1 = np.zeros_like(rvals)
potr_array = np.zeros_like(rvals)

for i in range(0,100):
  den0,den1,pot0,pot1,potr,pott,potp = read_sl.all_eval(rvals[i], 0.0, 0.0, scoeffs, sph_file, model_file)
  pvals[i] = pot0
  pvals1[i] = pot1
  dvals[i] = den0
  dvals1[i] = den1
  potr_array[i] = potr


plt.plot(rvals,(rvals*potr_array)**0.5)

  

rvals2 = np.linspace(0.,0.4,100)
plt.plot(rvals2,(rvals2*potr_array)**0.5)

'''


import numpy as np
import time
import sys

import halo_methods

from scipy.special import gammaln

import itertools
from multiprocessing import Pool, freeze_support

import multiprocessing


def get_halo_dens_pot_force(x, lmax, nmax, evtable, eftable, xi, d0, p0, cmap=0, scale=1.0):
    #
    # needs the potential table to be defined
    #
    numr = len(d0)
    dens_mat = np.zeros([lmax+1,nmax+1])
    pot_mat = np.zeros([lmax+1,nmax+1])
    force_mat = np.zeros([lmax+1,nmax+1])
    #if (which || !cmap):
    x = halo_methods.r_to_xi(x,cmap,scale);
    #
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    fac = halo_methods.d_xi_to_r(x,cmap,scale)/dxi;
    for l in range(0,lmax+1): #(int l=0; l<=lmax; l++) {
        #print x1,x2
        for n in range(1,nmax+1): #(int n=1; n<=nmax; n++) {
            dens_mat[l][n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])*np.sqrt(evtable[l,n]) * (x1*d0[indx] + x2*d0[indx+1]);
            force_mat[l,n] = fac * ((x2 - 0.5)*eftable[l,n,indx-1]*p0[indx-1] - 2.0*x2*eftable[l,n,indx]*p0[indx] + (x2 + 0.5)*eftable[l,n,indx+1]*p0[indx+1]) / np.sqrt(evtable[l,n]);
            pot_mat[l,n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])/np.sqrt(evtable[l,n]) * (x1*p0[indx] + x2*p0[indx+1]);
    return dens_mat,force_mat,pot_mat



def get_halo_dens(x, lmax, nmax, evtable, eftable, xi, d0, cmap=0, scale=1.0):#, int which):
    #
    # needs the potential table to be defined
    #
    numr = len(d0)
    mat = np.zeros([lmax+1,nmax+1])
    #if (which || !cmap):
    x = halo_methods.r_to_xi(x,cmap,scale);
    #
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    for l in range(0,lmax+1): #(int l=0; l<=lmax; l++) {
        #print x1,x2
        for n in range(1,nmax+1): #(int n=1; n<=nmax; n++) {
            mat[l][n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])*np.sqrt(evtable[l,n]) * (x1*d0[indx] + x2*d0[indx+1]);
    return mat



    
def get_halo_force(x, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0):
    #
    # needs the potential table to be defined and passed
    #
    numr = len(p0)
    mat = np.zeros([lmax+1,nmax+1])
    #if (which || !cmap):
    x = halo_methods.r_to_xi(x,cmap,scale);
    #print 'X',x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    if (indx < 1): indx = 1;
    if (indx > (numr-2) ): indx = numr - 2;
    p = (x - xi[indx])/dxi;
    fac = halo_methods.d_xi_to_r(x,cmap,scale)/dxi;
    for l in range(0,lmax+1): 
        for n in range(1,nmax+1):
            mat[l,n] = fac * ((p - 0.5)*eftable[l,n,indx-1]*p0[indx-1] - 2.0*p*eftable[l,n,indx]*p0[indx] + (p + 0.5)*eftable[l,n,indx+1]*p0[indx+1]) / np.sqrt(evtable[l,n]);
            #(p - 0.5)*eftable[l,n,indx-1]*p0[indx-1]
            #- 2.0*p*eftable[l,n,indx]*p0[indx]
            #(p + 0.5)*eftable[l,n,indx+1]*p0[indx+1]
    return mat



def get_halo_pot_matrix(x, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0):#, int which):
    #
    # needs the potential table to be defined
    #
    #if (which || !cmap):
    numr = len(p0)
    x = halo_methods.r_to_xi(x,cmap,scale);
    #print x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    #print x-np.min(xi), (x-np.min(xi))/(dxi)
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #print indx
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    mat = np.zeros([lmax+1,nmax+1])
    for l in range(0,lmax+1): #(int l=0; l<=lmax; l++) {
        for n in range(1,nmax+1): #(int n=1; n<=nmax; n++) {
            mat[l,n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])/np.sqrt(evtable[l,n]) * (x1*p0[indx] + x2*p0[indx+1]);
    return mat
    #return (x1*table[l].ef[n][indx] + x2*table[l].ef[n][indx+1])/sqrt(table[l].ev[n]) * sphpot(xi_to_r(x));




def get_halo_pot(x, l, n, evtable, eftable, xi, p0, cmap=0, scale=1.0):#, int which):
    #
    # needs the potential table to be defined
    #
    #if (which || !cmap):
    numr = len(p0)
    x = halo_methods.r_to_xi(x,cmap,scale);
    #print x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    #print x-np.min(xi), (x-np.min(xi))/(dxi)
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #print indx
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    return (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])/np.sqrt(evtable[l,n]) * (x1*p0[indx] + x2*p0[indx+1]);
    #return (x1*table[l].ef[n][indx] + x2*table[l].ef[n][indx+1])/sqrt(table[l].ev[n]) * sphpot(xi_to_r(x));



class particle_holder(object):
    xpos = None
    ypos = None
    zpos = None
    mass = None


def redistribute_particles(ParticleInstance,divisions):
    npart = np.zeros(divisions,dtype=object)
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


    




def compute_coefficients_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return compute_coefficients(*a_b)

def multi_compute_coefficients(holding,nprocs,sph_file,mod_file,verbose):
    pool = Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = sph_file
    third_arg = mod_file
    fourth_arg = [0 for i in range(0,nprocs)]
    fourth_arg[0] = verbose
    a_coeffs = pool.map(compute_coefficients_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),fourth_arg))
    pool.close()
    pool.join()
    return a_coeffs



def make_coefficients(O,sph_file,mod_file,verbose=1):
    nprocs = multiprocessing.cpu_count()
    holding = redistribute_particles(O,nprocs)
    t1 = time.time()
    freeze_support()
    a_coeffs = multi_compute_coefficients(holding,nprocs,sph_file,mod_file,verbose)
    if (verbose > 0): print 'spheresl.make_coefficients: accumulation took %3.2f seconds, or %4.2f microseconds per orbit.' %(time.time()-t1, 1.e6*(time.time()-t1)/len(O.mass))
    # sum over processes
    summed_coefs = np.sum(np.array(a_coeffs,dtype=object),axis=0)
    return summed_coefs#,a_coeffs




def all_eval_particles_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return all_eval_particles(*a_b)

def multi_all_eval_particles(holding,nprocs,expcoef,sph_file,mod_file,verbose):
    pool = Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = expcoef
    third_arg = sph_file
    fourth_arg = mod_file
    fifth_arg = [0 for i in range(0,nprocs)]
    fifth_arg[0] = verbose
    a_vals = pool.map(all_eval_particles_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg)\
                                                              ,itertools.repeat(fourth_arg),fifth_arg))
    pool.close()
    pool.join()
    return a_vals



def eval_particles(ParticleInstance,expcoef,sph_file,mod_file,nprocs=-1,verbose=1,l1=0,l2=1000):
    if nprocs == -1:
        nprocs = multiprocessing.cpu_count()

    if nprocs > 1:
        holding = redistribute_particles(ParticleInstance,nprocs)
        t1 = time.time()
        freeze_support()
        #
        if (verbose):
            print 'eof.make_coefficients_multi: %i processors, %i particles each.' %(nprocs,len(holding[0].mass))
        a_vals = multi_all_eval_particles(holding,nprocs,expcoef,sph_file,mod_file,verbose)
        #
        if (verbose):
            print 'spheresl.eval_particles: particle Evaluation took %3.2f seconds, or %4.2f microseconds per orbit.' %(time.time()-t1, 1.e6*(time.time()-t1)/len(ParticleInstance.mass))
        # sum over processes
        den0,den1,pot0,pot1,potr,pott,potp,rr = mix_outputs_sph(np.array(a_vals))
        

    else:
        den0,den1,pot0,pot1,potr,pott,potp,rr = all_eval_particles(ParticleInstance, expcoef, sph_file, mod_file,verbose,L1=l1,L2=l2)

        
    return den0,den1,pot0,pot1,potr,pott,potp,rr



def mix_outputs_sph(MultiOutput):
    n_instances = len(MultiOutput)
    n_part = 0
    for i in range(0,n_instances):
        n_part += len(MultiOutput[i][0])
    full_den0 = np.zeros(n_part)
    full_den1 = np.zeros(n_part)
    full_pot0 = np.zeros(n_part)
    full_pot1 = np.zeros(n_part)
    full_potr = np.zeros(n_part)
    full_pott = np.zeros(n_part)
    full_potp = np.zeros(n_part)
    full_rr = np.zeros(n_part)
    #
    #
    first_part = 0
    for i in range(0,n_instances):
        n_instance_part = len(MultiOutput[i][0])
        full_den0[first_part:first_part+n_instance_part] = MultiOutput[i][0]
        full_den1[first_part:first_part+n_instance_part] = MultiOutput[i][1]
        full_pot0[first_part:first_part+n_instance_part] = MultiOutput[i][2]
        full_pot1[first_part:first_part+n_instance_part] = MultiOutput[i][3]
        full_potr[first_part:first_part+n_instance_part] = MultiOutput[i][4]
        full_pott[first_part:first_part+n_instance_part] = MultiOutput[i][5]
        full_potp[first_part:first_part+n_instance_part] = MultiOutput[i][6]
        full_rr[first_part:first_part+n_instance_part] = MultiOutput[i][7]
        first_part += n_instance_part
    return full_den0,full_den1,full_pot0,full_pot1,full_potr,full_pott,full_potp,full_rr





import os


def compute_coefficients(ParticleInstance,sph_file,model_file,verbose):
    #
    # follows compute_coefficients from SphereSL.cc
    #
    fac0 = -4.0*np.pi;
    lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
    xi,r,p0,d0 = halo_methods.init_table(model_file,numr,rmin,rmax,cmap=cmap,scale=scale)
    #
    expcoef = np.zeros([lmax*(lmax+2)+1,nmax+1])
    #
    factorial = factorial_return(lmax)
    norb = len(ParticleInstance.mass)
    #print os.getpid()
    for p in range(0,norb): #(auto &p : part) {
        #try:
        #    if (os.getppid()==os.getpid()):
        #        if (p % 5000)==0: print 'Particle %i/%i' %(p,norb)
        #except:
        #    pass
        if (verbose > 0) & ( ((float(p)+1.) % 1000. == 0.0) | (p==0)): print_progress(p,norb,'spheresl.compute_coefficients')
        xx = ParticleInstance.xpos[p]
        yy = ParticleInstance.ypos[p]
        zz = ParticleInstance.zpos[p]
        mass = ParticleInstance.mass[p];
        #
        r2 = (xx*xx + yy*yy + zz*zz);
        r = np.sqrt(r2) + 1.0e-8;
        #
        #if (r<=RMAX) {
        costh = zz/r;
        phi = np.arctan2(yy,xx);
        #
        # now compute the legendre polynomial value tables
        legs = legendre_R(lmax, costh)#, legs);
        #cosm,sinm = sinecosine_R(lmax, phi)#, cosm, sinm);
        #
        #
        # get potential (needs an l/n loop wrapper?)
        #
        # go through loops
        #
        potd = get_halo_pot_matrix(r, lmax, nmax, evtable, eftable, xi, p0, cmap=cmap, scale=scale)
        loffset = 0
        for l in range(0,lmax+1): #(l=0, loffset=0; l<=LMAX; loffset+=(2*l+1), l++) {
            moffset = 0
            for m in range(0,l+1):#(m=0, moffset=0; m<=l; m++) {
                fac = factorial[l][m] * legs[l][m];
                if (m==0):
                    fac4 = potd[l]*fac*fac0;  
                    expcoef[loffset+moffset] += fac4 * mass #legs[l][m]*mass*fac0#/normM[l][n];
                    moffset += 1
                else:
                    fac1 = np.cos(phi*m);
                    fac2 = np.sin(phi*m);
                    fac4 = potd[l]*fac*fac0;
                    expcoef[loffset+moffset  ] += fac1 * fac4 * mass;
                    expcoef[loffset+moffset+1] += fac2 * fac4 * mass;
                    moffset+=2;
            loffset += (2*l+1)
    return expcoef







def legendre_R(lmax, x):#, Matrix &p, Matrix &dp)
    p = np.zeros([lmax+1,lmax+1])
    dp = np.zeros([lmax+1,lmax+1])
    p[0][0] = pll = 1.0;
    if (lmax > 0):# {
        somx2 = np.sqrt( (1.0 - x)*(1.0 + x) );
        fact = 1.0;
        for m in range(1,lmax+1):#(m=1; m<=lmax; m++) {
            pll *= -fact*somx2;
            p[m][m] = pll;
            fact += 2.0;
    #
    #
    for m in range(0,lmax):#(m=0; m<lmax; m++) {
        pl2 = p[m][m];
        p[m+1][m] = pl1 = x*(2*m+1)*pl2;
        for l in range(m+2,lmax+1):#(l=m+2; l<=lmax; l++) {
            p[l][m] = pll = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m);
            pl2 = pl1;
            pl1 = pll;
    return p





def dlegendre_R(lmax, x):#, Matrix &p, Matrix &dp)
    p = np.zeros([lmax+1,lmax+1])
    dp = np.zeros([lmax+1,lmax+1])
    p[0][0] = pll = 1.0;
    if (lmax > 0):# {
        somx2 = np.sqrt( (1.0 - x)*(1.0 + x) );
        fact = 1.0;
        for m in range(1,lmax+1):#(m=1; m<=lmax; m++) {
            pll *= -fact*somx2;
            p[m][m] = pll;
            fact += 2.0;
    #
    #
    for m in range(0,lmax):#(m=0; m<lmax; m++) {
        pl2 = p[m][m];
        p[m+1][m] = pl1 = x*(2*m+1)*pl2;
        for l in range(m+2,lmax+1):#(l=m+2; l<=lmax; l++) {
            p[l][m] = pll = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m);
            pl2 = pl1;
            pl1 = pll;
    #
    #
    MINEPS=1.e-8
    if (1.0-np.abs(x) < MINEPS):
        if (x>0): x =   1.0 - MINEPS;
        else    : x = -(1.0 - MINEPS);
    #
    #
    somx2 = 1.0/(x*x - 1.0);
    dp[0][0] = 0.0;
    for l in range(1,lmax+1):# (l=1; l<=lmax; l++):
        for m in range(0,l):#(m=0; m<l; m++):
            #print l
            dp[l][m] = somx2*(x*l*p[l][m] - (l+m)*p[l-1][m]);
            dp[l][l] = somx2*x*l*p[l][l];
    return p,dp


def factorial_return(lmax):
  factorial = np.zeros([lmax+1,lmax+1])
  for l in range(0,lmax+1):#=0; l<=lmax; l++) {
      for m in range(0,l+1):#(int m=0; m<=l; m++) {
          factorial[l][m] = np.sqrt( (0.5*l+0.25)/np.pi * np.exp(gammaln(1.0+l-m) - gammaln(1.0+l+m)) );
          if (m != 0): factorial[l][m] *= np.sqrt(2.)#M_SQRT2;
  return factorial




def get_pot_coefs(l, l_coef, l_potd, l_dpot):
    nmax = len(l_coef)
    pp = 0.0
    dpp = 0.0;
    for n in range(1,nmax+1):#(i=1; i<=NMAX; i++) {
        pp  += potd[i] * coef[i];
        dpp += dpot[i] * coef[i];
    return -1.*pp, -1.*dpp


def get_dens_coefs(l, l_coef, dend):
    nmax = len(l_coef)
    # pass this the l array of coefficients
    # and d array of coefficients
    pp = 0.0;
    for n in range(1,nmax+1):#(i=1; i<=NMAX; i++)
        pp  += dend[n] * l_coef[n];
    return pp




def all_eval_table(r, costh, phi, expcoef, sph_file, mod_file,L1=0,L2=-1):
  lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap=cmap,scale=scale)
  if (L2 == -1): L2 = lmax+1
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  # begin function
  sinth = -np.sqrt(np.abs(1.0 - costh*costh));
  fac1 = factorial[0][0];
  #
  # use the basis to get the density,potential,force arrays (which have already been read in somewhere?)
  dend = get_halo_dens(r, lmax, nmax, evtable, eftable, xi, d0, cmap=0, scale=1.0)
  potd = get_halo_pot_matrix(r, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0)
  dpot = get_halo_force(r, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0)
  #
  #
  legs,dlegs = dlegendre_R(lmax,costh)
  #
  den0 = np.sum(fac1 * expcoef[0]*dend[0]);
  pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
  potr = np.sum(fac1 * expcoef[0]*dpot[0]);
  den1 = 0.0;
  pot1 = 0.0;
  pott = 0.0;
  potp = 0.0;
  #
  # L loop
  #
  loffset = 1
  for l in range(1,lmax+1):#(int l=1, loffset=1; l<=lmax; loffset+=(2*l+1), l++) {
    # at end, add in loffset+=(2*l+1)
    if (l>(L2+1)) | (l<(L1+1)): continue
    #
    # M loop
    moffset = 0
    for m in range(0,l+1):#(int m=0, moffset=0; m<=l; m++) {
      #print l,m,loffset,moffset
      fac1 = factorial[l][m];
      if (m==0):
            #den1 += fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]);
            #pot1 += fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]);
            #dpot += fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]);
            #pott += fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]);
            den1 += np.sum(fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]));
            pot1 += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
            #dpot += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            potr += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            pott += np.sum(fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]));
            moffset+=1;
      else:
            cosm = np.cos(phi*m);
            sinm = np.sin(phi*m);
            #den1 += fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm );
            #pot1 += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm );
            #dpot += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm );
            #pott += fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm );
            #potp += fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm );
            den1 += np.sum(fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
            pot1 += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
            #dpot += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            potr += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            pott += np.sum(fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm ));
            potp += np.sum(fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm ));
            moffset +=2;
    loffset+=(2*l+1)
  #
  #
  #
  densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  potlfac = 1.0/scale;
  den0  *= densfac;
  den1  *= densfac;
  pot0  *= potlfac;
  pot1  *= potlfac;
  potr  *= potlfac/scale;
  pott  *= potlfac*sinth;
  potp  *= potlfac;
  return den0,den1,pot0,pot1,potr,pott,potp



def all_eval(r, costh, phi, expcoef,xi,rarr,p0,d0,lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable):
  # this version loads everything up first: lots of inputs, but no need to reread anything.
  #
  #lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  #xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap=cmap,scale=scale)
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  # begin function
  sinth = -np.sqrt(np.abs(1.0 - costh*costh));
  fac1 = factorial[0][0];
  #
  # use the basis to get the density,potential,force arrays
  #
  # these three need to be stuffed together into one call to save loops
  dend,potd,dpot = get_halo_dens_pot_force(r, lmax, nmax, evtable, eftable, xi, d0, p0, cmap=cmap, scale=scale)
  #
  #
  legs,dlegs = dlegendre_R(lmax,costh)
  #
  den0 = np.sum(fac1 * expcoef[0]*dend[0]);
  pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
  potr = np.sum(fac1 * expcoef[0]*dpot[0]);
  den1 = 0.0;
  pot1 = 0.0;
  pott = 0.0;
  potp = 0.0;
  #
  # L loop
  #
  loffset = 1
  for l in range(1,lmax+1):
    # M loop
    moffset = 0
    for m in range(0,l+1):
      fac1 = factorial[l][m];
      if (m==0):
            den1 += np.sum(fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]));
            pot1 += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
            potr += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            pott += np.sum(fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]));
            moffset+=1;
      else:
            cosm = np.cos(phi*m);
            sinm = np.sin(phi*m);
            den1 += np.sum(fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
            pot1 += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
            potr += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            pott += np.sum(fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm ));
            potp += np.sum(fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm ));
            moffset +=2;
    loffset+=(2*l+1)
  #
  #
  #
  densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  potlfac = 1.0/scale;
  den0  *= densfac;
  den1  *= densfac;
  pot0  *= potlfac;
  pot1  *= potlfac;
  potr  *= potlfac/scale;
  pott  *= potlfac*sinth;
  potp  *= potlfac;
  return den0,den1,pot0,pot1,potr,pott,potp





def all_eval_particles(Particles, expcoef, sph_file, mod_file,verbose,L1=0,L2=1000):
  lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap=cmap,scale=scale)
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  # begin function
  norb = len(Particles.xpos)
  r = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos + Particles.zpos*Particles.zpos)**0.5
  rr = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos)**0.5
  phi = np.arctan2(Particles.ypos,Particles.xpos)
  #
  # is this over planar or 3d r??
  costh = Particles.zpos/r
  #
  # allocate arrays
  den0 = np.zeros(norb)
  pot0 = np.zeros(norb)
  potr = np.zeros(norb)
  den1 = np.zeros(norb)
  pot1 = np.zeros(norb)
  pott = np.zeros(norb)
  potp = np.zeros(norb)
  #
  for part in range(0,norb):
      if (verbose > 0) & ( ((float(part)+1.) % 1000. == 0.0) | (part==0)): print_progress(part,norb,'spheresl.all_eval_particles')
      sinth = -np.sqrt(np.abs(1.0 - costh[part]*costh[part]));
      fac1 = factorial[0][0];
      #
      # use the basis to get the density,potential,force arrays
      #
      # these three need to be stuffed together into one call to save loops
      dend,dpot,potd = get_halo_dens_pot_force(r[part], lmax, nmax, evtable, eftable, xi, d0, p0, cmap=cmap, scale=scale)
      #
      #
      legs,dlegs = dlegendre_R(lmax,costh[part])
      #
      den0[part] = np.sum(fac1 * expcoef[0]*dend[0]);
      pot0[part] = np.sum(fac1 * expcoef[0]*potd[0]);
      potr[part] = np.sum(fac1 * expcoef[0]*dpot[0]);
      #
      # L loop
      #
      loffset = 1
      for l in range(1,lmax+1):
        if (l > (L2+1)) | (l < (L1+1)): continue
        # M loop
        moffset = 0
        for m in range(0,l+1):
          fac1 = factorial[l][m];
          if (m==0):
                den1[part] += np.sum(fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]));
                pot1[part] += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
                potr[part] += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
                pott[part] += np.sum(fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]));
                moffset+=1;
          else:
                cosm = np.cos(phi[part]*m);
                sinm = np.sin(phi[part]*m);
                den1[part] += np.sum(fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
                pot1[part] += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
                # this should be checked in other evaluations
                potr[part] += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
                pott[part] += np.sum(fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm ));
                potp[part] += np.sum(fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm ));
                moffset +=2;
        loffset+=(2*l+1)
  # END particle loop
  #
  #
  #
  densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  potlfac = 1.0/scale;
  den0  *= densfac;
  den1  *= densfac;
  pot0  *= potlfac;
  pot1  *= potlfac;
  potr  *= potlfac/scale;
  pott  *= potlfac*sinth;
  potp  *= potlfac;
  return den0,den1,pot0,pot1,potr,pott,potp,rr









def print_progress(current_n,total_orb,module):
    last = 0
    #print current_n,total_orb
    if float(current_n+1)==float(total_orb): last = 1

    #print last

    bar = ('=' * int(float(current_n)/total_orb * 20.)).ljust(20)
    percent = int(float(current_n)/total_orb * 100.)

    if last:
        bar = ('=' * int(20.)).ljust(20)
        percent = int(100)
        print "%s: [%s] %s%%" % (module, bar, percent)
        
    else:
        sys.stdout.write("%s: [%s] %s%%\r" % (module, bar, percent))
        sys.stdout.flush()





'''

# reversion version

def all_eval(r, costh, phi, expcoef, sph_file, mod_file):
  lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap=cmap,scale=scale)
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  # begin function
  sinth = -np.sqrt(np.abs(1.0 - costh*costh));
  fac1 = factorial[0][0];
  #
  # use the basis to get the density,potential,force arrays (which have already been read in somewhere?)
  dend = get_halo_dens(r, lmax, nmax, evtable, eftable, xi, d0, cmap=0, scale=1.0)
  potd = get_halo_pot_matrix(r, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0)
  dpot = get_halo_force(r, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0)
  #
  #
  legs,dlegs = dlegendre_R(lmax,costh)
  #
  den0 = np.sum(fac1 * expcoef[0]*dend[0]);
  pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
  potr = np.sum(fac1 * expcoef[0]*dpot[0]);
  den1 = 0.0;
  pot1 = 0.0;
  pott = 0.0;
  potp = 0.0;
  #
  # L loop
  #
  loffset = 1
  for l in range(1,lmax+1):#(int l=1, loffset=1; l<=lmax; loffset+=(2*l+1), l++) {
    # at end, add in loffset+=(2*l+1)
    #if (l<L1 || l>L2) continue;
    #
    # M loop
    moffset = 0
    for m in range(0,l+1):#(int m=0, moffset=0; m<=l; m++) {
      #print l,m,loffset,moffset
      fac1 = factorial[l][m];
      if (m==0):
            #den1 += fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]);
            #pot1 += fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]);
            #dpot += fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]);
            #pott += fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]);
            den1 += np.sum(fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]));
            pot1 += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
            dpot += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            pott += np.sum(fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]));
            moffset+=1;
      else:
            cosm = np.cos(phi*m);
            sinm = np.sin(phi*m);
            #den1 += fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm );
            #pot1 += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm );
            #dpot += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm );
            #pott += fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm );
            #potp += fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm );
            den1 += np.sum(fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
            pot1 += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
            dpot += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            pott += np.sum(fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm ));
            potp += np.sum(fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm ));
            moffset +=2;
    loffset+=(2*l+1)
  #
  #
  #
  densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  potlfac = 1.0/scale;
  den0  *= densfac;
  den1  *= densfac;
  pot0  *= potlfac;
  pot1  *= potlfac;
  potr  *= potlfac/scale;
  pott  *= potlfac*sinth;
  potp  *= potlfac;
  return den0,den1,pot0,pot1,potr,pott,potp



factorial = factorial_return(lmax)


def accumulate(x, y, z, mass, factorial, rscl=1.0):
    #double fac, fac1, fac2, fac4;
    # set the overall poisson normalizer
    fac0 = -4.0*np.pi;
    # set a guard
    dsmall = 1.0e-20;
    #
    # set the blank arrays
    expcoef.setsize(0, lmax*(lmax+2), 1, nmax);
    expcoef.zero();		// Need this?
    work1.setsize(1, nmax);
    #
    # skipping ability to compute covariance for now, could add later.
    #
    # get normalization factorial, will be read in
    #
    # check on how many particles go in
    #used = 0;
    #
    #//======================
    #// Compute coefficients 
    #//======================
    #
    r2 = (x*x + y*y + z*z);
    r = sqrt(r2) + dsmall;
    costh = z/r;
    phi = np.arctn2(y,x);
    rs = r/rscl;
	used+=1;
    #
    # get the potential matrix based on radius
    #sl->get_pot(potd, rs);
    potd = get_halo_pot(r)
    #
    # get the legendre polynomial values based on costh
    legs = legendre_R(lmax, costh);
    #
    #// L loop
    loffset = 0
    for l in range(0,lmax+1):#(int l=0, loffset=0; l<=lmax; loffset+=(2*l+1), l++) {
        # at the end, don't forget to update loffset
        #// M loop
        moffset = 0
        for m in range(0,l+1):#(int m=0, moffset=0; m<=l; m++) {
            if (m==0):# {
                fac = factorial[l][m] * legs[l][m];
                for (int n=1; n<=nmax; n++):
                    fac4 = potd[l][n]*fac*fac0;
                    expcoef[loffset+moffset][n] += fac4 * mass;
                moffset+=1;
            else:
                fac = factorial[l][m] * legs[l][m];
                fac1 = fac*np.cos(phi*m);
                fac2 = fac*np.sin(phi*m);
                for (int n=1; n<=nmax; n++):
                    fac4 = potd[l][n]*fac0;
                    expcoef[loffset+moffset  ][n] += fac1 * fac4 * mass;
                    expcoef[loffset+moffset+1][n] += fac2 * fac4 * mass;
                moffset+=2;
        loffset += (2*l+1)



def make_coefs():
    #
    # definition to collapse coefficients along the n axis?
{
  if (mpi) {

    for (int l=0; l<=lmax*(lmax+2); l++) {
      MPI_Allreduce(&expcoef[l][1], &work1[1], nmax, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      expcoef[l] = work1;
    }

    if (compute_covar) {
      for (int n=1; n<=nmat; n++) {
	MPI_Allreduce(&cc[n][1], &work2[1], nmat, MPI_DOUBLE,
		      MPI_SUM, MPI_COMM_WORLD);
	
	cc[n] = work2;
      }
    }
  }
}






phi = np.linspace(0.,np.pi,100.)

from scipy import special

xx = special.sph_harm(6,35,0.,phi)

phiphi = np.zeros([100.,numr])
for i in range(0,numr):
    phiphi[:,i] = xx


rr = np.zeros([100.,numr])
for i in range(0,100):
    rr[i,:] = eftable[6,25]

'''



