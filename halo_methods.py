#########################################################
#
#  Utility routines for Spherical expansion
#
#  MSP 5.1.2016
#
import numpy as np
from scipy import interpolate


def parse_slgrid(file,verbose=0):
    f = open(file,'rb')
    #
    # read the header
    #
    a = np.fromfile(f, dtype=np.uint32,count=4)
    lmax = a[0]
    nmax = a[1]
    numr = a[2]
    cmap = a[3]
    b = np.fromfile(f, dtype=np.float64,count=3)
    rmin = b[0]
    rmax = b[1]
    scale = b[2]
    
    if verbose > 0:
        print 'LMAX=',lmax
        print 'NMAX=',nmax
        print 'NUMR=',numr
        print 'CMAP=',cmap
        print 'RMIN=',rmin
        print 'RMAX=',rmax
        print 'SCALE=',scale

    f.close()
    
    return lmax,nmax,numr,cmap,rmin,rmax,scale




def read_cached_table(file,verbose=0,retall=True):
    f = open(file,'rb')
    #
    # read the header
    #
    a = np.fromfile(f, dtype=np.uint32,count=4)
    lmax = a[0]
    nmax = a[1]
    numr = a[2]
    cmap = a[3]
    a = np.fromfile(f, dtype='<f8',count=3) # this must be doubles
    rmin = a[0]
    rmax = a[1]
    scale = a[2]
    #
    # set up the matrices
    #
    ltable = np.zeros(lmax+1)
    evtable = np.ones([lmax+1,nmax+1])
    eftable = np.ones([lmax+1,nmax+1,numr])
    #
    #
    for l in range(0,lmax+1): # I think the padding needs to be here? test.
        #
        # The l integer
        #
        ltable[l] = np.fromfile(f, dtype=np.uint32,count=1)
        evtable[l,1:nmax+1] = np.fromfile(f,dtype='f8',count=nmax)
        for n in range(1,nmax+1):
            if verbose==1: print l,n
            #
            # loops for different levels go here
            #
            eftable[l,n,:] = np.fromfile(f,dtype='f8',count=numr)

    f.close()
    
    if retall:
        return lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable
    else:
        return ltable,evtable,eftable


def xi_to_r(xi,cmap,scale):
    if (cmap==1):
        if (xi<-1.0): print "xi < -1!" 
        if (xi>=1.0): print "xi >= 1!"
        ret =(1.0+xi)/(1.0 - xi) * scale;
    if (cmap==2):
        ret = np.exp(xi);
    if (cmap==0):
        if (xi<0.0): print "xi < 0!"
        ret = xi;
    return ret


def r_to_xi(r,cmap,scale):
    if (cmap==1):
        if (r<0.0): print "radius < 0!"
        ret =  (r/scale-1.0)/(r/scale+1.0);
    if (cmap==2):
        if (r<=0.0): print "radius <= 0!"
        ret = np.log(r);
    if (cmap==0):
        ret = r;
    return ret;


def d_xi_to_r(xi,cmap=0,scale=1.):
    if (cmap==1):
        if (xi<-1.0): print "xi < -1!" 
        if (xi>=1.0): print "xi >= 1!"
        ret = 0.5*(1.0-xi)*(1.0-xi)/scale;
    if (cmap==2):
        ret = np.exp(-xi);
    if (cmap==0):
        if (xi<0.0): print "xi < 0!"
        ret = 1.0;
    return ret


    



def read_sph_model_table(file):
    f = open(file)
    radius = []
    density = []
    mass = []
    potential = []
    for line in f:
        q = [d for d in line.split()]
        if len(q)==4:
            radius.append(float(q[0]))
            density.append(float(q[1]))
            mass.append(float(q[2]))
            potential.append(float(q[3]))

    f.close()
    
    return np.array(radius),np.array(density),np.array(mass),np.array(potential)


# R,D,M,P = halo_methods.read_sph_model_table(


def init_table(modelfile,numr,rmin,rmax,cmap=0,scale=1.0,spline=True):
    R1,D1,M1,P1 = read_sph_model_table(modelfile)
    fac0 = 4.*np.pi
    xi = np.zeros(numr)
    r = np.zeros(numr)
    p0 = np.zeros(numr)
    d0 = np.zeros(numr)
    if (cmap==1):
        xmin = (rmin/scale - 1.0)/(rmin/scale + 1.0);
        xmax = (rmax/scale - 1.0)/(rmax/scale + 1.0);           
    if (cmap==2):
        xmin = log(rmin);
        xmax = log(rmax);
    if (cmap==0):
        xmin = rmin;
        xmax = rmax;
    dxi = (xmax-xmin)/(numr-1);
    #
    #
    if spline==True:
        pfunc = interpolate.splrep(R1, P1, s=0)
        dfunc = interpolate.splrep(R1, fac0*D1, s=0)
    #
    #
    for i in range(0,numr):#(i=0; i<numr; i++):
        xi[i] = xmin + dxi*i;
        r[i] = xi_to_r(xi[i],cmap,scale);
        if spline==False:
            p0[i] = P1[ (abs(r[i]-R1)).argmin() ]; # this is the spherical potential at that radius
            d0[i] = fac0 * D1[ (abs(r[i]-R1)).argmin() ]; # this is the spherical density at that radius (4pi*dens)
        if spline==True:
            p0[i] = interpolate.splev(r[i], pfunc, der=0)
            d0[i] = interpolate.splev(r[i], dfunc, der=0)
    return xi,r,p0,d0


