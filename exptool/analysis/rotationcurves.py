
# tools to fit rotation curves. NEEDS MUCH CONSTRUCTION
# from NewHorizon bars project

import numpy as np
astronomicalG = 0.0000043009125 # gravitational constant, (km/s)^2 * kpc / Msun


def kuzmin_rotation(R,c,M,G=astronomicalG):
    """see BT08
    
    this assumes z=0, i.e. in-plane only.
    note that R is 2d in this case.
    """
    return np.sqrt(2*G*np.power(10.,M)*R*R*np.power(c*c+R*R,-1.5))

def plummer_rotation(r,b,M,G=astronomicalG):
    """see BT08 eq. 2.44a
    
    this is a spherical model, such that r is 3d.
    """
    return np.sqrt(2*G*np.power(10.,M)*r*r*np.power(b*b+r*r,-1.5))

def isochrone_rotation(R,b,M,G=astronomicalG):
    """see BT08"""
    a = np.sqrt(R*R+b*b)
    return np.sqrt(G*np.power(10.,M)*R*R*np.power(a*(b+a)*(b+a),-1))


def kuzmin_plummer(R,b,c,M,x,G=astronomicalG):
    """combine a Kuzmin and Plummer rotation model"""
    disc = kuzmin_rotation(R,c,M,G=astronomicalG)
    bulge = plummer_rotation(R,b,M,G=astronomicalG)
    return np.sqrt((1-x)*disc*disc+x*bulge*bulge)


def read_gal(gnum):
    infile = '/Volumes/External1/BarDetective/summaries/summary371_{}.txt'.format(gnum)
    aaa = np.genfromtxt(infile,skip_header=1,delimiter=',')
    # 0: rbins, 1: total rotation, 2: dh rotation, 3: stellar rotation, 4: gas rotation
    bbb = np.genfromtxt(infile,max_rows=1,delimiter=',')
    # 0: halo mass, 1: disc mass, 2: gas mass
    print(np.log10(bbb[0]))
    r,vcircd,vcirch = aaa[:,0],aaa[:,3],aaa[:,2]
    return r,vcircd,vcirch

def read_gal2(gnum):
    infile = '/Volumes/External1/BarDetective/summaries/summary_{}.txt'.format(gnum)
    aaa = np.genfromtxt(infile,skip_header=1,delimiter=',')
    # 0: rbins, 1: total rotation, 2: dh rotation, 3: stellar rotation, 4: gas rotation
    bbb = np.genfromtxt(infile,max_rows=1,delimiter=',')
    # 0: halo mass, 1: disc mass, 2: gas mass
    print(np.log10(bbb[0]))
    r,vcircd,vcirch = aaa[:,0],aaa[:,3],aaa[:,2]
    return r,vcircd,vcirch




"""
r,vcircd,vcirch = read_gal(1252)
r,vcircd,vcirch = read_gal(2)

popt, pcov = curve_fit(kuzmin_plummer, r,vcircd,bounds=([0.01,0.5,5.5,0.],[0.1*np.nanmax(r),0.6*np.nanmax(r),13,0.9]))
print(popt)
plt.plot(np.log10(r),kuzmin_plummer(r,*popt),color='red')

popt, pcov = curve_fit(kuzmin_rotation, r,vcircd,bounds=([0.01,5.5],[0.6*np.nanmax(r),11]))
print(popt)

plt.plot(np.log10(r),vcircd,color='grey')
plt.plot(np.log10(r),kuzmin_rotation(r,*popt),color='blue')

popt, pcov = curve_fit(plummer_rotation, r,vcirch,bounds=([0.01,5.5],[0.9*np.nanmax(r),12]))
print(popt)
plt.plot(np.log10(r),vcirch,color='grey',linestyle='dashed')
plt.plot(np.log10(r),plummer_rotation(r,*popt),color='blue',linestyle='dashed')
"""


def log_likelihood_kuzmin(theta, x, y, yerr):
    a,M = theta
    model = kuzmin_rotation(x,a,M)
    return -0.5 * np.sum( ((y - model)/yerr)**2.)

def log_likelihood_kuzmin_plummer(theta, x, y, yerr):
    b,c,M,x = theta
    model = kuzmin_plummer(x,b,c,M,x,G=astronomicalG)
    return -0.5 * np.sum( ((y - model)/yerr)**2.)

def lnprior_kuzmin(theta):
    a,M = theta
    if ((a>.01) & (a<50.) & (M>4) & (M<13)):
        return 0.0
    else:
        return -np.inf
    
def lnprior_kuzmin_plummer(theta):
    b,c,M,x = theta
    if ((c>.01) & (c<5.) & (M>4) & (M<13) & (b>.001) & (b<c) & (x>0.) & (x<0.9)):
        return 0.0
    else:
        return -np.inf
    
def lnprob_kuzmin(theta, x, y, yerr):
    lp = lnprior_kuzmin(theta)
    if not np.isfinite(lp):#check if lp is infinite:
        return -np.inf
    return lp + log_likelihood_kuzmin(theta, x, y, yerr) #recall if lp not -inf, its 0, so this just returns likelihood

def lnprob_kuzmin_plummer(theta, x, y, yerr):
    lp = lnprior_kuzmin_plummer(theta)
    if not np.isfinite(lp):#check if lp is infinite:
        return -np.inf
    return lp + log_likelihood_kuzmin_plummer(theta, x, y, yerr) #recall if lp not -inf, its 0, so this just returns likelihood



"""
import emcee


initial = np.array([0.1,1., 11.,0.]) + 0.1 * np.random.randn(32,4)
nwalkers, ndim = initial.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_kuzmin_plummer, args=(r,vcircd, (r**-.1)*vcircd))
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_kuzmin_plummer, args=(r,vcircd, 0.1*vcircd))



sampler.run_mcmc(initial, 2500, progress=True);
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

plt.scatter(np.log10(flat_samples[:,0]),flat_samples[:,1],color='black',s=1.)

print(np.nanpercentile(flat_samples[:,0],50.),\
                                    np.nanpercentile(flat_samples[:,1],50.),\
                                            np.nanpercentile(flat_samples[:,2],50.),\
                                            np.nanpercentile(flat_samples[:,3],50.))

"""


def hern_bulge_rotation(r,b):
    return np.sqrt((1./(r)) * hern_bulge_mass(r,b))

def hern_bulge_mass(r,b):
    """mass enclosed in a Hernquist bulge"""
    rb = r/b
    return ((rb*rb)/(2*(1+rb)**2.))

def exp_disc_rotation(r,a):
    return np.sqrt((1./(r)) * exp_disc_mass(r,a))

def exp_disc_mass(r,a):
    return 1. - (np.exp(-r/a) * (1.+(r/a)))


def disc_bulge_rotation(r,a,b,f=0.1):
    discmass = exp_disc_mass(r,a)
    bulgemass = hern_bulge_mass(r,b)
    
    return np.sqrt((1./r) * ((1-f)*discmass + f*bulgemass))
    
    
def find_peak(rvals,vvals):
    return rvals[np.where(vvals==np.nanmax(vvals))][0]




