###########################################################################################
#
#  centering.py
#     Tools to 
#
#
# 09 Jul 2021: First introduction
#
#
'''     

centering.py (part of exptool.basis)
    Tools for wrangling unruly n-body data
    Originally used for NewHorizon bar detection


# read in the test data
P = np.genfromtxt('testbar.small.part',names=True,dtype=None)

rtest,fpower,fangle,fvpower = fourier_tabulate(P['x'],P['y'],P['vx'],P['vy'],P['mass'])

# the simplest test is the ratio of m=4 to m=2 Fourier velocities

plt.plot(rtest,fvpower[4]/fvpower[2])
# when this ratio is over 1, it's the end of the bar.



'''

import numpy as np



import numpy as np

def xnorm(vec): return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

def compute_fourier(x,y,w,harmonic=2):
    phi = np.arctan2(y,x)
    A = np.nansum(w*np.cos(harmonic*phi))/np.nansum(w)
    B = np.nansum(w*np.sin(harmonic*phi))/np.nansum(w)
    mod = np.sqrt(A*A + B*B)
    angle = np.arctan2(B,A)/2.
    return mod,angle
    

def recentre_pos_and_vel(x,y,z,vx,vy,vz,mass,r_max):
    """compute center of mass and mass weighted velocity center within r_max and recentre position and velocity
    """
    mask = (x**2 + y**2 + z**2) < r_max**2
    mass_tot = np.sum(mass[mask])
    x_cm = np.sum(x[mask]*mass[mask])/mass_tot
    y_cm = np.sum(y[mask]*mass[mask])/mass_tot
    z_cm = np.sum(z[mask]*mass[mask])/mass_tot
    vx_cm = np.sum(vx[mask]*mass[mask])/mass_tot
    vy_cm = np.sum(vy[mask]*mass[mask])/mass_tot
    vz_cm = np.sum(vz[mask]*mass[mask])/mass_tot
    return x-x_cm,y-y_cm,z-z_cm,vx-vx_cm,vy-vy_cm,vz-vz_cm

def rotate(x,y,z,axis,angle):
    # HERE:
    x_rot=x
    y_rot=y
    z_rot=z
    axisx=axis[0]*np.ones(len(x))
    axisy=axis[1]*np.ones(len(x))
    axisz=axis[2]*np.ones(len(x))

    dot=x*axisx+y*axisy+z*axisz

    crossx=axisy*z-axisz*y
    crossy=axisz*x-axisx*z
    crossz=axisx*y-axisy*x

    cosa=np.cos(angle)
    sina=np.sin(angle)

    x_rot=x*cosa+crossx*sina+axis[0]*dot*(1-cosa)
    y_rot=y*cosa+crossy*sina+axis[1]*dot*(1-cosa)
    z_rot=z*cosa+crossz*sina+axis[2]*dot*(1-cosa)

    return x_rot,y_rot,z_rot

def compute_rotation_to_vec(x,y,z,vx,vy,vz,mass,vec):
    """compute rotation axis and anglr to vector vec
    """
    Lxtot = np.sum(y*vz*mass - z*vy*mass)
    Lytot = np.sum(z*vx*mass - x*vz*mass)
    Lztot = np.sum(x*vy*mass - y*vx*mass)
    
    L = np.zeros(3)
    L = np.hstack([Lxtot,Lytot,Lztot])

    L = L/np.linalg.norm(L)
    vec = vec/np.linalg.norm(vec)
    
    axis = np.zeros(3)
    axis = np.cross(L,vec)
    axis = axis/np.linalg.norm(axis)

    c = np.dot(L,vec)                 # cosine of the angle
    angle = np.arccos(np.clip(c, -1, 1)) # angle between L and vec
    #@@@angle = np.arccos(c) # angle between L and vec

    return axis, angle




def shrinking_sphere(x,y,z,vx,vy,vz,w,rmin=1.,stepsize=0.5,tol=0.001,verbose=0):
    #
    if stepsize >= 1.: 
        print('reverting to default step size:')
        stepsize=0.5
    #   
    tshiftx = np.nanmedian(np.nansum(w*x)/np.nansum(w))
    tshifty = np.nanmedian(np.nansum(w*y)/np.nansum(w))
    tshiftz = np.nanmedian(np.nansum(w*z)/np.nansum(w))
    #
    # first guess and normalisation
    x -= tshiftx
    y -= tshifty
    z -= tshiftz
    rval = np.sqrt(x*x + y*y + z*z)
    rmax = np.nanmax(rval)
    rmax0 = np.nanmax(rval)
    #
    if verbose: print('initial guess: {0:5.0f},{1:5.0f},{2:5.0f}'.format(tshiftx,tshifty,tshiftz))
    while rmax > rmin:
        #print(rmax)
        u = np.where(rval<stepsize*rmax)[0]
        #
        # also need a minimum particle guard here: minimum 1% of particles
        if float(u.size)/float(x.size) < tol: 
            print('too few particles to continue at radius ratio {}'.format(stepsize*rmax/rmax0))
            break
        #
        # compute the centre-of-mass
        comx = np.nanmedian(np.nansum(w[u]*x[u])/np.nansum(w[u]))
        comy = np.nanmedian(np.nansum(w[u]*y[u])/np.nansum(w[u]))
        comz = np.nanmedian(np.nansum(w[u]*z[u])/np.nansum(w[u]))
        #
        x -= comx
        y -= comy
        z -= comz
        #
        tshiftx += comx
        tshifty += comy
        tshiftz += comz
        #
        rval = np.sqrt(x*x + y*y + z*z)
        rmax *= stepsize
    #
    comvx = np.nanmedian(np.nansum(w[u]*vx[u])/np.nansum(w[u]))
    comvy = np.nanmedian(np.nansum(w[u]*vy[u])/np.nansum(w[u]))
    comvz = np.nanmedian(np.nansum(w[u]*vz[u])/np.nansum(w[u]))    
    if verbose: 
        print('final shift: {0:5.0f},{1:5.0f},{2:5.0f}'.format(tshiftx,tshifty,tshiftz))
        print('final velocity shift: {0:5.0f},{1:5.0f},{2:5.0f}'.format(comvx,comvy,comvz))
    vx -= comvx
    vy -= comvy
    vz -= comvz
    return x,y,z,vx,vy,vz



def compute_density_profile(R,W,rbins=10.**np.linspace(-3.7,0.3,100)):

    dens = np.zeros(rbins.size)
    menc = np.zeros(rbins.size)
    potp = np.zeros(rbins.size)
    
    astronomicalG = 0.0000043009125
    
    rbinstmp = np.concatenate([rbins,[2.*rbins[-1]-rbins[-2]]])

    for indx,val in enumerate(rbinstmp[0:-1]):
        w = np.where((R>rbinstmp[indx]) & (R<rbinstmp[indx+1]))[0]
        wenc = np.where((R<rbinstmp[indx+1]))[0]
        shellsize = (4/3.)*np.pi*(rbinstmp[indx+1]**3.-rbinstmp[indx]**3.)
        dens[indx] = np.nansum(W[w])/shellsize
        menc[indx] = np.nansum(W[wenc])
        potp[indx] = np.sqrt(astronomicalG*menc[indx]/(rbinstmp[indx+1]))
        
    return dens,menc,potp




def compute_fourier(x,y,w,harmonic=2):
    """compute Fourier moments"""
    phi = np.arctan2(y,x)
    A = np.nansum(w*np.cos(harmonic*phi))/np.nansum(w)
    B = np.nansum(w*np.sin(harmonic*phi))/np.nansum(w)
    mod = np.sqrt(A*A + B*B)
    angle = np.arctan2(B,A)/2.
    return mod,angle
    

def compute_fourier_vel(x,y,vx,vy,w,harmonic=2):
    """compute the velocity-weighted Fourier moments"""
    phi = np.arctan2(y,x)
    # the velocity perpendicular to the bar is the most powerful, but it requires knowing the bar position angle
    vel = vy
    # in the absence of knowing where the bar is, use tangential velocity
    vel = (x*vy - y*vx)/np.sqrt(x*x + y*y)
    A = np.nansum(w*vel*np.cos(harmonic*phi))/np.nansum(w)
    B = np.nansum(w*vel*np.sin(harmonic*phi))/np.nansum(w)
    mod = np.sqrt(A*A + B*B)
    return mod


def fourier_tabulate(xxr,yyr,vxxr,vyyr,mass):
    """compute the Fourier moments in radial bins"""
    rval = np.sqrt(xxr*xxr + yyr*yyr)
    rtest = np.linspace(0.,np.nanpercentile(rval,75),int(np.power(rval.size,0.25)))
    dr = rtest[1]-rtest[0]
    rbin = (np.floor(rval/dr)).astype('int')
    fpower  = np.zeros([5,rtest.size])
    fangle  = np.zeros([5,rtest.size])
    fvpower = np.zeros([5,rtest.size])
    #
    for ir,rv in enumerate(rtest):
        w = np.where( (rbin==ir))# & (np.abs(zr)<200./1.e6))
        for h in [0,1,2,3,4]:
            fpower[h][ir],fangle[h][ir] = compute_fourier(xxr[w],yyr[w],mass[w],harmonic=h)
            fvpower[h][ir] = compute_fourier_vel(xxr[w],yyr[w],vxxr[w],vyyr[w],mass[w],harmonic=h)

    return rtest,fpower,fangle,fvpower



"""
def print_galaxy(GAL,gal,outputnum):
    
    indir = '/Volumes/External1/BarDetective/'
    f = open(indir+'summary{0}_{1}.txt'.format(outputnum,gal),'w')
    
    print('{},{},{}'.format(GAL[gal]['dmass'],GAL[gal]['smass'],GAL[gal]['gmass']),file=f)
    
    for indx in range(0,len(GAL[gal]['rbins'])):
        print('{},{},{},{},{}'.format(GAL[gal]['rbins'][indx],\
              GAL[gal]['potpt'][indx],\
              GAL[gal]['potph'][indx],\
              GAL[gal]['potpd'][indx],\
              GAL[gal]['potpg'][indx]),file=f)
    f.close()
    
"""









def measure_bulge_mass(x,y,vx,vy,mass):
    rpos = np.sqrt(x*x + y*y)
    vtan = (x*vy- y*vx)/rpos
    # define prograde
    direction = np.sign(np.nanmean(vtan))
    vtan *= direction
    negvel = np.where(vtan<0)[0]
    print('50% bulge: {0:3.2}kpc'.format(np.nanpercentile(rpos[negvel],50.)))
    return 2.*np.nansum(mass[negvel])


