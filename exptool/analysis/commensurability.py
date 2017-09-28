'''
  ______   ______   .___  ___. .___  ___.  _______ .__   __.      _______. __    __  .______          ___      .______    __   __       __  .___________.____    ____ 
 /      | /  __  \  |   \/   | |   \/   | |   ____||  \ |  |     /       ||  |  |  | |   _  \        /   \     |   _  \  |  | |  |     |  | |           |\   \  /   / 
|  ,----'|  |  |  | |  \  /  | |  \  /  | |  |__   |   \|  |    |   (----`|  |  |  | |  |_)  |      /  ^  \    |  |_)  | |  | |  |     |  | `---|  |----` \   \/   /  
|  |     |  |  |  | |  |\/|  | |  |\/|  | |   __|  |  . `  |     \   \    |  |  |  | |      /      /  /_\  \   |   _  <  |  | |  |     |  |     |  |       \_    _/   
|  `----.|  `--'  | |  |  |  | |  |  |  | |  |____ |  |\   | .----)   |   |  `--'  | |  |\  \----./  _____  \  |  |_)  | |  | |  `----.|  |     |  |         |  |     
 \______| \______/  |__|  |__| |__|  |__| |_______||__| \__| |_______/     \______/  | _| `._____/__/     \__\ |______/  |__| |_______||__|     |__|         |__|     
commensurability.py: part of exptool
      tools to handle various commensurability finding items




 '''
from __future__ import absolute_import, division, print_function, unicode_literals

# standard imports
import numpy as np
from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# exptool imports
from exptool.utils import kde_3d
from exptool.utils import utils


# also check the scipy
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import filters
import scipy.ndimage.filters

# need a check here to see if this will actually import and a clause if not
from skimage.morphology import skeletonize







# read in orbit integrations

# follows file format of integrator output
def read_integrations(infile):
    f = open(infile,'r')
    D = {}
    D['R'] = {}
    D['V'] = {}
    D['dT'] = {}
    D['TX'] = {}
    D['TY'] = {}
    #
    linenum = 0
    for line in f:
        d = [float(q) for q in line.split()]
        npoints = int(d[0])
        D['R'][linenum] = d[1]
        D['V'][linenum] = d[2]
        D['dT'][linenum] = d[3]
        D['TX'][linenum] = d[4:npoints+4]
        D['TY'][linenum] = d[npoints+4:(2*npoints)+4]
        linenum += 1
    #
    f.close()
    return D





def make_orbit_plot(D,num,outdir='',T=''):
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_axes([0.25,0.25,0.6,0.6])
    ax.plot(D['TX'][num],D['TY'][num])
    ax.set_title('T'+T+', R'+str(np.round(D['R'][num],3))+', V'+str(np.round(D['V'][num],3)),size=18)
    ax.set_ylabel('Ybar',size=18)
    ax.set_xlabel('Xbar',size=18)
    ax.axis([-1.1*D['TX'][num][0],1.1*D['TX'][num][0],-1.1*D['TX'][num][0],1.1*D['TX'][num][0]])
    for label in ax.get_xticklabels(): label.set_rotation(30); label.set_horizontalalignment("right")
    plt.savefig(outdir+'T'+T+'R'+str(np.round(D['R'][num],3))+'V'+str(np.round(D['V'][num],3))+'.png')




'''
    
D = np.genfromtxt('/scratch/mpetersen/processed.txt')

Rarr = D[:,0].reshape(rads.size,vels.size)
Varr = D[:,1].reshape(rads.size,vels.size)
Aarr = D[:,2].reshape(rads.size,vels.size)





# pull a contour from the area array
c = cntr.Cntr(Rarr.T,Varr.T,Aarr.T/(np.pi*Rarr.T*Rarr.T))

res = c.trace(0.03,nchunk=2)

for j in range(0,len(res)):
    xcon = []
    ycon = []
    try:
        if len(res[j])>10:
            for i in range(0,len(res[j])):
    	        xcon.append(res[j][i][0])
    	        ycon.append(res[j][i][1])
            #print xcon,ycon
            XCON = np.array(xcon)
            YCON = np.array(ycon)
            ax2.plot(XCON,YCON,color='red')
        else:
	    pass
    except:
	pass


'''



def map_skeleton(Rarr,Varr,Aarr,sigma=(3.,3.),ridge_cutoff=0.5):
    
    area_map = 256.*Aarr.T/(np.pi*Rarr.T*Rarr.T)

    #
    # only do the needed derivatives
    #
    #rp_00 = filters.gaussian_filter(area_map,sigma,order=0)      # r
    #rp_01 = filters.gaussian_filter(area_map,sigma,order=(0,1))  # ry
    #rp_10 = filters.gaussian_filter(area_map,sigma,order=(1,0))  # rx
    rp_11 = filters.gaussian_filter(area_map,sigma,order=(1,1))  # rxy
    rp_20 = filters.gaussian_filter(area_map,sigma,order=(2,0))  # rxx
    rp_02 = filters.gaussian_filter(area_map,sigma,order=(0,2))  # ryy
    
    #
    # set up the hessian
    #
    hess = np.array([[rp_20.T,rp_11.T],[rp_11.T,rp_02.T]]).T

    #
    # solve the eigenvalue problem
    #
    w, v = np.linalg.eig(hess)
    
    #
    # find maximum eigenvalue
    # 
    ridge_map = np.max(w,axis=2)
    
    #
    # now threshold the map
    #
    threshold_map = np.zeros_like(ridge_map)
    good_values = np.where(ridge_map > ridge_cutoff)
    threshold_map[good_values] = 1.
    
    skeleton = skeletonize(threshold_map)

    #
    # zero out bad values
    #
    ridge_trace = np.zeros_like(ridge_map)
    good_values = np.where(skeleton < 0.002)
    ridge_trace[good_values] = np.nan
    
    
    return ridge_trace
    
    
def print_skeleton(infile,pskel,Rarr,Varr):

    f = open(infile,'w')

    pp = np.flipud(pskel).reshape(-1,)
    ccc = (Rarr.T).reshape(-1,)
    ddd = (Varr.T).reshape(-1,)

    for indx in range(0,len(pp)):
        print >>f,ccc[indx],ddd[indx],pp[indx]
    
    f.close()



