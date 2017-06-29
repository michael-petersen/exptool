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



