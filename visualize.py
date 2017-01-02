#
# exptool.visualize
#
#    a collection of routines to visualize first-check items
#
#    08-27-2016


'''

fig = visualize.show_dump('/scratch/mpetersen/Disk064a/OUT.run064a.01000','star')

'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# exptool routines
import psp_io
import kde_3d
import trapping


def show_dump(infile,comp,type='pos',transform=True,\
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1):

    # read in file

    PSPDump = psp_io.Input(infile,comp=comp)

    if transform:
        PSPDump = trapping.BarTransform(PSPDump)

    '''
    try:
        PSPDump = psp_io.convert_to_dict(PSPDump)

    except:
        print 'visualize.show_dump(): PSP Dump is already a dictionary. How did you do that?'

    '''
    # do a kde

    if type=='pos':

        # XY
        kdeX,kdeY,kdePOSXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

        print 'max:',np.max(kdePOSXY)
        # make a log guard
        eps = np.min(PSPDump.mass)

        

        # change to log surface density
        kdePOSXY = np.log10(kdePOSXY+eps)

        # XZ
        kdeXZx,kdeXZz,kdePOSXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                      gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                      weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)

        # change to log surface density
        kdePOSXZ = np.log10(kdePOSXZ+eps)

        # ZY
        kdeZYz,kdeZYy,kdePOSZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                                  gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                                  weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)

        # change to surface density
        kdePOSZY = np.log10(kdePOSZY+eps)

        # set up the figure

        maxlev_edge = np.max([np.max(kdePOSXZ),np.max(kdePOSZY)])
        
        levels_edge = np.round(np.linspace(np.log10(eps),maxlev_edge,cres),1)
        levels = np.round(np.linspace(np.log10(eps),np.max(kdePOSXY),cres),1)

        print 'Increase factor:',np.max(levels)/np.max(levels_edge)
        
        fig = plt.figure(figsize=(7.,7.))

        left_edge = 0.15
        wfac = 5.
        width_share = 1./wfac
        right_edge = 0.78
        width_share = (right_edge-left_edge)*width_share
        bottom_edge = 0.15
        ax1 = fig.add_axes([left_edge,bottom_edge,(wfac-1.)*width_share,(wfac-1.)*width_share])
        ax2 = fig.add_axes([left_edge+(wfac-1.)*width_share,bottom_edge,width_share,(wfac-1.)*width_share])
        ax3 = fig.add_axes([left_edge,bottom_edge+(wfac-1.)*width_share,(wfac-1.)*width_share,width_share])
        ax4 = fig.add_axes([0.82,bottom_edge,0.02,wfac*width_share])

        
        # XY
        cbar = ax1.contourf(kdeX,kdeY,kdePOSXY,levels,cmap=cm.gnuplot)
        ax1.axis([-0.05,0.05,-0.05,0.05])
        cbh = fig.colorbar(cbar,cax=ax4)
        

        # ZY
        ax2.contourf(kdeZYz,kdeZYy,kdePOSZY,levels_edge,cmap=cm.gnuplot)
        ax2.axis([-0.01,0.01,-0.05,0.05])
        ax2.set_yticklabels(())

        # XZ
        ax3.contourf(kdeXZx,kdeXZz,kdePOSXZ,levels_edge,cmap=cm.gnuplot)
        ax3.axis([-0.05,0.05,-0.01,0.01])
        ax3.set_xticklabels(())

        return fig




############################################################################################
def plot_image_velocity(O,ax,ax2,ax3,rmax=0.04,nsamp=257,levels = np.linspace(0.0,3.2,100),zlim=0.1):
    aval = np.sum( np.cos( 2.*np.arctan2(O.ypos,O.xpos) ) )
    bval = np.sum( np.sin( 2.*np.arctan2(O.ypos,O.xpos) ) )
    bpos = -np.arctan2(bval,aval)/2.
    #
    tX = O.xpos*np.cos(bpos) - O.ypos*np.sin(bpos)
    tY = -O.xpos*np.sin(bpos) - O.ypos*np.cos(bpos)
    tYv = -O.xvel*np.sin(bpos) - O.yvel*np.cos(bpos)
    #
    print 'ey'
    w = np.where( (abs(tX) < rmax) & (abs(tY) < rmax) & (abs(O.zpos) < zlim) )[0]
    extent = rmax#0.06
    kde_weight = tYv[w]
    #
    print 'ey'
    vv = kde_3d.fast_kde_two(tX[w],tY[w], gridsize=(nsamp,nsamp), extents=(-extent,extent,-extent,extent), nocorrelation=False, weights=kde_weight)
    kde_weight = tYv[w]**2.
    #
    ss = kde_3d.fast_kde_two(tX[w],tY[w], gridsize=(nsamp,nsamp), extents=(-extent,extent,-extent,extent), nocorrelation=False, weights=kde_weight)
    #
    #
    kde_weight = O.mass[w]
    tt = kde_3d.fast_kde_two(tX[w],tY[w], gridsize=(nsamp,nsamp), extents=(-extent,extent,-extent,extent), nocorrelation=False, weights=None)
    mm = kde_3d.fast_kde_two(tX[w],tY[w], gridsize=(nsamp,nsamp), extents=(-extent,extent,-extent,extent), nocorrelation=False, weights=kde_weight)
    avgvel = vv/tt
    sigma = ss/tt - (vv/tt)**2.
    # vizualize!
    xbins = np.linspace(-extent,extent,nsamp)
    xx,yy = np.meshgrid( xbins,xbins)
    effvolume = ((xbins[1]-xbins[0])*(xbins[1]-xbins[0]))#*(2.*zlim))
    ax.contourf(xx,yy,np.log10(mm/effvolume),levels,cmap=cm.spectral)
    ax2.contourf(xx,yy,avgvel,np.linspace(-1.4,1.4,72),cmap=cm.spectral)
    ax3.contourf(xx,yy,sigma,np.linspace(0.,1.,72),cmap=cm.spectral)



