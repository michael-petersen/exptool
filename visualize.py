#
# exptool.visualize
#
#    a collection of routines to visualize first-check items
#
#    08-27-2016
#    01-02-2017 first fixes


'''

fig = visualize.show_dump('/path/to/OUTFILE','comp')
ax1,ax2,ax3,ax4 = fig.get_axes()

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
    '''
    show_dump
        first ability to see a PSPDump in the simplest way possible


    '''

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

    if (type=='pos'):

        kdeX,kdeY,XY,\
          kdeZYz,kdeZYy,ZY,\
          kdeXZx,kdeXZz,XZ,\
          levels,levels_edge = kde_pos(PSPDump,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)


    if ( (type=='Xvel') | (type=='Rvel') |  (type=='Tvel')):

        if (type=='Xvel'): velarr = PSPDump.xvel
        if (type=='Rvel'): velarr = (PSPDump.xpos*PSPDump.xvel + PSPDump.ypos*PSPDump.yvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5
        if (type=='Tvel'): velarr = (PSPDump.xpos*PSPDump.yvel - PSPDump.ypos*PSPDump.xvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5


        kdeX,kdeY,XY,\
              kdeZYz,kdeZYy,ZY,\
              kdeXZx,kdeXZz,XZ,\
              levels,levels_edge = kde_xvel(PSPDump,velarr,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)


        # reset to positive velocities for tangential case.
        if (type=='Tvel'): levels = levels_edge = np.linspace(0.,np.max(levels),cres)


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

    cbar = ax1.contourf(kdeX,kdeY,XY,levels,cmap=cm.gnuplot)
    ax1.axis([-0.05,0.05,-0.05,0.05])
    cbh = fig.colorbar(cbar,cax=ax4)
        

    # ZY

    ax2.contourf(kdeZYz,kdeZYy,ZY,levels_edge,cmap=cm.gnuplot)
    ax2.axis([-0.01,0.01,-0.05,0.05])
    ax2.set_yticklabels(())

    # XZ
    ax3.contourf(kdeXZx,kdeXZz,XZ,levels_edge,cmap=cm.gnuplot)
    ax3.axis([-0.05,0.05,-0.01,0.01])
    ax3.set_xticklabels(())

    return fig



def kde_pos(PSPDump,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1):

    # XY
    kdeX,kdeY,kdePOSXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

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

    #print 'Increase factor:',np.max(levels)/np.max(levels_edge)

    XY = kdePOSXY
    ZY = kdePOSZY
    XZ = kdePOSXZ

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge


def kde_xvel(PSPDump,velarr,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,sncut=5.):

    # XY
    kdeX,kdeY,kdeNUMXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=None,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)
    kdeNUMXY += 0.0001
    kdeNUMXY[np.where(kdeNUMXY < sncut)] = 1.e10
    kdeX,kdeY,kdeVELXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=velarr,opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)


    # XZ
    kdeXZx,kdeXZz,kdeNUMXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=None,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)
    kdeNUMXZ += 0.0001
    kdeNUMXZ[np.where(kdeNUMXZ < sncut)] = 1.e10
    kdeXZx,kdeXZz,kdeVELXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=velarr,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)

    # ZY
    kdeZYz,kdeZYy,kdeNUMZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=None,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)
    kdeNUMZY += 0.0001
    kdeNUMZY[np.where(kdeNUMZY < sncut)] = 1.e10
    kdeZYz,kdeZYy,kdeVELZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=velarr,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)

    maxlev_edge = np.max([np.max(abs(kdeVELXY/kdeNUMXY)),np.max(abs(kdeVELZY/kdeNUMZY)),np.max(abs(kdeVELXZ/kdeNUMXZ))])

    levels_edge = np.round(np.linspace(-1.*maxlev_edge,maxlev_edge,cres),3)
    levels = np.round(np.linspace(-1.*maxlev_edge,maxlev_edge,cres),3)

    print 'Increase factor:',np.max(levels)/np.max(levels_edge)

    XY = kdeVELXY/kdeNUMXY
    ZY = kdeVELZY/kdeNUMZY
    XZ = kdeVELXZ/kdeNUMXZ

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge



def compare_dumps(infile1,infile2,comp,type='pos',transform=True,\
                  label1=None,label2=None,
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.025,slice_width=0.2):
    '''
    compare_dumps
        look at two different dumps for direct comparison


    '''

    # read in files

    PSPDump1 = psp_io.Input(infile1,comp=comp)
    PSPDump2 = psp_io.Input(infile2,comp=comp)

    if transform:
        PSPDump1 = trapping.BarTransform(PSPDump1)
        PSPDump2 = trapping.BarTransform(PSPDump2)


    # do the kde

    if (type=='pos'):

        kdeX1,kdeY1,XY1,\
          kdeZYz1,kdeZYy1,ZY1,\
          kdeXZx1,kdeXZz1,XZ1,\
          levels1,levels_edge1 = kde_pos(PSPDump1,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)

        kdeX2,kdeY2,XY2,\
          kdeZYz2,kdeZYy2,ZY2,\
          kdeXZx2,kdeXZz2,XZ2,\
          levels2,levels_edge2 = kde_pos(PSPDump2,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)

    if ( (type=='Xvel') | (type=='Rvel') |  (type=='Tvel')):
        if (type=='Xvel'):
            velarr1 = PSPDump1.xvel
            velarr2 = PSPDump2.xvel
        if (type=='Rvel'):
            velarr1 = (PSPDump1.xpos*PSPDump1.xvel + PSPDump1.ypos*PSPDump1.yvel)/(PSPDump1.xpos*PSPDump1.xpos + PSPDump1.ypos*PSPDump1.ypos)**0.5
            velarr2 = (PSPDump2.xpos*PSPDump2.xvel + PSPDump2.ypos*PSPDump2.yvel)/(PSPDump2.xpos*PSPDump2.xpos + PSPDump2.ypos*PSPDump2.ypos)**0.5
        if (type=='Tvel'):
            velarr1 = (PSPDump1.xpos*PSPDump1.yvel - PSPDump1.ypos*PSPDump1.xvel)/(PSPDump1.xpos*PSPDump1.xpos + PSPDump1.ypos*PSPDump1.ypos)**0.5
            velarr2 = (PSPDump2.xpos*PSPDump2.yvel - PSPDump2.ypos*PSPDump2.xvel)/(PSPDump2.xpos*PSPDump2.xpos + PSPDump2.ypos*PSPDump2.ypos)**0.5


        kdeX1,kdeY1,XY1,\
              kdeZYz1,kdeZYy1,ZY1,\
              kdeXZx1,kdeXZz1,XZ1,\
              levels1,levels_edge1 = kde_xvel(PSPDump1,velarr1,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)

        kdeX2,kdeY2,XY2,\
              kdeZYz2,kdeZYy2,ZY2,\
              kdeXZx2,kdeXZz2,XZ2,\
              levels2,levels_edge2 = kde_xvel(PSPDump2,velarr2,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)

        if (type=='Tvel'): levels1 = levels_edge1 = levels2 = levels_edge2 = np.linspace(0.,np.max(levels),cres)



    fig = plt.figure(figsize=(14.,7.))

    left_edge = 0.13
    midpoint = 0.47
    midbuffer = 0.04
    wfac = 5.
    width_share = 1./wfac
    right_edge = 0.83
    width_share_l = (midpoint-left_edge)*width_share
    width_share_r = (right_edge-midpoint)*width_share
    top_edge    = 0.85
    bottom_edge = 0.15
    width_share_h = (top_edge-bottom_edge)*width_share

    # dump 1
    ax1 = fig.add_axes([left_edge,                         bottom_edge,                         (wfac-1.)*width_share_l, (wfac-1.)*width_share_h])
    ax2 = fig.add_axes([left_edge+(wfac-1.)*width_share_l, bottom_edge,                         width_share_l,           (wfac-1.)*width_share_h])
    ax3 = fig.add_axes([left_edge,                         bottom_edge+(wfac-1.)*width_share_h, (wfac-1.)*width_share_l, width_share_h])

    # dump 1
    ax4 = fig.add_axes([midpoint+midbuffer,                         bottom_edge,                         (wfac-1.)*width_share_r, (wfac-1.)*width_share_h])
    ax5 = fig.add_axes([midpoint+midbuffer+(wfac-1.)*width_share_r, bottom_edge,                         width_share_r,           (wfac-1.)*width_share_h])
    ax6 = fig.add_axes([midpoint+midbuffer,                         bottom_edge+(wfac-1.)*width_share_h, (wfac-1.)*width_share_r, width_share_h])
    
    # colorbar
    ax7 = fig.add_axes([0.89,bottom_edge,0.01,wfac*width_share_h])

        
    # XY
    cbar = ax1.contourf(kdeX1,kdeY1,XY1,levels1,cmap=cm.gnuplot)
    ax1.axis([-0.05,0.05,-0.05,0.05])
    ax1.text(-0.04,0.04,label1)
    cbh = fig.colorbar(cbar,cax=ax7)
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")        

    # ZY
    ax2.contourf(kdeZYz1,kdeZYy1,ZY1,levels_edge1,cmap=cm.gnuplot)
    ax2.axis([-0.01,0.01,-0.05,0.05])
    ax2.set_yticklabels(())
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
        
    # XZ
    ax3.contourf(kdeXZx1,kdeXZz1,XZ1,levels_edge1,cmap=cm.gnuplot)
    ax3.axis([-0.05,0.05,-0.01,0.01])
    ax3.set_xticklabels(())

    # XY2
    cbar = ax4.contourf(kdeX2,kdeY2,XY2,levels1,cmap=cm.gnuplot)
    ax4.axis([-0.05,0.05,-0.05,0.05])
    ax4.set_yticklabels(())
    ax4.text(-0.04,0.04,label2)
    for label in ax4.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
        

    # ZY2
    ax5.contourf(kdeZYz2,kdeZYy2,ZY2,levels_edge1,cmap=cm.gnuplot)
    ax5.axis([-0.01,0.01,-0.05,0.05])
    ax5.set_yticklabels(())
    for label in ax5.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
        
    # XZ2
    ax6.contourf(kdeXZx2,kdeXZz2,XZ2,levels_edge1,cmap=cm.gnuplot)
    ax6.axis([-0.05,0.05,-0.01,0.01])
    ax6.set_xticklabels(())
    ax6.set_yticklabels(())

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



