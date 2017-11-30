#
# exptool.visualize
#
#    a collection of routines to visualize first-check items
#
#    08-27-2016
#    01-02-2017 first fixes
#    03-29-2017 compare_dump and label improvements

'''
____    ____  __       _______. __    __       ___       __       __   ________   _______ 
\   \  /   / |  |     /       ||  |  |  |     /   \     |  |     |  | |       /  |   ____|
 \   \/   /  |  |    |   (----`|  |  |  |    /  ^  \    |  |     |  | `---/  /   |  |__   
  \      /   |  |     \   \    |  |  |  |   /  /_\  \   |  |     |  |    /  /    |   __|  
   \    /    |  | .----)   |   |  `--'  |  /  _____  \  |  `----.|  |   /  /----.|  |____ 
    \__/     |__| |_______/     \______/  /__/     \__\ |_______||__|  /________||_______|
visualize.py : part of exptool
                      

# WISHLIST:
-add position overlays to velocity or dispersion plots (see velocity.py)
-specify colorbar levels (for movie making)


from exptool.observables import visualize

fig = visualize.show_dump('/path/to/OUTFILE','comp')
ax1,ax2,ax3,ax4 = fig.get_axes()

visualize.compare_dumps('/scratch/mpetersen/Disk001/OUT.run001.01000','/work/mpetersen/Disk001thick/OUT.run001t.01000','star',type='pos',label1='Fiducial',label2='Thick Basis')





'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# exptool routines
from exptool.io import psp_io
from exptool.utils import kde_3d
from exptool.analysis import trapping



def kde_pos(PSPDump,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1):
    '''
    kde_pos:
        take a PSP component structure and return slices.

    inputs
    ------------------------
    PSPDump
    gridsize=64
    cres=24
    face_extents=0.06
    edge_extents=0.02
    slice_width=0.1


    '''
    

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

    # add a prefac to eps to make sure lowest contour catches it
    levels_edge = np.round(np.linspace(np.log10(0.9*eps),maxlev_edge,cres),1)
    levels = np.round(np.linspace(np.log10(0.9*eps),np.max(kdePOSXY),cres),1)

    #print 'Increase factor:',np.max(levels)/np.max(levels_edge)

    XY = kdePOSXY
    ZY = kdePOSZY
    XZ = kdePOSXZ

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge


def kde_xvel(PSPDump,velarr,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,sncut=5.):
    #
    # do a velocity cut along the line of sight
    #
    sncut *= np.median(PSPDump.mass)

    # XY
    kdeX,kdeY,kdeNUMXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

    kdeNUMXY[np.where(kdeNUMXY < sncut)] = 1.e10
    kdeX,kdeY,kdeVELXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)


    # XZ
    kdeXZx,kdeXZz,kdeNUMXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)
    kdeNUMXZ[np.where(kdeNUMXZ < sncut)] = 1.e10
    kdeXZx,kdeXZz,kdeVELXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)

    # ZY
    kdeZYz,kdeZYy,kdeNUMZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)
    kdeNUMZY[np.where(kdeNUMZY < sncut)] = 1.e10
    kdeZYz,kdeZYy,kdeVELZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)

    maxlev_edge = np.max([np.max(abs(kdeVELXY/kdeNUMXY)),np.max(abs(kdeVELZY/kdeNUMZY)),np.max(abs(kdeVELXZ/kdeNUMXZ))])

    #
    # add a buffer 1.1 to capture lowest levels
    levels_edge = np.round(np.linspace(-1.1*maxlev_edge,maxlev_edge,cres),2)
    levels = np.round(np.linspace(-1.1*maxlev_edge,maxlev_edge,cres),2)


    XY = kdeVELXY/kdeNUMXY
    ZY = kdeVELZY/kdeNUMZY
    XZ = kdeVELXZ/kdeNUMXZ

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge



def kde_disp(PSPDump,velarr,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,sncut=5.):
    #
    # do a dispersion measurement along the line of sight
    #
    sncut *= np.median(PSPDump.mass)

    
    # XY
    kdeX,kdeY,kdeNUMXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

    # zero below an SN cut
    kdeNUMXY[np.where(kdeNUMXY < sncut)] = 1.e10
    
    kdeX,kdeY,kdeVELXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=velarr*PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

    kdeX,kdeY,kdeDISPXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                               gridsize=gridsize,extents=face_extents,weights=(velarr**2.)*PSPDump.mass,\
                                               opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)


    # XZ
    kdeXZx,kdeXZz,kdeNUMXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)
    kdeNUMXZ[np.where(kdeNUMXZ < sncut)] = 1.e10
    kdeXZx,kdeXZz,kdeVELXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)

    kdeXZx,kdeXZz,kdeDISPXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=(velarr**2.)*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width)
                                              
    # ZY
    kdeZYz,kdeZYy,kdeNUMZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)
    kdeNUMZY[np.where(kdeNUMZY < sncut)] = 1.e10
    kdeZYz,kdeZYy,kdeVELZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)

    kdeZYz,kdeZYy,kdeDISPZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=(velarr**2.)*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width)

                                              
    maxlev_edge = np.max([np.max(abs(kdeVELXY/kdeNUMXY)),np.max(abs(kdeVELZY/kdeNUMZY)),np.max(abs(kdeVELXZ/kdeNUMXZ))])

    levels_edge = np.round(np.linspace(-1.*maxlev_edge,maxlev_edge,cres),2)
    levels = np.round(np.linspace(-1.*maxlev_edge,maxlev_edge,cres),2)

    
    XY = kdeDISPXY/kdeNUMXY - (kdeVELXY/kdeNUMXY)**2.
    ZY = kdeDISPZY/kdeNUMZY - (kdeVELZY/kdeNUMZY)**2.
    XZ = kdeDISPXZ/kdeNUMXZ - (kdeVELXZ/kdeNUMXZ)**2.

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge




      

def show_dump(infile,comp,type='pos',transform=True,\
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1):
    '''
    show_dump
        first ability to see a PSPDump in the simplest way possible

    INPUTS
    ------------------------------
    infile
    comp
    type='pos'
    transform=True             : align bar axis to X axis
    gridsize=64                : evenly spaced bins between -face_extents, face_extents
    cres=24
    face_extents=0.06
    edge_extents=0.02
    slice_width=0.1

    OUTPUTS
    ------------------------------


    TODO
    ------------------------------
    1. allow for forcing contour levels

    '''

    # read in component(s)

    if np.array(comp).size == 1:
        PSPDump = psp_io.Input(infile,comp=comp)

    else:
        # allow for multiple components to be mixed together
        PartArray = [psp_io.Input(infile,comp=cc) for cc in comp]

        PSPDump = psp_io.mix_particles(PartArray)

        

    if transform:
        PSPDump = trapping.BarTransform(PSPDump)


    # do a kde

    if (type=='pos'):

        kdeX,kdeY,XY,\
          kdeZYz,kdeZYy,ZY,\
          kdeXZx,kdeXZz,XZ,\
          levels,levels_edge = kde_pos(PSPDump,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)


    if ( (type=='Xvel') | (type=='Yvel') | (type=='Zvel') | (type=='Rvel') |  (type=='Tvel')):

        if (type=='Xvel'): velarr = PSPDump.xvel
        if (type=='Yvel'): velarr = PSPDump.yvel
        if (type=='Zvel'): velarr = PSPDump.zvel
        if (type=='Rvel'): velarr = (PSPDump.xpos*PSPDump.xvel + PSPDump.ypos*PSPDump.yvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5
        if (type=='Tvel'): velarr = (PSPDump.xpos*PSPDump.yvel - PSPDump.ypos*PSPDump.xvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5


        kdeX,kdeY,XY,\
              kdeZYz,kdeZYy,ZY,\
              kdeXZx,kdeXZz,XZ,\
              levels,levels_edge = kde_xvel(PSPDump,velarr,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)


        # reset to positive velocities for tangential case.
        if (type=='Tvel'): levels = levels_edge = np.linspace(0.,np.max(levels),cres)


    if ( (type=='Xdisp') | (type=='Rdisp') |  (type=='disp')):

        if (type=='Xdisp'): velarr = PSPDump.xvel
        if (type=='Rdisp'): velarr = (PSPDump.xpos*PSPDump.xvel + PSPDump.ypos*PSPDump.yvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5
        if (type=='Tdisp'): velarr = (PSPDump.xpos*PSPDump.yvel - PSPDump.ypos*PSPDump.xvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5


        kdeX,kdeY,XY,\
              kdeZYz,kdeZYy,ZY,\
              kdeXZx,kdeXZz,XZ,\
              levels,levels_edge = kde_disp(PSPDump,velarr,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width)


        # reset to positive velocities for tangential case.
        levels = levels_edge = np.linspace(0.,np.max(levels),cres)



    fig = plt.figure(figsize=(7.8,7.5))

    left_edge = 0.22
    wfac = 5.
    width_share = 1./wfac
    right_edge = 0.78
    width_share = (right_edge-left_edge)*width_share
    bottom_edge = 0.2

    # the face on figure
    ax1 = fig.add_axes([left_edge,bottom_edge,(wfac-1.)*width_share,(wfac-1.)*width_share])

    # the YZ plane (right panel)
    ax2 = fig.add_axes([left_edge+(wfac-1.)*width_share,bottom_edge,width_share,(wfac-1.)*width_share])

    # the XZ plane (upper panel)
    ax3 = fig.add_axes([left_edge,bottom_edge+(wfac-1.)*width_share,(wfac-1.)*width_share,width_share])

    # the colorbar
    ax4 = fig.add_axes([right_edge+0.03,bottom_edge,0.02,wfac*width_share])

        
    # XY

    cbar = ax1.contourf(kdeX,kdeY,XY,levels,cmap=cm.gnuplot)
    ax1.axis([-0.05,0.05,-0.05,0.05])
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    ax1.set_xlabel('X',size=30)
    ax1.set_ylabel('Y',size=30)

    # colorbar
    cbh = fig.colorbar(cbar,cax=ax4)

    # set the colorbar label
    if (type=='pos'): ax4.set_ylabel('log Surface Density',size=20)
    if (type=='Xvel'): ax4.set_ylabel('X Velocity',size=20)
    if (type=='Yvel'): ax4.set_ylabel('Y Velocity',size=20)
    if (type=='Zvel'): ax4.set_ylabel('Z Velocity',size=20)
    if (type=='Rvel'): ax4.set_ylabel('Radial Velocity',size=20)
    if (type=='Tvel'): ax4.set_ylabel('Tangential Velocity',size=20)
    if (type=='Xdisp'): ax4.set_ylabel('X Velocity Dispersion ',size=20)
    if (type=='Rdisp'): ax4.set_ylabel('Radial Velocity Dispersion',size=20)
    if (type=='Tdisp'): ax4.set_ylabel('Tangential Velocity Dispersion',size=20)

        
    # ZY

    ax2.contourf(kdeZYz,kdeZYy,ZY,levels_edge,cmap=cm.gnuplot)
    ax2.axis([-0.01,0.01,-0.05,0.05])
    ax2.set_yticklabels(())
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("right")

    ax2.set_xlabel('Z',size=30)
    ax2.xaxis.labelpad = 18
        
    # XZ
    ax3.contourf(kdeXZx,kdeXZz,XZ,levels_edge,cmap=cm.gnuplot)
    ax3.axis([-0.05,0.05,-0.01,0.01])
    ax3.set_xticklabels(())
    for label in ax3.get_yticklabels():
        label.set_fontsize(10)

    ax3.set_ylabel('Z',size=30)
    ax3.yaxis.labelpad = 18

    short_infile = infile.split('/')[-1]
    breakdown = short_infile.split('.')

    #print(breakdown[1]+' '+breakdown[2]+': T={0:4.3f}'.format(PSPDump.time))
    ax3.set_title(breakdown[1]+' '+breakdown[2]+': T={0:4.3f}'.format(PSPDump.time),size=18)
    
    return fig



################################################
# for two side-by-side plots

def compare_dumps(infile1,infile2,comp,type='pos',transform=True,\
                  label1=None,label2=None,
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.025,slice_width=0.2):
    '''
    compare_dumps
        look at two different dumps for direct comparison

    TODO:
        enable multiple component comparison

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

    left_edge = 0.12
    midpoint = 0.47
    midbuffer = 0.03
    wfac = 5.
    width_share = 1./wfac
    right_edge = 0.83
    width_share_l = (midpoint-left_edge)*width_share
    width_share_r = (right_edge-midpoint)*width_share
    top_edge    = 0.85
    bottom_edge = 0.18
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
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("center")

    ax1.set_ylabel('Y',size=24)
    ax1.set_xlabel('X',size=24)

    cbh = fig.colorbar(cbar,cax=ax7)

    # set the colorbar label
    if (type=='pos'): ax7.set_ylabel('log Surface Density',size=20)
    if (type=='Xvel'): ax7.set_ylabel('X Velocity',size=20)
    if (type=='Yvel'): ax7.set_ylabel('Y Velocity',size=20)
    if (type=='Zvel'): ax7.set_ylabel('Z Velocity',size=20)
    if (type=='Rvel'): ax7.set_ylabel('Radial Velocity',size=20)
    if (type=='Tvel'): ax7.set_ylabel('Tangential Velocity',size=20)
    if (type=='Xdisp'): ax7.set_ylabel('X Velocity Dispersion ',size=20)
    if (type=='Rdisp'): ax7.set_ylabel('Radial Velocity Dispersion',size=20)
    if (type=='Tdisp'): ax7.set_ylabel('Tangential Velocity Dispersion',size=20)

       
    
    # ZY
    ax2.contourf(kdeZYz1,kdeZYy1,ZY1,levels_edge1,cmap=cm.gnuplot)
    ax2.axis([-0.01,0.01,-0.05,0.05])
    ax2.set_yticklabels(())
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("center")


    ax2.set_xlabel('Z',size=24)
    ax2.xaxis.labelpad = 18

    
    # XZ
    ax3.contourf(kdeXZx1,kdeXZz1,XZ1,levels_edge1,cmap=cm.gnuplot)
    ax3.axis([-0.05,0.05,-0.01,0.01])
    ax3.set_xticklabels(())
    for label in ax3.get_yticklabels():
        label.set_fontsize(10)

    ax3.set_ylabel('Z',size=24)
    ax3.yaxis.labelpad = 18

    # XY2
    cbar = ax4.contourf(kdeX2,kdeY2,XY2,levels1,cmap=cm.gnuplot)
    ax4.axis([-0.05,0.05,-0.05,0.05])
    ax4.set_yticklabels(())
    ax4.text(-0.04,0.04,label2)
    for label in ax4.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("center")

    ax4.set_xlabel('X',size=24)
        

    # ZY2
    ax5.contourf(kdeZYz2,kdeZYy2,ZY2,levels_edge1,cmap=cm.gnuplot)
    ax5.axis([-0.01,0.01,-0.05,0.05])
    ax5.set_yticklabels(())
    for label in ax5.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("center")

    ax5.set_xlabel('Z',size=24)
    ax5.xaxis.labelpad = 18
        
    # XZ2
    ax6.contourf(kdeXZx2,kdeXZz2,XZ2,levels_edge1,cmap=cm.gnuplot)
    ax6.axis([-0.05,0.05,-0.01,0.01])
    ax6.set_xticklabels(())
    ax6.set_yticklabels(())

    return fig


