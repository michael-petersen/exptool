"""
visualize.py
   a collection of routines to visualize first-check items

MSP 27 Aug 2016 Original commit
MSP  2 Jan 2017 Tested for compatibility
MSP 29 Mar 2017 compare_dump and label improvements
MSP 15 Mar 2019 improve documentation and write TO-DOs

 "I've stolen all the algorithms!"


TODO
-add position overlays to velocity or dispersion plots (see velocity.py)

GENERAL USAGE
from exptool.observables import visualize

fig = visualize.show_dump('/path/to/OUTFILE','comp')
ax1,ax2,ax3,ax4 = fig.get_axes()

visualize.compare_dumps('/scratch/mpetersen/Disk001/OUT.run001.01000','/work/mpetersen/Disk001thick/OUT.run001t.01000','star',type='pos',label1='Fiducial',label2='Thick Basis')

# turn the output files into a movie
ffmpeg -framerate 20 -i out%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ~/Desktop/Disk022trans.mp4

#To make a movie, set specifications make sense:
visualize.show_dump('/scratch/mpetersen/Disk004/OUT.run004.01000','star',type='pos',transform=True,gridsize=129,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,ktype='gaussian',npower=5.,cwheel='magma',barfile='/scratch/mpetersen/Disk004/run004_m2n1_barpos.dat')


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# exptool routines
from ..io import particle
from ..utils import kde_3d
from ..analysis import pattern

# bring in the exptool native plotting style
from ..utils import style


def kde_pos(PSPDump,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,ktype='gaussian',npower=6.,**kwargs):
    '''
    kde_pos:
        take a PSP component structure and return slices.

    inputs
    ------------------------
    PSPDump       : exptool.io.particle.Input instance
      the Input instance to visualize
    gridsize      : int, default=64
      number of cells per side of 2d grid
    cres          : int, default=24
      number of colour resolution elements
    face_extents  : float, default=0.06
      extent of the 2d grid (min/max)
    edge_extents  : float, default=0.02
      extent of the 2d grid for edge-on projections
    slice_width   : float, default=0.1
      maximum edge-on height to consider


    returns
    ------------------------
    kdeX
    kdeY
    XY
    kdeZYz
    kdeZYy
    ZY
    kdeXZx
    kdeXZz
    XZ
    levels
    levels_edge

    TODO:
    ------------------------
    # add kwargs to handle opt_third (?)

    '''
    
    # XY
    kdeX,kdeY,kdePOSXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,\
                                                  #opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width,\
                                                  ktype=ktype,npower=npower)

    # make a log guard
    eps = np.min(PSPDump.mass)

    # change to log surface density
    kdePOSXY = np.log10(kdePOSXY+eps)

    # XZ
    kdeXZx,kdeXZz,kdePOSXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  #weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
                                                  weights=PSPDump.mass,ktype=ktype,npower=npower)

    # change to log surface density
    kdePOSXZ = np.log10(kdePOSXZ+eps)

    # ZY
    kdeZYz,kdeZYy,kdePOSZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              #weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
                                              weights=PSPDump.mass,ktype=ktype,npower=npower)

    # change to surface density
    kdePOSZY = np.log10(kdePOSZY+eps)

    # set up the figure

    maxlev_edge = np.nanmax([np.nanmax(kdePOSXZ),np.nanmax(kdePOSZY)])
    minlev_edge = np.nanmin([np.nanmin(kdePOSXZ),np.nanmin(kdePOSZY)])

    # add a prefac to eps to make sure lowest contour catches it

    # fix if the resulting densities are negative
    levels = np.round(np.linspace(np.log10(0.9*eps),np.nanmax(kdePOSXY),cres),2)
    levels_edge = np.round(np.linspace(np.log10(0.9*eps),maxlev_edge,cres),2)


    # if an increase factor for the projection is desired...
    #print('Increase factor:',np.max(levels)/np.max(levels_edge))

    XY = kdePOSXY
    ZY = kdePOSZY
    XZ = kdePOSXZ

    return kdeX,kdeY,XY,\
      kdeZYz,kdeZYy,ZY,\
      kdeXZx,kdeXZz,XZ,\
      levels,levels_edge


def kde_xvel(PSPDump,velarr,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,sncut=5.,ktype='gaussian',npower=6.):
    #
    # do a velocity cut along the line of sight
    #
    sncut *= np.median(PSPDump.mass)

    # XY
    kdeX,kdeY,kdeNUMXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width)

    kdeNUMXY[np.where(kdeNUMXY < sncut)] = 1.e10
    kdeX,kdeY,kdeVELXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,gridsize=gridsize,extents=face_extents,weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)


    # XZ
    kdeXZx,kdeXZz,kdeNUMXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
    kdeNUMXZ[np.where(kdeNUMXZ < sncut)] = 1.e10
    kdeXZx,kdeXZz,kdeVELXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

    # ZY
    kdeZYz,kdeZYy,kdeNUMZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
    kdeNUMZY[np.where(kdeNUMZY < sncut)] = 1.e10
    kdeZYz,kdeZYy,kdeVELZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

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



def kde_disp(PSPDump,velarr,gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,sncut=5.,ktype='gaussian',npower=6.):
    #
    # do a dispersion measurement along the line of sight
    #
    sncut *= np.median(PSPDump.mass)

    
    # XY
    kdeX,kdeY,kdeNUMXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

    # zero below an SN cut
    kdeNUMXY[np.where(kdeNUMXY < sncut)] = 1.e10
    
    kdeX,kdeY,kdeVELXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=face_extents,weights=velarr*PSPDump.mass,\
                                              opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

    kdeX,kdeY,kdeDISPXY = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.ypos,\
                                               gridsize=gridsize,extents=face_extents,weights=(velarr**2.)*PSPDump.mass,\
                                               opt_third=abs(PSPDump.zpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)


    # XZ
    kdeXZx,kdeXZz,kdeNUMXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
    kdeNUMXZ[np.where(kdeNUMXZ < sncut)] = 1.e10
    kdeXZx,kdeXZz,kdeVELXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

    kdeXZx,kdeXZz,kdeDISPXZ = kde_3d.total_kde_two(PSPDump.xpos,PSPDump.zpos,\
                                                  gridsize=gridsize,extents=(-1.*face_extents,face_extents,-1.*edge_extents,edge_extents),\
                                                  weights=(velarr**2.)*PSPDump.mass,opt_third=abs(PSPDump.ypos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
                                              
    # ZY
    kdeZYz,kdeZYy,kdeNUMZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)
    kdeNUMZY[np.where(kdeNUMZY < sncut)] = 1.e10
    kdeZYz,kdeZYy,kdeVELZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=velarr*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

    kdeZYz,kdeZYy,kdeDISPZY = kde_3d.total_kde_two(PSPDump.zpos,PSPDump.ypos,\
                                              gridsize=gridsize,extents=(-1.*edge_extents,edge_extents,-1.*face_extents,face_extents),\
                                              weights=(velarr**2.)*PSPDump.mass,opt_third=abs(PSPDump.xpos),opt_third_constraint=slice_width,ktype=ktype,npower=npower)

                                              
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




      

def show_dump(infile,comp,nout=-1,type='pos',transform=True,\
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.02,slice_width=0.1,ktype='gaussian',npower=6.,cwheel=cm.gnuplot,**kwargs):
    '''
    show_dump
        first ability to see a PSPDump in the simplest way possible

    INPUTS
    ------------------------------
    infile                     : input PSP file
    comp                       : string name of component to visualize. also takes list in [,] format
    type                       : (string, default='pos') type of visualization ('pos','Xvel','Yvel','Zvel','Rvel','Tvel')
    transform                  : (boolean, default=True) align bar axis to X axis
    gridsize=64                : evenly spaced bins between -face_extents, face_extents
    cres=24                    : number of color elements
    face_extents=0.06          : in-plane extent for KDE mapping
    edge_extents=0.02          : vertical extent for KDE mapping
    slice_width=0.1            : slice width for density determination
    clevels                    : force contour levels
    barfile                    : filename for the bar
    ktype                      : kernel input for KDE
    npower                     : power-law exponent for the KDE kernel (lower means more compact)
    cwheel                     : the colormap to use

    OUTPUTS
    ------------------------------
    fig                        : figure with access to different axes (ax1,ax2,ax3,ax4)

    TODO
    ------------------------------
    -add the ability to scale the spatial dimensions as desired
    -eliminate slice_width, or at least make more utilitarian

    '''

    # read in component(s)


    if np.array(comp).size == 1:

        if nout < 0:
            PSPDump = particle.Input(infile,comp=comp,legacy=True)
        else:
            PSPDump = particle.Input(infile,comp=comp,nout=nout,legacy=True)

    else:
        # allow for multiple components to be mixed together
        PartArray = [particle.Input(infile,comp=cc,legacy=True) for cc in comp]

        PSPDump = particle.mix_particles(PartArray)

    # do we want a transformation?
    if transform:

        # check to see if a bar file was provided
        if 'barfile' in kwargs.keys():
            BarInstance = pattern.BarDetermine(file=kwargs['barfile'])
            bar_angle = pattern.find_barangle(PSPDump.time,BarInstance,interpolate=True)
            PSPDump = pattern.BarTransform(PSPDump,bar_angle=bar_angle)

        # calculate the position angle
        else:
            PSPDump = pattern.BarTransform(PSPDump)


    # do a kde

    if (type=='pos'):

        kdeX,kdeY,XY,\
          kdeZYz,kdeZYy,ZY,\
          kdeXZx,kdeXZz,XZ,\
          levels,levels_edge = kde_pos(PSPDump,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)


    if ( (type=='Xvel') | (type=='Yvel') | (type=='Zvel') | (type=='Rvel') |  (type=='Tvel')):

        if (type=='Xvel'): velarr = PSPDump.xvel
        if (type=='Yvel'): velarr = PSPDump.yvel
        if (type=='Zvel'): velarr = PSPDump.zvel
        if (type=='Rvel'): velarr = (PSPDump.xpos*PSPDump.xvel + PSPDump.ypos*PSPDump.yvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5
        if (type=='Tvel'): velarr = (PSPDump.xpos*PSPDump.yvel - PSPDump.ypos*PSPDump.xvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5


        kdeX,kdeY,XY,\
              kdeZYz,kdeZYy,ZY,\
              kdeXZx,kdeXZz,XZ,\
              levels,levels_edge = kde_xvel(PSPDump,velarr,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)


        # reset to positive velocities for tangential case.
        if (type=='Tvel'): levels = levels_edge = np.linspace(0.,np.max(levels),cres)


    if ( (type=='Xdisp') | (type=='Rdisp') |  (type=='disp')):

        if (type=='Xdisp'): velarr = PSPDump.xvel
        if (type=='Rdisp'): velarr = (PSPDump.xpos*PSPDump.xvel + PSPDump.ypos*PSPDump.yvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5
        if (type=='Tdisp'): velarr = (PSPDump.xpos*PSPDump.yvel - PSPDump.ypos*PSPDump.xvel)/(PSPDump.xpos*PSPDump.xpos + PSPDump.ypos*PSPDump.ypos)**0.5


        kdeX,kdeY,XY,\
              kdeZYz,kdeZYy,ZY,\
              kdeXZx,kdeXZz,XZ,\
              levels,levels_edge = kde_disp(PSPDump,velarr,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)


        # reset to positive velocities for tangential case.
        levels = np.linspace(0.,np.max(levels),cres)
        levels_edge = np.linspace(0.,np.max(levels),cres)



    if 'clevels' in kwargs.keys():
        # if clevels is specified, override the autoscaling for both edge on and face on
        levels = kwargs['clevels']

        if 'edge_factor' in kwargs.keys():
            levels_edge = kwargs['edge_factor']*kwargs['clevels']
        else:
            levels_edge = kwargs['clevels']

    if 'overplot' in kwargs.keys():
        fig = plt.gcf()
        
    else:
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
    ax4 = fig.add_axes([right_edge+0.01,bottom_edge,0.02,wfac*width_share])

        
    # XY

    cbar = ax1.contourf(kdeX,kdeY,XY,levels,cmap=cwheel)
    ax1.axis([-0.95*face_extents,0.95*face_extents,-0.95*face_extents,0.95*face_extents])
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

    ax2.contourf(kdeZYz,kdeZYy,ZY,levels_edge,cmap=cwheel)
    ax2.axis([-0.95*edge_extents,0.95*edge_extents,-0.95*face_extents,0.95*face_extents])
    ax2.set_yticklabels(())
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("right")

    ax2.set_xlabel('Z',size=30)
    ax2.xaxis.labelpad = 18
        
    # XZ
    ax3.contourf(kdeXZx,-kdeXZz,XZ,levels_edge,cmap=cwheel)
    ax3.axis([-0.95*face_extents,0.95*face_extents,-0.95*edge_extents,0.95*edge_extents])
    ax3.set_xticklabels(())
    for label in ax3.get_yticklabels():
        label.set_fontsize(10)

    ax3.set_ylabel('Z',size=30)
    ax3.yaxis.labelpad = 18

    short_infile = infile.split('/')[-1]
    breakdown = short_infile.split('.')

    ax3.set_title(breakdown[1]+' '+breakdown[2]+': T={0:4.3f}'.format(PSPDump.time),size=18)
    
    return fig



################################################
# for two side-by-side plots

def compare_dumps(infile1,infile2,comp,type='pos',transform=True,\
              # parameters for the plot
              gridsize=64,cres=24,face_extents=0.06,edge_extents=0.025,slice_width=0.2,cwheel=cm.magma,**kwargs):
    '''
    compare_dumps
        look at two different dumps for direct comparison

    TODO:
        enable multiple component comparison

    '''

    # read in files

    PSPDump1 = particle.Input(infile1,comp=comp,legacy=True)
    PSPDump2 = particle.Input(infile2,comp=comp,legacy=True)

    if transform:
        PSPDump1 = pattern.BarTransform(PSPDump1)
        PSPDump2 = pattern.BarTransform(PSPDump2)


    # do the kde
    if 'ktype' in kwargs.keys():
        ktype=kwargs['ktype']
    else:
        ktype='gaussian'

    if 'npower' in kwargs.keys():
        npower=kwargs['npower']
    else:
        npower=6.

    if (type=='pos'):

        kdeX1,kdeY1,XY1,\
          kdeZYz1,kdeZYy1,ZY1,\
          kdeXZx1,kdeXZz1,XZ1,\
          levels1,levels_edge1 = kde_pos(PSPDump1,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)

        kdeX2,kdeY2,XY2,\
          kdeZYz2,kdeZYy2,ZY2,\
          kdeXZx2,kdeXZz2,XZ2,\
          levels2,levels_edge2 = kde_pos(PSPDump2,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)

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
              levels1,levels_edge1 = kde_xvel(PSPDump1,velarr1,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)

        kdeX2,kdeY2,XY2,\
              kdeZYz2,kdeZYy2,ZY2,\
              kdeXZx2,kdeXZz2,XZ2,\
              levels2,levels_edge2 = kde_xvel(PSPDump2,velarr2,gridsize=gridsize,cres=cres,face_extents=face_extents,edge_extents=edge_extents,slice_width=slice_width,ktype=ktype,npower=npower)

        if (type=='Tvel'): levels1 = levels_edge1 = levels2 = levels_edge2 = np.linspace(0.,np.max(levels),cres)



    #if 'rescale' in kwargs.keys():
    #    levels_edge1 = levels_edge2 = 

    if 'clevels' in kwargs.keys():
        # if clevels is specified, override the autoscaling for both edge on and face on
        levels1 = levels_edge1 = levels2 = levels_edge2 = kwargs['clevels']



    if 'overplot' in kwargs.keys():
        fig = plt.gcf()

    else:
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
    ax7 = fig.add_axes([0.87,bottom_edge,0.01,wfac*width_share_h])

        
    # XY
    cbar = ax1.contourf(kdeX1,kdeY1,XY1,levels1,cmap=cwheel)
    ax1.axis([-0.05,0.05,-0.05,0.05])

    if 'label1' in kwargs.keys():
        ax1.text(-0.04,0.04,kwargs['label1'])
        
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
    ax2.contourf(kdeZYz1,kdeZYy1,ZY1,levels_edge1,cmap=cwheel)
    ax2.axis([-0.01,0.01,-0.05,0.05])
    ax2.set_yticklabels(())
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("center")


    ax2.set_xlabel('Z',size=24)
    ax2.xaxis.labelpad = 18

    
    # XZ
    ax3.contourf(kdeXZx1,kdeXZz1,XZ1,levels_edge1,cmap=cwheel)
    ax3.axis([-0.05,0.05,-0.01,0.01])
    ax3.set_xticklabels(())
    for label in ax3.get_yticklabels():
        label.set_fontsize(10)

    ax3.set_ylabel('Z',size=24)
    ax3.yaxis.labelpad = 18

    # XY2
    cbar = ax4.contourf(kdeX2,kdeY2,XY2,levels1,cmap=cwheel)
    ax4.axis([-0.05,0.05,-0.05,0.05])
    ax4.set_yticklabels(())

    if 'label2' in kwargs.keys():
        ax4.text(-0.04,0.04,kwargs['label2'])
    
    for label in ax4.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("center")

    ax4.set_xlabel('X',size=24)
        

    # ZY2
    ax5.contourf(kdeZYz2,kdeZYy2,ZY2,levels_edge1,cmap=cwheel)
    ax5.axis([-0.01,0.01,-0.05,0.05])
    ax5.set_yticklabels(())
    for label in ax5.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)
        label.set_horizontalalignment("center")

    ax5.set_xlabel('Z',size=24)
    ax5.xaxis.labelpad = 18
        
    # XZ2
    ax6.contourf(kdeXZx2,kdeXZz2,XZ2,levels_edge1,cmap=cwheel)
    ax6.axis([-0.05,0.05,-0.01,0.01])
    ax6.set_xticklabels(())
    ax6.set_yticklabels(())

    return fig


