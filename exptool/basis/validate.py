"""
exptool.basis.validate

Validation tools for different bases


TODO
-write a crawler to look for different bases

"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt



from . import eof




"""------------------------------------------------------------
CYLINDRICAL BASIS----------------------------------------------
"""------------------------------------------------------------
def show_eof_basis(eof_file,plot=False,sine=False):
    '''
    show_basis: demonstration plots for eof_files

    inputs
    ------------
    eof_file    : (string)
    plot        : (bool, default=False)
    sine        : (bool, default=False)


    returns
    ------------
    none


    '''
    potC,rforceC,zforceC,densC,potS,rforceS,zforceS,densS = eof.parse_eof(eof_file)
    rmin,rmax,numx,numy,MMAX,norder,ascale,hscale,cmap,dens = eof.eof_params(eof_file)
    XMIN,XMAX,dX,YMIN,YMAX,dY = eof.set_table_params(RMAX=rmax,RMIN=rmin,ASCALE=ascale,HSCALE=hscale,NUMX=numx,NUMY=numy,CMAP=cmap)

    xvals = np.array([xi_to_r(XMIN + i*dX,cmap,ascale) for i in range(0,numx+1)])
    zvals =  np.array([y_to_z(YMIN + i*dY,hscale) for i in range(0,numy+1)])

    print('eof.show_basis: plotting {0:d} azimuthal orders and {1:d} radial orders...'.format(MMAX,norder) )

    xgrid,zgrid = np.meshgrid(xvals,zvals)

    if sine:
        width=2
    else:
        width=1
    
    if plot:

        plt.subplots_adjust(hspace=0.001)
        
        for mm in range(0,MMAX+1):
            fig = plt.figure()

            for nn in range(0,norder):
                if mm > 0: ax = fig.add_subplot(norder,width,width*(nn)+1)
                else: ax = fig.add_subplot(norder,1,nn+1)

                ax.contourf(xgrid,zgrid,potC[mm,nn,:,:].T,cmap=cm.gnuplot)
                ax.axis([0.0,0.08,-0.03,0.03])

                if nn < (norder-1): ax.set_xticklabels(())

                if (mm > 0) & (sine):
                    ax2 = fig.add_subplot(norder,2,2*(nn)+2)

                    ax2.contourf(xgrid,zgrid,potS[mm,nn,:,:].T,cmap=cm.gnuplot)
                    
                    ax2.text(np.max(xgrid),0.,'N=%i' %nn)


