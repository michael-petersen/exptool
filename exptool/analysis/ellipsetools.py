'''
 _______  __       __       __  .______     _______. _______ .___________.  ______     ______    __          _______.
|   ____||  |     |  |     |  | |   _  \   /       ||   ____||           | /  __  \   /  __  \  |  |        /       |
|  |__   |  |     |  |     |  | |  |_)  | |   (----`|  |__   `---|  |----`|  |  |  | |  |  |  | |  |       |   (----`
|   __|  |  |     |  |     |  | |   ___/   \   \    |   __|      |  |     |  |  |  | |  |  |  | |  |        \   \    
|  |____ |  `----.|  `----.|  | |  |   .----)   |   |  |____     |  |     |  `--'  | |  `--'  | |  `----.----)   |   
|_______||_______||_______||__| | _|   |_______/    |_______|____|__|      \______/   \______/  |_______|_______/    
                                                           |______|                                                  
ellipse_tools.py: part of exptool
          basic ellipse fitting tools




                                                           
'''
from __future__ import absolute_import, division, print_function, unicode_literals




from exptool.utils import kde_3d
from exptool.utils import utils


import numpy as np
from matplotlib import _cntr as cntr
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib.cm as cm



#
# ellipse definitions for fitting
#

def gen_ellipse(th,a,b,c):
    '''
    returns generalized ellipse in polar coordinates

    for bar determination following Athanassoula 1990

    '''
    xcomp = ( abs(np.cos(th))**c) / a**c
    ycomp = ( abs(np.sin(th))**c) / b**c
    gell =  ( (xcomp + ycomp) )**(-1./c)
    return gell



def fixed_ellipse(th,a,b):
    '''
    returns c=2 ellipse in polar coordinates


    '''
    xcomp = (( abs(np.cos(th))**2.0) / a**2.0 ) 
    ycomp = (( abs(np.sin(th))**2.0) / b**2.0 ) 
    gell =  ( (xcomp + ycomp) )**(-1./2.0)
    return gell




def inside_ellipse(X,Y,A,B,C,rot=0.):
    '''
    inside_ellipse
        determine whether a set of points is inside of an ellipse
    
    # only tests in first quadrant for power safety



    '''
    rX,rY = X*np.cos(rot)-Y*np.sin(rot),-X*np.sin(rot)-Y*np.cos(rot)
    ellipse_radius = ((abs(rX)/A)**C + (abs(rY)/B)**C)
    yes_ellipse = np.where(ellipse_radius < 1.0)[0]
    ellipse_array = np.zeros(len(X))
    ellipse_array[yes_ellipse] = 1
    return ellipse_array
    #if ((abs(X)/A)**C + (abs(Y)/B)**C) < 1.0 : return 1
    #else: return 0







class genEllipse:
    '''

    #
    # least-squares (and soon to be Fourier?) fitting of generalized ellipses
    #

    '''
    def fitEllipse(self,O,theta_resolution=200,resolution=256,rmax=0.1,loggy=False,generalize=True,weights=None,ncbins=50,contourlevels=[None],SN=1.):
        # here, I have made the input a PSP call...but this could be generalized better.
        
        #
        # get a guess of second-order parameters (center, angle) from SOEllipse
        #
        E = EllipseFinder()

        kde_weights = None
        
        if weights=='normalized':
            kde_weights = O.mass/np.median(O.mass)

        if weights=='mass':
            kde_weights = O.mass

        extent = np.max([abs(O.xpos),abs(O.ypos)])

        desired_resolution = (2.*rmax)/resolution

        grid_size = int(np.ceil( (2.*extent)/desired_resolution))

        tmp_posarr = kde_3d.fast_kde_two(O.xpos, O.ypos, gridsize=(grid_size,grid_size), extents=[-extent,extent,-extent,extent], nocorrelation=False, weights=kde_weights)

        tmp_xarr,tmp_yarr = np.meshgrid( np.linspace(-extent,extent,grid_size),np.linspace(-extent,extent,grid_size))

        # cut the array down to the desired size
        wrows,wcols = np.where( (abs(tmp_xarr) < rmax) & (abs(tmp_yarr) < rmax))

        minx,maxx = np.min(wrows),np.max(wcols)
        miny,maxy = np.min(wcols),np.max(wcols)

        E.posarr,E.xarr, E.yarr = tmp_posarr[minx:maxx,miny:maxy],tmp_xarr[minx:maxx,miny:maxy],tmp_yarr[minx:maxx,miny:maxy]
        
        if loggy:
            pos_vals = E.posarr.reshape(-1,)
            eps = np.min( pos_vals[np.where(pos_vals > 0.)[0]])
            E.posarr = np.log10(E.posarr + eps)
        
        E.add_ellipse_field(check=0,cbins=ncbins,convals=contourlevels)

        # not always going to get 50, looks for non-degenerate ellipses
        ncbins = len(E.AVALS)

        self.R  = np.zeros([ncbins,theta_resolution])
        self.TH = np.zeros([ncbins,theta_resolution])
        self.A  = np.zeros(ncbins)
        self.B  = np.zeros(ncbins)
        self.C  = np.zeros(ncbins)
        self.Ae  = np.zeros(ncbins)
        self.Be  = np.zeros(ncbins)
        self.Ce  = np.zeros(ncbins)
        self.CEN = np.zeros([ncbins,2])
        self.ANG = np.zeros(ncbins)
        self.clevels = np.zeros(ncbins)

        #
        # cycle through all ellipse levels
        #
        k = 0
        indx_atmp = 0.
        
        for j in range(0,ncbins):


            #
            # adjust ellipse correspondingly to be least-squares fitted
            #
            adjx = E.FULLX[j] - E.CENTER[j][0]
            adjy = E.FULLY[j] - E.CENTER[j][1]

            rr = (adjx*adjx + adjy*adjy)**0.5
            th = np.arctan2(adjy,adjx) - E.PHITALLY[j]
            th2 = np.arctan(adjy/adjx) - E.PHITALLY[j]

            #
            # perform the fitting
            #
            thind = np.linspace(0.,2.*np.pi,theta_resolution)

            if generalize:
                popt, pcov = curve_fit(gen_ellipse, th2,rr)

                rind = gen_ellipse(thind,popt[0],popt[1],popt[2])

            else:
                popt, pcov = curve_fit(fixed_ellipse, th2, rr)
                rind = fixed_ellipse(thind,popt[0],popt[1])

            # calculate the errors on the parameters
            perr = np.sqrt(np.diag(pcov))

            
            # could put in a block for avals that fit too small
            atmp = np.max([popt[0],popt[1]])
            btmp = np.min([popt[0],popt[1]])

            aetmp = perr[np.where(atmp==popt)[0]]
            betmp = perr[np.where(btmp==popt)[0]]

            if (atmp > indx_atmp) & (atmp < 0.95*rmax):
                #
                # new guard for max aval 08.23.2016
                #

                #
                # new guard for S/N
                #
                if (atmp/aetmp > SN) & (btmp/betmp > SN):
                    
                    indx_atmp = atmp
                    self.R[k]  = rind
                    self.TH[k] = thind
                    self.A[k]  = atmp
                    self.B[k]  = btmp
                    self.Ae[k] = aetmp
                    self.Be[k] = betmp
                    if generalize:
                        self.C[k]  = popt[2]
                        self.Ce[k] = perr[2]
                    self.CEN[k] = [E.CENTER[j][0],E.CENTER[j][1]]
                    self.ANG[k] = E.PHITALLY[j]
                    self.clevels[k] = E.clevels[j]

                    k += 1

                else:

                    print('ellipse_tools.fitEllipse: Rejected for SN.')

        # trim the extras
        self.R = self.R[0:k]
        self.TH = self.TH[0:k]
        self.A = self.A[0:k]
        self.B = self.B[0:k]
        self.Ae = self.Ae[0:k]
        self.Be = self.Be[0:k]
        if generalize:
            self.C = self.C[0:k]
            self.Ce = self.Ce[0:k]
        self.CEN = self.CEN[0:k]
        self.ANG = self.ANG[0:k]
        self.clevels = self.clevels[0:k]
            
        #self.PH = np.arctan(-self.B/self.A)

        #
        # grab the relevant things from EllipseFinder to be self'ed here
        #
        self.xarr = E.xarr
        self.yarr = E.yarr
        self.posarr = E.posarr
        self.CONX = E.CONX
        self.CONY = E.CONY
        self.FULLX = E.FULLX
        self.FULLY = E.FULLY
        
        
    def plot_contours(self,ellipses=True,fignum=None):
        '''
        plot_contours
            plot the contours on top of the surface density plot.

        '''
        if fignum:
            plt.figure(fignum)
        else:
            plt.figure()

        try:
            defined = (self.xarr[0] == self.xarr[1])
        except:
            print('ellipse_tools: genEllipse.fitEllipse() must be called prior to genEllipse.plot_contours().')
            
        plt.contourf(self.xarr,self.yarr,self.posarr,36,cmap=cm.gnuplot)

        plt.colorbar()

        if (ellipses):
            # plot the generalized ellipses
            for j in range(0,len(self.A)):
                #if T.A[j] < ellipse_tools.ellip_drop(T.A,T.B,drop=0.4): # only plot if a bar contour
                _ = plt.plot(self.R[j]*np.cos(self.TH[j]+self.ANG[j])+self.CEN[j,0],self.R[j]*np.sin(self.TH[j]+self.ANG[j])+self.CEN[j,1],color='black',lw=1.)

    def plot_ellipse_diagnostics(self,fignum=None):
        '''
        plot_ellipse_diagnostics
            plot the ellipticity versus major axis (left) and phase angle versus major axis (right)
            

        '''
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        ax1.scatter(self.A,1.-self.B/self.A,color='black',s=10.)
        ax1.set_xlabel('Semi-major axis')
        ax1.set_ylabel('Ellipticity')
        
        ax2.scatter(self.A,self.ANG,color='black',s=10.)



def ellip_drop(A,B,drop=0.4):
    '''
    given a list of axis lengths, calculate the length of the bar based on some specified ellipticity drop

    '''
    found = False
    j = 2
    while found==False:
        d = (1.-B[j-1]/A[j-1]) - (1.-B[j]/A[j])
        if d > drop:
            found = True
            print('INDEX VALUE is {0:d}'.format(j-1))
        j += 1
        if j==len(A):
            found = True
            j=2
    return A[j-2]



def ellip_drop_below(A,B,drop=0.4):
    '''
    where does the ellipticity first drop below some value?

    '''
    d = (1.-B/A)
    lessthan = np.where( d >= drop )[0]
    if len(lessthan) > 0:
        if np.max(lessthan) < 1:
            minbin = 1
        else:
            minbin = np.max(lessthan)
    else:
        minbin = 1
    return A[minbin]



def max_ellip_drop(A,B):
    edrop = np.ediff1d((1.-B/A),to_end=0.)
    return A[ np.where(np.min(edrop)==edrop)[0]]




#
# MUNOZ13 proposes several bar length metrics, reproduced here:
#
def max_ellip(A,B):
    e = (1.-B/A)
    return A[ np.where(np.max(e)==e)[0]]

        
def ellip_change(A,B,change=0.1):
    e = (1.-B/A)
    ellip_index = np.where(np.max(e)==e)[0]
    max_ellip_value = e[ellip_index]
    ellip_diff = 0.
    while (ellip_diff < change):
        ellip_index += 1
        ellip_diff = abs(e[ellip_index] - max_ellip_value)
    return A[ellip_index-1]


def pa_change(A,B,change=10.):
    # change must be in degrees
    e = (1.-B/A)
    pa = np.arctan(B/A)
    ellip_index = np.where(np.max(e)==e)[0]
    pa_value = pa[ellip_index]
    pa_diff = 0.
    while (pa_diff < change):
        ellip_index += 1
        pa_diff = abs(pa[ellip_index] - pa_value)*180./np.pi
    return A[ellip_index-1]





class SOEllipse(object):
    '''
    #
    # Conic Ellipse fitter
    #     exploiting the quadratic curve nature of the ellipse
    #

    advantages: fast

    disadvantages: does not have flexibility
    
    '''
    @staticmethod
    def fitEllipse(x,y):
        #
        # Take a set of x,y points at fit an ellipse to it
        #
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        
        return a

    @staticmethod
    def ellipse_center(a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
            
        return np.array([x0,y0])

    @staticmethod
    def ellipse_angle_of_rotation( a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        
        return 0.5*np.arctan(2*b/(a-c))

    @staticmethod
    def ellipse_axis_length( a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        
        return np.array([res1, res2])





class EllipseFinder():
    """

    Try EllipseFinder(help=True) to get print_usage.

    """
    def __init__(self,help=False):

        if help: EllipseFinder.print_usage()
            
        return None

    def print_usage(self):

        print(\
        "\n\n*****************Ellipse Fitting Tools*********************\n\
        \nMEMBER DEFINITIONS:\
        \n   *generate_flat_field(xpos,ypos,zpos,mass)\
        \n   *add_ellipse_field(xarr,yarr,posarr,check=1)\
        \n   *add_single_ellipse\
        \n\n***********************************************************\n\n")


    def full_ellipse(self,IO_obj,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        EllipseFinder.generate_flat_field(IO_obj.xpos,IO_obj.ypos,IO_obj.zpos,IO_obj.mass,xres=xres,xbins=xbins,normamass=normamass,numdens=numdens,logvals=logvals)

        

       
    def generate_flat_field(self,xpos,ypos,zpos,mass,zcut=0.1,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        # ToDo : add support for zpos to make slices.

        if xbins[0] == 0:
            print('ellipse_tools.EllipseFinder.generate_flat_field: xbins is not user specified. using default...')
            
            self.xbins = np.linspace(-np.max(xpos),np.max(xpos),xres)

        else:

            self.xbins = xbins


        if normamass:
            massuse = mass/np.median(mass)
        else:
            massuse = mass

        
        self.xarr,self.yarr,self.posarr = utils.quick_contour(self.xbins,self.xbins,xpos,ypos,massuse)

        if numdens:
            xt,yt,pt = utils.quick_contour(self.xbins,self.xbins,xpos,ypos,np.ones(len(xpos)))
        
        if logvals:
            self.posarr = np.log10(self.posarr+np.min(massuse))


    def generate_flat_field_kde(self,xpos,ypos,zpos,mass,zcut=0.1,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        # ToDo : add support for zpos to make slices.

        if xbins[0] == 0:
            print('ellipse_tools.EllipseFinder.generate_flat_field: xbins is not user specified. using default...')
            
            self.xbins = np.linspace(-np.max(xpos),np.max(xpos),xres)

        else:

            self.xbins = xbins
            xres = len(self.xbins)

        
        if normamass:
            massuse = mass/np.median(mass)
        else:
            massuse = mass


        if numdens: massuse = None
        

        extent = np.max(xbins)
        tt = kde_3d.fast_kde(xpos,ypos,zpos, gridsize=(xres+2,xres+2,xres+2), extents=[-extent,extent,-extent,extent,-0.05,0.05], nocorrelation=False, weights=massuse)
        
        self.posarr = np.sum(tt[1:(xres+1),1:(xres+1),1:(xres+1)],axis=0)



        if logvals:
            self.posarr = np.log10(self.posarr+np.min(massuse))


        self.xarr,self.yarr = np.meshgrid(self.xbins,self.xbins)


        
    def determine_contour_levels(self,cbins=50,vertices=30):
        '''
        determine_contour_levels
             intelligently select the surface density values for fitting ellipses

        inputs
        ------
        self     : object
        cbins    : (int) the number of output contours desired
        vertices : (int) the number of points in a contour that must exist to fit ellipse


        returns
        -------
        self.clevels : array of surface density values to fit ellipses
        
        '''

        # use matplotlib's marching squares contour finder for this. to be improved with a better algorithm later...
        c = cntr.Cntr(self.xarr,self.yarr,self.posarr)

        # how about a smarter way to decide where to lay the contours?
        startval = 0.90*np.max(self.posarr)                  # maximum limit for contours

        eps = 1.e-10*np.max(self.posarr)                     # make an epsilon in case there are zero bins
        endval = 1.0*np.min(self.posarr) + eps               # minimum limit for contours

        stepsize = (startval-endval)/1000.

        # iterate down and up to find where contours exist       
        conlevels = np.zeros(1001)

        indval = startval
        j = 0
        
        while (indval > endval):
            res = c.trace(indval)
            if (len(res) > 0):             # does the contour level exist?
                if (len(res[0])>vertices):       # only accept those with greater than 30 vertices
                    conlevels[j] = indval
                    j += 1
                    
            indval -= stepsize
            

        cvals = conlevels[0:j]            # truncate list to number of valid contours


        # redefine startval and stepsize
        startval = np.max(cvals)
        endval = np.min(cvals)
        stepsize = (startval-endval)/float(cbins)

        # define the contour levels
        self.clevels = np.array([ (startval - stepsize*x) for x in range(0,cbins)])

        
                        
    def add_ellipse_field(self,xarr=None,yarr=None,posarr=None,check=0,cbins=50,convals=[None]):

        
        try:
            y = self.xarr[0,0]
        except:
            try:
                y = xarr[0,0]
                self.xarr = xarr
                self.yarr = yarr
                self.posarr = posarr
            except:
                print('EllipseFinder.add_ellipse_field: no valid arrays input.')
                return None


        # initialize internal blank arrays
        tt = []
        avals  =[]
        bvals = []
        phitally = []
        levout = []
        xxtot = []
        yytot = []
        fullx = []
        fully = []
        centera = []


        # decide on contours to be fit if not specified
        if convals[0] == None:
            EllipseFinder.determine_contour_levels(self,cbins=cbins)

        else:
            self.clevels = convals
            # this has to be carefully done in order to make sure it matches log call, unless I change to better density determination??
        
        # ^^ generated self.clevels

        # instantiate the contour class object
        c = cntr.Cntr(self.xarr,self.yarr,self.posarr)

        
        indx_aval = 0.
        
        for contour_indx,contour_value in enumerate(self.clevels):

            res = c.trace(contour_value,nchunk=4)                                     # trace the individual contours
            
            if (len(res)>0):                                      # guard against non-existent contours
                
                if (len(res[0])>10):                              # guard against singular matrices

                    # construct arrays for particular contour level
                    xcon = []
                    ycon = []
                    for i in range(0,len(res[0])):
                        xcon.append(res[0][i][0])
                        ycon.append(res[0][i][1])
                    XCON = np.array(xcon)
                    YCON = np.array(ycon)

                    fullx.append(XCON)
                    fully.append(YCON)

                    # perform the second-order ellipse fitting
                    ell = SOEllipse().fitEllipse(XCON,YCON)
                    phi = SOEllipse().ellipse_angle_of_rotation(ell)
                    center = SOEllipse().ellipse_center(ell)
                    
                    if (center[0]**2.+center[1]**2.)**0.5 < 0.007: # guard against bad center detections (upgrade to binsize fraction??)
                        axes = SOEllipse().ellipse_axis_length(ell)              
                        a, b = axes

                        if indx_aval < np.max([a,b]):
                            indx_aval = np.max([a,b])
                            #print indx_aval

                            # print to screen?
                            
                            r = np.linspace(-0.02,0.02,100)
                            R = np.arange(0,2*np.pi, 0.01)
                            xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                            yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
                            xxtot.append(xx)
                            yytot.append(yy)
                            
                            if check==1:
                                plt.plot(xx,yy,color='black')
                            
                            avals.append(np.max([a,b]))               # force major axis
                            bvals.append(np.min([a,b]))               # force minor axis
                            levout.append(contour_value)
                            phitally.append(phi)
                            centera.append([center[0],center[1]])
                        
            else:
                # if no fit:
                avals.append(0.0)
                levout.append(contour_value)
                bvals.append(0.0)
                phitally.append(0.0)
                centera.append([0.,0.])



        #
        # want to resample values so that avals is increasing.
        #
            
        self.AVALS    = np.array(avals)      # the major axis values
        self.BVALS    = np.array(bvals)      # the minor axis values
        self.LEVOUT   = np.array(levout)     # the contour levels of the output array
        self.PHITALLY = np.array(phitally)   # the angle values
        self.CONX     = np.array(xxtot)      # full x positions for contours
        self.CONY     = np.array(yytot)      # full y positions for contours
        self.FULLX    = np.array(fullx)      # full x positions for ellipses
        self.FULLY    = np.array(fully)      # full y positions for ellipses
        self.CENTER   = np.array(centera)    # positions of the fitted ellipse centers
        
        #return AVALS,BVALS,PHITALLY,LEVOUT,np.array(xxtot),np.array(yytot)



'''
    def add_single_ellipse(self,xarr,yarr,posarr,level,frac=True):
        
        c = cntr.Cntr(xarr,yarr,posarr)
        aval = 0.
        bval = 0.
        xxtot = []
        yytot = []
        if frac==True: levelin = level*np.max(posarr)
        else: levelin = level
        res =c.trace(levelin)
        if (len(res)>0):                                      # guard against non-existent contours
                if (len(res[0])>10):                              # guard against singular matrices
                    xcon = []
                    ycon = []
                    for i in range(0,len(res[0])):
                        xcon.append(res[0][i][0])
                        ycon.append(res[0][i][1])
                    XCON = np.array(xcon)
                    YCON = np.array(ycon)
                    ell = fitEllipse(XCON,YCON)
                    phi = ellipse_angle_of_rotation(ell)
                    center = ellipse_center(ell)
                    if (center[0]**2.+center[1]**2.)**0.5 < 0.007: # guard against bad center detections (upgrade to binsize fraction??)
                        axes = ellipse_axis_length(ell)              
                        a, b = axes
                        r = np.linspace(-0.02,0.02,100)
                        R = np.arange(0,2*np.pi, 0.01)
                        xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                        yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
                        xxtot.append(xx)
                        yytot.append(yy)
                        aval = np.max([a,b])               # force major axis
                        bval = np.min([a,b])               # force minor axis
        else:
            print('Invalid ellipse chosen.')
        return aval,bval,np.array(xxtot),np.array(yytot)

'''

