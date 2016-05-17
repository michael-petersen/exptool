

import helpers
import numpy as np
from matplotlib import _cntr as cntr
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import kde_3d
from scipy.optimize import curve_fit


'''
import psp_io
import ellipse_tools

# Specify a simulation output
#
O = psp_io.Input('/Volumes/SIMSET/OUT.run064a.00800',comp='star')

O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='star')


O = psp_io.Input('/scratch/mpetersen/Disk013/OUT.run013p.01000',comp='star')

#
# Two methods to fit ellipses to a simulation output.
#
#     First, using standard ellipses

# instantiate the class
E = ellipse_tools.EllipseFinder()

# make a field to fit ellipses to
import time
t1 = time.time()
E.generate_flat_field_kde(O.xpos,O.ypos,O.zpos,O.mass,xbins=np.linspace(-0.05,0.05,121),normamass=True,logvals=True)
print time.time()-t1


# several fields are now defined: E.xarr, E.yarr, E.posarr

# plot the contours
plt.figure(0)
plt.contourf(E.xarr,E.yarr,E.posarr,36,cmap=cm.gnuplot)


# add a field of ellipses
E.add_ellipse_field(E,check=0,cbins=80)
# many more fields are defined now, some examples are below


bar_length_ellipse = ellip_drop(E.AVALS,E.BVALS,drop=0.4)



# would like to compare the the bar extent in all simulations with the ellipse fit to see if they are different
# Look at any orbit that passes into the bar ellipse and take the time average of those, plus time average of angular momentum with some thoughts about what it means


TX,TY = insta_transform(O.xpos,O.ypos,np.pi/2.-0.08)

E = ellipse_tools.EllipseFinder()

ellipse_tools.EllipseFinder.generate_flat_field(E,TX,TY,O.zpos,O.mass,xbins=np.linspace(-0.05,0.05,51),normamass=True,logvals=True)


F = ellipse_tools.EllipseFinder()
TX,TY = insta_transform(O.xpos,O.ypos,np.pi/2.-0.08)
TVX,TVY = insta_transform(O.xvel,O.yvel,np.pi/2.-0.08)

R = (O.xpos**2. + O.ypos**2.)**0.5
VTAN = (O.xpos*O.yvel - O.ypos*O.xvel)/R

VRAD = (O.xpos*O.xvel + O.ypos*O.yvel)/R

plt.scatter(R[0:20000],abs(VRAD[0:20000]),s=1.,color='black')


H = ellipse_tools.EllipseFinder()
# make a field to fit ellipses to
ellipse_tools.EllipseFinder.generate_flat_field(H,TX,TY,O.zpos,VRAD,xbins=np.linspace(-0.05,0.05,51),normamass=False,logvals=False)



ellipse_tools.EllipseFinder.generate_flat_field(F,TX,TY,O.zpos,VTAD,xbins=np.linspace(-0.05,0.05,51),normamass=False,logvals=False)


G = ellipse_tools.EllipseFinder()


# make a field to fit ellipses to
ellipse_tools.EllipseFinder.generate_flat_field(G,TX,TY,O.zpos,np.ones(len(O.mass)),xbins=np.linspace(-0.05,0.05,51),normamass=False,logvals=False)


plt.contourf(F.xarr,F.yarr,H.posarr/G.posarr,36,cmap=cm.jet)



plt.figure(2)
plt.contourf(E.xarr,E.yarr,E.posarr,36)


ellipse_tools.EllipseFinder.add_ellipse_field(E,check=0,cbins=80)


# plot the bar ellipses
for j in range(0,len(E.AVALS)):
     if E.AVALS[j] < ellipse_tools.ellip_drop(E.AVALS,E.BVALS,drop=0.4): # only plot if a bar contour
              _ = plt.plot(E.CONX[j],E.CONY[j],color='black',lw=1.)

#



plt.figure(2)
plt.contourf(E.xarr,E.yarr,E.posarr,36)

j=24
_ = plt.plot(E.CONX[j],E.CONY[j],color='black',lw=3.)



for j in [28]: 
     print j
     if E.AVALS[j] < ellipse_tools.ellip_drop(E.AVALS,E.BVALS,drop=0.4): # only plot if a bar contour


#


plt.figure(1)

# plot the ellipticity as a function of semi-major axis. Where this drops significantly is the end of the bar.
plt.plot(E.AVALS,1.-E.BVALS/E.AVALS)

# ...which can be defined with this tool.
bar_length_ellipse = ellipse_tools.ellip_drop(E.AVALS,E.BVALS,drop=0.4)
print 'The length of the bar in ellipse measurements is %4.3f' %bar_length_ellipse

#
# A more sophisticated approach follows that of Athanassoula (1990) in fitting generalized ellipses
#

# instantiate a generalized ellipse object
T = ellipse_tools.genEllipse()

# use the simulation output from above to 
T.fitEllipse(O,xbins=np.linspace(-0.05,0.05,51),loggy=True)

j=24
_ = plt.plot(T.R[j]*np.cos(T.TH[j]+T.ANG[j]-np.pi/2.-0.08)+T.CEN[j,0],T.R[j]*np.sin(T.TH[j]+T.ANG[j]-np.pi/2.-0.08)+T.CEN[j,1],color='black',lw=1.)


#

# plot the contours again
plt.figure(2)
plt.contourf(E.xarr,E.yarr,E.posarr,36)

# plot the generalized ellipses
for j in range(0,len(T.A)):
     if T.A[j] < ellipse_tools.ellip_drop(T.A,T.B,drop=0.4): # only plot if a bar contour
              _ = plt.plot(T.R[j]*np.cos(T.TH[j]+T.ANG[j])+T.CEN[j,0],T.R[j]*np.sin(T.TH[j]+T.ANG[j])+T.CEN[j,1],color='black',lw=1.)

#
# observe that this really only has meaning for the bar region; generalized ellipses are failing at larger radii
#    (as determined by the overlap in ellipses)
#

# plot over the ellipse drop of the other formalism
plt.figure(1)
plt.plot(T.A,1.-T.B/T.A)

# ...which can be defined with this tool.
bar_length_ellipse = ellipse_tools.ellip_drop(T.A,T.B,drop=0.4)
print 'The length of the bar in ellipse measurements is %4.3f' %bar_length_ellipse


# also examine the generalized fit parameter, $c$
plt.figure(3)
plt.plot(T.A,T.C)


'''

#
# ellipse definitions for fitting
#

def gen_ellipse(th,a,b,c):
    # returns generalized ellipse in polar coordinates
    xcomp = ( abs(np.cos(th))**c) / a**c
    ycomp = ( abs(np.sin(th))**c) / b**c
    gell =  ( (xcomp + ycomp) )**(-1./c)
    return gell



def fixed_ellipse(th,a,b):
    # returns c=2 ellipse in polar coordinates
    xcomp = ( abs(np.cos(th))**2.0) / a**2.0
    ycomp = ( abs(np.sin(th))**2.0) / b**2.0
    gell =  ( (xcomp + ycomp) )**(-1./2.0)
    return gell





class genEllipse:

    #
    # least-squares (and soon to be Fourier) fitting of generalized ellipses
    #
    
    def fitEllipse(self,O,thres=200,xbins=np.linspace(-0.03,0.03,51),loggy=False,generalize=True):
        # here, I have made the input a PSP call...but this could be generalized better.
        
        #
        # get a guess of second-order parameters (center, angle) from SOEllipse
        #
        E = EllipseFinder()

        EllipseFinder.generate_flat_field_kde(E,O.xpos,O.ypos,O.zpos,O.mass,xbins=xbins,logvals=loggy)

        ncbins = 50
        EllipseFinder.add_ellipse_field(E,check=0,cbins=ncbins)

        # not always going to get 50, looks for non-degenerate ellipses
        ncbins = len(E.AVALS)

        self.R  = np.zeros([ncbins,thres])
        self.TH = np.zeros([ncbins,thres])
        self.A  = np.zeros(ncbins)
        self.B  = np.zeros(ncbins)
        self.C  = np.zeros(ncbins)
        self.CEN = np.zeros([ncbins,2])
        self.ANG = np.zeros(ncbins)

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
            thind = np.linspace(0.,2.*np.pi,thres)

            if generalize:
                popt, pcov = curve_fit(gen_ellipse, th2,rr)

                rind = gen_ellipse(thind,popt[0],popt[1],popt[2])

            else:
                popt, pcov = curve_fit(fixed_ellipse, th2, rr)
                rind = fixed_ellipse(thind,popt[0],popt[1])

            # could put in a block for avals that fit too small
            atmp = np.max([popt[0],popt[1]])
            btmp = np.min([popt[0],popt[1]])

            if atmp > indx_atmp:
                indx_atmp = atmp
                self.R[k]  = rind
                self.TH[k] = thind
                self.A[k]  = atmp
                self.B[k]  = btmp
                if generalize: self.C[k]  = popt[2]
                self.CEN[k] = [E.CENTER[j][0],E.CENTER[j][1]]
                self.ANG[k] = E.PHITALLY[j]

                k += 1

        # trim the extras
        self.R = self.R[0:k]
        self.TH = self.TH[0:k]
        self.A = self.A[0:k]
        self.B = self.B[0:k]
        if generalize: self.C = self.C[0:k]
        self.CEN = self.CEN[0:k]
        self.ANG = self.ANG[0:k]
            
        #
        # analyze for bar lengths? this would ideally be separate, but overhead might be very small here
        #

        #self.PH = np.arctan(-self.B/self.A)
        


def ellip_drop(A,B,drop=0.4):
    found = False
    j = 2
    while found==False:
        d = (1.-B[j-1]/A[j-1]) - (1.-B[j]/A[j])
        if d > drop:
            found = True
            print 'INDEX VALUE is %i' %(j-1)
        j += 1
        if j==len(A):
            found = True
            j=2
    return A[j-2]



def max_ellip_drop(A,B):
    edrop = np.ediff1d((1.-B/A),to_end=0.)
    return A[ np.where(np.min(edrop)==edrop)[0]]

        

class SOEllipse:

    #
    # Second-Order Ellipse fitter
    #     exploiting the quadratic curve nature of the ellipse
    #
    
    def fitEllipse(self,x,y):
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

    def ellipse_center(self,a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
            
        return np.array([x0,y0])

    def ellipse_angle_of_rotation(self, a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        
        return 0.5*np.arctan(2*b/(a-c))

    def ellipse_axis_length(self, a ):
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

        print \
        "\n\n*****************Ellipse Fitting Tools*********************\n\
        \nMEMBER DEFINITIONS:\
        \n   *generate_flat_field(xpos,ypos,zpos,mass)\
        \n   *add_ellipse_field(xarr,yarr,posarr,check=1)\
        \n   *add_single_ellipse\
        \n\n***********************************************************\n\n"


    def full_ellipse(self,IO_obj,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        EllipseFinder.generate_flat_field(IO_obj.xpos,IO_obj.ypos,IO_obj.zpos,IO_obj.mass,xres=xres,xbins=xbins,normamass=normamass,numdens=numdens,logvals=logvals)

        

       
    def generate_flat_field(self,xpos,ypos,zpos,mass,zcut=0.1,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        # ToDo : add support for zpos to make slices.

        if xbins[0] == 0:
            print 'EllipseFinder.generate_flat_field: xbins is not user specified. using default...'
            
            self.xbins = np.linspace(-np.max(xpos),np.max(xpos),xres)

        else:

            self.xbins = xbins


        if normamass:
            massuse = mass/np.median(mass)
        else:
            massuse = mass

        
        self.xarr,self.yarr,self.posarr = helpers.quick_contour(self.xbins,self.xbins,xpos,ypos,massuse)

        if numdens:
            xt,yt,pt = helpers.quick_contour(self.xbins,self.xbins,xpos,ypos,np.ones(len(xpos)))
        
        if logvals:
            self.posarr = np.log10(self.posarr+np.min(massuse))


    def generate_flat_field_kde(self,xpos,ypos,zpos,mass,zcut=0.1,xres=30,xbins=[0],normamass=False,numdens=False,logvals=False):

        # ToDo : add support for zpos to make slices.

        if xbins[0] == 0:
            print 'EllipseFinder.generate_flat_field: xbins is not user specified. using default...'
            
            self.xbins = np.linspace(-np.max(xpos),np.max(xpos),xres)

        else:

            self.xbins = xbins
            xres = len(self.xbins)

        if normamass:
            massuse = mass/np.median(mass)
        else:
            massuse = mass


        tt = kde_3d.fast_kde(xpos,ypos,zpos, gridsize=(xres+2,xres+2,xres+2), extents=[np.min(xbins),np.max(xbins),np.min(xbins),np.max(xbins),-zcut,zcut], nocorrelation=False, weights=massuse)
        
        self.posarr = np.sum(tt[1:(xres+1),1:(xres+1),1:(xres+1)],axis=0)

        self.xarr,self.yarr = np.meshgrid(self.xbins,self.xbins)


        
        #self.xarr,self.yarr,self.posarr = helpers.quick_contour(self.xbins,self.xbins,xpos,ypos,massuse)

        if numdens:
            print 'ellipse_tools.generate_flat_field_kde: numdens is not currently accepted.'
            #xt,yt,pt = helpers.quick_contour(self.xbins,self.xbins,xpos,ypos,np.ones(len(xpos)))
        
        if logvals:
            self.posarr = np.log10(self.posarr+np.min(massuse))


            

    def determine_contour_levels(self,cbins=50):

        # cbins is the number of output contours desired


        # use matplotlib's marching squares contour finder for this. to be improved with a better algorithm later...
        c = cntr.Cntr(self.xarr,self.yarr,self.posarr)


        # how about a smarter way to decide where to lay the contours?
        startval = 0.90*np.max(self.posarr)                  # maximum limit for contours

        eps = 1.e-10*np.max(self.posarr)                     # make an epsilon in case there are zero bins
        endval = 1.0*np.min(self.posarr) + eps               # minimum limit for contours

        stepsize = (startval-endval)/1000.
        #print 'The INITIAL Stepsize is ',stepsize,'(',startval,endval,')'

        # iterate down and up to find where contours exist       
        conlevels = np.zeros(1000)

        indval = startval
        j = 0
        
        while (indval > endval):
            res = c.trace(indval)
            if (len(res) > 0):             # does the contour level exist?
                if (len(res[0])>10):       # only accept those with greater than 10 vertices
                    conlevels[j] = indval
                    j += 1
                    
            indval -= stepsize
            

        cvals = conlevels[0:j]            # truncate list to number of valid contours


        # redefine startval and stepsize
        startval = np.max(cvals)
        endval = np.min(cvals)
        stepsize = (startval-endval)/float(cbins)
        #print 'The FINAL Stepsize is ',stepsize,'(',startval,endval,')'

        # define the contour levels
        self.clevels = np.array([ (stepsize*x + startval) for x in range(0,cbins)])

        
                        
    def add_ellipse_field(self,xarr=None,yarr=None,posarr=None,check=0,cbins=50):

        if cbins >= 1000:
            print 'cbins must be <= 999.'
            return
        

        try:
            y = self.xarr[0,0]
        except:
            try:
                y = xarr[0,0]
                self.xarr = xarr
                self.yarr = yarr
                self.posarr = posarr
            except:
                print 'EllipseFinder.add_ellipse_field: no valid arrays input.'
                return None

        
        # break matplotlib's contour finder for this. to be improved with a better algorithm later...
        c = cntr.Cntr(self.xarr,self.yarr,self.posarr)

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

        # how about a smarter way to decide where to lay the contours?
        startval = 0.90*np.max(self.posarr)                  # maximum limit for contours

        eps = 1.e-10*np.max(self.posarr)                     # make an epsilon in case there are zero bins
        endval = 1.1*np.min(self.posarr) + eps               # minimum limit for contours

        stepsize = (startval-endval)/1000.
        #print 'The INITIAL Stepsize is ',stepsize,'(',startval,endval,')'

        # iterate down and up to find where contours start exist       
        conlevels = np.zeros(1000)
        #existingc

        indval = startval
        j = 0
        
        while (indval > endval):
            res = c.trace(indval)
            if (len(res) > 0):
                if (len(res[0])>10):
                    #existingc.append(indval)
                    conlevels[j] = indval
                    j += 1
                    
            indval -= stepsize
            

        #cvals = np.array(existingc)
        cvals = conlevels[0:j]


        # redefine startval and stepsize
        startval = np.max(cvals)
        endval = np.min(cvals)
        stepsize = (startval-endval)/float(cbins)
        #print 'The FINAL Stepsize is ',stepsize,'(',startval,endval,')'

        indx_aval = 0.
        
        while startval > endval:                            # minimum limit for contours
            
            res =c.trace(startval,nchunk=4)                                     # trace the individual contours

            
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
                            levout.append(startval)
                            phitally.append(phi)
                            centera.append([center[0],center[1]])
                        
            else:
                # if no fit:
                avals.append(0.0)
                levout.append(0.0)
                bvals.append(0.0)
                phitally.append(0.0)
                centera.append([0.,0.])

            # advance to next step    
            startval -= stepsize


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
            print 'Invalid ellipse chosen.'
        return aval,bval,np.array(xxtot),np.array(yytot)

'''

