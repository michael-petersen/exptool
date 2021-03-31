'''
rotationcurve.py
Models for rotation curves

31-Mar-2021: First written

inspired by some very helpful notes at
https://astrohchung.com/project/mangarc/

see Bouche+ 2015 summary of models:
https://ui.adsabs.harvard.edu/abs/2015AJ....150...92B/abstract

TODO
1. this would also be an ideal place to describe the fits advocated by Sanders et al. (https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract)
2. study Lucy-Richardson deconvolution (see https://ui.adsabs.harvard.edu/abs/2020arXiv200804313C/abstract)
     u^{n+1} = u^n ( (d/(u^n [2dconvolve] p)) [2dconvolve] p )
   where u^n is the nth estimate of the 2d maximum likelihood solution (u^0=d), d is the original
   PSF-convolved image, p is the 2d PSF, and [2dconvolve] is the 2d convolution.
3. more practically, turn this all into fits

'''

import numpy as np


class rotationcurve():

    def __init__(self,vrotation,inclination=0,phi0=0,vsys=0,rt=1,modeltype='flat'):
        """

        are we happy with all the defaults, or
        should we switch to **kwargs for a variety of extra parameters?

        """

        # this needs to be a relationship with radius
        self.vrot = vrotation

        # check in radians?
        self.inc  = inclination
        
        self.phi0 = phi0

        self.vsys = vsys

        self.rt   = rt

        if modeltype=='arctan':
            self.model = arctanmodel
        elif modeltype=='tanh':
            self.model = tanhmodel            
        elif modeltype=='exp':
            self.model = expmodel
        elif modeltype=='tanhlinear':
            self.model = tanhlinearmodel
        else:
            self.model = flatmodel

    def vlos(self,r,phi=0):
        """return the line-of-sight velocity"""
        return self.vsys + self.model(r,self.vrot,self.rt) * np.sin(self.inc) * np.cos(phi-self.phi0)


    def flatmodel(r,vrot,rt):
        """flat rotation curve model

        """
        return vrot


    def arctanmodel(r,vrot,rt):
        """return arctan model

        e.g. Puech et al. 2008

        """
        return vrot*(2./np.pi)*np.arctan(r/rt)

    def tanhmodel(r,vrot,rt):
        """return tanh model

        e.g. Andersen & Bershady (2013)

        """
        return vrot*np.tanh(r/rt)

    def expmodel(r,vrot,rt):
        """return inverted exp model

        e.g. Feng & Gallo (2011)

        """
        return vrot*(1-np.exp(r/rt))

    def tanhlinearmodel(r,vrot,rt):
        """return tanh+linear model

        see https://ui.adsabs.harvard.edu/abs/2020arXiv200804313C/abstract

        'By adding one linear term on the previous model, we can fit the slope of the galaxy rotation curve at its outskirts.'

        pass rt as a tuple!
        """
        return vrot*(np.tanh(r/rt[0]) + r/rt[1])


    def dispersionprofile(r,sigma0,rt,sigmaprime=1.):
        """a simple dispersion profile fit
        see https://ui.adsabs.harvard.edu/abs/2020arXiv200804313C/abstract
        """

        return sigma0/(sigmaprime*r/rt + 1)

        
