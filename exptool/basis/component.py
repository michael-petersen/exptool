

'''
  ______   ______   .___  ___. .______     ______   .__   __.  _______ .__   __. .___________.
 /      | /  __  \  |   \/   | |   _  \   /  __  \  |  \ |  | |   ____||  \ |  | |           |
|  ,----'|  |  |  | |  \  /  | |  |_)  | |  |  |  | |   \|  | |  |__   |   \|  | `---|  |----`
|  |     |  |  |  | |  |\/|  | |   ___/  |  |  |  | |  . `  | |   __|  |  . `  |     |  |     
|  `----.|  `--'  | |  |  |  | |  |      |  `--'  | |  |\   | |  |____ |  |\   |     |  |     
 \______| \______/  |__|  |__| | _|       \______/  |__| \__| |_______||__| \__|     |__|     
component.py
       handle the input of various components that can then be passed to potential analysis





 '''


# standard libraries
import numpy as np
import time

# exptool classes
from exptool.io import psp_io
from exptool.basis import eof
from exptool.basis import spheresl

#from exptool.utils import halo_methods
#from exptool.utils import utils


# for interpolation
#from scipy.interpolate import UnivariateSpline



class Component():


    def __init__(self,PSPComponent,nmaxbods=-1):
        '''

        inputs
        ---------
        PSPComponent :
        nmaxbods     :  maximum number of bodies to tabulate for expansion

        '''

        if PSPComponent['expansion'] == 'cylinder':

            Component.cyl_expansion(self)

        if PSPComponent['expansion'] == 'sphereSL':

            Component.sph_expansion(self)



    def cyl_expansion(self):

        pass


    def sph_expansion(self):

        pass

        
            
