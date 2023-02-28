###########################################################################################
#
#  exponentialdisc.py
#     Helper functions for exponential discs
#
# 18 Aug 2021: First construction
#
#
'''

exponentialdisc (part of exptool.models)
    Implementation of exponential disc models

'''
from __future__ import absolute_import, division, print_function, unicode_literals

# general python imports
import numpy as np

# special function imports
from scipy.special import iv,kv

def disc_rotation_curve(rvals,rd,Mdisc,G=1):
  """build a rough analytic disc rotation curve for model generation

  the exponential disc: https://ned.ipac.caltech.edu/level5/Sept16/Sofue/Sofue4.html"""
  y = rvals/(2*rd)
  sigma0 = Mdisc/(2*np.pi*rd*rd)

  # conversely, this means that
  # Mdisc = 2*np.pi*rd*rd*sigma0

  vd = np.sqrt(4*np.pi*G*sigma0*rd*y*y*(iv(0,y)*kv(0,y) - iv(1,y)*kv(1,y)))
  return vd
