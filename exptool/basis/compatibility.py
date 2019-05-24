'''
backup versions of C code in case compatibility fails



'''


import numpy as np


#############################################################################################
#
# mapping functions
#
def r_to_xi(r, cmap, scale):
  '''
  bonus written for arrays!

  '''
  #
  r = np.asarray(r)
  scalar_input = False
  if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True
  #
  subzero = np.where(r < 0.)
  #
  if ( cmap == 1 ):
    #
    outval = (r/scale-1.0)/(r/scale+1.0)
    outval[subzero] = 0.
  #
  #
  elif ( cmap == 2 ):
    #
    outval = np.log(r)
    outval[subzero] = 0.
  #
  else:    
    outval = r;
    outval[subzero] = 0.
  #
  if scalar_input:
        return np.squeeze(outval)
  return outval


#
# these have the guards removed now...
#

def xi_to_r(xi, cmap, scale):
  if ( cmap == 1 ):
    return (1.0+xi)/(1.0 - xi) * scale;
  #
  elif (cmap==2):
    return np.exp(xi);
  #
  else:
    return xi;


def d_xi_to_r(xi, cmap, scale):
    '''
    d_xi_to_r
        compute delta r as a function of xi (radial mapping coordinate)

    TODO: needs error handling

    '''
    if ( cmap == 1 ) :
        return 0.5*(1.0-xi)*(1.0-xi)/scale;
    #
    elif (cmap==2):
        return np.exp(-xi);
    #
    else:
        return 1.0;


def z_to_y(z, hscale):
    '''
    compute z position to y table mapping

    22-5-2019, precision work:
    if we increase MIN_DOUBLE, does that help push everything into the plane?

    '''
    return (z /( np.abs(z)+1.e-8)) * np.arcsinh( np.abs(z/hscale));


def y_to_z(y, hscale):
    '''
    compute y table to z position mapping

    '''
    return hscale*np.sinh(y)



