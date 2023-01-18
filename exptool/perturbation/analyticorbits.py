"""



#plt.plot(model.rcurve,find_jmax(model.rcurve,1.,model))


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from scipy.interpolate import UnivariateSpline

class spherical_model(object):
    """initialize the values in the spherical model

    """
    rcurve = 0
    potcurve = 0
    dpotcurve = 0



class orbit(object):
    """initialize the values in the orbit model

    """
    r_circ = 0.
    r_apo = 0.
    r_peri = 0.
    ee = 0.
    jj = 0.
    kappa = 0.
  



def make_sph_model(filename):
    """reads a spherical model file text file and generates interpolated values

    Args:
        filename:
    Returns:
        model:



    """

    M = np.loadtxt(filename, dtype={'names': ('rcurve', 'potcurve', 'dpotcurve'),'formats': ('f4', 'f4', 'f4')},skiprows=1)
    
    model = spherical_model()
    model.rcurve = M['rcurve']
    model.potcurve = M['potcurve']
    model.dpotcurve = M['dpotcurve']

    model.rcurve = M['rcurve']
    model.potcurve = UnivariateSpline(model.rcurve,M['potcurve'],k=3)
    model.dpotcurve = UnivariateSpline(model.rcurve,M['dpotcurve'],k=3)

    return model


#

def find_j(r,kappa,model):
    """find the (non-normalized) angular momentum for the orbit

    Args:
        r: the radius of the orbit
        kappa: the normalized angular momentum
        model: the spherical model class
    Returns:
        J: the angular momentum of the orbit

    """
    
    dudr = model.dpotcurve(r)#model.dpotcurve[ (abs(r-model.rcurve)).argmin()]
    jmax = np.sqrt(r*r*r*dudr);
    J = jmax*kappa;
    
    return J




def Ecirc(r,E,model):
    """find the energy of a circular orbit

    """
    ur = model.potcurve(r)#model.potcurve[ (abs(r-model.rcurve)).argmin()]
    dudr = model.dpotcurve(r)#model.dpotcurve[ (abs(r-model.rcurve)).argmin()]
    return  E - 0.5*r*dudr - ur

  
def denom(r,E,J,model):
    """compute the denominator of the potential equation

    """
    ur = model.potcurve(r)#model.potcurve[ (abs(r-model.rcurve)).argmin()]
    return 2.0*(E-ur)*r*r - J*J;



def make_orbit(orbit,E,K,model):
  """determine orbit properties for a given energy and kappa in a spherical model

  Args:
      orbit:
      E:
      K:
      model:
  Returns:
      orbit: updated orbit class with new attributes


  """
  orbit.ee = E
  orbit.r_circ = brentq(Ecirc,0.0,0.1,args=(orbit.ee,model))
  orbit.kappa = K
  orbit.jj = find_j(orbit.r_circ,orbit.kappa,model)
  orbit.r_apo = brentq(denom,orbit.r_circ,0.1,args=(orbit.ee,orbit.jj,model))
  orbit.r_peri = brentq(denom,0.0,orbit.r_circ,args=(orbit.ee,orbit.jj,model))
  return orbit




def compute_frequencies(orbit,model):
  """for a given orbit, determine the frequencies using the centered rectangle technique



  """
  ap = 0.5*(orbit.r_apo + orbit.r_peri);
  am = 0.5*(orbit.r_apo - orbit.r_peri);
  sp = ap/(orbit.r_apo*orbit.r_peri);
  sm = am/(orbit.r_apo*orbit.r_peri);
  #
  accum0 = 0.0;
  accum1 = 0.0;
  accum2 = 0.0;
  #
  FRECS = 16
  #
  dt = np.pi/FRECS;
  #
  # the 'centered rectangle technique'
  #
  t = 0.5*(dt-np.pi)
  for i in range(0,FRECS):
      #
      r = ap + am*np.sin(t)
      #
      ur = model.potcurve(r)#model.potcurve[ (abs(r-model.rcurve)).argmin()]
      cost = np.cos(t)
      #t +=
      #
      tmp = np.sqrt(2.0*(orbit.ee-ur) - (orbit.jj*orbit.jj)/(r*r));
      accum0 += cost * tmp;
      accum1 += cost / tmp;
      s = sp + sm*np.sin(t);
      ur = model.potcurve(1.0/s)#model.potcurve[ (abs( (1.0/s)-model.rcurve)).argmin()]
      accum2 += cost/np.sqrt(2.0*(orbit.ee-ur) - (orbit.jj*orbit.jj*s*s));
      t += dt
  #
  #
  orbit.freq = np.zeros(3)
  orbit.action = np.zeros(3)
  #    
  orbit.freq[0] = np.pi/(am*accum1*dt);
  orbit.freq[1] = orbit.freq[0]*orbit.jj * sm*accum2*dt/np.pi;
  orbit.freq[2] = 0.0;
  #
  orbit.action[0] = am*accum0*dt/np.pi;
  orbit.action[1] = orbit.jj;
  orbit.action[2] = 0.0;
  return orbit



def locate(E,K,L1,L2,MM,OMEGA,model):
  """

  Args:
      E: input orbit energy
      K: input orbit kappa (normalized angular momentum)
      L1: radial frequency integer
      L2: azimuthal frequency integer
      MM: pattern frequency integer
      OMEGA: pattern frequency
      model: input spherical model, see class above
  Returns:
      the resonance value (i.e. 0 means the value is resonant)


  """

  # initialize an orbit class object
  O = orbit()

  # make an orbit in the model
  O = make_orbit(O,E,K,model)

  # compute frequencies for given orbit in the model
  O = compute_frequencies(O,model)

  # return the resonance value
  return O.freq[0]*L1 + O.freq[1]*L2 - OMEGA*MM;



def find_resonance(Rres,Tres,Pres,OMEGA,model,kappares=200):
    """search energy kappa space for a particular resonance

    Args:
        Rres: radial frequency integer
        Tres: azimuthal frequency integer
        Pres: pattern frequency integer
        Omega: pattern frequency
        model: input spherical model, see class above
        kappares: (200) the number of evenly-spaced kappa values to use to locate the resonance
    Returns:
        list of kappa values and list of energy values


    use brentq to find where the resonance equation is satisfied using locate (above)


    """
    # initialize kappa within known boundaries (0,1)
    krange = np.linspace(0.01,0.99,kappares)
    erange = np.zeros_like(krange)

    # for each kappa value, find the energy value which satisfies the resonance equation. if not available, skip.
    for index,value in enumerate(krange):
        try:
            erange[index] = brentq(locate,model.potcurve(np.min(model.rcurve)),model.potcurve(np.max(model.rcurve)),args=(value,Rres,Tres,Pres,OMEGA,model))
        except:
            pass

    # find energy and kappa ranges which satisfy the resonance
    gvals = np.where( (erange > model.potcurve(np.min(model.rcurve))) & (erange < model.potcurve(np.max(model.rcurve))) & (erange != 0.) )[0]
    
    return krange[gvals],erange[gvals]

