'''
resonancemodel.py: part of exptool
      basic resonance (or frequency) finding in potential models




'''

import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import brentq




class spherical_model(object):
    """class structure for defining spherical models"""
    rcurve = 0
    potcurve = 0
    dpotcurve = 0

class orbit(object):
    """class structure for defining orbits"""
    r_circ = 0.
    r_apo = 0.
    r_peri = 0.
    ee = 0.
    jj = 0.
    kappa = 0.



def find_j(r,kappa,model):
    """
    find angular momentum given kappa and a model
    """
    dudr = model.dpotcurve(r)#model.dpotcurve[ (abs(r-model.rcurve)).argmin()]
    jmax = np.sqrt(r*r*r*dudr);
    J = jmax*kappa;
    return J

def Ecirc(r,E,model):
    """
    #// Function to iteratively locate radius of circular orbit with energy EE
    """
    ur = model.potcurve(r)#model.potcurve[ (abs(r-model.rcurve)).argmin()]
    dudr = model.dpotcurve(r)#model.dpotcurve[ (abs(r-model.rcurve)).argmin()]
    return  E - 0.5*r*dudr - ur


def denom(r,E,J,model):
    """solve the denominator"""
    ur = model.potcurve(r)#model.potcurve[ (abs(r-model.rcurve)).argmin()]
    return 2.0*(E-ur)*r*r - J*J;


def make_orbit(orbit,E,K,model):
    """make an orbit"""
      orbit.ee = E
      #
      # this should work, the boundaries are in radius...
      orbit.r_circ = brentq(Ecirc,np.min(model.rcurve),np.max(model.rcurve),args=(orbit.ee,model))
      orbit.kappa = K
      orbit.jj = find_j(orbit.r_circ,orbit.kappa,model)
      orbit.r_apo = brentq(denom,orbit.r_circ,np.max(model.rcurve),args=(orbit.ee,orbit.jj,model))
      orbit.r_peri = brentq(denom,np.min(model.rcurve),orbit.r_circ,args=(orbit.ee,orbit.jj,model))
      return orbit




def compute_frequencies(orbit,model):
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




def make_sph_model_file(filename):
    #
    M = np.loadtxt(filename, dtype={'names': ('rcurve', 'potcurve', 'dpotcurve'),'formats': ('f4', 'f4', 'f4')},skiprows=1)
    #
    model = spherical_model()
    model.rcurve = M['rcurve']
    model.potcurve = M['potcurve']
    model.dpotcurve = M['dpotcurve']
    #
    model.rcurve = M['rcurve']
    model.potcurve = UnivariateSpline(model.rcurve,M['potcurve'],k=2)
    model.dpotcurve = UnivariateSpline(model.rcurve,M['dpotcurve'],k=2)
    #
    model.potcurve =  interp1d(model.rcurve,M['potcurve'],kind='cubic')
    model.dpotcurve =  interp1d(model.rcurve,M['dpotcurve'],kind='cubic')
    #
    model.potcurve =  interp1d(model.rcurve,M['potcurve'])
    model.dpotcurve =  interp1d(model.rcurve,M['dpotcurve'])
    #
    #
    return model



def make_sph_model(PotInstance,rads=np.linspace(0.,0.1,100)):
    model = spherical_model()
    PotInstance.compute_axis_potential(rvals=rads)
    #
    model.rcurve = PotInstance.rvals
    model.potcurve = interp1d(PotInstance.rvals,PotInstance.total_pot)
    model.dpotcurve = interp1d(PotInstance.rvals,PotInstance.total_dpdr)
    #
    return model



def find_resonance(Rres,Tres,Pres,OMEGA,model):
    krange = np.linspace(0.01,0.995,50)
    erange = np.zeros_like(krange)
    for index,value in enumerate(krange):
        # check the boundary values
        try:
            #print(brentq(locate,0.99*model.potcurve(np.min(model.rcurve)),model.potcurve(0.99*np.max(model.rcurve)),args=(kappa,0,2,2,patt,model)))
            erange[index] = brentq(locate,0.99*model.potcurve(np.min(model.rcurve)),model.potcurve(0.99*np.max(model.rcurve)),args=(value,Rres,Tres,Pres,OMEGA,model))
        except:
            pass
    #print(erange)
    gvals = np.where( (erange > model.potcurve(np.min(model.rcurve))) & (erange < model.potcurve(np.max(model.rcurve))) & (erange != 0.) )[0]
    return krange[gvals],erange[gvals]



def locate(E,K,L1,L2,MM,OMEGA,model):
  O = orbit()
  O = make_orbit(O,E,K,model)
  O = compute_frequencies(O,model)
  return O.freq[0]*L1 + O.freq[1]*L2 - OMEGA*MM;




def make_resonance_model(PotInstance,\
                         rads=np.linspace(0.,0.06,50),\
                         vels=np.linspace(0.0,1.5,50)):
    #
    #
    rr,vv = np.meshgrid(rads,vels)
    PotInstance.disk_use_m = 0
    PotInstance.halo_use_l = 0
    PotInstance.rotation_curve(rvals=rads)
    PotInstance.compute_axis_potential(rvals=rads)
    #
    model = make_sph_model(PotInstance,rads=rads)
    #
    tp = np.tile(model.potcurve(rads),(vels.size,1))
    tv = np.tile((rads*model.dpotcurve(rads))**0.5,(vels.size,1))
    ee = 0.5*vv*vv + tp
    kk = (vv)/(tv)
    #
    kk[kk>1.] = 1.
    kk[kk<-1.] = -1.
    #
    #
    model.rfunc = interpolate.interp2d(ee,kk,rr,kind='cubic')
    model.vfunc = interpolate.interp2d(ee,kk,vv,kind='cubic')
    return model
