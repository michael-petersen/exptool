###########################################################################################
#
#  rk4.py
#     Simple Runga-Kutta 4-th order integrator
#
# 09 Jul 2021: Introduction
#
#
'''

rk4 (part of exptool.utils)
    Implementation of runga-kutta 4 for quick-and-dirty calculations
    Currently requires specialised versions of the potentials -- to be
    made more flexible?

it might also be fun to implement other integrators (including error estimates?)
http://www.math-cs.gordon.edu/courses/ma342/python/diffeq.py

and also some ODE-solving standards:
https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html

'''



def rk4(P,ri,vi,dt,n):
  """very simple runga-kutta 4 integrator
  
  http://spiff.rit.edu/richmond/nbody/OrbitRungeKutta4.pdf
  (with typos corrected)

  inputs
  ---------
  P   : (potential) potential instance, must have .get_cartesian_forces_array defined
  ri  : (3-array) initial positions
  vi  : (3-array) initial velocities
  dt  : (float) timestep
  n   : (int) number of steps


  """

  r,v = np.zeros([n,3]),np.zeros([n,3])
  r[0] = ri
  v[0] = vi

  for n in range(1,n):
    k1r = v[n-1]
    k1v = P.get_cartesian_forces_array(r[n-1])

    k2r = v[n-1] + k1v*dt*0.5
    k2v = P.get_cartesian_forces_array(r[n-1]+k1r*dt*0.5)

    k3r = v[n-1] + k2v*dt*0.5
    k3v = P.get_cartesian_forces_array(r[n-1]+k2r*dt*0.5)

    k4r = v[n-1] + k3v*dt
    k4v = P.get_cartesian_forces_array(r[n-1]+k3r*dt)

    v[n] = v[n-1] + (1/6)*dt*(k1v+2*k2v+2*k3v+k4v)
    r[n] = r[n-1] + (1/6)*dt*(k1r+2*k2r+2*k3r+k4r)
  
  return r.T,v.T


def add_rotation_acceleration(xarr,vxarr,omegab=1):
  """
  add the pseudoforces from a rotating frame
  
  always defined to be in the plane
  """
  e1 = np.array([0,0,1])
  x = xarr[0]
  y = xarr[1]
  vx = vxarr[0]
  vy = vxarr[1]

  z = 0
  vz = 0

  # Binney & Tremaine 3.117
  coriolis = 2*np.cross(omegab*e1,np.array([vx,vy,vz]))

  # take the derivative of the Phi_eff term, which in the planar case
  # is simply given by 0.5*omega^2*R^2
  # see equation 3.114
  centrifugal = omegab*omegab * np.array([x,y,z])

  #print(centrifugal,coriolis)
  return - centrifugal + coriolis



def rk4_rotation(P,ri,vi,dt,n,omegab=1):
  """very simple runga-kutta 4 integrator
  
  http://spiff.rit.edu/richmond/nbody/OrbitRungeKutta4.pdf
  (with typos corrected)

  inputs
  ---------
  P   : (potential) potential instance, must have .get_cartesian_forces_array defined
  ri  : (3-array) initial positions
  vi  : (3-array) initial velocities
  dt  : (float) timestep
  n   : (int) number of steps


  """

  r,v = np.zeros([n,3]),np.zeros([n,3])
  r[0] = ri
  v[0] = vi

  for n in range(1,n):
    k1r = v[n-1]
    k1v = P.get_cartesian_forces_array(r[n-1])
    k1v += add_rotation_acceleration(r[n-1],k1r,omegab=omegab)

    k2r = v[n-1] + k1v*dt*0.5
    k2v = P.get_cartesian_forces_array(r[n-1]+k1r*dt*0.5)
    k2v += add_rotation_acceleration(r[n-1]+k1r*dt*0.5,k2r,omegab=omegab)

    k3r = v[n-1] + k2v*dt*0.5
    k3v = P.get_cartesian_forces_array(r[n-1]+k2r*dt*0.5)
    k3v += add_rotation_acceleration(r[n-1]+k2r*dt*0.5,k3r,omegab=omegab)

    k4r = v[n-1] + k3v*dt
    k4v = P.get_cartesian_forces_array(r[n-1]+k3r*dt)
    k4v += add_rotation_acceleration(r[n-1]+k3r*dt,k4r,omegab=omegab)

    v[n] = v[n-1] + (1/6)*dt*(k1v+2*k2v+2*k3v+k4v)
    r[n] = r[n-1] + (1/6)*dt*(k1r+2*k2r+2*k3r+k4r)
  
  return r.T,v.T
