"""
surface

simply generate points on the surface of a sphere
can be used to test simple prolate/oblate models

04-Mar-2021: introduction


"""
import numpy as np

def gen_sphere_surface(npoints,cartesian=True):
    """generate evenly spaced points on the surface of a sphere
    
    inputs
    -------------
    npoints    : integer
        the maximum number of points to generate on the surface of the sphere
    cartesian  : boolean, default True
        if true, return (x,y,z). If false, return (phi,theta)
    
    note that npoints is the maximum, because we are enforcing
    equal spacing: not all npoints values return even spacing.
    
    returns
    -------------
    x,y,z      : arrays, npoints long
        cartesian location of points on the unit sphere
    phi,theta  : arrays, npoints long
        surface of sphere points in radians
    
    
    """
    npoint = 0
    points = np.zeros([npoints,3])
    sphpoints = np.zeros([npoints,2])
    
    a = 4*np.pi/npoints
    d = np.sqrt(a)
    Mtheta = np.floor(np.pi/d).astype('int')
    dtheta = np.pi/Mtheta
    dphi   = a/dtheta
    
    for m in range(0,Mtheta):
        
        theta = np.pi*(m+0.5)/Mtheta
        Mphi  = np.floor(2*np.pi*np.sin(theta)/dphi).astype('int')
        
        for n in range(0,Mphi):
            phi = 2*np.pi*n/Mphi
            points[npoint] = np.array([np.sin(theta)*np.cos(phi),
                                       np.sin(theta)*np.sin(phi),
                                       np.cos(theta)])
            sphpoints[npoint] = np.array([phi,theta])
            npoint += 1
            
    if cartesian:
        return points[:npoint,0],points[:npoint,1],points[:npoint,2]
    else:
        return sphpoints[:npoint,0],sphpoints[:npoint,1]
            


