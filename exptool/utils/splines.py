"""
splines.py

ports of direct Numerical Recipes spline implementations; for
compatibility and verification more than actual use.


"""


import numpy as np


def Spline(x, y, yp1, ypn):
    """return spline coefficients
    
    inputs
    --------
    
    
    
    returns
    --------
    y2   : spline coefficients
    
    exputil/Spline.cc
    
    """
    i1 = 0
    i2 = x.size - 1
    u = np.zeros(x.size)#Vector(i1, i2-1);
    y2 = np.zeros(x.size)

    #/*     Boundary conditions obtained by fixing third derivative as computed
    #   by divided differences */
    if (yp1 < -0.99e30):
        y2[i1+0]=1.0;
        d2 = ((y[i1+3]-y[i1+2])/(x[i1+3]-x[i1+2]) - (y[i1+2]-y[i1+1])/(x[i1+2]-x[i1+1]))/(x[i1+3]-x[i1+1]);
        d1 = ((y[i1+2]-y[i1+1])/(x[i1+2]-x[i1+1]) - (y[i1+1]-y[i1+0])/(x[i1+1]-x[i1+0]))/(x[i1+2]-x[i1+0]);
        u[i1+0] = -6.0*(d2-d1)*(x[i1+1]-x[i1+0])/(x[i1+3]-x[i1+0]);

    #/*     "Normal" zero second derivative boundary conditions */
    elif (yp1 > 0.99e30):
        y2[i1+0]=u[i1+0]=0.0;

    #/*      Known first derivative */
    else:
        y2[i1+0] = -0.5;
        u[i1+0]=(3.0/(x[i1+1]-x[i1+0]))*((y[i1+1]-y[i1+0])/(x[i1+1]-x[i1+0])-yp1);
        
        
    for i in range(i1+1,i2):#(i=i1+1;i<i2;i++) {
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
        p=sig*y2[i-1]+2.0;
        y2[i]=(sig-1.0)/p;
        u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
        u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;


    #/*     Boundary conditions obtained by fixing third derivative as computed
    # by divided differences */
    
    if (ypn < -0.99e30):
        d2 = ((y[i2]-y[i2-1])/(x[i2]-x[i2-1]) - (y[i2-1]-y[i2-2])/(x[i2-1]-x[i2-2]))/(x[i2]-x[i2-2]);
        d1 = ((y[i2-1]-y[i2-2])/(x[i2-1]-x[i2-2]) - (y[i2-2]-y[i2-3])/(x[i2-2]-x[i2-3]))/(x[i2-1]-x[i2-3]);
        qn = -1.0;
        un = 6.0*(d2-d1)*(x[i2]-x[i2-1])/(x[i2]-x[i2-3]);
        
    #/*     "Normal" zero second derivative boundary conditions */
    elif (ypn > 0.99e30):
        qn=un=0.0;
    #/*      Known first derivative */
    else:
        qn=0.5;
        un=(3.0/(x[i2]-x[i2-1]))*(ypn-(y[i2]-y[i2-1])/(x[i2]-x[i2-1]));

    y2[i2]=(un-qn*u[i2-1])/(qn*y2[i2-1]+1.0);
  
    for k in range(i2-1,i1-1,-1):#(k=i2-1;k>=i1;k--)
        y2[k]=y2[k]*y2[k+1]+u[k];

    return y2

#d0spline = Spline(rarrhalo, d0halo, 0.0, 0.0)
#p0spline = Spline(rarrhalo, p0halo, -1.0e30, -1.0e30)
#Spline(x, y, yp1, ypn)

def Splint1(xa, ya, y2a, x, even=0):
    """
    
    
    
    exputil/SplintE.cc
    
    """
    n1 = 0#.getlow();
    n2 = xa.size-1#.gethigh();

    if (even):
        klo=(( (x-xa[n1])/(xa[n2]-xa[n1])*(n2-n1) ) + n1).astype('int')
        #klo=klo<n1 ? n1 : klo;
        #klo=klo<n2 ? klo : n2-1;
        khi=klo+1;
    else:
        klo = n1;
        khi = n2;
        while (khi-klo > 1):
          k = (khi+klo) >> 1;
          if (xa[k] > x): khi = k;
          else: klo = k;

    h=xa[khi]-xa[klo];
  
    a=(xa[khi]-x)/h;
    b=(x-xa[klo])/h;
    y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
    
    return y
    
    
#Splint1(rarrhalo, d0halo, d0spline, 0.01)
#Splint1(rarrhalo, p0halo, p0spline, 0.01)



