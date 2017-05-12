
#include <stdio.h>
#include <math.h>

#include "accumulate.h"

double r_to_xi(double r, int cmap, double scale)
{
  if ( cmap == 1 ) {
    
    if (r<0.0) {
      printf("radius < 0!");
      return 0.0;
	} else {
      return (r/scale-1.0)/(r/scale+1.0);
    }

  } else if ( cmap == 2 ) {
    if (r<=0.0) {
      printf("radius <= 0!");
      return 0.0;
    } else {
    return log(r);
    }
    
  } else {
    
    if (r<0.0) {
      printf("radius < 0!");
      return 0.0;
    } else {
    
    return r;
    }
  }
}


    
double xi_to_r(double xi, int cmap, double scale)
{
  if ( cmap == 1 ) {
    if (xi<-1.0) printf("xi < -1!");
    if (xi>=1.0) printf("xi >= 1!");

    // this is a dumb spot for this, but verify that C code can depend only upon itself inside the python wrapper (it can)
    //say_hello();

    return (1.0+xi)/(1.0 - xi) * scale;
    
     } else if (cmap==2) {
    return exp(xi);
    
  } else {
    return xi;
  }

}


double d_xi_to_r(double xi, int cmap, double scale)
{
  if ( cmap == 1 ) {
    if (xi<-1.0) printf("xi < -1!");
    if (xi>=1.0) printf("xi >= 1!");

    return 0.5*(1.0-xi)*(1.0-xi)/scale;
    
      } else if (cmap==2) {
    return exp(-xi);
    
  } else {
    return 1.0;
  }
}


double z_to_y(double z, double hscale)
{

  return z /( fabs(z)+1.e-10) * asinh( fabs(z/hscale));
      }

double y_to_z(double y, double hscale)
{
  return hscale*sinh(y);
      }





void say_hello(void)
{
  printf("Hello!\n");
    }


/*
// I am desirous to convert this to a C program

def return_bins(r,z,rmin=0,dR=0,zmin=0,dZ=0,numx=0,numy=0,ASCALE=0.01,HSCALE=0.001,CMAP=0):
    #
    # routine to return the integer bin numbers based on dimension mapping
    # 
    X = (r_to_xi(r,CMAP,ASCALE) - rmin)/dR
    Y = (z_to_y(z,hscale=HSCALE) - zmin)/dZ
    ix = int( np.floor((r_to_xi(r,CMAP,ASCALE) - rmin)/dR) )
    iy = int( np.floor((z_to_y(z,hscale=HSCALE) - zmin)/dZ) )
    #
    # check the boundaries and set guards
    if ix < 0:
        ix = 0
        X = 0
    if ix >= numx:
        ix = numx - 1
        X = numx - 1
    if iy < 0:
        iy = 0
        Y = 0
    if iy >= numy:
        iy = numy - 1
        Y = numy - 1
    return X,Y,ix,iy

*/

