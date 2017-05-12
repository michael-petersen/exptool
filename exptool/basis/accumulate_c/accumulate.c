
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

      } else if (cmap==2) {
    return exp(xi);

    return (1.0+xi)/(1.0 - xi) * scale;
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

void say_hello(void)
{
  printf("Hello!\n");
    }

