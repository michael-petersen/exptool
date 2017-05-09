
#include <stdio.h>

#include "accumulate.h"

double r_to_xi(double r, int cmap, double scale)
{
  if (cmap) {
    
    if (r<0.0) {
      printf("radius < 0!");
      return 0.0;
	} else {
      return (r/scale-1.0)/(r/scale+1.0);
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
  if (cmap) {
    if (xi<-1.0) printf("xi < -1!");
    if (xi>=1.0) printf("xi >= 1!");

    return (1.0+xi)/(1.0 - xi) * scale;
  } else {
    return xi;
  }

}


double d_xi_to_r(double xi, int cmap, double scale)
{
  if (cmap) {
    if (xi<-1.0) printf("xi < -1!");
    if (xi>=1.0) printf("xi >= 1!");

    return 0.5*(1.0-xi)*(1.0-xi)/scale;
  } else {
    return 1.0;
  }
}


