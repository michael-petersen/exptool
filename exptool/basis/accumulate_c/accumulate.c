
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
    
