#include "accumulate.h"

double r_to_xi(double r, int cmap, double scale)
{
  if (cmap) {
    //if (r<0.0) cout << "radius < 0!";
    return (r/scale-1.0)/(r/scale+1.0);
  } else {
    // if (r<0.0) cout << "radius < 0!";
    return r;
  }
}
    
