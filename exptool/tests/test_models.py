
import numpy as np

from exptool.models.mndisc import MiyamotoNagai
from exptool.models.nfw import NFW
from exptool.models.plummer import Plummer
from exptool.models.hernquist import Hernquist
from exptool.models.logpot import LogPot


################################################################################

# Create a MiyamotoNagai model
mn_model = MiyamotoNagai(a=1.0, b=0.2, M=1.0, G=1.0)

# Test potential at (R, z)
R = 2.0
z = 0.5
potential = mn_model.potential(R, z)
print(f"Potential at ({R}, {z}): {potential}")

# Test mass enclosed at (R, z)
mass_enclosed = mn_model.mass(R, z)
print(f"Mass enclosed at ({R}, {z}): {mass_enclosed}")

# Test density at (R, z)
density = mn_model.density(R, z)
print(f"Density at ({R}, {z}): {density}")

# Test vertical force at (R, z)
zforce = mn_model.zforce(R, z)
print(f"Vertical force at ({R}, {z}): {zforce}")

# Test radial force at (R, z)
rforce = mn_model.rforce(R, z)
print(f"Radial force at ({R}, {z}): {rforce}")


################################################################################

# Create an NFW model
nfw_model = NFW(rscl=1.0, G=1.0, Mvir=1.0, Rvir=1.0)

# Test density at radius 'r'
r = 0.5
density = nfw_model.get_dens(r)
print(f"Density at r={r}: {density}")

# Test mass enclosed at radius 'r'
mass_enclosed = nfw_model.get_mass(r)
print(f"Mass enclosed at r={r}: {mass_enclosed}")

# Test gravitational potential at radius 'r'
potential = nfw_model.get_pot(r)
print(f"Gravitational potential at r={r}: {potential}")

################################################################################

# Create a LogPot model
logpot_model = LogPot(rscl=1.0, q=0.8, v0=1.0)

# Test potential at (x, y)
x = 1.0
y = 2.0
potential = logpot_model.get_pot(x, y)
print(f"Potential at (x={x}, y={y}): {potential}")

# Test x-force at (x, y)
xforce = logpot_model.get_xforce(x, y)
print(f"x-force at (x={x}, y={y}): {xforce}")

# Test y-force at (x, y)
yforce = logpot_model.get_yforce(x, y)
print(f"y-force at (x={x}, y={y}): {yforce}")

################################################################################

# Create a Hernquist model
hernquist_model = Hernquist(rscl=1.0, G=1.0, M=1.0)

# Test density at radius 'r'
r = 0.5
density = hernquist_model.get_dens(r)
print(f"Density at r={r}: {density}")

# Test mass enclosed at radius 'r'
mass_enclosed = hernquist_model.get_mass(r)
print(f"Mass enclosed at r={r}: {mass_enclosed}")

# Test gravitational potential at radius 'r'
potential = hernquist_model.get_pot(r)
print(f"Gravitational potential at r={r}: {potential}")

################################################################################

# Create a Plummer model
plummer_model = Plummer(rscl=0.1)

# Test density at radius 'r'
r = 1.0
density = plummer_model.get_dens(r)
print(f"Density at r={r}: {density}")

# Test mass enclosed at radius 'r'
mass_enclosed = plummer_model.get_mass(r)
print(f"Mass enclosed at r={r}: {mass_enclosed}")

# Test gravitational potential at radius 'r'
potential = plummer_model.get_pot(r)
print(f"Gravitational potential at r={r}: {potential}")

################################################################################
