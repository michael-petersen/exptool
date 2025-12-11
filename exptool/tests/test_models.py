"""
Test suite for exptool.models using pytest.

This module contains unit tests for various analytical models including:
- MiyamotoNagai disc model
- NFW dark matter halo model
- Logarithmic potential model
- Hernquist spherical model
- Plummer spherical model
"""

import numpy as np
import pytest

from exptool.models.mndisc import MiyamotoNagai
from exptool.models.nfw import NFW
from exptool.models.plummer import Plummer
from exptool.models.hernquist import Hernquist
from exptool.models.logpot import LogPot


class TestMiyamotoNagai:
    """Test suite for MiyamotoNagai disc model."""
    
    @pytest.fixture
    def mn_model(self):
        """Create a MiyamotoNagai model instance for testing."""
        return MiyamotoNagai(a=1.0, b=0.2, M=1.0, G=1.0)
    
    def test_potential(self, mn_model):
        """Test gravitational potential calculation."""
        R, z = 2.0, 0.5
        potential = mn_model.potential(R, z)
        # Expected: -GM / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)
        # = -1.0 / sqrt(4.0 + (1.0 + sqrt(0.25 + 0.04))^2)
        # = -1.0 / sqrt(4.0 + (1.0 + 0.5385...)^2)
        expected = -1.0 / np.sqrt(4.0 + (1.0 + np.sqrt(0.29))**2)
        assert np.isclose(potential, expected, rtol=1e-10)
    
    def test_mass(self, mn_model):
        """Test spherical enclosed mass calculation."""
        R, z = 2.0, 0.5
        mass = mn_model.mass(R, z)
        rad = np.sqrt(R*R + z*z)
        expected = rad * (-mn_model.potential(R, z))
        assert np.isclose(mass, expected, rtol=1e-10)
    
    def test_density(self, mn_model):
        """Test density calculation."""
        R, z = 2.0, 0.5
        density = mn_model.density(R, z)
        assert density > 0
        assert np.isfinite(density)
    
    def test_zforce(self, mn_model):
        """Test vertical force calculation."""
        R, z = 2.0, 0.5
        zforce = mn_model.zforce(R, z)
        assert np.isfinite(zforce)
        # For z > 0, vertical force should be negative (pointing toward midplane)
        assert zforce < 0
    
    def test_rforce(self, mn_model):
        """Test radial force calculation."""
        R, z = 2.0, 0.5
        rforce = mn_model.rforce(R, z)
        assert np.isfinite(rforce)
        # For R > 0, radial force should be negative (pointing toward center)
        assert rforce < 0


class TestNFW:
    """Test suite for NFW dark matter halo model."""
    
    @pytest.fixture
    def nfw_model(self):
        """Create an NFW model instance for testing."""
        return NFW(rscl=1.0, G=1.0, Mvir=1.0, Rvir=1.0)
    
    def test_density(self, nfw_model):
        """Test density calculation."""
        r = 0.5
        density = nfw_model.get_dens(r)
        assert density > 0
        assert np.isfinite(density)
    
    def test_mass(self, nfw_model):
        """Test enclosed mass calculation."""
        r = 0.5
        mass = nfw_model.get_mass(r)
        assert mass > 0
        assert mass <= nfw_model.Mvir  # Enclosed mass should not exceed total mass
        assert np.isfinite(mass)
    
    def test_potential(self, nfw_model):
        """Test gravitational potential calculation."""
        r = 0.5
        potential = nfw_model.get_pot(r)
        assert potential < 0  # Potential should be negative
        assert np.isfinite(potential)
    
    def test_mass_monotonic(self, nfw_model):
        """Test that enclosed mass increases monotonically with radius."""
        r1, r2 = 0.5, 1.0
        mass1 = nfw_model.get_mass(r1)
        mass2 = nfw_model.get_mass(r2)
        assert mass2 > mass1


class TestLogPot:
    """Test suite for logarithmic potential model."""
    
    @pytest.fixture
    def logpot_model(self):
        """Create a LogPot model instance for testing."""
        return LogPot(rscl=1.0, q=0.8, v0=1.0)
    
    def test_potential(self, logpot_model):
        """Test gravitational potential calculation."""
        x, y = 1.0, 2.0
        potential = logpot_model.get_pot(x, y)
        # Expected: 0.5 * v0^2 * ln(rscl^2 + x^2 + y^2/q^2)
        expected = 0.5 * 1.0 * np.log(1.0 + 1.0 + 4.0 / 0.64)
        assert np.isclose(potential, expected, rtol=1e-10)
    
    def test_xforce(self, logpot_model):
        """Test x-component of force calculation."""
        x, y = 1.0, 2.0
        xforce = logpot_model.get_xforce(x, y)
        assert np.isfinite(xforce)
        # For x > 0, force should be negative (toward origin)
        assert xforce < 0
    
    def test_yforce(self, logpot_model):
        """Test y-component of force calculation."""
        x, y = 1.0, 2.0
        yforce = logpot_model.get_yforce(x, y)
        assert np.isfinite(yforce)
        # For y > 0, force should be negative (toward origin)
        assert yforce < 0
    
    def test_forces_at_origin(self, logpot_model):
        """Test that forces are zero at the origin."""
        xforce = logpot_model.get_xforce(0.0, 0.0)
        yforce = logpot_model.get_yforce(0.0, 0.0)
        assert np.isclose(xforce, 0.0, atol=1e-10)
        assert np.isclose(yforce, 0.0, atol=1e-10)


class TestHernquist:
    """Test suite for Hernquist spherical model."""
    
    @pytest.fixture
    def hernquist_model(self):
        """Create a Hernquist model instance for testing."""
        return Hernquist(rscl=1.0, G=1.0, M=1.0)
    
    def test_density(self, hernquist_model):
        """Test density calculation."""
        r = 0.5
        density = hernquist_model.get_dens(r)
        assert density > 0
        assert np.isfinite(density)
    
    def test_mass(self, hernquist_model):
        """Test enclosed mass calculation."""
        r = 0.5
        mass = hernquist_model.get_mass(r)
        assert mass > 0
        assert mass <= hernquist_model.M  # Enclosed mass should not exceed total mass
        assert np.isfinite(mass)
    
    def test_potential(self, hernquist_model):
        """Test gravitational potential calculation."""
        r = 0.5
        potential = hernquist_model.get_pot(r)
        assert potential < 0  # Potential should be negative
        assert np.isfinite(potential)
    
    def test_mass_at_infinity(self, hernquist_model):
        """Test that enclosed mass approaches total mass at large radius."""
        r = 1000.0
        mass = hernquist_model.get_mass(r)
        # At large radius, enclosed mass should be close to total mass
        assert np.isclose(mass, hernquist_model.M, rtol=0.01)


class TestPlummer:
    """Test suite for Plummer spherical model."""
    
    @pytest.fixture
    def plummer_model(self):
        """Create a Plummer model instance for testing."""
        return Plummer(rscl=0.1)
    
    def test_density(self, plummer_model):
        """Test density calculation."""
        r = 1.0
        density = plummer_model.get_dens(r)
        assert density > 0
        assert np.isfinite(density)
    
    def test_mass(self, plummer_model):
        """Test enclosed mass calculation."""
        r = 1.0
        mass = plummer_model.get_mass(r)
        assert mass > 0
        assert mass <= 1.0  # For normalized Plummer, mass approaches 1 at infinity
        assert np.isfinite(mass)
    
    def test_potential(self, plummer_model):
        """Test gravitational potential calculation."""
        r = 1.0
        potential = plummer_model.get_pot(r)
        assert potential < 0  # Potential should be negative
        assert np.isfinite(potential)
    
    def test_force(self, plummer_model):
        """Test gravitational force calculation."""
        r = 1.0
        force = plummer_model.get_force(r)
        assert force > 0  # Force magnitude should be positive
        assert np.isfinite(force)
    
    def test_cartesian_forces(self, plummer_model):
        """Test Cartesian force components."""
        x, y, z = 1.0, 0.5, 0.3
        fx, fy, fz = plummer_model.get_cartesian_forces(x, y, z)
        assert np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fz)
        # Forces should point outward from origin (positive for positive coordinates)
        # This is consistent with get_force returning force magnitude
        assert fx > 0 and fy > 0 and fz > 0
        # Verify force components are proportional to position
        r = np.sqrt(x*x + y*y + z*z)
        force_mag = plummer_model.get_force(r)
        assert np.isclose(fx, x * force_mag / r, rtol=1e-10)
        assert np.isclose(fy, y * force_mag / r, rtol=1e-10)
        assert np.isclose(fz, z * force_mag / r, rtol=1e-10)
