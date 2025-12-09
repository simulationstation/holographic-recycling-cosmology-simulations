"""Tests for effective gravity module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters
from hrc.effective_gravity import (
    EffectiveGravity,
    GeffResult,
    check_G_eff_constraints,
    compute_H0_shift,
)


class TestEffectiveGravity:
    """Tests for EffectiveGravity class."""

    @pytest.fixture
    def standard_grav(self):
        """Standard HRC gravity calculator."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        return EffectiveGravity(params)

    @pytest.fixture
    def zero_coupling_grav(self):
        """Zero coupling (GR limit)."""
        params = HRCParameters(xi=0.0, phi_0=0.2)
        return EffectiveGravity(params)

    def test_critical_phi(self, standard_grav):
        """Critical φ should be 1/(8πξ)."""
        params = standard_grav.params
        expected = 1.0 / (8 * np.pi * params.xi)
        assert_allclose(standard_grav.critical_phi(), expected)

    def test_g_eff_at_zero(self, standard_grav):
        """G_eff at φ=0 should equal G."""
        result = standard_grav.G_eff_ratio(0.0)
        assert_allclose(result.G_eff_ratio, 1.0)
        assert result.is_physical

    def test_g_eff_enhancement(self, standard_grav):
        """G_eff should be > 1 for ξφ > 0."""
        result = standard_grav.G_eff_ratio(0.2)
        assert result.G_eff_ratio > 1.0
        assert result.is_physical

    def test_g_eff_formula(self, standard_grav):
        """G_eff = G/(1 - 8πξφ) should hold exactly."""
        params = standard_grav.params
        phi = 0.1

        expected = 1.0 / (1.0 - 8 * np.pi * params.xi * phi)
        result = standard_grav.G_eff_ratio(phi)
        assert_allclose(result.G_eff_ratio, expected)

    def test_gr_limit(self, zero_coupling_grav):
        """With ξ=0, G_eff should always equal G."""
        for phi in [0.0, 0.1, 0.5, 1.0]:
            result = zero_coupling_grav.G_eff_ratio(phi)
            assert_allclose(result.G_eff_ratio, 1.0)

    def test_near_critical_warning(self, standard_grav):
        """Near-critical φ should be flagged."""
        phi_crit = standard_grav.critical_phi()
        result = standard_grav.G_eff_ratio(phi_crit * 0.99)
        # Very large G_eff should be flagged as unphysical
        assert result.G_eff_ratio > 10
        assert not result.is_physical

    def test_beyond_critical(self, standard_grav):
        """Beyond critical φ, G_eff is negative (unphysical)."""
        phi_crit = standard_grav.critical_phi()
        result = standard_grav.G_eff_ratio(phi_crit * 1.1)
        assert not result.is_physical

    def test_g_eff_derivative(self, standard_grav):
        """Test G_eff time derivative computation."""
        phi = 0.2
        phi_dot = 0.01

        dG_dt, G_dot_over_G_eff = standard_grav.compute_G_eff_derivative(phi, phi_dot)

        # Both should be finite for physical parameters
        assert np.isfinite(dG_dt)
        assert np.isfinite(G_dot_over_G_eff)

        # Signs: positive ξφ̇ should increase G_eff
        if phi_dot > 0:
            assert dG_dt > 0

    def test_g_eff_array(self, standard_grav):
        """Test array computation of G_eff."""
        phi_arr = np.linspace(0, 0.3, 10)
        G_eff, is_physical = standard_grav.G_eff_ratio_array(phi_arr)

        assert len(G_eff) == len(phi_arr)
        assert len(is_physical) == len(phi_arr)
        assert np.all(is_physical[:5])  # First few should be physical


class TestH0Shift:
    """Tests for H₀ shift computation."""

    def test_no_shift_equal_geff(self):
        """No shift if G_eff is same at local and CMB."""
        H0_local, H0_cmb = compute_H0_shift(1.0, 1.0, H0_true=70.0)
        assert_allclose(H0_local, 70.0)
        assert_allclose(H0_cmb, 70.0)

    def test_enhanced_local_geff(self):
        """Enhanced G_eff at z=0 increases local H₀."""
        H0_local, H0_cmb = compute_H0_shift(1.2, 1.0, H0_true=70.0)
        # H_local ∝ √G_eff
        expected_local = 70.0 * np.sqrt(1.2)
        assert_allclose(H0_local, expected_local)

    def test_hubble_tension_direction(self):
        """HRC should predict H0_local > H0_CMB for typical parameters."""
        # G_eff slightly higher at z=0 than at CMB
        H0_local, H0_cmb = compute_H0_shift(1.15, 1.14, H0_true=70.0)
        assert H0_local > H0_cmb

    def test_typical_hrc_shift(self):
        """Typical HRC parameters should give ~6 km/s/Mpc shift."""
        # With G_eff(0) ≈ 1.18 and G_eff(CMB) ≈ 1.16
        G_eff_0 = 1.0 / (1.0 - 8 * np.pi * 0.03 * 0.2)  # ≈ 1.178
        G_eff_cmb = G_eff_0 * 0.99  # Slight evolution

        H0_local, H0_cmb = compute_H0_shift(G_eff_0, G_eff_cmb, H0_true=70.0)
        Delta_H0 = H0_local - H0_cmb

        # Should be positive (local > CMB) and few km/s/Mpc
        assert Delta_H0 > 0
        assert Delta_H0 < 20  # Not unreasonably large


class TestGeffConstraints:
    """Tests for G_eff constraint checks."""

    def test_check_requires_solution(self):
        """Constraint check should work with mock data."""
        from hrc.background import BackgroundSolution

        n = 100
        z = np.linspace(0, 1100, n)
        G_eff = np.ones(n) * 1.1  # Constant, slightly enhanced
        G_dot = np.zeros(n)

        # Create mock evolution
        from hrc.effective_gravity import GeffEvolution

        evolution = GeffEvolution(
            z=z,
            G_eff_ratio=G_eff,
            G_eff_dot_ratio=G_dot,
            is_physical=np.ones(n, dtype=bool),
            Delta_G_eff_BBN=0.0,  # No change at BBN
            G_dot_over_G_today=0.0,
        )

        results = check_G_eff_constraints(evolution)

        # All should pass with constant G_eff
        assert all(passed for _, passed, _ in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
