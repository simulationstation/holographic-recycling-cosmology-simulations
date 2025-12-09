"""Tests for cosmological and astrophysical constraints."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters
from hrc.constraints.bbn import check_bbn_constraint, BBNConstraint
from hrc.constraints.ppn import (
    check_ppn_constraints,
    check_G_dot_constraint,
    compute_ppn_gamma,
)
from hrc.constraints.stellar import check_stellar_constraints
from hrc.constraints.structure_growth import GrowthCalculator, GrowthSolution


class TestBBNConstraint:
    """Tests for BBN constraints on G_eff."""

    def test_no_variation(self):
        """Should pass if G_eff is constant."""
        result = check_bbn_constraint(
            G_eff_bbn=1.0,
            G_eff_today=1.0,
            params=HRCParameters(xi=0.0),
        )
        assert result.allowed
        assert_allclose(result.value, 0.0)

    def test_small_variation(self):
        """Should pass for small variations."""
        result = check_bbn_constraint(
            G_eff_bbn=1.05,
            G_eff_today=1.0,
            params=HRCParameters(),
        )
        # 5% variation should pass conservative bound
        assert result.allowed

    def test_large_variation(self):
        """Should fail for large variations."""
        result = check_bbn_constraint(
            G_eff_bbn=1.5,  # 50% variation
            G_eff_today=1.0,
            params=HRCParameters(),
        )
        assert not result.allowed

    def test_result_structure(self):
        """Result should have all required fields."""
        result = check_bbn_constraint(
            G_eff_bbn=1.0,
            G_eff_today=1.0,
            params=HRCParameters(),
        )
        assert isinstance(result, BBNConstraint)
        assert hasattr(result, "allowed")
        assert hasattr(result, "value")
        assert hasattr(result, "bound")
        assert hasattr(result, "z_bbn")


class TestPPNConstraints:
    """Tests for PPN (solar system) constraints."""

    def test_gr_limit(self):
        """GR (ξ=0) should satisfy all constraints."""
        params = HRCParameters(xi=0.0, phi_0=0.0)
        passed, results = check_ppn_constraints(
            phi_0=0.0, phi_dot_0=0.0, params=params
        )
        assert passed
        assert len(results) == 3  # G_dot, gamma, Nordtvedt

    def test_small_coupling(self):
        """Small coupling should satisfy constraints."""
        params = HRCParameters(xi=0.001, phi_0=0.1)
        passed, results = check_ppn_constraints(
            phi_0=0.1, phi_dot_0=0.0, params=params
        )
        assert passed

    def test_gamma_near_one(self):
        """PPN γ should be near 1 for small coupling."""
        params = HRCParameters(xi=0.01, phi_0=0.1)
        gamma = compute_ppn_gamma(0.1, params)
        assert_allclose(gamma, 1.0, atol=0.01)

    def test_gamma_deviation(self):
        """γ should deviate from 1 for larger coupling."""
        params = HRCParameters(xi=0.1, phi_0=0.3)
        gamma = compute_ppn_gamma(0.3, params)
        assert gamma != 1.0

    def test_g_dot_static_field(self):
        """Ġ/G should be zero for static field."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        result = check_G_dot_constraint(phi=0.2, phi_dot=0.0, params=params)
        assert result.allowed
        assert_allclose(result.value, 0.0)


class TestStellarConstraints:
    """Tests for stellar evolution constraints."""

    def test_requires_solution(self):
        """Should work with a background solution."""
        from hrc.background import BackgroundSolution
        import numpy as np

        n = 50
        solution = BackgroundSolution(
            z=np.linspace(0, 10, n),
            a=1.0 / (1 + np.linspace(0, 10, n)),
            H=np.sqrt(0.3 * (1 + np.linspace(0, 10, n))**3 + 0.7),
            phi=np.full(n, 0.2),
            phi_dot=np.zeros(n),
            G_eff_ratio=np.full(n, 1.1),  # Constant G_eff
            rho_m=0.3 * (1 + np.linspace(0, 10, n))**3,
            rho_r=np.full(n, 1e-4),
            rho_phi=np.full(n, 0.01),
            R=np.zeros(n),
            success=True,
        )

        passed, results = check_stellar_constraints(solution)
        # Constant G_eff should pass all stellar constraints
        assert passed
        assert len(results) == 3  # helio, WD, GC


class TestGrowthCalculator:
    """Tests for structure growth calculations."""

    @pytest.fixture
    def standard_params(self):
        return HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)

    def test_initialization(self, standard_params):
        """Calculator should initialize."""
        calc = GrowthCalculator(standard_params)
        assert calc.params == standard_params

    def test_solve(self, standard_params):
        """Should solve growth equation."""
        calc = GrowthCalculator(standard_params)
        solution = calc.solve(z_max=5, z_points=50)

        assert solution.success
        assert len(solution.z) == 50
        assert len(solution.D) == 50

    def test_growth_normalization(self, standard_params):
        """D(z=0) should be normalized to 1."""
        calc = GrowthCalculator(standard_params)
        solution = calc.solve(z_max=5, z_points=50)

        D_0 = solution.D_at(0.0)
        assert_allclose(D_0, 1.0, rtol=0.01)

    def test_growth_decreases_with_z(self, standard_params):
        """D(z) should decrease with z (looking back)."""
        calc = GrowthCalculator(standard_params)
        solution = calc.solve(z_max=5, z_points=50)

        # D(z=0) = 1 (normalized), D(z>0) < 1 (less structure in past)
        # Array is sorted from z_max to 0, so D[0] is at high z, D[-1] is at z=0
        assert solution.D[0] < solution.D[-1]  # D at high z < D at z=0

    def test_growth_rate_positive(self, standard_params):
        """Growth rate f should be positive."""
        calc = GrowthCalculator(standard_params)
        solution = calc.solve(z_max=2, z_points=50)

        assert np.all(solution.f >= 0)

    def test_sigma8_evolution(self, standard_params):
        """σ₈(z) should decrease with z."""
        calc = GrowthCalculator(standard_params, sigma8_0=0.8)
        solution = calc.solve(z_max=2, z_points=50)

        # σ₈ = σ₈(0) × D(z), so σ₈ at high z should be smaller
        # Array: sigma8[0] at high z, sigma8[-1] at z=0
        assert solution.sigma8[0] < solution.sigma8[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
