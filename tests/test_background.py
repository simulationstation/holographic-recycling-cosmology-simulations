"""Tests for background cosmology module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters, PotentialConfig
from hrc.background import BackgroundCosmology, BackgroundSolution


class TestHRCParameters:
    """Tests for HRCParameters validation."""

    def test_default_parameters_valid(self):
        """Default parameters should be valid."""
        params = HRCParameters()
        valid, errors = params.validate()
        assert valid, f"Default params invalid: {errors}"

    def test_omega_m_property(self):
        """Omega_m should be sum of baryons and CDM."""
        params = HRCParameters(Omega_b=0.05, Omega_c=0.25)
        assert params.Omega_m == 0.30

    def test_omega_lambda_closure(self):
        """Omega_Lambda should close the universe."""
        params = HRCParameters(Omega_b=0.05, Omega_c=0.25, Omega_r=0.0, Omega_k=0.0)
        assert_allclose(params.Omega_Lambda, 0.70, rtol=1e-10)

    def test_h0_from_h(self):
        """H0 should be 100*h."""
        params = HRCParameters(h=0.7)
        assert params.H0 == 70.0

    def test_invalid_g_eff_divergence(self):
        """Should detect G_eff divergence condition."""
        # 8πξφ = 1 when ξ = 0.1, φ = 1/(8π*0.1) ≈ 0.398
        params = HRCParameters(xi=0.1, phi_0=0.5)  # Beyond critical
        valid, errors = params.validate()
        assert not valid
        assert any("diverge" in e.lower() for e in errors)

    def test_valid_hrc_parameters(self):
        """Standard HRC parameters should be valid."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        valid, errors = params.validate()
        assert valid, f"Standard HRC params invalid: {errors}"


class TestBackgroundCosmology:
    """Tests for BackgroundCosmology solver."""

    @pytest.fixture
    def standard_params(self):
        """Standard HRC parameters."""
        return HRCParameters(xi=0.03, phi_0=0.2, h=0.7)

    @pytest.fixture
    def lcdm_params(self):
        """ΛCDM-like parameters (no HRC modification)."""
        return HRCParameters(xi=0.0, phi_0=0.0, h=0.7)

    def test_initialization(self, standard_params):
        """BackgroundCosmology should initialize correctly."""
        cosmo = BackgroundCosmology(standard_params)
        assert cosmo.params == standard_params

    def test_g_eff_ratio_positive(self, standard_params):
        """G_eff/G should be positive for valid parameters."""
        cosmo = BackgroundCosmology(standard_params)
        ratio = cosmo.G_eff_ratio(standard_params.phi_0)
        assert ratio > 0

    def test_g_eff_ratio_enhancement(self, standard_params):
        """G_eff should be enhanced (>1) for positive ξ and φ."""
        cosmo = BackgroundCosmology(standard_params)
        ratio = cosmo.G_eff_ratio(standard_params.phi_0)
        # G_eff = G / (1 - 8πξφ), so for ξφ > 0, G_eff > G
        assert ratio > 1.0

    def test_g_eff_lcdm_limit(self, lcdm_params):
        """G_eff should equal G when ξ=0."""
        cosmo = BackgroundCosmology(lcdm_params)
        ratio = cosmo.G_eff_ratio(0.0)
        assert_allclose(ratio, 1.0, rtol=1e-10)

    def test_solve_success(self, standard_params):
        """Background evolution should solve successfully."""
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=10, z_points=50)
        assert solution.success

    def test_solve_z_range(self, standard_params):
        """Solution should cover requested z range."""
        z_max = 100
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=z_max, z_points=100)
        assert solution.z[0] == 0.0
        assert_allclose(solution.z[-1], z_max, rtol=0.01)

    def test_h_increases_with_z(self, standard_params):
        """H(z) should generally increase with z."""
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=10, z_points=50)
        # H(z) should increase from z=0 to z=10
        assert solution.H[-1] > solution.H[0]

    def test_phi_evolution(self, standard_params):
        """Scalar field should evolve from initial condition."""
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=10, z_points=50)
        # φ(z=0) should match initial condition
        assert_allclose(solution.phi[0], standard_params.phi_0, rtol=0.1)

    def test_interpolation(self, standard_params):
        """Interpolation methods should work."""
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=10, z_points=50)

        # Test H interpolation
        H_mid = solution.H_at(5.0)
        assert H_mid > 0

        # Test phi interpolation
        phi_mid = solution.phi_at(5.0)
        assert np.isfinite(phi_mid)

        # Test G_eff interpolation
        G_eff_mid = solution.G_eff_at(5.0)
        assert G_eff_mid > 0


class TestBackgroundSolution:
    """Tests for BackgroundSolution data structure."""

    @pytest.fixture
    def sample_solution(self):
        """Create a sample solution."""
        n = 10
        return BackgroundSolution(
            z=np.linspace(0, 10, n),
            a=1.0 / (1 + np.linspace(0, 10, n)),
            H=np.sqrt(0.3 * (1 + np.linspace(0, 10, n))**3 + 0.7),
            phi=np.full(n, 0.2),
            phi_dot=np.zeros(n),
            G_eff_ratio=np.full(n, 1.1),
            rho_m=0.3 * (1 + np.linspace(0, 10, n))**3,
            rho_r=np.full(n, 1e-4),
            rho_phi=np.full(n, 0.01),
            R=np.zeros(n),
            success=True,
        )

    def test_solution_shapes(self, sample_solution):
        """All arrays should have same length."""
        n = len(sample_solution.z)
        assert len(sample_solution.a) == n
        assert len(sample_solution.H) == n
        assert len(sample_solution.phi) == n

    def test_scale_factor_relation(self, sample_solution):
        """a = 1/(1+z) should hold."""
        expected_a = 1.0 / (1 + sample_solution.z)
        assert_allclose(sample_solution.a, expected_a, rtol=1e-10)


class TestPotentialConfig:
    """Tests for scalar field potential configuration."""

    def test_quadratic_potential(self):
        """Quadratic potential V = V₀ + ½m²φ²."""
        pot = PotentialConfig(form="quadratic", V0=0.1, m=1.0)

        V = pot.V(0.5)
        expected = 0.1 + 0.5 * 1.0**2 * 0.5**2
        assert_allclose(V, expected)

        dV = pot.dV(0.5)
        expected_dV = 1.0**2 * 0.5
        assert_allclose(dV, expected_dV)

    def test_quartic_potential(self):
        """Quartic potential V = V₀ + ½m²φ² + ¼λφ⁴."""
        pot = PotentialConfig(form="quartic", V0=0.0, m=1.0, lambda_4=0.1)

        V = pot.V(1.0)
        expected = 0.5 * 1.0 + 0.25 * 0.1 * 1.0
        assert_allclose(V, expected)

    def test_exponential_potential(self):
        """Exponential potential V = V₀ exp(-αφ)."""
        pot = PotentialConfig(form="exponential", V0=1.0, alpha_exp=2.0)

        V = pot.V(0.5)
        expected = 1.0 * np.exp(-2.0 * 0.5)
        assert_allclose(V, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
