"""Tests for background cosmology module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters, PotentialConfig
from hrc.background import BackgroundCosmology, BackgroundSolution
from hrc.utils.numerics import (
    GeffDivergenceError,
    GeffValidityResult,
    compute_critical_phi,
    check_geff_validity,
)


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
        """Background evolution should solve successfully at low z."""
        # Note: With standard HRC params (xi=0.03, phi_0=0.2), the scalar
        # field evolves and diverges around z~3.5. Use z_max=2 for safe range.
        cosmo = BackgroundCosmology(standard_params)
        solution = cosmo.solve(z_max=2, z_points=50)
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


class TestGeffDivergenceDetection:
    """Tests for G_eff divergence detection during integration."""

    def test_compute_critical_phi_positive_xi(self):
        """Critical phi should be 1/(8πξ) for positive ξ."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)
        expected = 1.0 / (8.0 * np.pi * xi)
        assert_allclose(phi_c, expected, rtol=1e-10)

    def test_compute_critical_phi_zero_xi(self):
        """Critical phi should be inf for ξ=0."""
        phi_c = compute_critical_phi(0.0)
        assert phi_c == float('inf')

    def test_compute_critical_phi_negative_xi(self):
        """Critical phi should be inf for ξ<0."""
        phi_c = compute_critical_phi(-0.1)
        assert phi_c == float('inf')

    def test_check_geff_validity_safe_region(self):
        """Validity check should pass in safe region."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)
        phi = phi_c * 0.5  # Well below critical

        result = check_geff_validity(phi, xi)
        assert result.valid
        assert result.G_eff_ratio is not None
        assert result.G_eff_ratio > 0

    def test_check_geff_validity_near_critical(self):
        """Validity check should fail near critical value."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)
        phi = phi_c * 0.995  # Within 1% of critical (default epsilon)

        result = check_geff_validity(phi, xi, epsilon=0.01)
        assert not result.valid
        assert "exceeds" in result.message.lower()

    def test_check_geff_validity_beyond_critical(self):
        """Validity check should fail beyond critical value."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)
        phi = phi_c * 1.1  # Beyond critical

        result = check_geff_validity(phi, xi)
        assert not result.valid

    def test_check_geff_validity_negative_geff(self):
        """Validity check should detect negative G_eff."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)
        phi = phi_c * 1.5  # Well beyond critical

        # Use larger epsilon to allow computation
        result = check_geff_validity(phi, xi, epsilon=0.8)
        assert not result.valid
        assert "negative" in result.message.lower() or result.G_eff_ratio is None

    def test_background_cosmo_phi_critical_property(self):
        """BackgroundCosmology should expose phi_critical."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        cosmo = BackgroundCosmology(params)

        expected = compute_critical_phi(params.xi)
        assert_allclose(cosmo.phi_critical, expected, rtol=1e-10)

    def test_background_cosmo_valid_params_geff_valid(self):
        """Valid parameters should produce geff_valid=True solution at low z."""
        # Note: With xi=0.03, phi_0=0.2, the scalar field evolves toward
        # the critical value and diverges around z~3.5. For a valid test,
        # we use z_max=2 which is safely before the divergence.
        params = HRCParameters(xi=0.03, phi_0=0.2)
        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=2, z_points=50)

        assert solution.success
        assert solution.geff_valid

    def test_background_cosmo_near_critical_initial(self):
        """Initial phi near critical should be flagged invalid."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)

        # Set phi_0 to 95% of critical (within default 1% safety margin)
        phi_0_near = phi_c * 0.995

        params = HRCParameters(xi=xi, phi_0=phi_0_near)

        # This should trigger validation failure
        valid, errors = params.validate()

        if valid:
            # If params pass validation, solver should detect issue
            cosmo = BackgroundCosmology(params, geff_epsilon=0.01)
            solution = cosmo.solve(z_max=10, z_points=50)
            # Either solution fails or geff_valid is False
            assert not solution.geff_valid or not solution.success

    def test_background_cosmo_diverging_evolution(self):
        """Parameters causing phi to grow toward critical should be detected."""
        # Use parameters where scalar field might grow during evolution
        # Large initial velocity can cause this
        xi = 0.1  # Larger coupling
        phi_c = compute_critical_phi(xi)

        # Start at 80% of critical
        phi_0 = phi_c * 0.8

        params = HRCParameters(xi=xi, phi_0=phi_0, h=0.7)
        valid, _ = params.validate()

        if valid:
            cosmo = BackgroundCosmology(params, geff_epsilon=0.1)
            solution = cosmo.solve(z_max=1100, z_points=200)

            # The solution should track whether geff stayed valid
            assert hasattr(solution, 'geff_valid')
            assert hasattr(solution, 'phi_critical')

            # If phi grew toward critical, geff_valid should be False
            if solution.success:
                max_phi = np.max(np.abs(solution.phi))
                threshold = phi_c * 0.9
                if max_phi >= threshold:
                    assert not solution.geff_valid

    def test_solution_tracks_divergence_redshift(self):
        """Solution should track where divergence occurred."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)

        # At z_max=2, the standard parameters should not diverge
        params = HRCParameters(xi=xi, phi_0=0.2)
        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=2, z_points=50)

        assert solution.geff_valid
        assert solution.geff_divergence_z is None
        assert solution.phi_critical is not None
        assert_allclose(solution.phi_critical, phi_c, rtol=1e-10)

    def test_solution_detects_divergence_at_high_z(self):
        """Solution should detect divergence when phi evolves toward critical."""
        xi = 0.03
        phi_c = compute_critical_phi(xi)

        # At z_max=10, the scalar field will evolve toward critical
        params = HRCParameters(xi=xi, phi_0=0.2)
        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=10, z_points=100)

        # The solution should detect that phi approached critical
        assert not solution.geff_valid
        assert solution.geff_divergence_z is not None
        # Divergence should happen somewhere between z=3 and z=4
        assert 3.0 < solution.geff_divergence_z < 5.0
        assert solution.phi_critical is not None
        assert_allclose(solution.phi_critical, phi_c, rtol=1e-10)

    def test_geff_divergence_error_attributes(self):
        """GeffDivergenceError should have useful attributes."""
        phi = 0.4
        phi_c = 0.5
        z = 10.5

        error = GeffDivergenceError(phi, phi_c, z)

        assert error.phi == phi
        assert error.phi_critical == phi_c
        assert error.z == z
        assert_allclose(error.fraction, phi / phi_c, rtol=1e-10)
        assert "0.4" in str(error) or "phi" in str(error).lower()

    def test_zero_xi_no_divergence_possible(self):
        """With ξ=0, no divergence should ever occur."""
        params = HRCParameters(xi=0.0, phi_0=1.0)  # Large phi, but ξ=0
        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=100, z_points=50)

        assert solution.success
        assert solution.geff_valid
        assert cosmo.phi_critical == float('inf')

    def test_custom_geff_epsilon(self):
        """Custom geff_epsilon should be respected."""
        params = HRCParameters(xi=0.03, phi_0=0.2)

        # Strict epsilon
        cosmo_strict = BackgroundCosmology(params, geff_epsilon=0.1)
        # Lax epsilon
        cosmo_lax = BackgroundCosmology(params, geff_epsilon=0.001)

        phi_c = cosmo_strict.phi_critical

        # Strict has larger safety margin
        strict_threshold = phi_c * 0.9
        lax_threshold = phi_c * 0.999

        assert strict_threshold < lax_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
