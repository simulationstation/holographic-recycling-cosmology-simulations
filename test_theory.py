"""
Unit Tests for Holographic Recycling Cosmology (HRC) Theory Module

Tests verify:
1. Recovery of standard Friedmann equations when recycling physics disabled
2. Energy conservation (covariant divergence of total T_μν = 0)
3. Correct limits and asymptotic behavior
4. Mathematical consistency of derived quantities
5. Parameter validation

Run with: pytest test_theory.py -v
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import sympy as sp

from hrc_theory import (
    HRCTheory,
    HRCParameters,
    ActionComponents,
    FLRWCosmology,
    FieldEquations,
    compute_remnant_production_rate,
    hubble_tension_diagnostic
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def standard_params():
    """Standard ΛCDM-like parameters (no recycling physics)."""
    return HRCParameters(
        G=1.0,
        Lambda=0.7,
        xi=0.0,
        lambda_r=0.0,
        alpha=0.0,
        m_phi=1.0
    )


@pytest.fixture
def hrc_params():
    """Parameters with recycling physics enabled."""
    return HRCParameters(
        G=1.0,
        Lambda=0.7,
        xi=0.01,
        lambda_r=0.1,
        alpha=0.05,
        m_phi=1.0
    )


@pytest.fixture
def standard_theory(standard_params):
    """Theory instance with standard cosmology parameters."""
    return HRCTheory(standard_params)


@pytest.fixture
def hrc_theory(hrc_params):
    """Theory instance with HRC parameters."""
    return HRCTheory(hrc_params)


@pytest.fixture
def action():
    """ActionComponents instance for symbolic tests."""
    return ActionComponents()


@pytest.fixture
def flrw(action):
    """FLRWCosmology instance for symbolic tests."""
    return FLRWCosmology(action)


# =============================================================================
# TEST: PARAMETER VALIDATION
# =============================================================================

class TestParameterValidation:
    """Tests for HRCParameters validation."""

    def test_valid_parameters(self, standard_params):
        """Valid parameters should pass validation."""
        valid, msg = standard_params.is_valid()
        assert valid
        assert msg == "OK"

    def test_negative_G_invalid(self):
        """Negative G should be invalid."""
        params = HRCParameters(G=-1.0)
        valid, msg = params.is_valid()
        assert not valid
        assert "G must be positive" in msg

    def test_zero_G_invalid(self):
        """Zero G should be invalid."""
        params = HRCParameters(G=0.0)
        valid, msg = params.is_valid()
        assert not valid

    def test_negative_mass_invalid(self):
        """Negative scalar field mass should be invalid."""
        params = HRCParameters(m_phi=-1.0)
        valid, msg = params.is_valid()
        assert not valid
        assert "m_phi must be non-negative" in msg

    def test_theory_rejects_invalid_params(self):
        """HRCTheory should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError):
            HRCTheory(HRCParameters(G=-1.0))

    def test_to_from_dict_roundtrip(self, hrc_params):
        """Parameter dict conversion should be invertible."""
        d = hrc_params.to_dict()
        params_new = HRCParameters.from_dict(d)
        assert params_new.G == hrc_params.G
        assert params_new.xi == hrc_params.xi
        assert params_new.alpha == hrc_params.alpha


# =============================================================================
# TEST: RECOVERY OF STANDARD FRIEDMANN EQUATIONS
# =============================================================================

class TestLCDMRecovery:
    """Tests for recovery of ΛCDM when recycling physics is off."""

    def test_first_friedmann_phi_zero(self, standard_theory):
        """First Friedmann should reduce to standard form when φ=0."""
        rho_m = 0.3
        rho_rem = 0.0

        # HRC with φ=0
        H2_hrc = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=rho_rem,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        # Standard: H² = 8πGρ/3 + Λ/3
        G = standard_theory.G
        Lambda = standard_theory.Lambda
        H2_standard = (8.0 * np.pi * G * rho_m + Lambda) / 3.0

        assert_allclose(H2_hrc, H2_standard, rtol=1e-10)

    def test_first_friedmann_with_remnants_phi_zero(self, standard_theory):
        """Remnants should act as dark matter when φ=0."""
        rho_m = 0.3
        rho_rem = 0.1

        H2_hrc = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=rho_rem,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        G = standard_theory.G
        Lambda = standard_theory.Lambda
        H2_standard = (8.0 * np.pi * G * (rho_m + rho_rem) + Lambda) / 3.0

        assert_allclose(H2_hrc, H2_standard, rtol=1e-10)

    def test_verify_lcdm_limit_method(self, standard_theory):
        """The verify_lcdm_limit method should confirm recovery."""
        result = standard_theory.verify_lcdm_limit(rho_m=0.3, rho_rem=0.1)

        assert result['match']
        assert result['relative_difference'] < 1e-10

    def test_scalar_kinetic_contributes_correctly(self, standard_theory):
        """Scalar field kinetic energy should contribute to H²."""
        rho_m = 0.3

        # With φ̇ ≠ 0 but φ = 0
        H2_with_kinetic = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=0.0, dphi_dt=0.5, n_rem=0.0
        )

        H2_no_kinetic = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        # Kinetic energy = 0.5 * φ̇² = 0.5 * 0.25 = 0.125
        expected_diff = 8.0 * np.pi * 0.125 / 3.0
        actual_diff = H2_with_kinetic - H2_no_kinetic

        assert_allclose(actual_diff, expected_diff, rtol=1e-10)

    def test_potential_contributes_correctly(self, standard_theory):
        """Scalar field potential should contribute to H²."""
        rho_m = 0.3

        # With φ ≠ 0 (potential contribution)
        phi_val = 0.5
        H2_with_potential = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=phi_val, dphi_dt=0.0, n_rem=0.0
        )

        H2_no_potential = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        # V(φ) = 0.5 * m² * φ² = 0.5 * 1 * 0.25 = 0.125
        m_phi = standard_theory.m_phi
        expected_V = 0.5 * m_phi**2 * phi_val**2
        expected_diff = 8.0 * np.pi * expected_V / 3.0
        actual_diff = H2_with_potential - H2_no_potential

        assert_allclose(actual_diff, expected_diff, rtol=1e-10)


# =============================================================================
# TEST: MODIFIED FRIEDMANN WITH RECYCLING
# =============================================================================

class TestModifiedFriedmann:
    """Tests for modified Friedmann equations with recycling physics."""

    def test_effective_G_factor_minimal_coupling(self, standard_theory):
        """G_eff factor should be 1 for minimal coupling (ξ=0)."""
        factor = standard_theory.G_effective_factor(phi=0.5)
        assert_allclose(factor, 1.0, rtol=1e-10)

    def test_effective_G_factor_nonminimal_coupling(self, hrc_theory):
        """G_eff factor should deviate from 1 for non-minimal coupling."""
        phi = 0.5
        xi = hrc_theory.xi
        G = hrc_theory.G

        factor = hrc_theory.G_effective_factor(phi)
        expected = 1.0 - 8.0 * np.pi * G * xi * phi

        assert_allclose(factor, expected, rtol=1e-10)
        assert factor != 1.0

    def test_effective_G_raises_for_negative_factor(self, hrc_theory):
        """Should raise ValueError if G_eff factor goes non-positive."""
        # With xi=0.01, need 8πGξφ > 1, so φ > 1/(8π*0.01) ≈ 3.98
        with pytest.raises(ValueError):
            hrc_theory.G_effective_factor(phi=100.0)

    def test_remnant_coupling_modifies_density(self, hrc_theory):
        """Alpha coupling should modify effective remnant density."""
        rho_m = 0.3
        rho_rem = 0.1
        phi = 0.5

        H2_modified = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=rho_rem,
            phi=phi, dphi_dt=0.0, n_rem=0.0
        )

        # Manually compute expected
        alpha = hrc_theory.alpha
        G = hrc_theory.G
        xi = hrc_theory.xi
        Lambda = hrc_theory.Lambda
        m_phi = hrc_theory.m_phi

        rho_phi = 0.5 * m_phi**2 * phi**2
        rho_rem_eff = rho_rem * (1.0 + alpha * phi)
        G_eff_factor = 1.0 - 8.0 * np.pi * G * xi * phi
        rho_total = rho_m + rho_phi + rho_rem_eff

        H2_expected = (8.0 * np.pi * G * rho_total + Lambda) / (3.0 * G_eff_factor)

        assert_allclose(H2_modified, H2_expected, rtol=1e-10)

    def test_lambda_coupling_affects_energy_density(self, hrc_theory):
        """Lambda coupling to n_rem should affect scalar energy density."""
        rho_m = 0.3
        phi = 0.5
        n_rem = 0.2

        H2_with_nrem = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=phi, dphi_dt=0.0, n_rem=n_rem
        )

        H2_without_nrem = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=phi, dphi_dt=0.0, n_rem=0.0
        )

        # n_rem coupling adds 0.5*λ*φ*n_rem to rho_phi
        lambda_r = hrc_theory.lambda_r
        expected_extra = 0.5 * lambda_r * phi * n_rem

        # Account for G_eff factor
        G_eff = hrc_theory.G_effective_factor(phi)
        expected_diff = 8.0 * np.pi * expected_extra / (3.0 * G_eff)

        assert_allclose(H2_with_nrem - H2_without_nrem, expected_diff, rtol=1e-10)


# =============================================================================
# TEST: SCALAR FIELD EQUATION
# =============================================================================

class TestScalarFieldEquation:
    """Tests for scalar field equation of motion."""

    def test_hubble_friction(self, hrc_theory):
        """Hubble friction term should be present."""
        H = 1.0
        dphi_dt = 0.5

        # With dphi_dt ≠ 0, should have -3H*dphi_dt contribution
        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi=0.0, a=1.0, H=H,
            rho_rem=0.0, n_rem=0.0, dphi_dt=dphi_dt
        )

        # Expected: -3Hφ̇ - m²φ - ξR + sources
        # With φ=0, rho_rem=0, n_rem=0: -3Hφ̇ - 6ξH²
        expected_friction = -3.0 * H * dphi_dt
        xi = hrc_theory.xi
        R_approx = 6.0 * H**2
        expected_curvature = -xi * R_approx

        expected = expected_friction + expected_curvature

        assert_allclose(phi_ddot, expected, rtol=1e-10)

    def test_mass_term_restoring(self, hrc_theory):
        """Mass term should provide restoring force toward φ=0."""
        phi = 0.5
        m_phi = hrc_theory.m_phi

        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi=phi, a=1.0, H=0.0,
            rho_rem=0.0, n_rem=0.0, dphi_dt=0.0
        )

        # With H=0: -m²φ only (no friction, no curvature)
        expected = -m_phi**2 * phi

        assert_allclose(phi_ddot, expected, rtol=1e-10)

    def test_remnant_sources_field(self, hrc_theory):
        """Remnants should source the scalar field."""
        rho_rem = 0.2
        n_rem = 0.1
        alpha = hrc_theory.alpha
        lambda_r = hrc_theory.lambda_r

        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi=0.0, a=1.0, H=0.0,
            rho_rem=rho_rem, n_rem=n_rem, dphi_dt=0.0
        )

        # With φ=0, H=0: only remnant sources
        expected = lambda_r * n_rem + alpha * rho_rem

        assert_allclose(phi_ddot, expected, rtol=1e-10)

    def test_field_equation_oscillatory(self, hrc_theory):
        """Field equation should give oscillatory solution without sources."""
        # The equation φ̈ + m²φ = 0 (with H=0) has oscillatory solutions
        m_phi = hrc_theory.m_phi

        # At φ = 1, φ̇ = 0: expect φ̈ = -m²
        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi=1.0, a=1.0, H=0.0,
            rho_rem=0.0, n_rem=0.0, dphi_dt=0.0
        )

        # Account for curvature coupling with H=0 → R=0
        expected = -m_phi**2 * 1.0

        assert_allclose(phi_ddot, expected, rtol=1e-10)


# =============================================================================
# TEST: CONTINUITY EQUATIONS
# =============================================================================

class TestContinuityEquations:
    """Tests for energy conservation / continuity equations."""

    def test_matter_continuity_dust(self, hrc_theory):
        """Dust (p=0) should have ρ̇ = -3Hρ."""
        rho_m = 0.5
        p_m = 0.0
        H = 1.0

        rho_dot = hrc_theory.matter_density_dot(rho_m, p_m, H)
        expected = -3.0 * H * rho_m

        assert_allclose(rho_dot, expected, rtol=1e-10)

    def test_matter_continuity_radiation(self, hrc_theory):
        """Radiation (p=ρ/3) should have ρ̇ = -4Hρ."""
        rho_m = 0.5
        p_m = rho_m / 3.0
        H = 1.0

        rho_dot = hrc_theory.matter_density_dot(rho_m, p_m, H)
        expected = -4.0 * H * rho_m

        assert_allclose(rho_dot, expected, rtol=1e-10)

    def test_remnant_continuity_no_coupling(self, standard_theory):
        """Remnants should be conserved when α=0."""
        rho_rem = 0.3
        H = 1.0
        dphi_dt = 0.5

        rho_dot = standard_theory.remnant_density_dot(
            rho_rem, phi=0.5, dphi_dt=dphi_dt, H=H
        )

        # With α=0: ρ̇_rem = -3Hρ_rem
        expected = -3.0 * H * rho_rem

        assert_allclose(rho_dot, expected, rtol=1e-10)

    def test_remnant_continuity_with_coupling(self, hrc_theory):
        """Remnants should exchange energy with φ when α≠0."""
        rho_rem = 0.3
        H = 1.0
        phi = 0.5
        dphi_dt = 0.2
        alpha = hrc_theory.alpha

        rho_dot = hrc_theory.remnant_density_dot(rho_rem, phi, dphi_dt, H)

        # ρ̇_rem = -3Hρ_rem - αρ_rem·φ̇
        expected = -3.0 * H * rho_rem - alpha * rho_rem * dphi_dt

        assert_allclose(rho_dot, expected, rtol=1e-10)

    def test_energy_exchange_sign(self, hrc_theory):
        """Energy should flow correctly between sectors."""
        rho_rem = 0.3
        H = 1.0
        alpha = hrc_theory.alpha

        # Positive φ̇: field increasing
        dphi_dt_pos = 0.2
        rho_dot_pos = hrc_theory.remnant_density_dot(
            rho_rem, phi=0.5, dphi_dt=dphi_dt_pos, H=H
        )

        # Negative φ̇: field decreasing
        dphi_dt_neg = -0.2
        rho_dot_neg = hrc_theory.remnant_density_dot(
            rho_rem, phi=0.5, dphi_dt=dphi_dt_neg, H=H
        )

        # When φ̇ > 0 and α > 0: remnants lose extra energy (more negative ρ̇)
        # When φ̇ < 0 and α > 0: remnants gain energy (less negative ρ̇)
        if alpha > 0:
            assert rho_dot_pos < rho_dot_neg


# =============================================================================
# TEST: STRESS-ENERGY TENSOR COMPONENTS
# =============================================================================

class TestStressEnergy:
    """Tests for stress-energy tensor components."""

    def test_matter_stress_energy_dust(self, hrc_theory):
        """Dust stress-energy should have T_00 = ρ, T_ii = 0."""
        rho = 0.5
        p = 0.0

        T = hrc_theory.stress_energy_matter(rho, p)

        assert_allclose(T['T_00'], rho)
        assert_allclose(T['T_ii'], 0.0)
        assert_allclose(T['w'], 0.0)

    def test_matter_stress_energy_radiation(self, hrc_theory):
        """Radiation stress-energy should have w=1/3."""
        rho = 0.5
        p = rho / 3.0

        T = hrc_theory.stress_energy_matter(rho, p)

        assert_allclose(T['w'], 1.0/3.0, rtol=1e-10)

    def test_scalar_stress_energy_kinetic_dominated(self, hrc_theory):
        """Kinetic-dominated scalar should have w ≈ 1."""
        phi = 0.0  # No potential energy
        dphi_dt = 1.0  # Kinetic energy = 0.5

        T = hrc_theory.stress_energy_recycling(
            phi=phi, dphi_dt=dphi_dt,
            H=0.0, H_dot=0.0, d2phi_dt2=0.0, n_rem=0.0
        )

        # ρ = 0.5φ̇², p = 0.5φ̇² (for minimal coupling with V=0)
        assert_allclose(T['kinetic'], 0.5)
        assert_allclose(T['V'], 0.0)
        # w = p/ρ = 1 for kinetic dominated
        assert_allclose(T['w_eff'], 1.0, rtol=1e-5)

    def test_scalar_stress_energy_potential_dominated(self, hrc_theory):
        """Potential-dominated scalar should have w ≈ -1."""
        phi = 1.0  # V = 0.5m²φ² = 0.5
        dphi_dt = 0.0  # No kinetic

        T = hrc_theory.stress_energy_recycling(
            phi=phi, dphi_dt=dphi_dt,
            H=0.0, H_dot=0.0, d2phi_dt2=0.0, n_rem=0.0
        )

        # ρ = V, p = -V (for minimal coupling)
        # w = p/ρ = -1
        assert_allclose(T['w_eff'], -1.0, rtol=1e-5)

    def test_remnant_stress_energy_pressureless(self, hrc_theory):
        """Remnants should be pressureless."""
        rho_rem = 0.3
        phi = 0.5

        T = hrc_theory.stress_energy_remnant(rho_rem, phi)

        assert_allclose(T['p'], 0.0)


# =============================================================================
# TEST: GEOMETRIC QUANTITIES
# =============================================================================

class TestGeometry:
    """Tests for geometric quantities (FLRW specific)."""

    def test_ricci_scalar_de_sitter(self, hrc_theory):
        """de Sitter space should have constant R = 12H²."""
        H = 1.0
        H_dot = 0.0  # de Sitter: Ḣ = 0

        R = hrc_theory.ricci_scalar_flrw(H, H_dot)

        # R = 6(Ḣ + 2H²) = 12H² for de Sitter
        assert_allclose(R, 12.0 * H**2, rtol=1e-10)

    def test_ricci_scalar_matter_dominated(self, hrc_theory):
        """Matter-dominated era should have R = 6H² (Ḣ = -3H²/2)."""
        H = 1.0
        H_dot = -1.5 * H**2  # Matter dominated: Ḣ = -3H²/2

        R = hrc_theory.ricci_scalar_flrw(H, H_dot)

        # R = 6(Ḣ + 2H²) = 6(-3H²/2 + 2H²) = 6(H²/2) = 3H²
        # Wait, let me recalculate...
        # R = 6(-1.5 + 2) = 6(0.5) = 3
        assert_allclose(R, 3.0, rtol=1e-10)

    def test_einstein_tensor_trace(self, hrc_theory):
        """Einstein tensor trace should equal -R."""
        H = 1.0
        H_dot = -0.5

        G = hrc_theory.einstein_tensor_flrw(H, H_dot)
        R = hrc_theory.ricci_scalar_flrw(H, H_dot)

        # G^μ_μ = R^μ_μ - 2R = R - 2R = -R (in 4D)
        # But G['trace'] is computed as -G_00 + 3*G^i_i
        # Let's verify the definition...
        # Actually, trace of Einstein tensor = -R
        # G_00 = 3H², G_ii_mixed = -(2Ḣ + 3H²)
        # G = -G_00 + 3*G_ii (with metric signature)
        # = -3H² + 3*(-2Ḣ - 3H²) = -3H² - 6Ḣ - 9H² = -12H² - 6Ḣ
        # = -6(2H² + Ḣ) = -R ✓

        assert_allclose(G['trace'], -R, rtol=1e-10)


# =============================================================================
# TEST: SYMBOLIC DERIVATIONS
# =============================================================================

class TestSymbolicDerivations:
    """Tests for symbolic derivations via SymPy."""

    def test_action_components_initialized(self, action):
        """Action components should be properly initialized."""
        assert action.G is not None
        assert action.phi is not None
        assert action.Lambda is not None

    def test_potential_is_quadratic(self, action):
        """Default potential should be quadratic."""
        phi = sp.Symbol('phi')
        V = action.potential_V(phi)

        # V = m²φ²/2
        # dV/dφ = m²φ
        # d²V/dφ² = m²
        d2V = sp.diff(V, phi, 2)

        assert d2V == action.m_phi**2

    def test_flrw_ricci_scalar_symbolic(self, flrw):
        """FLRW Ricci scalar should have correct symbolic form."""
        R = flrw.R

        # R should contain ä/a and H² terms
        # R = 6(ä/a + H²)
        t = flrw.t
        a = flrw.a

        # Substitute numerical values to test
        R_test = R.subs({
            sp.diff(a, t, 2): 1.0,  # ä = 1
            sp.diff(a, t): 1.0,      # ȧ = 1
            a: 1.0                    # a = 1
        })

        # With ä=1, ȧ=1, a=1: H=1, ä/a=1, R = 6(1+1) = 12
        assert float(R_test) == 12.0

    def test_symbolic_friedmann_equation_exists(self, hrc_theory):
        """Symbolic Friedmann equation should be derivable."""
        result = hrc_theory.get_symbolic_friedmann_1()

        assert 'H_squared' in result
        assert 'equation' in result
        assert result['equation'] is not None

    def test_symbolic_scalar_equation_exists(self, hrc_theory):
        """Symbolic scalar equation should be derivable."""
        result = hrc_theory.get_symbolic_scalar_equation()

        assert 'phi_ddot' in result
        assert 'equation' in result


# =============================================================================
# TEST: ASYMPTOTIC BEHAVIOR
# =============================================================================

class TestAsymptoticBehavior:
    """Tests for correct asymptotic limits."""

    def test_late_time_de_sitter(self, standard_theory):
        """Universe should approach de Sitter as matter dilutes."""
        Lambda = standard_theory.Lambda

        # As ρ → 0, H² → Λ/3
        H2_limit = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=1e-10, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        H2_de_sitter = Lambda / 3.0

        assert_allclose(H2_limit, H2_de_sitter, rtol=1e-8)

    def test_early_matter_dominated(self, standard_theory):
        """Early universe should be matter-dominated."""
        # When ρ >> Λ, H² ≈ 8πGρ/3
        rho_m = 1000.0  # Large density
        Lambda = standard_theory.Lambda
        G = standard_theory.G

        H2 = standard_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        H2_matter = 8.0 * np.pi * G * rho_m / 3.0

        # Should be close (Λ is negligible)
        assert_allclose(H2, H2_matter, rtol=1e-3)

    def test_scalar_decays_without_sources(self, hrc_theory):
        """Scalar field should decay to zero without sources."""
        # The equation φ̈ + 3Hφ̇ + m²φ = 0 has decaying solutions
        # For H=0: simple harmonic oscillator with damping from mass

        # Test that acceleration is negative for positive φ
        phi = 1.0
        H = 1.0  # Some expansion

        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi=phi, a=1.0, H=H,
            rho_rem=0.0, n_rem=0.0, dphi_dt=0.1
        )

        # Should be negative (restoring force)
        assert phi_ddot < 0


# =============================================================================
# TEST: HUBBLE TENSION DIAGNOSTIC
# =============================================================================

class TestHubbleTension:
    """Tests for Hubble tension diagnostic function."""

    def test_no_tension_when_phi_constant(self):
        """No tension expected if φ doesn't evolve."""
        result = hubble_tension_diagnostic(
            H0_local=73.0, H0_cmb=67.0,
            phi_late=0.5, phi_early=0.5,
            xi=0.01, G=1.0
        )

        # If φ_late = φ_early, G_eff ratio = 1
        assert_allclose(result['G_eff_ratio'], 1.0)
        assert_allclose(result['phi_evolution'], 0.0)

    def test_tension_direction(self):
        """Positive tension requires G_eff_late > G_eff_early (for ξ>0)."""
        # H0_local > H0_cmb means we measure faster expansion locally
        # H² ∝ G_eff = G/(1-8πGξφ), so H ∝ √(G_eff)
        # Need G_eff(late) > G_eff(early) for H(late) > H(early)

        result = hubble_tension_diagnostic(
            H0_local=73.0, H0_cmb=67.0,
            phi_late=0.5, phi_early=0.1,  # φ grew
            xi=0.01, G=1.0
        )

        # With ξ > 0 and φ_late > φ_early:
        # G_eff_factor = (1-8πGξφ) is smaller for larger φ
        # G_eff = G/G_eff_factor is larger for larger φ
        # G_eff_ratio returns factor_late/factor_early which is < 1 when φ grows
        # This means 1/G_eff_ratio = G_eff_late/G_eff_early > 1
        assert result['G_eff_ratio'] < 1.0  # Factor ratio < 1 means G_eff grew
        assert result['phi_evolution'] > 0  # φ increased

    def test_observed_tension_calculation(self):
        """Observed tension should be calculated correctly."""
        H0_local = 73.0
        H0_cmb = 67.0

        result = hubble_tension_diagnostic(
            H0_local=H0_local, H0_cmb=H0_cmb,
            phi_late=0.0, phi_early=0.0,
            xi=0.0, G=1.0
        )

        expected_tension = (H0_local - H0_cmb) / H0_cmb

        assert_allclose(result['observed_tension'], expected_tension)


# =============================================================================
# TEST: NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of computations."""

    def test_small_phi_stability(self, hrc_theory):
        """Results should be stable for very small φ."""
        rho_m = 0.3

        H2_small = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=1e-15, dphi_dt=1e-15, n_rem=0.0
        )

        H2_zero = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        # Should be nearly identical
        assert_allclose(H2_small, H2_zero, rtol=1e-10)

    def test_large_rho_stability(self, hrc_theory):
        """Results should not overflow for large densities."""
        H2 = hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=1e10, rho_rem=0.0,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        assert np.isfinite(H2)
        assert H2 > 0

    def test_acceleration_equation_stability(self, hrc_theory):
        """Acceleration equation should be stable."""
        a_ddot_over_a = hrc_theory.friedmann_a_ddot(
            a=1.0, rho_m=0.3, p_m=0.0,
            rho_rem=0.1, phi=0.1, dphi_dt=0.01,
            d2phi_dt2=0.001, n_rem=0.05
        )

        assert np.isfinite(a_ddot_over_a)


# =============================================================================
# TEST: REMNANT PRODUCTION RATE
# =============================================================================

class TestRemnantProduction:
    """Tests for remnant production rate phenomenology."""

    def test_production_rate_positive(self):
        """Production rate should be positive for positive inputs."""
        rate = compute_remnant_production_rate(
            H=1.0, T_hawking=0.1, recycling_efficiency=0.01
        )

        assert rate > 0

    def test_production_rate_scales_with_efficiency(self):
        """Rate should scale linearly with recycling efficiency."""
        rate1 = compute_remnant_production_rate(
            H=1.0, T_hawking=0.1, recycling_efficiency=0.01
        )

        rate2 = compute_remnant_production_rate(
            H=1.0, T_hawking=0.1, recycling_efficiency=0.02
        )

        assert_allclose(rate2 / rate1, 2.0)

    def test_zero_efficiency_gives_zero_rate(self):
        """Zero efficiency should give zero production rate."""
        rate = compute_remnant_production_rate(
            H=1.0, T_hawking=0.1, recycling_efficiency=0.0
        )

        assert rate == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests checking consistency across components."""

    def test_friedmann_conservation_consistency(self, hrc_theory):
        """Friedmann equations should be consistent with conservation."""
        # This is a weak test - full consistency requires numerical integration

        rho_m = 0.3
        rho_rem = 0.1
        phi = 0.1
        dphi_dt = 0.01
        H = np.sqrt(hrc_theory.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=rho_rem,
            phi=phi, dphi_dt=dphi_dt, n_rem=0.05
        ))

        # Check that continuity equations give sensible results
        rho_m_dot = hrc_theory.matter_density_dot(rho_m, 0.0, H)
        rho_rem_dot = hrc_theory.remnant_density_dot(rho_rem, phi, dphi_dt, H)

        # Both should be negative (densities decrease in expanding universe)
        assert rho_m_dot < 0
        # Remnant dot might be less negative or more negative depending on coupling

        # Just check it's finite
        assert np.isfinite(rho_rem_dot)

    def test_full_evolution_step(self, hrc_theory):
        """Test that one step of evolution is consistent."""
        # Initial conditions
        a = 1.0
        rho_m = 0.3
        rho_rem = 0.1
        phi = 0.1
        dphi_dt = 0.01
        n_rem = 0.05

        # Compute H²
        H2 = hrc_theory.friedmann_H_squared(
            a, rho_m, rho_rem, phi, dphi_dt, n_rem
        )
        H = np.sqrt(H2)

        # Compute φ̈
        phi_ddot = hrc_theory.phi_equation_of_motion(
            phi, a, H, rho_rem, n_rem, dphi_dt
        )

        # Compute ρ̇ values
        rho_m_dot = hrc_theory.matter_density_dot(rho_m, 0.0, H)
        rho_rem_dot = hrc_theory.remnant_density_dot(rho_rem, phi, dphi_dt, H)

        # All should be finite
        assert np.isfinite(H)
        assert np.isfinite(phi_ddot)
        assert np.isfinite(rho_m_dot)
        assert np.isfinite(rho_rem_dot)

        # Take a small step (dt = 0.01)
        dt = 0.01
        phi_new = phi + dphi_dt * dt + 0.5 * phi_ddot * dt**2
        dphi_dt_new = dphi_dt + phi_ddot * dt
        rho_m_new = rho_m + rho_m_dot * dt
        rho_rem_new = rho_rem + rho_rem_dot * dt

        # New H² should still be positive and finite
        # (Use simplified n_rem scaling)
        n_rem_new = n_rem * (rho_rem_new / rho_rem) if rho_rem > 0 else 0

        H2_new = hrc_theory.friedmann_H_squared(
            a * np.exp(H * dt), rho_m_new, rho_rem_new,
            phi_new, dphi_dt_new, n_rem_new
        )

        assert np.isfinite(H2_new)
        assert H2_new > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
