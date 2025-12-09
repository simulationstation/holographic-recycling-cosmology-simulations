"""Tests for stability checks."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters
from hrc.perturbations.stability_checks import (
    check_effective_planck_mass,
    check_no_ghost,
    check_gradient_stability,
    check_tensor_stability,
    check_all_stability,
    compute_effective_planck_mass_squared,
    StabilityChecker,
)


class TestEffectivePlanckMass:
    """Tests for effective Planck mass computation."""

    def test_gr_limit(self):
        """M_eff² = M_Pl²/(8π) when ξ=0."""
        params = HRCParameters(xi=0.0, phi_0=0.0)
        M_eff_sq = compute_effective_planck_mass_squared(0.0, params)
        expected = 1.0 / (8 * np.pi)  # In Planck units
        assert_allclose(M_eff_sq, expected)

    def test_enhanced_gravity(self):
        """M_eff² decreases when G_eff increases."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        M_eff_sq_0 = compute_effective_planck_mass_squared(0.0, params)
        M_eff_sq_phi = compute_effective_planck_mass_squared(0.2, params)
        # M_eff² = (1 - 8πξφ)/(8π), decreases with φ
        assert M_eff_sq_phi < M_eff_sq_0

    def test_positivity_check(self):
        """Check should pass for positive M_eff²."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        result = check_effective_planck_mass(0.2, params)
        assert result.passed
        assert result.value > 0

    def test_negativity_detection(self):
        """Check should fail for negative M_eff²."""
        params = HRCParameters(xi=0.1, phi_0=0.0)
        phi_crit = 1.0 / (8 * np.pi * 0.1)
        result = check_effective_planck_mass(phi_crit * 1.1, params)
        assert not result.passed


class TestNoGhost:
    """Tests for no-ghost condition."""

    def test_stable_parameters(self):
        """Standard HRC parameters should be ghost-free."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        result = check_no_ghost(0.2, 0.0, 1.0, params)
        assert result.passed
        assert result.value > 0

    def test_gr_limit(self):
        """GR limit should be ghost-free."""
        params = HRCParameters(xi=0.0, phi_0=0.0)
        result = check_no_ghost(0.0, 0.0, 1.0, params)
        assert result.passed


class TestGradientStability:
    """Tests for gradient stability."""

    def test_canonical_field(self):
        """Canonical scalar field should have c_s² = 1."""
        params = HRCParameters(xi=0.0, phi_0=0.0)
        result = check_gradient_stability(0.0, 0.0, 1.0, params)
        assert result.passed
        assert_allclose(result.value, 1.0, rtol=0.1)

    def test_slow_evolution(self):
        """Slowly evolving field should be gradient stable."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        result = check_gradient_stability(0.2, 0.001, 1.0, params)
        assert result.passed


class TestTensorStability:
    """Tests for tensor perturbation stability."""

    def test_standard_params(self):
        """Standard parameters should have stable tensors."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        result = check_tensor_stability(0.2, 0.0, 1.0, params)
        assert result.passed

    def test_gr_limit(self):
        """GR should have stable tensors."""
        params = HRCParameters(xi=0.0, phi_0=0.0)
        result = check_tensor_stability(0.0, 0.0, 1.0, params)
        assert result.passed


class TestAllStability:
    """Tests for combined stability check."""

    def test_all_pass_standard(self):
        """All checks should pass for standard parameters."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        all_passed, results = check_all_stability(0.2, 0.0, 1.0, params)
        assert all_passed
        assert len(results) == 4  # Four stability checks

    def test_abort_on_failure(self):
        """Should raise exception when abort_on_failure=True."""
        params = HRCParameters(xi=0.1, phi_0=0.0)
        phi_crit = 1.0 / (8 * np.pi * 0.1)

        with pytest.raises(ValueError, match="Stability"):
            check_all_stability(phi_crit * 1.1, 0.0, 1.0, params, abort_on_failure=True)


class TestStabilityChecker:
    """Tests for StabilityChecker class."""

    def test_initialization(self):
        """Checker should initialize with params."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        checker = StabilityChecker(params)
        assert checker.params == params
        assert len(checker.history) == 0

    def test_check_at_z(self):
        """Check at specific redshift should work."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        checker = StabilityChecker(params)

        passed, results = checker.check_at_z(z=0.0, phi=0.2, phi_dot=0.0, H=1.0)
        assert passed
        assert len(checker.history) == 1

    def test_summary(self):
        """Summary should report statistics."""
        params = HRCParameters(xi=0.03, phi_0=0.2)
        checker = StabilityChecker(params)

        # Run a few checks
        for z in [0.0, 0.5, 1.0]:
            checker.check_at_z(z, 0.2, 0.0, 1.0)

        summary = checker.get_stability_summary()
        assert summary["checked"]
        assert summary["n_points"] == 3
        assert summary["all_stable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
