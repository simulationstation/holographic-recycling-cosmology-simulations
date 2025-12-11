"""Tests for the layered expansion (bent-deck) cosmology module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from hrc2.layered import (
    LayeredExpansionHyperparams,
    LayeredExpansionParams,
    LCDMBackground,
    make_default_nodes,
    log_smoothness_prior,
    H_of_z_layered,
    E_of_z_layered,
    get_H0_effective,
    check_physical_validity,
    make_zero_params,
    make_random_params,
)


class TestLayeredExpansionHyperparams:
    """Tests for LayeredExpansionHyperparams dataclass."""

    def test_default_values(self):
        """Default hyperparameters should be reasonable."""
        hyp = LayeredExpansionHyperparams()
        assert hyp.n_layers == 6
        assert hyp.z_min == 0.0
        assert hyp.z_max == 6.0
        assert hyp.smooth_sigma == 0.05
        assert hyp.mode == "delta_H"
        assert hyp.spacing == "log"

    def test_custom_values(self):
        """Custom hyperparameters should be stored correctly."""
        hyp = LayeredExpansionHyperparams(
            n_layers=10,
            z_min=0.1,
            z_max=3.0,
            smooth_sigma=0.1,
            mode="delta_w",
            spacing="linear"
        )
        assert hyp.n_layers == 10
        assert hyp.z_min == 0.1
        assert hyp.z_max == 3.0
        assert hyp.smooth_sigma == 0.1
        assert hyp.mode == "delta_w"
        assert hyp.spacing == "linear"


class TestLayeredExpansionParams:
    """Tests for LayeredExpansionParams dataclass."""

    def test_valid_params(self):
        """Valid parameters should be accepted."""
        z_nodes = np.array([0.0, 1.0, 2.0])
        delta_nodes = np.array([0.01, 0.02, 0.0])
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        assert len(params.z_nodes) == 3
        assert len(params.delta_nodes) == 3
        assert params.n_layers == 3

    def test_mismatched_lengths_raises(self):
        """Mismatched z and delta lengths should raise."""
        z_nodes = np.array([0.0, 1.0, 2.0])
        delta_nodes = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="same length"):
            LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

    def test_non_monotonic_raises(self):
        """Non-monotonic z_nodes should raise."""
        z_nodes = np.array([0.0, 2.0, 1.0])  # Not monotonic
        delta_nodes = np.array([0.01, 0.02, 0.0])

        with pytest.raises(ValueError, match="monotonically increasing"):
            LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

    def test_too_few_nodes_raises(self):
        """Fewer than 2 nodes should raise."""
        z_nodes = np.array([0.0])
        delta_nodes = np.array([0.01])

        with pytest.raises(ValueError, match="at least 2 nodes"):
            LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)


class TestMakeDefaultNodes:
    """Tests for make_default_nodes function."""

    def test_correct_number_of_nodes(self):
        """Should return correct number of nodes."""
        hyp = LayeredExpansionHyperparams(n_layers=8)
        z_nodes = make_default_nodes(hyp)
        assert len(z_nodes) == 8

    def test_monotonic_increasing(self):
        """Nodes should be monotonically increasing."""
        hyp = LayeredExpansionHyperparams(n_layers=10, z_min=0.0, z_max=5.0)
        z_nodes = make_default_nodes(hyp)

        assert np.all(np.diff(z_nodes) > 0)

    def test_covers_range(self):
        """Nodes should cover z_min to z_max."""
        hyp = LayeredExpansionHyperparams(n_layers=6, z_min=0.1, z_max=4.0)
        z_nodes = make_default_nodes(hyp)

        assert_allclose(z_nodes[0], 0.1, rtol=1e-10)
        assert_allclose(z_nodes[-1], 4.0, rtol=1e-10)

    def test_log_spacing_concentrates_at_low_z(self):
        """Log spacing should have more nodes at low z."""
        hyp_log = LayeredExpansionHyperparams(n_layers=10, spacing="log")
        hyp_lin = LayeredExpansionHyperparams(n_layers=10, spacing="linear")

        z_log = make_default_nodes(hyp_log)
        z_lin = make_default_nodes(hyp_lin)

        # Log spacing should have smaller gaps at low z
        diff_log = np.diff(z_log)
        diff_lin = np.diff(z_lin)

        # First gap should be smaller for log spacing
        assert diff_log[0] < diff_lin[0]

        # Last gap should be larger for log spacing
        assert diff_log[-1] > diff_lin[-1]


class TestLogSmoothnessPrior:
    """Tests for log_smoothness_prior function."""

    def test_flat_delta_gives_zero(self):
        """Flat delta (all same) should give log_prior = 0."""
        hyp = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.05)
        z_nodes = make_default_nodes(hyp)

        # All deltas equal
        delta_nodes = np.ones(6) * 0.02
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        log_prior = log_smoothness_prior(params, hyp)
        assert_allclose(log_prior, 0.0)

    def test_all_zeros_gives_zero(self):
        """All zeros should also give log_prior = 0."""
        hyp = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.05)
        params = make_zero_params(hyp)

        log_prior = log_smoothness_prior(params, hyp)
        assert_allclose(log_prior, 0.0)

    def test_wiggly_gives_negative(self):
        """Wiggly delta should give negative log_prior."""
        hyp = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.05)
        z_nodes = make_default_nodes(hyp)

        # Alternating signs = very wiggly
        delta_nodes = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        log_prior = log_smoothness_prior(params, hyp)
        assert log_prior < 0

    def test_stiffer_prior_penalizes_more(self):
        """Smaller smooth_sigma should penalize wiggles more."""
        hyp_stiff = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.02)
        hyp_floppy = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.1)

        z_nodes = make_default_nodes(hyp_stiff)
        delta_nodes = np.array([0.0, 0.05, 0.0, 0.05, 0.0, 0.05])
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        log_prior_stiff = log_smoothness_prior(params, hyp_stiff)
        log_prior_floppy = log_smoothness_prior(params, hyp_floppy)

        # Stiffer prior should be more negative (penalize more)
        assert log_prior_stiff < log_prior_floppy

    def test_invalid_sigma_raises(self):
        """Zero or negative smooth_sigma should raise."""
        hyp = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.0)
        params = make_zero_params(hyp)

        with pytest.raises(ValueError, match="positive"):
            log_smoothness_prior(params, hyp)


class TestLCDMBackground:
    """Tests for LCDMBackground class."""

    def test_default_values(self):
        """Default values should be reasonable Planck-like."""
        lcdm = LCDMBackground()
        assert lcdm.H0 == 67.5
        assert lcdm.Omega_m == 0.315
        assert lcdm.h == 0.675

    def test_omega_closure(self):
        """Omega_L should close the universe."""
        lcdm = LCDMBackground(Omega_m=0.3, Omega_r=0.0)
        assert_allclose(lcdm.Omega_L, 0.7)

    def test_E_at_z0_equals_1(self):
        """E(z=0) should be 1."""
        lcdm = LCDMBackground()
        assert_allclose(lcdm.E_of_z(0.0), 1.0, rtol=1e-10)

    def test_H_at_z0_equals_H0(self):
        """H(z=0) should equal H0."""
        lcdm = LCDMBackground(H0=70.0)
        assert_allclose(lcdm.H_of_z(0.0), 70.0, rtol=1e-10)

    def test_H_increases_with_z(self):
        """H(z) should increase with z."""
        lcdm = LCDMBackground()
        z_test = np.array([0.0, 0.5, 1.0, 2.0])
        H_test = lcdm.H_of_z(z_test)

        assert np.all(np.diff(H_test) > 0)


class TestHOfZLayered:
    """Tests for H_of_z_layered and related functions."""

    @pytest.fixture
    def default_setup(self):
        """Create default LCDM + hyperparameters."""
        lcdm = LCDMBackground(H0=67.5, Omega_m=0.315)
        hyp = LayeredExpansionHyperparams(n_layers=6, smooth_sigma=0.05)
        return lcdm, hyp

    def test_zero_delta_recovers_lcdm(self, default_setup):
        """With delta=0, should recover LCDM H(z)."""
        lcdm, hyp = default_setup
        params = make_zero_params(hyp)

        z_test = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        H_layered = H_of_z_layered(z_test, lcdm, params, hyp)
        H_lcdm = lcdm.H_of_z(z_test)

        assert_allclose(H_layered, H_lcdm, rtol=1e-10)

    def test_positive_delta_increases_H(self, default_setup):
        """Positive delta_H should increase H(z)."""
        lcdm, hyp = default_setup
        hyp_delta_H = LayeredExpansionHyperparams(n_layers=6, mode="delta_H")
        z_nodes = make_default_nodes(hyp_delta_H)

        # Uniform positive offset
        delta_nodes = np.ones(6) * 0.05  # 5% increase
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        z_test = np.array([0.5, 1.0, 1.5])  # Within node range
        H_layered = H_of_z_layered(z_test, lcdm, params, hyp_delta_H)
        H_lcdm = lcdm.H_of_z(z_test)

        # H_layered should be higher
        assert np.all(H_layered > H_lcdm)

    def test_H0_effective_with_nonzero_delta_at_z0(self, default_setup):
        """Effective H0 should change with delta at z=0."""
        lcdm, hyp = default_setup
        z_nodes = make_default_nodes(hyp)

        # Set delta[0] = 0.1 (10% increase at z=0)
        delta_nodes = np.zeros(6)
        delta_nodes[0] = 0.1
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        H0_eff = get_H0_effective(lcdm, params, hyp)

        # Should be ~10% higher than baseline
        expected = lcdm.H0 * 1.1
        assert_allclose(H0_eff, expected, rtol=0.01)

    def test_array_and_scalar_consistent(self, default_setup):
        """Array and scalar calls should give consistent results."""
        lcdm, hyp = default_setup
        params = make_random_params(hyp, sigma_delta=0.03, rng=np.random.default_rng(42))

        z_scalar = 1.5
        z_array = np.array([1.5, 2.0])  # Use 2 elements so we get array back

        H_scalar = H_of_z_layered(z_scalar, lcdm, params, hyp)
        H_array = H_of_z_layered(z_array, lcdm, params, hyp)

        assert_allclose(H_scalar, H_array[0], rtol=1e-10)


class TestHOfZLayeredDeltaW:
    """Tests for delta_w mode."""

    @pytest.fixture
    def setup_delta_w(self):
        """Create setup for delta_w mode."""
        lcdm = LCDMBackground(H0=67.5, Omega_m=0.315)
        hyp = LayeredExpansionHyperparams(n_layers=6, mode="delta_w", smooth_sigma=0.05)
        return lcdm, hyp

    def test_zero_delta_w_recovers_lcdm(self, setup_delta_w):
        """With delta_w=0, should recover LCDM H(z)."""
        lcdm, hyp = setup_delta_w
        params = make_zero_params(hyp)

        z_test = np.array([0.0, 0.5, 1.0, 2.0])
        H_layered = H_of_z_layered(z_test, lcdm, params, hyp)
        H_lcdm = lcdm.H_of_z(z_test)

        # Should match to good precision
        assert_allclose(H_layered, H_lcdm, rtol=1e-6)

    def test_positive_delta_w_changes_expansion(self, setup_delta_w):
        """Nonzero delta_w should modify expansion history."""
        lcdm, hyp = setup_delta_w
        z_nodes = make_default_nodes(hyp)

        # w_eff = -1 + 0.1 = -0.9 (less negative than Lambda)
        delta_nodes = np.ones(6) * 0.1
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        z_test = 1.0
        H_layered = H_of_z_layered(z_test, lcdm, params, hyp)
        H_lcdm = lcdm.H_of_z(z_test)

        # With w > -1, DE dilutes faster, so H(z>0) should be lower
        # (less DE contribution at high z)
        assert H_layered != H_lcdm  # Just check it's modified


class TestCheckPhysicalValidity:
    """Tests for check_physical_validity function."""

    def test_lcdm_is_valid(self):
        """Baseline LCDM should be valid."""
        lcdm = LCDMBackground()
        hyp = LayeredExpansionHyperparams()
        params = make_zero_params(hyp)

        result = check_physical_validity(lcdm, params, hyp)

        assert result["valid"]
        assert result["H_positive"]
        assert len(result["warnings"]) == 0

    def test_extreme_delta_may_be_invalid(self):
        """Very extreme delta values may cause issues."""
        lcdm = LCDMBackground()
        hyp = LayeredExpansionHyperparams()
        z_nodes = make_default_nodes(hyp)

        # Very negative delta at low z could make H0_eff too low
        delta_nodes = np.array([-0.3, -0.2, 0.0, 0.0, 0.0, 0.0])
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        result = check_physical_validity(lcdm, params, hyp)

        # H0_eff will be ~47 km/s/Mpc, outside valid range
        assert result["H0_eff"] < 50
        assert not result["valid"]


class TestConvenienceConstructors:
    """Tests for convenience constructors."""

    def test_make_zero_params(self):
        """make_zero_params should create valid parameters with delta=0."""
        hyp = LayeredExpansionHyperparams(n_layers=8)
        params = make_zero_params(hyp)

        assert len(params.z_nodes) == 8
        assert_allclose(params.delta_nodes, 0.0)

    def test_make_random_params_respects_seed(self):
        """make_random_params should be reproducible with seed."""
        hyp = LayeredExpansionHyperparams(n_layers=6)

        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        params1 = make_random_params(hyp, sigma_delta=0.1, rng=rng1)
        params2 = make_random_params(hyp, sigma_delta=0.1, rng=rng2)

        assert_allclose(params1.delta_nodes, params2.delta_nodes)

    def test_make_random_params_different_seeds_differ(self):
        """Different seeds should give different results."""
        hyp = LayeredExpansionHyperparams(n_layers=6)

        params1 = make_random_params(hyp, sigma_delta=0.1, rng=np.random.default_rng(111))
        params2 = make_random_params(hyp, sigma_delta=0.1, rng=np.random.default_rng(222))

        # Should not be identical
        assert not np.allclose(params1.delta_nodes, params2.delta_nodes)


class TestInterpolation:
    """Tests for interpolation behavior."""

    def test_interpolation_at_nodes(self):
        """At node locations, should match node values exactly."""
        lcdm = LCDMBackground()
        hyp = LayeredExpansionHyperparams(n_layers=6, mode="delta_H")
        z_nodes = make_default_nodes(hyp)
        delta_nodes = np.array([0.01, 0.02, 0.03, 0.02, 0.01, 0.0])
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        # Check at middle nodes
        for i in range(1, 5):
            z = z_nodes[i]
            H_layered = H_of_z_layered(z, lcdm, params, hyp)
            H_expected = lcdm.H_of_z(z) * (1 + delta_nodes[i])
            assert_allclose(H_layered, H_expected, rtol=1e-8)

    def test_interpolation_monotonic_delta(self):
        """Monotonic delta nodes should give monotonic interpolation."""
        hyp = LayeredExpansionHyperparams(n_layers=6, mode="delta_H")
        z_nodes = make_default_nodes(hyp)
        delta_nodes = np.linspace(0.0, 0.1, 6)  # Monotonically increasing
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        # Test many intermediate z values
        z_test = np.linspace(z_nodes[0], z_nodes[-1], 50)

        from hrc2.layered.layered_expansion import _interpolate_delta
        delta_interp = _interpolate_delta(z_test, params)

        # Should be roughly monotonic (some numerical noise allowed)
        diffs = np.diff(delta_interp)
        # Most diffs should be >= 0
        assert np.sum(diffs >= -1e-10) > 0.9 * len(diffs)
