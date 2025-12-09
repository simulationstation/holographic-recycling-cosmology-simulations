"""Tests for observational likelihoods."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hrc.utils.config import HRCParameters
from hrc.observables.h0_likelihoods import (
    SH0ESLikelihood,
    TRGBLikelihood,
    CMBDistanceLikelihood,
    GaussianH0Likelihood,
    SHOES_MEASUREMENT,
)
from hrc.observables.bao import BAOLikelihood, BAODataPoint
from hrc.observables.supernovae import SNeLikelihood, SNDataPoint
from hrc.observables.distances import DistanceCalculator, luminosity_distance


class TestGaussianH0Likelihood:
    """Tests for Gaussian H₀ likelihood."""

    def test_chi2_at_central(self):
        """χ² should be 0 at central value."""
        like = SH0ESLikelihood()
        chi2 = like.chi2(SHOES_MEASUREMENT.H0)
        assert_allclose(chi2, 0.0)

    def test_chi2_at_one_sigma(self):
        """χ² should be 1 at 1σ."""
        like = SH0ESLikelihood()
        chi2 = like.chi2(SHOES_MEASUREMENT.H0 + SHOES_MEASUREMENT.sigma)
        assert_allclose(chi2, 1.0)

    def test_log_likelihood_negative(self):
        """log-likelihood should be negative."""
        like = SH0ESLikelihood()
        logL = like.log_likelihood(70.0)
        assert logL <= 0

    def test_log_likelihood_maximum_at_central(self):
        """log-likelihood should be maximum at central value."""
        like = SH0ESLikelihood()
        logL_central = like.log_likelihood(SHOES_MEASUREMENT.H0)
        logL_offset = like.log_likelihood(SHOES_MEASUREMENT.H0 + 5)
        assert logL_central > logL_offset


class TestSH0ESLikelihood:
    """Tests for SH0ES likelihood."""

    def test_default_values(self):
        """Default should use SH0ES measurement."""
        like = SH0ESLikelihood()
        assert like.H0_obs == SHOES_MEASUREMENT.H0
        assert like.sigma == SHOES_MEASUREMENT.sigma

    def test_custom_values(self):
        """Should accept custom values."""
        like = SH0ESLikelihood(H0=74.0, sigma=1.5)
        assert like.H0_obs == 74.0
        assert like.sigma == 1.5


class TestCMBDistanceLikelihood:
    """Tests for CMB distance prior likelihood."""

    def test_initialization(self):
        """Should initialize with Planck values."""
        like = CMBDistanceLikelihood()
        assert like.z_star > 1000  # Recombination redshift

    def test_compute_theta_star(self):
        """Should compute θ* from parameters."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        like = CMBDistanceLikelihood()
        theta = like.compute_theta_star(params)
        assert theta > 0
        assert theta < 0.1  # θ* ≈ 0.01 rad


class TestBAOLikelihood:
    """Tests for BAO likelihood."""

    def test_initialization(self):
        """Should initialize with default data."""
        like = BAOLikelihood()
        assert len(like.data) > 0

    def test_custom_data(self):
        """Should accept custom data."""
        custom_data = [
            BAODataPoint(z=0.5, observable="DM_rs", value=14.0, sigma=0.3),
        ]
        like = BAOLikelihood(data=custom_data)
        assert len(like.data) == 1

    def test_log_likelihood_finite(self):
        """log-likelihood should be finite for reasonable params."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        like = BAOLikelihood()
        logL = like.log_likelihood(params)
        assert np.isfinite(logL)

    def test_chi2_positive(self):
        """χ² should be non-negative."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        like = BAOLikelihood()
        chi2 = like.chi2(params)
        assert chi2 >= 0


class TestSNeLikelihood:
    """Tests for Type Ia supernovae likelihood."""

    @pytest.fixture
    def sample_data(self):
        """Sample SNe data."""
        return [
            SNDataPoint("SN1", 0.01, 0.011, 32.5, 0.15, "test"),
            SNDataPoint("SN2", 0.05, 0.051, 36.0, 0.12, "test"),
            SNDataPoint("SN3", 0.10, 0.101, 38.0, 0.13, "test"),
        ]

    def test_initialization(self, sample_data):
        """Should initialize with data."""
        like = SNeLikelihood(sample_data)
        assert len(like.data) == 3
        assert len(like.z_array) == 3

    def test_compute_distance_moduli(self, sample_data):
        """Should compute distance moduli."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        like = SNeLikelihood(sample_data)
        mu_pred = like.compute_distance_moduli(params)
        assert len(mu_pred) == 3
        # Distance modulus should increase with z
        assert mu_pred[2] > mu_pred[0]

    def test_chi2_positive(self, sample_data):
        """χ² should be non-negative."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        like = SNeLikelihood(sample_data)
        chi2 = like.chi2(params)
        assert chi2 >= 0


class TestDistanceCalculator:
    """Tests for distance calculations."""

    def test_hubble_distance(self):
        """d_H = c/H should be correct."""
        params = HRCParameters(h=0.7)
        calc = DistanceCalculator(params)

        d_H = calc.hubble_distance(0.0)
        # c/H0 ≈ 4280 Mpc for H0 = 70
        assert_allclose(d_H, 4280, rtol=0.01)

    def test_comoving_distance_increases(self):
        """Comoving distance should increase with z."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        calc = DistanceCalculator(params)

        d1 = calc.comoving_distance(0.5)
        d2 = calc.comoving_distance(1.0)
        assert d2 > d1

    def test_distance_relations(self):
        """d_L = (1+z)² d_A should hold."""
        params = HRCParameters(h=0.7, Omega_b=0.05, Omega_c=0.25)
        calc = DistanceCalculator(params)

        z = 0.5
        d_A = calc.angular_diameter_distance(z)
        d_L = calc.luminosity_distance(z)

        # For flat universe: d_L = (1+z)² d_A
        expected_d_L = (1 + z)**2 * d_A
        assert_allclose(d_L, expected_d_L, rtol=1e-6)

    def test_luminosity_distance_function(self):
        """Standalone luminosity distance function."""
        d_L = luminosity_distance(z=1.0, H0=70.0, Omega_m=0.3, Omega_Lambda=0.7)
        # Should be roughly 6700 Mpc for z=1 with these parameters
        assert 6000 < d_L < 7500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
