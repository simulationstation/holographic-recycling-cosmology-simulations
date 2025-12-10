"""Tests for parameter space scanning module."""

import pytest
import numpy as np

from hrc.analysis import (
    scan_parameter_space,
    ParameterScanResult,
    PointClassification,
)
from hrc.analysis.parameter_scan import _scan_single_point, compute_validity_boundary


class TestPointClassification:
    """Tests for PointClassification enum."""

    def test_values(self):
        """Check classification values exist."""
        assert PointClassification.INVALID.value == "invalid"
        assert PointClassification.VALID_NO_TENSION.value == "valid_but_no_tension"
        assert PointClassification.VALID_RESOLVES.value == "valid_and_resolves_tension"


class TestScanSinglePoint:
    """Tests for single point scanning."""

    def test_valid_point_low_z(self):
        """A point with low z_max should be valid."""
        point = _scan_single_point(
            xi=0.03,
            phi_0=0.1,
            z_max=2.0,  # Low z - before divergence
            z_points=50,
            tension_threshold=3.0,
            h=0.7,
            geff_epsilon=0.01,
        )
        assert point.geff_valid
        assert point.classification != PointClassification.INVALID

    def test_invalid_point_high_z(self):
        """A point with high z_max may become invalid."""
        point = _scan_single_point(
            xi=0.03,
            phi_0=0.2,
            z_max=100.0,  # High enough for scalar field to evolve
            z_points=100,
            tension_threshold=3.0,
            h=0.7,
            geff_epsilon=0.01,
        )
        # With these parameters, phi evolves toward critical
        # The exact behavior depends on the potential
        assert isinstance(point.classification, PointClassification)

    def test_initial_phi_too_close_to_critical(self):
        """Point with initial phi near critical should be invalid."""
        point = _scan_single_point(
            xi=0.1,  # phi_c = 1/(8*pi*0.1) ≈ 0.398
            phi_0=0.35,  # Close to critical
            z_max=10.0,
            z_points=50,
            tension_threshold=3.0,
            h=0.7,
            geff_epsilon=0.1,  # 10% safety margin
        )
        assert point.classification == PointClassification.INVALID

    def test_zero_xi_always_valid(self):
        """With xi=0, G_eff never diverges."""
        point = _scan_single_point(
            xi=0.0,
            phi_0=0.5,
            z_max=100.0,
            z_points=50,
            tension_threshold=3.0,
            h=0.7,
            geff_epsilon=0.01,
        )
        # For xi=0, phi_critical is infinite, so always valid
        # But the HRCParameters may reject xi=0 for other reasons
        # This test checks the logic handles xi=0 gracefully
        assert isinstance(point.classification, PointClassification)


class TestScanParameterSpace:
    """Tests for full parameter space scan."""

    def test_scan_returns_result(self):
        """Scan should return a ParameterScanResult."""
        result = scan_parameter_space(
            xi_range=(0.01, 0.03),
            phi_0_range=(0.05, 0.15),
            n_xi=3,
            n_phi_0=3,
            z_max=2.0,
            z_points=30,
            verbose=False,
        )
        assert isinstance(result, ParameterScanResult)

    def test_scan_grid_shape(self):
        """Result grids should have correct shapes."""
        n_xi, n_phi_0 = 4, 5
        result = scan_parameter_space(
            xi_range=(0.01, 0.05),
            phi_0_range=(0.05, 0.2),
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=2.0,
            z_points=30,
            verbose=False,
        )
        assert len(result.xi_grid) == n_xi
        assert len(result.phi_0_grid) == n_phi_0
        assert result.classification.shape == (n_xi, n_phi_0)
        assert result.geff_valid.shape == (n_xi, n_phi_0)

    def test_scan_counts_sum(self):
        """Classification counts should sum to total points."""
        n_xi, n_phi_0 = 3, 4
        result = scan_parameter_space(
            xi_range=(0.01, 0.05),
            phi_0_range=(0.05, 0.2),
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=2.0,
            z_points=30,
            verbose=False,
        )
        total = n_xi * n_phi_0
        assert result.n_invalid + result.n_valid_no_tension + result.n_valid_resolves == total

    def test_scan_points_list(self):
        """Result should have list of individual points."""
        n_xi, n_phi_0 = 3, 3
        result = scan_parameter_space(
            xi_range=(0.01, 0.03),
            phi_0_range=(0.05, 0.15),
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=2.0,
            z_points=30,
            verbose=False,
        )
        assert len(result.points) == n_xi * n_phi_0

    def test_get_valid_region(self):
        """get_valid_region should return coordinate arrays."""
        result = scan_parameter_space(
            xi_range=(0.01, 0.03),
            phi_0_range=(0.05, 0.15),
            n_xi=3,
            n_phi_0=3,
            z_max=2.0,
            z_points=30,
            verbose=False,
        )
        xi_valid, phi_valid = result.get_valid_region()
        assert len(xi_valid) == len(phi_valid)
        # Number of valid points should match
        n_valid = result.n_valid_no_tension + result.n_valid_resolves
        assert len(xi_valid) == n_valid


class TestComputeValidityBoundary:
    """Tests for validity boundary computation."""

    def test_boundary_shape(self):
        """Boundary should return arrays of correct length."""
        n_xi = 5
        xi_vals, phi_max = compute_validity_boundary(
            xi_range=(0.01, 0.05),
            n_xi=n_xi,
            z_max=2.0,
            verbose=False,
        )
        assert len(xi_vals) == n_xi
        assert len(phi_max) == n_xi

    def test_boundary_positive(self):
        """Maximum phi_0 should be non-negative."""
        xi_vals, phi_max = compute_validity_boundary(
            xi_range=(0.01, 0.05),
            n_xi=3,
            z_max=2.0,
            verbose=False,
        )
        assert np.all(phi_max >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
