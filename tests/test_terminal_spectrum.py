"""
Tests for the terminal_spectrum module (mode_spectrum.py).

Tests cover:
- TerminalMode and TerminalSpectrumParams dataclasses
- Mode profile functions
- δH/H computation
- Modified H(z) computation
- Physical validity checks
"""

import pytest
import numpy as np

from hrc2.terminal_spectrum import (
    TerminalMode,
    TerminalSpectrumParams,
    SpectrumCosmoConfig,
    mode_profile_ln_a,
    delta_H_over_H_ln_a,
    delta_H_over_H_of_z,
    E_squared_LCDM,
    H_LCDM,
    compute_modified_H_of_z,
    E_modified,
    get_H0_effective,
    check_physical_validity,
    make_3mode_template,
    make_single_mode,
    make_zero_spectrum,
)


class TestTerminalMode:
    """Tests for TerminalMode dataclass."""

    def test_basic_creation(self):
        """Test basic mode creation."""
        mode = TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.3, amplitude=0.01)
        assert mode.mu_ln_a == -4.6
        assert mode.sigma_ln_a == 0.3
        assert mode.amplitude == 0.01
        assert mode.phase == 0.0  # Default

    def test_with_phase(self):
        """Test mode with phase parameter."""
        mode = TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.3, amplitude=0.01, phase=np.pi/2)
        assert mode.phase == pytest.approx(np.pi/2)

    def test_invalid_sigma(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError):
            TerminalMode(mu_ln_a=-4.6, sigma_ln_a=-0.1, amplitude=0.01)

    def test_zero_sigma(self):
        """Test that zero sigma raises ValueError."""
        with pytest.raises(ValueError):
            TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.0, amplitude=0.01)


class TestTerminalSpectrumParams:
    """Tests for TerminalSpectrumParams dataclass."""

    def test_empty_modes(self):
        """Test creation with no modes."""
        spec = TerminalSpectrumParams(modes=[])
        assert spec.n_modes == 0

    def test_with_modes(self):
        """Test creation with modes."""
        modes = [
            TerminalMode(mu_ln_a=-8.0, sigma_ln_a=0.3, amplitude=0.01),
            TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.3, amplitude=-0.02),
        ]
        spec = TerminalSpectrumParams(modes=modes)
        assert spec.n_modes == 2

    def test_invalid_max_deltaH(self):
        """Test that non-positive max_deltaH_fraction raises ValueError."""
        with pytest.raises(ValueError):
            TerminalSpectrumParams(modes=[], max_deltaH_fraction=0.0)

    def test_get_amplitude_vector(self):
        """Test amplitude vector extraction."""
        modes = [
            TerminalMode(mu_ln_a=-8.0, sigma_ln_a=0.3, amplitude=0.01),
            TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.3, amplitude=-0.02),
            TerminalMode(mu_ln_a=0.0, sigma_ln_a=0.3, amplitude=0.03),
        ]
        spec = TerminalSpectrumParams(modes=modes)
        amps = spec.get_amplitude_vector()
        np.testing.assert_array_almost_equal(amps, [0.01, -0.02, 0.03])


class TestModeProfileFunction:
    """Tests for mode_profile_ln_a function."""

    def test_peak_at_center(self):
        """Test that profile peaks at mode center."""
        mode = TerminalMode(mu_ln_a=-4.6, sigma_ln_a=0.3, amplitude=1.0)
        ln_a = np.array([-4.6])
        profile = mode_profile_ln_a(ln_a, mode)
        assert profile[0] == pytest.approx(1.0)  # Peak value

    def test_gaussian_decay(self):
        """Test Gaussian decay away from center."""
        mode = TerminalMode(mu_ln_a=0.0, sigma_ln_a=1.0, amplitude=1.0)
        ln_a = np.array([0.0, 1.0, 2.0])
        profile = mode_profile_ln_a(ln_a, mode)

        # exp(-0) = 1, exp(-0.5) ≈ 0.606, exp(-2) ≈ 0.135
        assert profile[0] == pytest.approx(1.0)
        assert profile[1] == pytest.approx(np.exp(-0.5), rel=1e-5)
        assert profile[2] == pytest.approx(np.exp(-2.0), rel=1e-5)

    def test_phase_modulation(self):
        """Test phase modulation."""
        mode_no_phase = TerminalMode(mu_ln_a=0.0, sigma_ln_a=1.0, amplitude=1.0, phase=0.0)
        mode_with_phase = TerminalMode(mu_ln_a=0.0, sigma_ln_a=1.0, amplitude=1.0, phase=np.pi/2)

        ln_a = np.array([0.0])
        profile_no_phase = mode_profile_ln_a(ln_a, mode_no_phase)
        profile_with_phase = mode_profile_ln_a(ln_a, mode_with_phase)

        # cos(0) = 1, cos(π/2) ≈ 0
        assert profile_no_phase[0] == pytest.approx(1.0)
        assert abs(profile_with_phase[0]) < 1e-10


class TestDeltaHOverH:
    """Tests for delta_H_over_H functions."""

    def test_zero_modes_gives_zero(self):
        """Test that zero modes give δH/H = 0."""
        spec = TerminalSpectrumParams(modes=[])
        ln_a = np.linspace(-10, 0, 100)
        delta = delta_H_over_H_ln_a(ln_a, spec)

        np.testing.assert_array_equal(delta, 0.0)

    def test_single_mode(self):
        """Test single mode contribution."""
        mode = TerminalMode(mu_ln_a=-5.0, sigma_ln_a=0.5, amplitude=0.05)
        spec = TerminalSpectrumParams(modes=[mode])

        # At mode center
        delta_center = delta_H_over_H_ln_a(np.array([-5.0]), spec)
        assert delta_center[0] == pytest.approx(0.05, rel=1e-5)

        # Far from mode center
        delta_far = delta_H_over_H_ln_a(np.array([0.0]), spec)
        assert abs(delta_far[0]) < 0.01  # Should be very small

    def test_clipping(self):
        """Test that large amplitudes are clipped."""
        # Large amplitude mode
        mode = TerminalMode(mu_ln_a=0.0, sigma_ln_a=0.5, amplitude=0.5)
        spec = TerminalSpectrumParams(modes=[mode], max_deltaH_fraction=0.2)

        ln_a = np.array([0.0])
        delta = delta_H_over_H_ln_a(ln_a, spec)

        # Should be clipped to max_deltaH_fraction
        assert delta[0] == pytest.approx(0.2, rel=1e-5)

    def test_negative_clipping(self):
        """Test that large negative amplitudes are clipped."""
        mode = TerminalMode(mu_ln_a=0.0, sigma_ln_a=0.5, amplitude=-0.5)
        spec = TerminalSpectrumParams(modes=[mode], max_deltaH_fraction=0.2)

        ln_a = np.array([0.0])
        delta = delta_H_over_H_ln_a(ln_a, spec)

        assert delta[0] == pytest.approx(-0.2, rel=1e-5)


class TestHLCDM:
    """Tests for ΛCDM background functions."""

    def test_H0_at_z0(self):
        """Test that H(z=0) ≈ H0 (small radiation contribution)."""
        cosmo = SpectrumCosmoConfig(H0=70.0, Omega_m=0.3, Omega_L=0.7)
        H0 = H_LCDM(0.0, cosmo)
        # Small deviation due to radiation component
        assert H0 == pytest.approx(70.0, rel=1e-3)

    def test_E_squared_z0(self):
        """Test that E²(z=0) = 1."""
        cosmo = SpectrumCosmoConfig(H0=70.0, Omega_m=0.3, Omega_L=0.7, Omega_r=0.0, Omega_k=0.0)
        E2 = E_squared_LCDM(0.0, cosmo)
        assert E2 == pytest.approx(1.0, rel=1e-10)

    def test_H_increases_with_z(self):
        """Test that H increases with redshift."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)
        z_vals = [0, 0.5, 1.0, 2.0]
        H_vals = [H_LCDM(z, cosmo) for z in z_vals]

        for i in range(len(H_vals) - 1):
            assert H_vals[i+1] > H_vals[i]


class TestModifiedHz:
    """Tests for modified H(z) computation."""

    def test_zero_modes_equals_LCDM(self):
        """Test that zero modes gives H_LCDM."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)
        spec = TerminalSpectrumParams(modes=[])

        z_test = np.array([0, 0.5, 1.0, 2.0, 10.0])
        H_mod = compute_modified_H_of_z(z_test, cosmo, spec)
        H_lcdm = H_LCDM(z_test, cosmo)

        np.testing.assert_array_almost_equal(H_mod, H_lcdm)

    def test_positive_amplitude_increases_H(self):
        """Test that positive amplitude increases H."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)

        # Mode at z ~ 1 (ln_a = 0)
        mode = TerminalMode(mu_ln_a=-0.69, sigma_ln_a=0.3, amplitude=0.05)
        spec = TerminalSpectrumParams(modes=[mode])

        z = 1.0  # Near mode center
        H_mod = compute_modified_H_of_z(z, cosmo, spec)
        H_lcdm = H_LCDM(z, cosmo)

        assert H_mod > H_lcdm

    def test_H0_effective(self):
        """Test effective H0 computation."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)

        # Mode at z=0 (ln_a = 0)
        mode = TerminalMode(mu_ln_a=0.0, sigma_ln_a=0.3, amplitude=0.03)
        spec = TerminalSpectrumParams(modes=[mode])

        H0_eff = get_H0_effective(cosmo, spec)

        # H0_eff = H0_LCDM * (1 + δH/H(z=0))
        # At z=0, ln_a = 0, profile = 1, so δH/H = amplitude
        expected = cosmo.H0 * (1 + 0.03)
        assert H0_eff == pytest.approx(expected, rel=1e-3)


class TestPhysicalValidity:
    """Tests for physical validity checks."""

    def test_zero_modes_valid(self):
        """Test that zero modes is always valid."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)
        spec = TerminalSpectrumParams(modes=[])

        result = check_physical_validity(cosmo, spec)
        assert result["valid"]
        assert result["H_positive"]

    def test_small_amplitude_valid(self):
        """Test that small amplitudes are valid."""
        cosmo = SpectrumCosmoConfig(H0=67.5, Omega_m=0.315, Omega_L=0.685)
        mode = TerminalMode(mu_ln_a=-5.0, sigma_ln_a=0.3, amplitude=0.02)
        spec = TerminalSpectrumParams(modes=[mode])

        result = check_physical_validity(cosmo, spec)
        assert result["valid"]
        assert result["H_positive"]
        assert result["max_delta_H_H"] < 0.1


class TestConvenienceConstructors:
    """Tests for convenience constructor functions."""

    def test_make_3mode_template(self):
        """Test 3-mode template creation."""
        spec = make_3mode_template(
            z_centers=(3000, 100, 1),
            sigma_ln_a=0.3,
            amplitudes=(0.01, -0.02, 0.03)
        )

        assert spec.n_modes == 3
        amps = spec.get_amplitude_vector()
        np.testing.assert_array_almost_equal(amps, [0.01, -0.02, 0.03])

    def test_make_single_mode(self):
        """Test single mode creation."""
        spec = make_single_mode(z_center=100, sigma_ln_a=0.5, amplitude=0.02)
        assert spec.n_modes == 1
        assert spec.modes[0].amplitude == pytest.approx(0.02)

    def test_make_zero_spectrum(self):
        """Test zero spectrum creation."""
        spec = make_zero_spectrum()
        assert spec.n_modes == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
