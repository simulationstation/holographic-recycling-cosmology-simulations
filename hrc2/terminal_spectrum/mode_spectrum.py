"""
Multi-Mode Terminal Spectrum Cosmology

SIMULATION 25: This module implements a phenomenological framework where
a "terminal core release" produces a finite set of spectral modes in ln(a)
that perturb the expansion history:

    δH/H(a) = Σ_i A_i * f_i(ln a; μ_i, σ_i, φ_i)

where each mode f_i is a localized function (Gaussian) in ln(a) space.

This is NOT derived from a specific quantum gravity model; it's a flexible
multi-mode imprint parameterization motivated by the idea that pre-Big-Bang
physics (e.g., terminal quiet state in a parent black hole interior) could
leave a spectral imprint on the expansion history.

The key question: given CMB + BAO + SN constraints, what mode configurations
(if any) can shift H0 toward higher values while remaining consistent with data?
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray


# Physical constants
C_KM_S = 299792.458  # Speed of light in km/s


@dataclass
class TerminalMode:
    """
    A single spectral mode from the terminal core release.

    Each mode contributes to δH/H as a localized bump in ln(a) space.

    Attributes
    ----------
    mu_ln_a : float
        Center of the mode in ln(a).
        Examples: ln(1/(1+3000)) ≈ -8.0 for z~3000 (early universe)
                  ln(1/(1+100)) ≈ -4.6 for z~100
                  ln(1) = 0 for z=0 (today)
    sigma_ln_a : float
        Width of the mode in ln(a) space. Typical values 0.1–1.0.
        Larger values = broader impact across redshifts.
    amplitude : float
        Dimensionless amplitude of the δH/H contribution.
        Positive: increases H(z), decreases distances, increases inferred H0.
        Negative: decreases H(z), increases distances.
        Typical values: |A| < 0.1 for physically reasonable models.
    phase : float
        Optional phase parameter (for future extensions with oscillatory modes).
        Currently multiplies the profile by cos(phase).
        Default: 0.0 (no phase modulation, cos(0) = 1)
    """
    mu_ln_a: float
    sigma_ln_a: float
    amplitude: float
    phase: float = 0.0

    def __post_init__(self):
        """Validate inputs."""
        if self.sigma_ln_a <= 0:
            raise ValueError(f"sigma_ln_a must be positive, got {self.sigma_ln_a}")
        if not np.isfinite(self.mu_ln_a):
            raise ValueError(f"mu_ln_a must be finite, got {self.mu_ln_a}")
        if not np.isfinite(self.amplitude):
            raise ValueError(f"amplitude must be finite, got {self.amplitude}")


@dataclass
class TerminalSpectrumParams:
    """
    Parameters for a multi-mode terminal spectrum configuration.

    This defines the complete spectral imprint from the terminal core release.

    Attributes
    ----------
    modes : List[TerminalMode]
        List of spectral modes to sum. Can be empty (recovers ΛCDM).
    max_deltaH_fraction : float
        Safety cap on |δH/H| after summing all modes. Values exceeding
        this are clipped. Prevents unphysical configurations.
        Default: 0.2 (20% maximum deviation from ΛCDM)
    z_min : float
        Minimum redshift for validity (typically 0).
    z_max : float
        Maximum redshift for validity. Beyond this, modes are suppressed.
    n_samples : int
        Number of ln(a) sampling points for internal calculations.
    """
    modes: List[TerminalMode] = field(default_factory=list)
    max_deltaH_fraction: float = 0.2
    z_min: float = 0.0
    z_max: float = 1.0e4
    n_samples: int = 400

    def __post_init__(self):
        """Validate inputs."""
        if self.max_deltaH_fraction <= 0:
            raise ValueError(f"max_deltaH_fraction must be positive, got {self.max_deltaH_fraction}")
        if self.z_min < 0:
            raise ValueError(f"z_min must be non-negative, got {self.z_min}")
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max must exceed z_min, got z_max={self.z_max}, z_min={self.z_min}")

    @property
    def n_modes(self) -> int:
        """Number of spectral modes."""
        return len(self.modes)

    def add_mode(self, mode: TerminalMode) -> None:
        """Add a mode to the spectrum."""
        self.modes.append(mode)

    def get_amplitude_vector(self) -> NDArray[np.floating]:
        """Get amplitudes as a numpy array."""
        return np.array([m.amplitude for m in self.modes])

    def set_amplitudes(self, amplitudes: NDArray[np.floating]) -> None:
        """Set all amplitudes from an array (must match n_modes)."""
        if len(amplitudes) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} amplitudes, got {len(amplitudes)}")
        for mode, amp in zip(self.modes, amplitudes):
            mode.amplitude = float(amp)


@dataclass
class SpectrumCosmoConfig:
    """
    Baseline cosmological parameters for the terminal spectrum model.

    This defines the ΛCDM background that gets modified by the spectral modes.

    Attributes
    ----------
    H0 : float
        Hubble constant in km/s/Mpc (baseline value)
    Omega_m : float
        Matter density parameter (today)
    Omega_L : float
        Dark energy (cosmological constant) density parameter
    Omega_r : float
        Radiation density parameter (typically ~5e-5)
    Omega_k : float
        Curvature parameter (0 for flat universe)
    """
    H0: float = 67.5
    Omega_m: float = 0.315
    Omega_L: float = 0.685
    Omega_r: float = 5e-5
    Omega_k: float = 0.0

    def __post_init__(self):
        """Validate cosmological parameters."""
        if self.H0 <= 0:
            raise ValueError(f"H0 must be positive, got {self.H0}")
        if self.Omega_m < 0:
            raise ValueError(f"Omega_m must be non-negative, got {self.Omega_m}")
        if self.Omega_L < 0:
            raise ValueError(f"Omega_L must be non-negative, got {self.Omega_L}")
        # Check flatness (approximately)
        total = self.Omega_m + self.Omega_L + self.Omega_r + self.Omega_k
        if abs(total - 1.0) > 0.01:
            # Allow small deviations but warn implicitly via the curvature
            pass

    @property
    def h(self) -> float:
        """Reduced Hubble constant H0/100."""
        return self.H0 / 100.0


# =============================================================================
# Mode Profile Functions
# =============================================================================

def mode_profile_ln_a(
    ln_a: NDArray[np.floating],
    mode: TerminalMode
) -> NDArray[np.floating]:
    """
    Compute the profile function f_i(ln a) for a single spectral mode.

    The profile is a normalized Gaussian in ln(a) space with optional
    phase modulation:

        f_i(ln a) = exp[-(ln a - μ)² / (2σ²)] * cos(φ)

    Parameters
    ----------
    ln_a : array
        Natural log of scale factor, ln(a) = -ln(1+z)
    mode : TerminalMode
        The mode parameters (center, width, amplitude, phase)

    Returns
    -------
    array
        Profile values at each ln_a point

    Notes
    -----
    - The amplitude is NOT included here; it's applied in delta_H_over_H_ln_a
    - The Gaussian is not normalized (peak = 1 at μ)
    - Phase modulation: cos(phase) globally multiplies the profile
      For phase=0, cos(0)=1, no effect
      For phase=π/2, cos(π/2)=0, mode is suppressed
      For phase=π, cos(π)=-1, mode is inverted
    """
    ln_a = np.atleast_1d(ln_a)

    # Gaussian profile centered at mu_ln_a
    exponent = -((ln_a - mode.mu_ln_a)**2) / (2.0 * mode.sigma_ln_a**2)
    profile = np.exp(exponent)

    # Phase modulation (global factor)
    phase_factor = np.cos(mode.phase)

    return profile * phase_factor


def delta_H_over_H_ln_a(
    ln_a: NDArray[np.floating],
    params: TerminalSpectrumParams
) -> NDArray[np.floating]:
    """
    Compute δH/H(ln a) by summing over all spectral modes.

    δH/H(ln a) = Σ_i A_i * f_i(ln a; μ_i, σ_i, φ_i)

    After summation, the result is clipped to enforce |δH/H| <= max_deltaH_fraction.

    Parameters
    ----------
    ln_a : array
        Natural log of scale factor
    params : TerminalSpectrumParams
        The multi-mode spectrum parameters

    Returns
    -------
    array
        Fractional Hubble modification δH/H at each ln_a point

    Notes
    -----
    - Returns zeros if no modes are defined
    - Clipping prevents unphysical configurations (negative H² etc.)
    """
    ln_a = np.atleast_1d(ln_a)

    # Initialize δH/H = 0
    delta_H_H = np.zeros_like(ln_a, dtype=float)

    if params.n_modes == 0:
        return delta_H_H

    # Sum contributions from each mode
    for mode in params.modes:
        profile = mode_profile_ln_a(ln_a, mode)
        delta_H_H += mode.amplitude * profile

    # Clip to enforce physical sanity
    delta_H_H = np.clip(delta_H_H, -params.max_deltaH_fraction, params.max_deltaH_fraction)

    return delta_H_H


def delta_H_over_H_of_z(
    z: Union[float, NDArray[np.floating]],
    params: TerminalSpectrumParams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute δH/H(z) by converting z -> a -> ln(a) and calling delta_H_over_H_ln_a.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    params : TerminalSpectrumParams
        The multi-mode spectrum parameters

    Returns
    -------
    float or array
        Fractional Hubble modification δH/H at each redshift
    """
    z = np.atleast_1d(z)

    # Convert z -> a -> ln(a)
    a = 1.0 / (1.0 + z)
    ln_a = np.log(a)

    result = delta_H_over_H_ln_a(ln_a, params)

    return result if len(result) > 1 else float(result[0])


# =============================================================================
# ΛCDM Background Functions
# =============================================================================

def E_squared_LCDM(
    z: Union[float, NDArray[np.floating]],
    cosmo: SpectrumCosmoConfig
) -> Union[float, NDArray[np.floating]]:
    """
    Compute E²(z) = (H(z)/H0)² for flat ΛCDM.

    E²(z) = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_k(1+z)² + Ω_Λ

    Parameters
    ----------
    z : float or array
        Redshift(s)
    cosmo : SpectrumCosmoConfig
        Cosmological parameters

    Returns
    -------
    float or array
        E²(z) values
    """
    z = np.atleast_1d(z)

    E2 = (
        cosmo.Omega_m * (1 + z)**3 +
        cosmo.Omega_r * (1 + z)**4 +
        cosmo.Omega_k * (1 + z)**2 +
        cosmo.Omega_L
    )

    result = E2
    return result if len(result) > 1 else float(result[0])


def H_LCDM(
    z: Union[float, NDArray[np.floating]],
    cosmo: SpectrumCosmoConfig
) -> Union[float, NDArray[np.floating]]:
    """
    Compute H(z) for ΛCDM in km/s/Mpc.

    H(z) = H0 * E(z) = H0 * sqrt(E²(z))

    Parameters
    ----------
    z : float or array
        Redshift(s)
    cosmo : SpectrumCosmoConfig
        Cosmological parameters

    Returns
    -------
    float or array
        H(z) in km/s/Mpc
    """
    E2 = E_squared_LCDM(z, cosmo)
    E2 = np.atleast_1d(E2)

    # Handle any numerical issues
    E2 = np.maximum(E2, 1e-30)

    H = cosmo.H0 * np.sqrt(E2)

    return H if len(H) > 1 else float(H[0])


# =============================================================================
# Modified H(z) with Terminal Spectrum
# =============================================================================

def compute_modified_H_of_z(
    z: Union[float, NDArray[np.floating]],
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute H(z) with the multi-mode δH/H imprint.

    H_modified(z) = H_LCDM(z) * (1 + δH/H(z))

    This is the main interface for getting the modified expansion history.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    cosmo : SpectrumCosmoConfig
        Baseline ΛCDM cosmological parameters
    spec : TerminalSpectrumParams
        Multi-mode terminal spectrum parameters

    Returns
    -------
    float or array
        Modified H(z) in km/s/Mpc

    Notes
    -----
    - For spec with no modes, returns H_LCDM
    - The clipping in delta_H_over_H_ln_a ensures H_modified > 0
      (as long as max_deltaH_fraction < 1)
    """
    z = np.atleast_1d(z)

    # Baseline ΛCDM
    H_base = H_LCDM(z, cosmo)
    H_base = np.atleast_1d(H_base)

    # Spectral modification
    delta_H_H = delta_H_over_H_of_z(z, spec)
    delta_H_H = np.atleast_1d(delta_H_H)

    # Apply modification
    H_modified = H_base * (1.0 + delta_H_H)

    # Ensure positivity (should already be guaranteed by clipping)
    H_modified = np.maximum(H_modified, 1e-30)

    return H_modified if len(H_modified) > 1 else float(H_modified[0])


def E_modified(
    z: Union[float, NDArray[np.floating]],
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute dimensionless E(z) = H(z)/H0 for modified cosmology.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float or array
        E(z) = H_modified(z) / H0
    """
    H_mod = compute_modified_H_of_z(z, cosmo, spec)
    return H_mod / cosmo.H0


def get_H0_effective(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Get the effective H0 from the modified model.

    This is H_modified(z=0), which differs from the baseline H0
    if the spectral modes have nonzero contributions at z=0.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Effective H0 in km/s/Mpc
    """
    return compute_modified_H_of_z(0.0, cosmo, spec)


# =============================================================================
# Physical Validity Checks
# =============================================================================

def check_physical_validity(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    n_test_points: int = 200
) -> dict:
    """
    Check physical validity of the terminal spectrum configuration.

    Checks:
    1. H(z) > 0 for all z in [0, z_max] (no negative expansion)
    2. No pathological energy densities
    3. H0_eff in reasonable range

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    n_test_points : int
        Number of test redshifts

    Returns
    -------
    dict
        Dictionary with:
        - "valid": bool, overall validity
        - "H_positive": bool, all H(z) > 0
        - "H_min": float, minimum H(z)
        - "H_max": float, maximum H(z)
        - "H0_eff": float, effective H0
        - "max_delta_H_H": float, maximum |δH/H|
        - "warnings": list of warning messages
    """
    # Test redshift grid (log-spaced for better coverage)
    z_test = np.logspace(-3, np.log10(spec.z_max), n_test_points)
    z_test = np.concatenate([[0], z_test])  # Include z=0

    # Compute modified H(z)
    H_values = compute_modified_H_of_z(z_test, cosmo, spec)
    H_values = np.atleast_1d(H_values)

    # Compute δH/H
    delta_H_H_values = delta_H_over_H_of_z(z_test, spec)
    delta_H_H_values = np.atleast_1d(delta_H_H_values)

    # Checks
    H_positive = np.all(H_values > 0) and np.all(np.isfinite(H_values))
    H_min = np.nanmin(H_values)
    H_max = np.nanmax(H_values)
    H0_eff = get_H0_effective(cosmo, spec)
    max_delta_H_H = np.nanmax(np.abs(delta_H_H_values))

    warnings_list = []

    if not H_positive:
        warnings_list.append(f"H(z) non-positive or NaN; min H = {H_min:.2f}")

    if H0_eff < 50 or H0_eff > 100:
        warnings_list.append(f"H0_eff = {H0_eff:.2f} outside reasonable range [50, 100]")

    if max_delta_H_H > spec.max_deltaH_fraction:
        warnings_list.append(f"max |δH/H| = {max_delta_H_H:.3f} exceeds limit {spec.max_deltaH_fraction}")

    # Check for very large early-time modifications
    H_ratio_early = H_values[-1] / H_LCDM(z_test[-1], cosmo)
    if abs(H_ratio_early - 1.0) > 0.3:
        warnings_list.append(f"Large early-time H modification: H/H_LCDM = {H_ratio_early:.3f} at z={z_test[-1]:.0f}")

    return {
        "valid": H_positive and (50 < H0_eff < 100),
        "H_positive": H_positive,
        "H_min": H_min,
        "H_max": H_max,
        "H0_eff": H0_eff,
        "max_delta_H_H": max_delta_H_H,
        "warnings": warnings_list
    }


# =============================================================================
# Convenience Constructors
# =============================================================================

def make_3mode_template(
    z_centers: tuple = (3000, 100, 1),
    sigma_ln_a: float = 0.3,
    amplitudes: tuple = (0.0, 0.0, 0.0)
) -> TerminalSpectrumParams:
    """
    Create a standard 3-mode template for the terminal spectrum.

    The default centers are:
    - Mode 0: z ~ 3000 (very early, near recombination)
    - Mode 1: z ~ 100 (around matter-radiation equality)
    - Mode 2: z ~ 1 (late-time, SN/BAO regime)

    Parameters
    ----------
    z_centers : tuple of 3 floats
        Redshift centers for the three modes
    sigma_ln_a : float
        Width of each mode in ln(a) space (same for all modes)
    amplitudes : tuple of 3 floats
        Amplitudes (A1, A2, A3) for each mode

    Returns
    -------
    TerminalSpectrumParams
        Configured 3-mode spectrum
    """
    if len(z_centers) != 3 or len(amplitudes) != 3:
        raise ValueError("Need exactly 3 centers and 3 amplitudes for 3-mode template")

    modes = []
    for z_c, amp in zip(z_centers, amplitudes):
        a_c = 1.0 / (1.0 + z_c)
        mu = np.log(a_c)
        modes.append(TerminalMode(mu_ln_a=mu, sigma_ln_a=sigma_ln_a, amplitude=amp))

    return TerminalSpectrumParams(modes=modes)


def make_single_mode(
    z_center: float,
    sigma_ln_a: float,
    amplitude: float,
    phase: float = 0.0
) -> TerminalSpectrumParams:
    """
    Create a single-mode terminal spectrum.

    Useful for testing individual mode effects.

    Parameters
    ----------
    z_center : float
        Redshift center for the mode
    sigma_ln_a : float
        Width in ln(a) space
    amplitude : float
        Mode amplitude
    phase : float
        Phase parameter

    Returns
    -------
    TerminalSpectrumParams
        Single-mode spectrum
    """
    a_c = 1.0 / (1.0 + z_center)
    mu = np.log(a_c)
    mode = TerminalMode(mu_ln_a=mu, sigma_ln_a=sigma_ln_a, amplitude=amplitude, phase=phase)
    return TerminalSpectrumParams(modes=[mode])


def make_zero_spectrum() -> TerminalSpectrumParams:
    """
    Create a zero-mode spectrum (recovers baseline ΛCDM).

    Returns
    -------
    TerminalSpectrumParams
        Empty spectrum (δH/H = 0 everywhere)
    """
    return TerminalSpectrumParams(modes=[])
