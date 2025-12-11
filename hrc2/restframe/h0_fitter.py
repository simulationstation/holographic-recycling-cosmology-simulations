"""
hrc2.restframe.h0_fitter - H0 fitting with rest-frame corrections

This module provides:
1. fit_H0_with_frame_correction: Fit H0 after correcting to a specific rest frame
2. fit_H0_wrong_frame: Fit H0 using the wrong rest frame (to compute bias)
3. compute_H0_bias_from_frame_mismatch: Compute the H0 bias from frame mismatch
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import optimize

from .frames import (
    RestFrameDefinition,
    correct_redshift_to_frame,
    C_LIGHT,
)
from .sn_catalog import SNSample


# =============================================================================
# H0 Fit Result
# =============================================================================

@dataclass
class H0FitResult:
    """
    Result of an H0 fit from SN data.

    Attributes
    ----------
    H0 : float
        Best-fit Hubble constant [km/s/Mpc]
    H0_err : float
        Statistical uncertainty on H0 [km/s/Mpc]
    M_B : float
        Best-fit absolute magnitude (degenerate with H0)
    chi2 : float
        Chi-squared of the fit
    dof : int
        Degrees of freedom
    n_sn : int
        Number of SNe used in fit
    """
    H0: float
    H0_err: float
    M_B: float
    chi2: float
    dof: int
    n_sn: int


# =============================================================================
# H0 Fitting Functions
# =============================================================================

def mu_theory(z: np.ndarray, H0: float) -> np.ndarray:
    """
    Theoretical distance modulus in the low-z Hubble law approximation.

    μ = 5 * log10(cz / H0) + 25

    This is valid for the Hubble flow where z >> v_pec/c.
    """
    z_safe = np.maximum(z, 1e-10)
    return 5.0 * np.log10(C_LIGHT * z_safe / H0) + 25.0


def fit_H0_simple(
    z: np.ndarray,
    mu_obs: np.ndarray,
    sigma_mu: np.ndarray,
) -> H0FitResult:
    """
    Simple H0 fit from redshifts and distance moduli.

    Uses the Hubble law: μ = 5 log10(cz/H0) + 25

    Parameters
    ----------
    z : ndarray
        Redshifts (in the assumed rest frame)
    mu_obs : ndarray
        Observed distance moduli
    sigma_mu : ndarray
        Distance modulus uncertainties

    Returns
    -------
    H0FitResult
        Best-fit H0 and associated statistics
    """
    # Weight by inverse variance
    w = 1.0 / sigma_mu**2

    # Minimize chi-squared
    def chi2_fn(H0):
        mu_th = mu_theory(z, H0)
        return np.sum(w * (mu_obs - mu_th)**2)

    # Find best-fit H0 (bracket reasonable range)
    result = optimize.minimize_scalar(chi2_fn, bounds=(40, 120), method='bounded')
    H0_best = result.fun is not None and result.x or 70.0

    # Actually get the value
    result = optimize.minimize_scalar(chi2_fn, bounds=(40, 120), method='bounded')
    H0_best = result.x

    # Compute chi-squared
    mu_th_best = mu_theory(z, H0_best)
    chi2_best = np.sum(w * (mu_obs - mu_th_best)**2)

    # Estimate uncertainty from curvature
    # d²χ²/dH0² ≈ (χ²(H0+δ) - 2χ²(H0) + χ²(H0-δ)) / δ²
    delta = 0.1
    d2chi2 = (chi2_fn(H0_best + delta) - 2 * chi2_fn(H0_best) + chi2_fn(H0_best - delta)) / delta**2
    H0_err = np.sqrt(2.0 / max(d2chi2, 1e-10))

    # Compute M_B from the intercept (assuming M_B = -19.3 as reference)
    # μ = m - M_B, so M_B = m - μ
    # This is degenerate with H0 in the distance ladder
    M_B = -19.3  # Placeholder

    n_sn = len(z)
    dof = n_sn - 1  # 1 free parameter (H0)

    return H0FitResult(
        H0=H0_best,
        H0_err=H0_err,
        M_B=M_B,
        chi2=chi2_best,
        dof=dof,
        n_sn=n_sn,
    )


def fit_H0_with_frame_correction(
    sample: SNSample,
    assumed_helio_velocity: RestFrameDefinition,
) -> H0FitResult:
    """
    Fit H0 after correcting heliocentric redshifts to an assumed rest frame.

    Parameters
    ----------
    sample : SNSample
        SN sample with heliocentric redshifts
    assumed_helio_velocity : RestFrameDefinition
        The velocity we assume for the heliocentric correction

    Returns
    -------
    H0FitResult
        Best-fit H0 using the assumed rest frame
    """
    # Correct heliocentric redshifts to assumed rest frame
    z_corrected = np.array([
        correct_redshift_to_frame(z_h, l, b, assumed_helio_velocity)
        for z_h, l, b in zip(sample.z_helio, sample.l_gal, sample.b_gal)
    ])

    # Ensure positive redshifts
    z_corrected = np.maximum(z_corrected, 1e-6)

    # Fit H0
    return fit_H0_simple(z_corrected, sample.mu_obs, sample.sigma_mu)


def compute_H0_bias_from_frame_mismatch(
    sample: SNSample,
    H0_true: float,
    true_helio_velocity: RestFrameDefinition,
    wrong_helio_velocity: RestFrameDefinition,
) -> Tuple[float, float, float]:
    """
    Compute H0 bias from using the wrong rest frame.

    Parameters
    ----------
    sample : SNSample
        SN sample with heliocentric redshifts
    H0_true : float
        True Hubble constant
    true_helio_velocity : RestFrameDefinition
        The correct heliocentric velocity (e.g., radio dipole)
    wrong_helio_velocity : RestFrameDefinition
        The wrong velocity used for correction (e.g., CMB dipole)

    Returns
    -------
    H0_fit : float
        Fitted H0 using wrong frame
    delta_H0 : float
        H0_fit - H0_true (the bias)
    frac_bias : float
        Fractional bias (delta_H0 / H0_true)
    """
    # Fit with wrong frame correction
    result = fit_H0_with_frame_correction(sample, wrong_helio_velocity)

    H0_fit = result.H0
    delta_H0 = H0_fit - H0_true
    frac_bias = delta_H0 / H0_true

    return H0_fit, delta_H0, frac_bias


def fit_H0_true_frame(
    sample: SNSample,
) -> H0FitResult:
    """
    Fit H0 using the true cosmological redshifts (control case).

    This is what we'd get if we knew the true rest frame perfectly.
    """
    return fit_H0_simple(sample.z_cosmo, sample.mu_obs, sample.sigma_mu)


# =============================================================================
# Analytical Estimates
# =============================================================================

def estimate_H0_bias_analytical(
    v_true: float,
    v_assumed: float,
    z_mean: float,
    sky_coverage: str = "isotropic",
) -> float:
    """
    Estimate H0 bias analytically from frame velocity mismatch.

    For isotropic sky coverage, the dipole averages out and the bias
    is zero. For partial sky coverage, there can be a residual bias.

    Parameters
    ----------
    v_true : float
        True heliocentric velocity [km/s]
    v_assumed : float
        Assumed heliocentric velocity [km/s]
    z_mean : float
        Mean redshift of the sample
    sky_coverage : str
        "isotropic", "toward_apex", or "away_from_apex"

    Returns
    -------
    float
        Expected fractional H0 bias (delta_H0 / H0)

    Notes
    -----
    The velocity correction affects the inferred redshift:
    z_corr = z_helio - v_los/c

    If we use v_assumed instead of v_true:
    z_corr_wrong = z_true + (v_true - v_assumed) * cos(theta) / c

    For isotropic sky, <cos(theta)> = 0, so bias averages out.
    For hemispherical coverage, <cos(theta)> ≈ ±0.5.
    """
    delta_v = v_true - v_assumed

    if sky_coverage == "isotropic":
        # For isotropic sky, dipole averages to zero
        # Second-order effect from z-dependence
        # delta_H0/H0 ≈ (delta_v / c)^2 / (2 * z_mean)
        frac_bias = 0.5 * (delta_v / C_LIGHT)**2 / z_mean
    elif sky_coverage == "toward_apex":
        # <cos(theta)> ≈ 0.5 for hemisphere toward apex
        # delta_H0/H0 ≈ delta_v / (c * z_mean) * <cos(theta)>
        frac_bias = 0.5 * delta_v / (C_LIGHT * z_mean)
    elif sky_coverage == "away_from_apex":
        # <cos(theta)> ≈ -0.5 for hemisphere away from apex
        frac_bias = -0.5 * delta_v / (C_LIGHT * z_mean)
    else:
        frac_bias = 0.0

    return frac_bias


def estimate_H0_scatter_from_dipole(
    v_amplitude: float,
    z_mean: float,
    n_sn: int,
) -> float:
    """
    Estimate the scatter in H0 from uncorrected dipole.

    Even with isotropic sky coverage, the dipole introduces scatter
    in distance modulus residuals that increases H0 uncertainty.

    Parameters
    ----------
    v_amplitude : float
        Velocity amplitude [km/s]
    z_mean : float
        Mean redshift
    n_sn : int
        Number of SNe

    Returns
    -------
    float
        Additional H0 scatter [km/s/Mpc]
    """
    # RMS velocity along random lines of sight
    v_rms = v_amplitude / np.sqrt(3)

    # This translates to distance modulus scatter
    sigma_mu_dipole = (5.0 / np.log(10)) * (v_rms / C_LIGHT) / z_mean

    # And H0 scatter (roughly)
    # sigma_H0 / H0 ≈ sigma_mu / (5 / ln(10))
    # sigma_H0 ≈ H0 * v_rms / (c * z_mean)
    H0_nominal = 70.0
    sigma_H0 = H0_nominal * v_rms / (C_LIGHT * z_mean) / np.sqrt(n_sn)

    return sigma_H0
