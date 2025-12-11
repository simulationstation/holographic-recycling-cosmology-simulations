#!/usr/bin/env python3
"""
Two-step distance ladder pipeline for SN Ia H0 measurement.

Implements SH0ES-like ladder:
1. Calibrator step: Estimate M_B from SNe with known distances (Cepheid/TRGB hosts)
2. Hubble-flow step: Fit H0 using M_B from calibrators

The naive fitter ignores systematics that nature includes.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from .cosmology_baseline import TrueCosmology, mu_of_z
from .snia_salt2 import SNSystematicParameters11B


@dataclass
class LadderFitResult:
    """Result of two-step ladder fit."""
    H0_fit: float           # Fitted H0 [km/s/Mpc]
    M_B_fit: float          # Calibrated absolute magnitude
    chi2_flow: float        # Chi-squared for Hubble flow
    dof_flow: int           # Degrees of freedom
    N_calib: int            # Number of calibrators used
    N_flow: int             # Number of flow SNe used
    H0_true: float          # True H0 for reference
    delta_H0: float         # H0_fit - H0_true


def calibrate_M_B_from_mu(
    calib_sample: Dict[str, np.ndarray],
    mu_calib: np.ndarray,
    params: SNSystematicParameters11B,
) -> float:
    """
    Generic calibrator step: given observed SN quantities and
    (possibly biased) distance moduli mu_calib, estimate M_B_fit.

    The fitter assumes:
        m_B = M_B - alpha_fit * x1 + beta_fit * c + mu_calib

    So:
        M_B_est = m_B + alpha_fit * x1 - beta_fit * c - mu_calib

    Args:
        calib_sample: Calibrator SN sample
        mu_calib: Distance moduli for calibrators (may include biases)
        params: Systematic parameters (uses alpha_fit, beta_fit)

    Returns:
        Estimated M_B
    """
    m_B = calib_sample["m_B_obs"]
    x1 = calib_sample["x1"]
    c = calib_sample["c"]

    # Naive fitter: no host mass step correction
    M_est = m_B + params.alpha_fit * x1 - params.beta_fit * c - mu_calib

    # If fitter assumes a host mass step, apply it
    if params.delta_M_step_fit != 0.0:
        high_mass = calib_sample["high_mass_mask"]
        M_est[high_mass] = M_est[high_mass] - params.delta_M_step_fit

    return float(np.mean(M_est))


def calibrate_M_B(
    calib_sample: Dict[str, np.ndarray],
    params: SNSystematicParameters11B,
) -> float:
    """
    Estimate M_B from calibrator sample with known distances.

    Backwards-compatible wrapper: uses true mu from cosmology
    (no calibrator biases).

    Args:
        calib_sample: Calibrator SN sample (must have mu_true)
        params: Systematic parameters (uses alpha_fit, beta_fit)

    Returns:
        Estimated M_B
    """
    mu_true = calib_sample["mu_true"]
    return calibrate_M_B_from_mu(calib_sample, mu_true, params)


def compute_standardized_mag(
    sample: Dict[str, np.ndarray],
    params: SNSystematicParameters11B,
) -> np.ndarray:
    """
    Compute standardized apparent magnitude from raw m_B, x1, c.

    m_B_standardized = m_B + alpha_fit * x1 - beta_fit * c

    This is what the fitter uses to compare with the model.
    """
    m_B = sample["m_B_obs"]
    x1 = sample["x1"]
    c = sample["c"]

    m_std = m_B + params.alpha_fit * x1 - params.beta_fit * c

    # Apply host mass step correction if fitter uses one
    if params.delta_M_step_fit != 0.0:
        high_mass = sample["high_mass_mask"]
        m_std[high_mass] = m_std[high_mass] - params.delta_M_step_fit

    return m_std


def chi2_flow_for_H0(
    H0: float,
    flow_sample: Dict[str, np.ndarray],
    M_B_fit: float,
    params: SNSystematicParameters11B,
    Omega_m: float = 0.315,
    sigma_total: float = 0.13,
) -> float:
    """
    Compute chi-squared for Hubble flow sample at given H0.

    Model: m_B_standardized = M_B_fit + mu(z; H0)

    Args:
        H0: Trial H0 value
        flow_sample: Hubble flow SN sample
        M_B_fit: Calibrated absolute magnitude
        params: Systematic parameters
        Omega_m: Matter density
        sigma_total: Total uncertainty per SN

    Returns:
        Chi-squared value
    """
    z = flow_sample["z"]
    m_std = compute_standardized_mag(flow_sample, params)

    # Compute model distance moduli
    cosmo = TrueCosmology(H0=H0, Omega_m=Omega_m, Omega_L=1.0 - Omega_m)
    mu_model = np.array([mu_of_z(zi, cosmo) for zi in z])

    # Model apparent magnitude
    m_model = M_B_fit + mu_model

    # Chi-squared
    residuals = (m_std - m_model) / sigma_total
    return float(np.sum(residuals**2))


def fit_H0_from_flow(
    flow_sample: Dict[str, np.ndarray],
    M_B_fit: float,
    params: SNSystematicParameters11B,
    H0_min: float = 60.0,
    H0_max: float = 80.0,
    n_H0: int = 201,
) -> Tuple[float, float, int]:
    """
    Fit H0 by minimizing chi-squared over a grid.

    Args:
        flow_sample: Hubble flow SN sample
        M_B_fit: Calibrated M_B from calibrators
        params: Systematic parameters
        H0_min, H0_max: H0 grid bounds
        n_H0: Number of grid points

    Returns:
        (H0_best, chi2_min, dof)
    """
    H0_grid = np.linspace(H0_min, H0_max, n_H0)
    chi2_grid = np.array([
        chi2_flow_for_H0(H0, flow_sample, M_B_fit, params)
        for H0 in H0_grid
    ])

    idx_best = np.argmin(chi2_grid)
    H0_best = H0_grid[idx_best]
    chi2_min = chi2_grid[idx_best]
    dof = len(flow_sample["z"]) - 1  # Only H0 is free

    return float(H0_best), float(chi2_min), dof


def run_ladder(
    calib_sample: Dict[str, np.ndarray],
    flow_sample: Dict[str, np.ndarray],
    params: SNSystematicParameters11B,
    cosmo_true: TrueCosmology,
) -> LadderFitResult:
    """
    Run full two-step ladder fit.

    Step 1: Calibrate M_B from calibrators (known distances)
    Step 2: Fit H0 from Hubble flow using that M_B

    Args:
        calib_sample: Calibrator SN sample
        flow_sample: Hubble flow SN sample
        params: Systematic parameters
        cosmo_true: True cosmology (for computing bias)

    Returns:
        LadderFitResult with fitted parameters and bias
    """
    # Step 1: Calibrate M_B
    M_B_fit = calibrate_M_B(calib_sample, params)

    # Step 2: Fit H0
    H0_fit, chi2_flow, dof_flow = fit_H0_from_flow(flow_sample, M_B_fit, params)

    # Compute bias
    delta_H0 = H0_fit - cosmo_true.H0

    return LadderFitResult(
        H0_fit=H0_fit,
        M_B_fit=M_B_fit,
        chi2_flow=chi2_flow,
        dof_flow=dof_flow,
        N_calib=len(calib_sample["z"]),
        N_flow=len(flow_sample["z"]),
        H0_true=cosmo_true.H0,
        delta_H0=delta_H0,
    )
