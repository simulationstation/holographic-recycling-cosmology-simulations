#!/usr/bin/env python3
"""
Naive distance ladder fitter for SN Ia systematics simulation.

This emulates a "standard candle" analysis that ignores:
- Population drift in M_B with redshift
- Metallicity dependence
- Malmquist/selection bias
- Dust R_V variations

The fitter assumes constant M_B and fits (H0, M_B) to minimize chi^2.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize

from .cosmology_baseline import TrueCosmology, mu_of_z


@dataclass
class NaiveFitResult:
    """Result of naive ladder fit."""
    H0_fit: float      # Fitted H0 in km/s/Mpc
    M_B_fit: float     # Fitted absolute magnitude
    chi2: float        # Chi-squared value
    dof: int           # Degrees of freedom
    chi2_per_dof: float  # Reduced chi-squared
    success: bool      # Whether optimization converged


def chi2_naive(
    params_vec: np.ndarray,
    data: Dict[str, np.ndarray],
    Omega_m: float = 0.315,
    sigma_total: float = 0.13,
) -> float:
    """
    Compute chi-squared for naive fit.

    The naive fitter assumes:
    - Constant M_B (no evolution)
    - No correction for metallicity
    - No correction for Malmquist bias
    - Perfect dust correction (though in practice the true extinction
      differs from what fitter assumes due to R_V mismatch)

    Model: m_predicted = M_B + mu(z; H0)

    Args:
        params_vec: [H0, M_B]
        data: Dictionary with 'z' and 'm_obs' arrays
        Omega_m: Fixed matter density
        sigma_total: Total magnitude uncertainty per SN

    Returns:
        Chi-squared value
    """
    H0, M_B = params_vec

    # Ensure H0 is physical
    if H0 <= 0 or H0 > 200:
        return 1e30

    z = data["z"]
    m_obs = data["m_obs"]

    # Create cosmology with fitted H0
    cosmo = TrueCosmology(H0=H0, Omega_m=Omega_m, Omega_L=1.0 - Omega_m)

    # Compute model apparent magnitudes
    mu_model = np.array([mu_of_z(zi, cosmo) for zi in z])
    m_model = M_B + mu_model

    # Chi-squared
    residuals = (m_obs - m_model) / sigma_total
    chi2 = np.sum(residuals**2)

    return chi2


def fit_naive_H0_M_B(
    data: Dict[str, np.ndarray],
    H0_init: float = 70.0,
    M_B_init: float = -19.3,
    Omega_m: float = 0.315,
    sigma_total: float = 0.13,
) -> NaiveFitResult:
    """
    Fit H0 and M_B using naive ladder model.

    Minimizes chi-squared assuming constant M_B with no systematics.

    Args:
        data: Dictionary with 'z' and 'm_obs' arrays
        H0_init: Initial guess for H0
        M_B_init: Initial guess for M_B
        Omega_m: Fixed matter density
        sigma_total: Total magnitude uncertainty per SN

    Returns:
        NaiveFitResult with fitted parameters
    """
    x0 = np.array([H0_init, M_B_init])

    def objective(x):
        return chi2_naive(x, data, Omega_m, sigma_total)

    # Try Nelder-Mead first
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 0.01, 'fatol': 0.1}
    )

    H0_fit, M_B_fit = result.x
    chi2_val = result.fun
    dof = len(data["z"]) - 2  # N - 2 free parameters

    return NaiveFitResult(
        H0_fit=float(H0_fit),
        M_B_fit=float(M_B_fit),
        chi2=float(chi2_val),
        dof=dof,
        chi2_per_dof=float(chi2_val / max(dof, 1)),
        success=result.success,
    )


def fit_with_dust_correction(
    data: Dict[str, np.ndarray],
    R_V_assumed: float = 3.1,
    H0_init: float = 70.0,
    M_B_init: float = -19.3,
    Omega_m: float = 0.315,
    sigma_total: float = 0.13,
) -> NaiveFitResult:
    """
    Fit H0 and M_B with dust correction.

    The fitter "corrects" for dust using the assumed R_V, but if the
    true R_V differs, this introduces a systematic bias.

    Model: m_predicted = M_B + mu(z; H0) + R_V_assumed * E(B-V)

    But: m_obs was generated with R_V_true * E(B-V)

    If R_V_assumed != R_V_true, there's a residual:
    delta_A_V = (R_V_true - R_V_assumed) * E(B-V)

    Args:
        data: Dictionary with 'z', 'm_obs', and 'E_BV' arrays
        R_V_assumed: R_V value assumed by fitter
        H0_init, M_B_init: Initial guesses
        Omega_m: Fixed matter density
        sigma_total: Total magnitude uncertainty

    Returns:
        NaiveFitResult with fitted parameters
    """
    z = data["z"]
    m_obs = data["m_obs"]
    E_BV = data.get("E_BV", np.zeros_like(z))

    # Create dust-corrected apparent magnitudes
    # (subtract the fitter's assumed extinction)
    A_V_assumed = R_V_assumed * E_BV
    m_corrected = m_obs - A_V_assumed

    # Now fit with corrected magnitudes
    corrected_data = {"z": z, "m_obs": m_corrected}

    return fit_naive_H0_M_B(
        corrected_data,
        H0_init=H0_init,
        M_B_init=M_B_init,
        Omega_m=Omega_m,
        sigma_total=sigma_total,
    )


def analytic_H0_shift_estimate(
    delta_m_avg: float,
    z_eff: float = 0.1,
    H0_true: float = 67.5,
) -> float:
    """
    Estimate H0 shift from average magnitude offset.

    For small shifts: delta_m ~ 5 * log10(H0_fit / H0_true)
                    => delta_H0 / H0 ~ delta_m / (5 * log10(e))
                                     ~ delta_m / 2.17

    At low z where mu ~ 5*log10(cz/H0) + const:
    delta_m => delta_H0 / H0 ~ -delta_m / 2.17

    Negative delta_m (brighter SNe) => higher inferred H0.

    Args:
        delta_m_avg: Average magnitude offset (positive = fainter)
        z_eff: Effective redshift of sample
        H0_true: True H0

    Returns:
        Estimated H0 shift in km/s/Mpc
    """
    # delta_mu = 5 * log10(D_L,fit / D_L,true)
    # For D_L ~ c*z/H0 at low z:
    # delta_mu = 5 * log10(H0_true / H0_fit) ~ -5/ln(10) * (delta_H0/H0)
    # So: delta_H0/H0 ~ -delta_m * ln(10)/5 ~ -delta_m * 0.46

    delta_H0_frac = -delta_m_avg * 0.46
    return delta_H0_frac * H0_true
