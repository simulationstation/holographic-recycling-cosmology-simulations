#!/usr/bin/env python3
"""
SALT2-like SN Ia population model with comprehensive systematics.

Implements:
- SALT2 parameters (m_B, x1, c) with standardization (alpha, beta)
- Population drift in M_B with redshift
- Metallicity-dependent luminosity
- Host-mass step
- Dust law mismatch (R_V_true vs R_V_fit, beta_true vs beta_fit)
- Magnitude-limit selection (Malmquist bias)
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from .cosmology_baseline import TrueCosmology, mu_of_z
from .host_population import HostGalaxy


@dataclass
class SNSystematicParameters11B:
    """
    Parameters controlling SN Ia systematics for SH0ES-like ladder.

    SALT2-like standardization:
        m_B_standardized = m_B + alpha * x1 - beta * c

    True absolute mag includes corrections for:
        M_B = M_B_0 + pop_drift + metallicity + host_mass_step
    """
    # True intrinsic absolute magnitude (z=0, solar metallicity, low-mass host)
    M_B_0: float = -19.3

    # SALT2-like standardization coefficients
    alpha_true: float = 0.14      # True stretch coefficient
    beta_true: float = 3.1        # True color coefficient
    alpha_fit: float = 0.14       # Assumed in fit
    beta_fit: float = 3.1         # Assumed in fit

    # Population drift: M_B(z) = M_B_0 + alpha_pop * (z / z_ref_pop)
    # Positive = fainter at higher z
    alpha_pop: float = 0.0
    z_ref_pop: float = 0.5

    # Metallicity effect: M_B += gamma_Z * Z (mag per dex)
    gamma_Z: float = 0.0

    # Host mass step: ΔM at logM_star > M_step_threshold
    M_step_threshold: float = 10.5
    delta_M_step_true: float = 0.0    # True step (what nature does)
    delta_M_step_fit: float = 0.0     # What fitter assumes (often 0)

    # Dust law
    R_V_true: float = 3.1
    R_V_fit: float = 3.1

    # Magnitude limits for selection
    m_lim_flow: float = 19.5      # Hubble flow limit
    m_lim_calib: float = 18.5     # Calibrator limit (brighter)

    # Intrinsic scatter and measurement noise
    sigma_int: float = 0.10       # Intrinsic scatter in standardized mag
    sigma_meas: float = 0.08      # Measurement uncertainty

    # x1 and c distributions (true underlying population)
    x1_mean: float = 0.0
    x1_sigma: float = 1.0
    c_mean: float = 0.0
    c_sigma: float = 0.08


def simulate_snia_with_hosts(
    z_array: np.ndarray,
    hosts: List[HostGalaxy],
    params: SNSystematicParameters11B,
    cosmo_true: TrueCosmology,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Generate SN Ia observables given host properties and systematics.

    The SALT2 model:
        m_B_obs = M_B_true + mu(z) - alpha_true * x1 + beta_true * c + noise

    Where M_B_true includes:
        M_B_0 + alpha_pop * (z/z_ref) + gamma_Z * Z + delta_M_step * high_mass

    Args:
        z_array: Redshifts
        hosts: List of HostGalaxy objects
        params: Systematic parameters
        cosmo_true: True underlying cosmology
        rng: Random number generator

    Returns:
        Dictionary of SN observables and properties
    """
    N = len(z_array)
    assert len(hosts) == N

    # Draw SALT2 parameters from population
    x1 = rng.normal(params.x1_mean, params.x1_sigma, size=N)
    c = rng.normal(params.c_mean, params.c_sigma, size=N)

    # Compute true distance moduli
    mu_true = np.array([mu_of_z(z, cosmo_true) for z in z_array])

    # Build true absolute magnitudes
    M_true = np.full(N, params.M_B_0)

    # Population drift with redshift
    if params.alpha_pop != 0.0:
        M_true = M_true + params.alpha_pop * (z_array / params.z_ref_pop)

    # Metallicity dependence
    if params.gamma_Z != 0.0:
        host_Z = np.array([h.Z for h in hosts])
        M_true = M_true + params.gamma_Z * host_Z

    # Host mass step (true)
    host_logM = np.array([h.logM_star for h in hosts])
    high_mass_mask = host_logM > params.M_step_threshold
    if params.delta_M_step_true != 0.0:
        M_true[high_mass_mask] = M_true[high_mass_mask] + params.delta_M_step_true

    # Dust extinction from host E(B-V)
    E_BV = np.array([h.E_BV for h in hosts])
    A_V = params.R_V_true * E_BV

    # Construct true apparent magnitude:
    # m_B = M_B + mu - alpha*x1 + beta*c + A_B
    # where A_B ~ A_V / R_V * (R_V + 1) for B-band (simplified: A_B ~ 1.3 * A_V)
    # For SALT2, the color term beta*c absorbs most of this
    # So we add residual dust not captured by beta*c

    # True observed m_B (before noise):
    # m_B_true = M_true + mu_true - alpha_true * x1 + beta_true * c
    # Plus any dust not captured by beta*c (e.g., if R_V differs from assumed)
    m_B_true = M_true + mu_true - params.alpha_true * x1 + params.beta_true * c

    # Add residual dust effect if R_V differs
    # The color c is roughly related to E(B-V) but not perfectly
    # If R_V_true != R_V implied by beta, there's a residual
    # For simplicity, add a small dust-related term
    dust_residual = (params.R_V_true - 3.1) * E_BV * 0.3  # Rough approximation
    m_B_true = m_B_true + dust_residual

    # Add intrinsic + measurement noise
    sigma_tot = np.sqrt(params.sigma_int**2 + params.sigma_meas**2)
    noise = rng.normal(0.0, sigma_tot, size=N)
    m_B_obs = m_B_true + noise

    return {
        "z": z_array,
        "mu_true": mu_true,
        "M_true": M_true,
        "m_B_obs": m_B_obs,
        "m_B_true": m_B_true,
        "x1": x1,
        "c": c,
        "host_logM": host_logM,
        "host_Z": np.array([h.Z for h in hosts]),
        "E_BV": E_BV,
        "high_mass_mask": high_mass_mask,
    }


def apply_magnitude_limit(
    sample_dict: Dict[str, np.ndarray],
    m_lim: float,
) -> Dict[str, np.ndarray]:
    """
    Apply magnitude-limit selection (Malmquist bias).

    Only keeps SNe with m_B_obs <= m_lim.

    Args:
        sample_dict: SN sample dictionary
        m_lim: Magnitude limit (fainter = larger m)

    Returns:
        Filtered sample dictionary
    """
    mask = sample_dict["m_B_obs"] <= m_lim
    return {k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in sample_dict.items()}


def count_after_selection(
    sample_dict: Dict[str, np.ndarray],
    m_lim: float,
) -> int:
    """Count how many SNe pass magnitude limit."""
    return int(np.sum(sample_dict["m_B_obs"] <= m_lim))
