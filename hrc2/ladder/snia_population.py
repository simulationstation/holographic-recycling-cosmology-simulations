#!/usr/bin/env python3
"""
SN Ia population model with systematics for distance ladder simulation.

Implements:
- Population drift in M_B with redshift
- Metallicity-dependent luminosity
- Dust extinction with possible R_V miscalibration
- Malmquist/selection bias
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .cosmology_baseline import TrueCosmology, mu_of_z


@dataclass
class SNSystematicParameters:
    """
    Parameters controlling SN Ia systematics.

    All magnitude offsets are in the sense that POSITIVE values make
    SNe FAINTER (larger m).
    """
    # Intrinsic M_B at z=0, solar metallicity
    M_B_0: float = -19.3

    # Population drift: M_B(z) = M_B_0 + alpha_pop * (z / z_ref_pop)
    # Positive alpha_pop means SNe are intrinsically fainter at higher z
    alpha_pop: float = 0.0
    z_ref_pop: float = 0.5

    # Metallicity dependence: M_B += gamma_Z * Z
    # where Z = log10(Z_env / Z_solar)
    # Positive gamma_Z means metal-rich environment -> fainter
    gamma_Z: float = 0.0

    # Dust law parameters
    R_V_true: float = 3.1   # True R_V in nature
    R_V_fit: float = 3.1    # R_V assumed by fitter

    # Malmquist bias: preferential selection of brighter SNe at high z
    # delta_m_malm < 0 means observed sample is brighter than average
    # (we observe brighter-than-average SNe)
    delta_m_malm: float = 0.0   # magnitude offset for z > z_malm
    z_malm: float = 0.1         # redshift above which bias applies

    # Scatter and noise
    sigma_int: float = 0.10     # Intrinsic scatter (mag)
    sigma_meas: float = 0.08    # Measurement uncertainty (mag)


def true_M_B(z: float, Z: float, params: SNSystematicParameters) -> float:
    """
    True absolute magnitude as a function of redshift and metallicity.

    Args:
        z: Redshift
        Z: log10(metallicity / solar metallicity)
        params: Systematic parameters

    Returns:
        True absolute magnitude M_B
    """
    M = params.M_B_0

    # Population drift with redshift
    if params.alpha_pop != 0.0:
        M += params.alpha_pop * (z / params.z_ref_pop)

    # Metallicity dependence
    if params.gamma_Z != 0.0:
        M += params.gamma_Z * Z

    return M


def dust_extinction_true(E_BV: float, params: SNSystematicParameters) -> float:
    """
    True dust extinction A_V in magnitudes.

    Args:
        E_BV: Color excess E(B-V)
        params: Systematic parameters

    Returns:
        A_V = R_V_true * E(B-V)
    """
    return params.R_V_true * E_BV


def dust_extinction_fit(E_BV: float, params: SNSystematicParameters) -> float:
    """
    Dust extinction assumed by the fitter.

    Args:
        E_BV: Color excess E(B-V)
        params: Systematic parameters

    Returns:
        A_V_fit = R_V_fit * E(B-V)
    """
    return params.R_V_fit * E_BV


def malmquist_bias(z: float, params: SNSystematicParameters) -> float:
    """
    Malmquist/selection bias magnitude offset.

    Returns a negative value (brightening) for z > z_malm.

    Args:
        z: Redshift
        params: Systematic parameters

    Returns:
        Magnitude offset (negative = brighter selection)
    """
    if z >= params.z_malm:
        return -abs(params.delta_m_malm)  # Negative = brighter
    return 0.0


def simulate_snia_sample(
    z_array: np.ndarray,
    params: SNSystematicParameters,
    cosmo: TrueCosmology,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Simulate a synthetic SN Ia sample with systematics.

    Args:
        z_array: Array of redshifts
        params: Systematic parameters
        cosmo: True cosmology
        rng: Random number generator

    Returns:
        Dictionary containing:
        - z: Redshifts
        - mu_true: True distance moduli
        - M_B_true: True absolute magnitudes
        - m_obs: Observed apparent magnitudes
        - Z: Metallicities
        - E_BV: Color excesses
        - A_V_true: True dust extinctions
        - delta_m_malm: Malmquist bias values
        - m_true: True apparent magnitudes (before noise)
    """
    N = len(z_array)

    # Generate metallicity distribution
    # Z = log10(Z_env / Z_solar), typically -0.3 to +0.3
    Z = rng.normal(0.0, 0.2, size=N)

    # Generate dust distribution
    # E(B-V) typically 0-0.2 with mean ~0.05
    E_BV = np.clip(rng.exponential(0.05, size=N), 0.0, 0.5)

    # Compute true distance moduli
    mu_true = np.array([mu_of_z(z, cosmo) for z in z_array])

    # Compute true absolute magnitudes (with population effects)
    M_B_true = np.array([true_M_B(z, z_metal, params)
                         for z, z_metal in zip(z_array, Z)])

    # True dust extinction
    A_V_true = dust_extinction_true(E_BV, params)

    # True apparent magnitude (before selection bias)
    m_true = M_B_true + mu_true + A_V_true

    # Malmquist bias (makes observed sample brighter)
    delta_malm = np.array([malmquist_bias(z, params) for z in z_array])

    # Apply Malmquist bias to get "selected" apparent magnitude
    m_selected = m_true + delta_malm

    # Add intrinsic scatter and measurement noise
    total_scatter = np.sqrt(params.sigma_int**2 + params.sigma_meas**2)
    noise = rng.normal(0.0, total_scatter, size=N)
    m_obs = m_selected + noise

    return {
        "z": z_array,
        "mu_true": mu_true,
        "M_B_true": M_B_true,
        "m_obs": m_obs,
        "Z": Z,
        "E_BV": E_BV,
        "A_V_true": A_V_true,
        "delta_m_malm": delta_malm,
        "m_true": m_true,
        "m_selected": m_selected,
    }


def generate_realistic_z_distribution(
    n_total: int = 300,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a realistic redshift distribution for distance ladder SNe.

    Mimics the distribution used in SH0ES-like analyses:
    - Low-z anchors: z ~ 0.01-0.03
    - Hubble flow: z ~ 0.03-0.08
    - Higher-z extension: z ~ 0.08-0.2

    Args:
        n_total: Total number of SNe
        rng: Random number generator

    Returns:
        Array of redshifts
    """
    if rng is None:
        rng = np.random.default_rng()

    # Split into three bins
    n_low = int(0.15 * n_total)      # ~15% in anchors
    n_mid = int(0.35 * n_total)      # ~35% in Hubble flow
    n_high = n_total - n_low - n_mid  # ~50% in extension

    z_low = rng.uniform(0.01, 0.03, size=n_low)
    z_mid = rng.uniform(0.03, 0.08, size=n_mid)
    z_high = rng.uniform(0.08, 0.20, size=n_high)

    z_all = np.concatenate([z_low, z_mid, z_high])
    rng.shuffle(z_all)

    return z_all
