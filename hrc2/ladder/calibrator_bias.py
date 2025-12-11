#!/usr/bin/env python3
"""
Calibrator distance bias model for SN Ia distance ladder simulation.

Models systematic biases in Cepheid/TRGB distance measurements:
- Global zero-point offset
- Metallicity-dependent distance bias
- Crowding/blending effects in high-mass hosts
- Field-to-field calibration scatter
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .host_population import HostGalaxy


@dataclass
class CalibratorBiasParameters:
    """
    Parameters controlling Cepheid/TRGB distance systematics.

    These biases affect the calibrator distances used to anchor M_B,
    propagating to the final H0 measurement.
    """
    # Global zero-point offset in distance modulus (mag).
    # E.g., +0.05 mag means all calibrator distances are biased too large
    # (objects appear farther than they are -> M_B too bright -> H0 too high).
    delta_mu_global: float = 0.0

    # Metallicity dependence of distance modulus (mag per dex in Z or [O/H]).
    # E.g., +0.05 mag/dex means higher-metallicity hosts get larger μ.
    # This can arise from Cepheid P-L relation metallicity dependence.
    k_mu_Z: float = 0.0

    # Crowding/blending term (mag) applied to high surface brightness / high-mass hosts.
    # Positive = distances biased high in crowded fields.
    delta_mu_crowd: float = 0.0
    logM_crowd_threshold: float = 10.5

    # Random field-to-field calibration scatter (mag).
    # This represents systematic uncertainties that vary per host galaxy.
    sigma_field: float = 0.0


def apply_calibrator_biases(
    mu_true: np.ndarray,
    hosts: List[HostGalaxy],
    bias_params: CalibratorBiasParameters,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Given true distance moduli mu_true for calibrators and their host properties,
    compute biased mu_calib used by the ladder.

    The bias model is:
      mu_biased = mu_true
                  + delta_mu_global
                  + k_mu_Z * Z
                  + delta_mu_crowd * I(logM > logM_crowd_threshold)
                  + field scatter (optional)

    Args:
        mu_true: True distance moduli for calibrator SNe
        hosts: List of HostGalaxy objects for calibrator hosts
        bias_params: CalibratorBiasParameters specifying biases
        rng: Random number generator for field scatter

    Returns:
        Biased distance moduli mu_biased
    """
    N = len(mu_true)
    assert len(hosts) >= N, f"Need at least {N} hosts, got {len(hosts)}"

    mu = np.array(mu_true, dtype=float, copy=True)

    # Extract host properties (only for the calibrators we have)
    Z = np.array([hosts[i].Z for i in range(N)])
    logM = np.array([hosts[i].logM_star for i in range(N)])

    # Global zero-point bias
    if bias_params.delta_mu_global != 0.0:
        mu += bias_params.delta_mu_global

    # Metallicity-dependent bias
    if bias_params.k_mu_Z != 0.0:
        mu += bias_params.k_mu_Z * Z

    # Crowding/blending bias for high-mass hosts
    if bias_params.delta_mu_crowd != 0.0:
        mask_crowd = logM > bias_params.logM_crowd_threshold
        mu[mask_crowd] += bias_params.delta_mu_crowd

    # Field-to-field scatter
    if bias_params.sigma_field > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        # One field-level offset per calibrator host
        field_offsets = rng.normal(0.0, bias_params.sigma_field, size=N)
        mu += field_offsets

    return mu


def compute_total_calibrator_bias(
    hosts: List[HostGalaxy],
    bias_params: CalibratorBiasParameters,
) -> float:
    """
    Compute the mean calibrator distance bias given host properties.

    This is the average bias that will shift M_B_fit and thus H0.

    Args:
        hosts: List of HostGalaxy objects for calibrators
        bias_params: CalibratorBiasParameters

    Returns:
        Mean bias in distance modulus (mag)
    """
    N = len(hosts)
    Z = np.array([h.Z for h in hosts])
    logM = np.array([h.logM_star for h in hosts])

    bias = bias_params.delta_mu_global

    if bias_params.k_mu_Z != 0.0:
        bias += bias_params.k_mu_Z * np.mean(Z)

    if bias_params.delta_mu_crowd != 0.0:
        frac_crowd = np.mean(logM > bias_params.logM_crowd_threshold)
        bias += bias_params.delta_mu_crowd * frac_crowd

    return float(bias)
