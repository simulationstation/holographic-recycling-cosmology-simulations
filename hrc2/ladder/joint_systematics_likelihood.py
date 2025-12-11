#!/usr/bin/env python3
"""
SIMULATION 15: Joint Hierarchical Systematics Likelihood

This module provides:
1. Data generation for synthetic SH0ES-like ladder data
2. Functions to apply systematics to the ladder
3. Log-likelihood computation
4. Log-posterior for MCMC sampling

The likelihood is based on:
- Calibrator step: SNe with Cepheid/TRGB distances → M_B
- Hubble flow step: SNe with M_B → H0
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .cosmology_baseline import TrueCosmology, mu_of_z
from .snia_salt2 import SNSystematicParameters11B, simulate_snia_with_hosts, apply_magnitude_limit
from .host_population import HostGalaxy, HostPopulationParams, sample_hosts
from .cepheid_calibration import (
    Anchor, CepheidPLParameters, CepheidHost, TRGBParameters,
    get_default_anchors, get_default_cepheid_hosts,
    compute_calibrator_mu_from_chain,
)
from .joint_systematics_model import (
    JointSystematicsParameters,
    JointSystematicsPriors,
    theta_to_params,
    log_prior,
    NDIM,
    PARAM_NAMES,
)


# =============================================================================
# Synthetic Data Structure
# =============================================================================

@dataclass
class SyntheticLadderData:
    """
    Complete synthetic dataset for SH0ES-like H0 inference.

    This is generated once and held fixed during MCMC sampling.
    """
    # True cosmology used to generate data
    H0_true: float
    Omega_m: float

    # Calibrator SNe (have Cepheid/TRGB distances)
    calib_z: np.ndarray
    calib_m_B_obs: np.ndarray
    calib_x1: np.ndarray
    calib_c: np.ndarray
    calib_mu_true: np.ndarray       # True distance moduli
    calib_host_logM: np.ndarray
    calib_host_Z: np.ndarray
    calib_high_mass_mask: np.ndarray

    # Hubble flow SNe
    flow_z: np.ndarray
    flow_m_B_obs: np.ndarray
    flow_x1: np.ndarray
    flow_c: np.ndarray
    flow_mu_true: np.ndarray
    flow_host_logM: np.ndarray
    flow_host_Z: np.ndarray
    flow_high_mass_mask: np.ndarray

    # Cepheid hosts info
    cepheid_hosts: List[CepheidHost]
    anchors: List[Anchor]

    # Observational uncertainties
    sigma_m_B: float
    sigma_mu_calib: float


# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_ladder_data(
    H0_true: float = 67.5,
    Omega_m: float = 0.315,
    N_calib: int = 40,
    N_flow: int = 200,
    z_min_flow: float = 0.023,
    z_max_flow: float = 0.15,
    sigma_m_B: float = 0.13,
    sigma_mu_calib: float = 0.05,
    seed: int = 42,
) -> SyntheticLadderData:
    """
    Generate a complete synthetic SH0ES-like ladder dataset.

    The data is generated with no systematic biases (truth),
    and the likelihood will evaluate how well various H0 + systematics
    combinations fit this data.

    Args:
        H0_true: True Hubble constant
        Omega_m: Matter density
        N_calib: Number of calibrator SNe
        N_flow: Number of Hubble flow SNe
        z_min_flow, z_max_flow: Redshift range for flow sample
        sigma_m_B: Total uncertainty on m_B
        sigma_mu_calib: Uncertainty on calibrator mu from Cepheids
        seed: Random seed

    Returns:
        SyntheticLadderData
    """
    rng = np.random.default_rng(seed)

    cosmo_true = TrueCosmology(H0=H0_true, Omega_m=Omega_m, Omega_L=1.0 - Omega_m)

    # Get anchors and Cepheid hosts
    anchors = get_default_anchors()
    cepheid_hosts = get_default_cepheid_hosts(H0_true)

    # SN parameters (no systematics for truth generation)
    sn_params = SNSystematicParameters11B()

    # =========================================================================
    # Generate Calibrator SNe
    # =========================================================================

    # Distribute calibrators among Cepheid host galaxies
    n_hosts = len(cepheid_hosts)
    n_per_host = max(1, N_calib // n_hosts)

    calib_z_list = []
    calib_mu_true_list = []
    calib_hosts_list = []

    for host in cepheid_hosts:
        for _ in range(n_per_host):
            # Calibrator redshifts are very low (local)
            z = 0.001 + rng.exponential(0.003)
            calib_z_list.append(z)
            calib_mu_true_list.append(host.mu_true)
            calib_hosts_list.append(HostGalaxy(
                logM_star=host.logM_star,
                Z=host.Z,
                E_BV=rng.exponential(0.03),
            ))

    calib_z = np.array(calib_z_list)[:N_calib]
    calib_mu_true = np.array(calib_mu_true_list)[:N_calib]
    calib_hosts = calib_hosts_list[:N_calib]

    # Generate calibrator SN observables
    calib_sample = simulate_snia_with_hosts(
        calib_z, calib_hosts, sn_params, cosmo_true, rng
    )
    # The SN simulation computes mu_true from cosmology, but calibrators have
    # distances from Cepheid hosts. We need to adjust m_B_obs to be consistent
    # with the Cepheid-host mu_true.
    # Original: m_B_obs = M_true + mu_from_cosmo - alpha*x1 + beta*c + noise
    # We want: m_B_obs = M_true + mu_from_cepheid - alpha*x1 + beta*c + noise
    # So: m_B_obs_corrected = m_B_obs - (mu_from_cosmo - mu_from_cepheid)
    mu_correction = calib_sample["mu_true"] - calib_mu_true
    calib_sample["m_B_obs"] = calib_sample["m_B_obs"] - mu_correction
    calib_sample["m_B_true"] = calib_sample["m_B_true"] - mu_correction
    calib_sample["mu_true"] = calib_mu_true

    # =========================================================================
    # Generate Hubble Flow SNe
    # =========================================================================

    # Flow redshifts: volume-weighted distribution
    flow_z_raw = rng.uniform(z_min_flow, z_max_flow, N_flow * 3)
    # Volume weighting ~ z^2
    weights = (flow_z_raw / z_max_flow) ** 2
    accept = rng.random(len(flow_z_raw)) < weights
    flow_z = flow_z_raw[accept][:N_flow]

    # Generate flow hosts
    pop_params = HostPopulationParams()
    flow_hosts = sample_hosts(len(flow_z), "flow", pop_params, rng)

    # Generate flow SN observables
    flow_sample = simulate_snia_with_hosts(
        flow_z, flow_hosts, sn_params, cosmo_true, rng
    )

    # Apply magnitude limit for Malmquist bias
    flow_sample = apply_magnitude_limit(flow_sample, m_lim=sn_params.m_lim_flow)

    return SyntheticLadderData(
        H0_true=H0_true,
        Omega_m=Omega_m,
        calib_z=calib_z,
        calib_m_B_obs=calib_sample["m_B_obs"],
        calib_x1=calib_sample["x1"],
        calib_c=calib_sample["c"],
        calib_mu_true=calib_mu_true,
        calib_host_logM=calib_sample["host_logM"],
        calib_host_Z=calib_sample["host_Z"],
        calib_high_mass_mask=calib_sample["high_mass_mask"],
        flow_z=flow_sample["z"],
        flow_m_B_obs=flow_sample["m_B_obs"],
        flow_x1=flow_sample["x1"],
        flow_c=flow_sample["c"],
        flow_mu_true=flow_sample["mu_true"],
        flow_host_logM=flow_sample["host_logM"],
        flow_host_Z=flow_sample["host_Z"],
        flow_high_mass_mask=flow_sample["high_mass_mask"],
        cepheid_hosts=cepheid_hosts,
        anchors=anchors,
        sigma_m_B=sigma_m_B,
        sigma_mu_calib=sigma_mu_calib,
    )


# =============================================================================
# Likelihood Computation
# =============================================================================

def compute_biased_calibrator_mu(
    data: SyntheticLadderData,
    params: JointSystematicsParameters,
) -> np.ndarray:
    """
    Compute biased calibrator distance moduli given systematics.

    The systematics affect:
    1. Anchor distances (delta_mu_anchor)
    2. Cepheid PL calibration (delta_M_W0)
    3. Crowding (delta_mu_crowd)
    4. Instrument effects (delta_ZP_inst, delta_color_inst)

    Returns:
        Biased mu_calib array
    """
    # Start with true calibrator mu
    mu_calib = data.calib_mu_true.copy()

    # Apply Cepheid/anchor systematics
    # delta_M_W0: PL zero-point bias (brighter M_W0 → smaller mu)
    mu_calib -= params.delta_M_W0

    # delta_mu_anchor: anchor distance bias (direct shift)
    mu_calib += params.delta_mu_anchor

    # delta_mu_crowd: crowding bias (typically positive, overestimates distance)
    mu_calib += params.delta_mu_crowd

    # Instrument systematics (smaller effect, applies to Cepheid photometry)
    mu_calib += params.delta_ZP_inst

    return mu_calib


def compute_standardized_mag_with_systematics(
    m_B_obs: np.ndarray,
    x1: np.ndarray,
    c: np.ndarray,
    host_logM: np.ndarray,
    host_Z: np.ndarray,
    z: np.ndarray,
    high_mass_mask: np.ndarray,
    params: JointSystematicsParameters,
    alpha_fit: float = 0.14,
    beta_fit: float = 3.1,
    delta_M_step_fit: float = 0.0,
    M_step_threshold: float = 10.5,
) -> np.ndarray:
    """
    Compute standardized magnitudes including SN systematics biases.

    The fitter computes:
        m_std = m_B + alpha_fit * x1 - beta_fit * c - delta_M_step_fit * high_mass

    But nature has:
        m_true = m_B + alpha_true * x1 - beta_true * c - delta_M_step_true * high_mass
                 + population_drift + metallicity_effect

    So the difference introduces bias.

    Returns:
        Standardized apparent magnitudes with systematic biases
    """
    # Base standardization (what fitter does)
    m_std = m_B_obs + alpha_fit * x1 - beta_fit * c

    # Apply host mass step correction if fitter uses one
    if delta_M_step_fit != 0.0:
        m_std[high_mass_mask] -= delta_M_step_fit

    # Now add biases from systematics that fitter doesn't know about:

    # Color-law mismatch: delta_beta * c
    # If beta_true > beta_fit, red SNe appear fainter than model expects
    m_std += params.delta_beta * c

    # Host mass step mismatch: delta_M_step * high_mass
    # This is the RESIDUAL step after any fitter correction
    m_std[high_mass_mask] += params.delta_M_step

    # Population drift: alpha_pop * (z / z_ref)
    z_ref = 0.5
    m_std += params.alpha_pop * (z / z_ref)

    # Metallicity effect: gamma_Z * host_Z
    m_std += params.gamma_Z * host_Z

    return m_std


def compute_log_likelihood(
    data: SyntheticLadderData,
    params: JointSystematicsParameters,
) -> float:
    """
    Compute log-likelihood of the data given H0 and systematics.

    The likelihood has two components:
    1. Calibrator chi-squared: how well calibrators constrain M_B
    2. Flow chi-squared: how well flow SNe constrain H0 given M_B

    Returns:
        Log-likelihood value
    """
    H0 = params.H0
    sigma_m = data.sigma_m_B

    # =========================================================================
    # Step 1: Calibrate M_B from calibrators
    # =========================================================================

    # Get biased calibrator distances from Cepheid systematics
    mu_calib_biased = compute_biased_calibrator_mu(data, params)

    # Get standardized calibrator magnitudes with SN systematics
    m_std_calib = compute_standardized_mag_with_systematics(
        m_B_obs=data.calib_m_B_obs,
        x1=data.calib_x1,
        c=data.calib_c,
        host_logM=data.calib_host_logM,
        host_Z=data.calib_host_Z,
        z=data.calib_z,
        high_mass_mask=data.calib_high_mass_mask,
        params=params,
    )

    # Estimate M_B from calibrators
    # m_std = M_B + mu_calib => M_B = m_std - mu_calib
    M_B_estimates = m_std_calib - mu_calib_biased
    M_B_fit = np.mean(M_B_estimates)

    # Calibrator chi-squared (scatter around mean M_B)
    sigma_calib = np.sqrt(sigma_m**2 + data.sigma_mu_calib**2)
    residuals_calib = M_B_estimates - M_B_fit
    chi2_calib = np.sum((residuals_calib / sigma_calib) ** 2)

    # =========================================================================
    # Step 2: Fit H0 from Hubble flow
    # =========================================================================

    # Get standardized flow magnitudes with SN systematics
    m_std_flow = compute_standardized_mag_with_systematics(
        m_B_obs=data.flow_m_B_obs,
        x1=data.flow_x1,
        c=data.flow_c,
        host_logM=data.flow_host_logM,
        host_Z=data.flow_host_Z,
        z=data.flow_z,
        high_mass_mask=data.flow_high_mass_mask,
        params=params,
    )

    # Model prediction: m_std = M_B_fit + mu(z; H0)
    cosmo = TrueCosmology(H0=H0, Omega_m=data.Omega_m, Omega_L=1.0 - data.Omega_m)
    mu_model = np.array([mu_of_z(z, cosmo) for z in data.flow_z])
    m_model = M_B_fit + mu_model

    # Flow chi-squared
    residuals_flow = (m_std_flow - m_model) / sigma_m
    chi2_flow = np.sum(residuals_flow ** 2)

    # =========================================================================
    # Total log-likelihood
    # =========================================================================

    # Gaussian likelihood: -0.5 * chi2 (ignoring normalization constants)
    log_like = -0.5 * (chi2_calib + chi2_flow)

    return float(log_like)


def log_posterior(
    theta: np.ndarray,
    data: SyntheticLadderData,
    priors: JointSystematicsPriors,
) -> float:
    """
    Compute log-posterior for MCMC sampling.

    log p(theta | data) = log p(data | theta) + log p(theta) + const

    Args:
        theta: Flat parameter vector of length NDIM
        data: Synthetic ladder data
        priors: Prior hyperparameters

    Returns:
        Log-posterior value (or -inf if outside prior bounds)
    """
    # Unpack theta to parameters
    params = theta_to_params(theta)

    # Evaluate prior
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf

    # Evaluate likelihood
    ll = compute_log_likelihood(data, params)
    if not np.isfinite(ll):
        return -np.inf

    return float(lp + ll)


# =============================================================================
# Utility Functions for MCMC
# =============================================================================

def test_log_posterior(
    data: SyntheticLadderData,
    priors: JointSystematicsPriors,
    H0_test: float = 70.0,
) -> float:
    """
    Test log_posterior at a simple point to verify it works.

    Returns log-posterior value at H0_test with zero systematics.
    """
    theta = np.array([
        H0_test,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
    ])
    return log_posterior(theta, data, priors)


def compute_chi2_at_point(
    data: SyntheticLadderData,
    H0: float,
    systematics: Optional[JointSystematicsParameters] = None,
) -> Tuple[float, float, float]:
    """
    Compute chi-squared components at a specific H0 and systematics.

    Returns:
        (chi2_total, chi2_calib, chi2_flow)
    """
    if systematics is None:
        systematics = JointSystematicsParameters(
            H0=H0,
            alpha_pop=0.0,
            gamma_Z=0.0,
            delta_M_step=0.0,
            delta_beta=0.0,
            delta_M_W0=0.0,
            delta_mu_anchor=0.0,
            delta_mu_crowd=0.0,
            delta_ZP_inst=0.0,
            delta_color_inst=0.0,
        )

    sigma_m = data.sigma_m_B

    # Calibrators
    mu_calib_biased = compute_biased_calibrator_mu(data, systematics)
    m_std_calib = compute_standardized_mag_with_systematics(
        m_B_obs=data.calib_m_B_obs,
        x1=data.calib_x1,
        c=data.calib_c,
        host_logM=data.calib_host_logM,
        host_Z=data.calib_host_Z,
        z=data.calib_z,
        high_mass_mask=data.calib_high_mass_mask,
        params=systematics,
    )
    M_B_estimates = m_std_calib - mu_calib_biased
    M_B_fit = np.mean(M_B_estimates)

    sigma_calib = np.sqrt(sigma_m**2 + data.sigma_mu_calib**2)
    chi2_calib = np.sum(((M_B_estimates - M_B_fit) / sigma_calib) ** 2)

    # Flow
    m_std_flow = compute_standardized_mag_with_systematics(
        m_B_obs=data.flow_m_B_obs,
        x1=data.flow_x1,
        c=data.flow_c,
        host_logM=data.flow_host_logM,
        host_Z=data.flow_host_Z,
        z=data.flow_z,
        high_mass_mask=data.flow_high_mass_mask,
        params=systematics,
    )

    cosmo = TrueCosmology(H0=H0, Omega_m=data.Omega_m, Omega_L=1.0 - data.Omega_m)
    mu_model = np.array([mu_of_z(z, cosmo) for z in data.flow_z])
    m_model = M_B_fit + mu_model
    chi2_flow = np.sum(((m_std_flow - m_model) / sigma_m) ** 2)

    return chi2_calib + chi2_flow, chi2_calib, chi2_flow


def find_best_fit_H0_no_systematics(
    data: SyntheticLadderData,
    H0_min: float = 60.0,
    H0_max: float = 80.0,
    n_grid: int = 101,
) -> Tuple[float, float]:
    """
    Find best-fit H0 with no systematics (baseline case).

    Returns:
        (H0_best, chi2_min)
    """
    H0_grid = np.linspace(H0_min, H0_max, n_grid)
    chi2_values = np.array([
        compute_chi2_at_point(data, H0)[0] for H0 in H0_grid
    ])

    idx_best = np.argmin(chi2_values)
    return H0_grid[idx_best], chi2_values[idx_best]
