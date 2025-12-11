#!/usr/bin/env python3
"""
SIMULATION 15: Joint Hierarchical Systematics Model

This module defines the parameter space and priors for a hierarchical Bayesian
model that combines all astrophysical and calibration systematics:

1. SN Ia systematics (SIM 11/11B/11C): population drift, metallicity, host-mass step, color-law
2. Cepheid/TRGB systematics (SIM 12): PL zero-point, anchor biases, crowding
3. Instrument systematics (SIM 13): HST vs JWST zero-point and color terms

The goal is to infer P(H0 | data, LCDM, systematics) with realistic priors.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


# =============================================================================
# Prior Definitions
# =============================================================================

@dataclass
class JointSystematicsPriors:
    """
    Hyperparameters defining Gaussian prior widths on systematics.

    These are based on realistic ranges from SIM 11-14 and literature.
    All sigmas are 1-sigma Gaussian prior widths.
    """
    # H0 prior: uniform in [H0_min, H0_max]
    H0_min: float = 50.0   # km/s/Mpc
    H0_max: float = 90.0   # km/s/Mpc

    # =========================================================================
    # SN Ia systematics (from SIM 11/11B/11C)
    # =========================================================================

    # Population drift: M_B(z) = M_B_0 + alpha_pop * (z / z_ref)
    # Prior: alpha_pop ~ N(0, sigma_alpha_pop^2)
    # SH0ES samples z_mean ~ 0.03, so delta_M ~ 0.1 mag for alpha_pop ~ 0.05
    sigma_alpha_pop: float = 0.05  # mag per z/z_ref

    # Metallicity effect: M_B += gamma_Z * Z
    # Prior: gamma_Z ~ N(0, sigma_gamma_Z^2)
    # Typical host Z range is ±0.3 dex, so effect ~ 0.015 mag
    sigma_gamma_Z: float = 0.05  # mag per dex

    # Host mass step: true value differs from fitted
    # delta_M_step = M_step_true - M_step_fit
    # Prior: delta_M_step ~ N(0, sigma_delta_M_step^2)
    sigma_delta_M_step: float = 0.03  # mag

    # Color-law mismatch: delta_beta = beta_true - beta_fit
    # Prior: delta_beta ~ N(0, sigma_delta_beta^2)
    # Standard beta ~ 3.1, uncertainty ~ 0.3
    sigma_delta_beta: float = 0.3

    # =========================================================================
    # Cepheid/TRGB calibration systematics (from SIM 12)
    # =========================================================================

    # PL zero-point shift: M_W0_true - M_W0_fit
    # Prior: delta_M_W0 ~ N(0, sigma_delta_M_W0^2)
    sigma_delta_M_W0: float = 0.03  # mag

    # Anchor distance bias (global offset applied to all anchors)
    # Prior: delta_mu_anchor ~ N(0, sigma_delta_mu_anchor^2)
    sigma_delta_mu_anchor: float = 0.02  # mag

    # Crowding bias in host Cepheid fields
    # Prior: delta_mu_crowd ~ N(0, sigma_delta_mu_crowd^2)
    sigma_delta_mu_crowd: float = 0.03  # mag

    # TRGB zero-point offset (if TRGB is used)
    sigma_delta_M_TRGB: float = 0.03  # mag

    # =========================================================================
    # Instrument systematics (from SIM 13)
    # =========================================================================

    # Instrumental zero-point offset (HST vs JWST)
    # Prior: delta_ZP_inst ~ N(0, sigma_delta_ZP_inst^2)
    sigma_delta_ZP_inst: float = 0.02  # mag

    # Instrumental color term difference
    # Prior: delta_color_inst ~ N(0, sigma_delta_color_inst^2)
    sigma_delta_color_inst: float = 0.02  # mag/mag


# =============================================================================
# Model Parameters
# =============================================================================

@dataclass
class JointSystematicsParameters:
    """
    Complete set of model parameters for joint hierarchical inference.

    These are the free parameters that the MCMC samples.
    """
    # Cosmological parameter of interest
    H0: float  # km/s/Mpc

    # SN Ia systematics
    alpha_pop: float       # Population drift coefficient
    gamma_Z: float         # Metallicity dependence
    delta_M_step: float    # Host mass step bias
    delta_beta: float      # Color-law mismatch

    # Cepheid/TRGB systematics
    delta_M_W0: float      # PL zero-point bias
    delta_mu_anchor: float # Anchor distance bias
    delta_mu_crowd: float  # Crowding bias

    # Instrument systematics
    delta_ZP_inst: float   # Zero-point offset
    delta_color_inst: float # Color term difference

    # Optional TRGB offset
    delta_M_TRGB: Optional[float] = None


# Parameter ordering for flat theta vector
PARAM_NAMES = [
    "H0",
    "alpha_pop",
    "gamma_Z",
    "delta_M_step",
    "delta_beta",
    "delta_M_W0",
    "delta_mu_anchor",
    "delta_mu_crowd",
    "delta_ZP_inst",
    "delta_color_inst",
]

NDIM = len(PARAM_NAMES)


def theta_to_params(theta: np.ndarray) -> JointSystematicsParameters:
    """Convert flat theta vector to JointSystematicsParameters."""
    return JointSystematicsParameters(
        H0=theta[0],
        alpha_pop=theta[1],
        gamma_Z=theta[2],
        delta_M_step=theta[3],
        delta_beta=theta[4],
        delta_M_W0=theta[5],
        delta_mu_anchor=theta[6],
        delta_mu_crowd=theta[7],
        delta_ZP_inst=theta[8],
        delta_color_inst=theta[9],
        delta_M_TRGB=None,
    )


def params_to_theta(params: JointSystematicsParameters) -> np.ndarray:
    """Convert JointSystematicsParameters to flat theta vector."""
    return np.array([
        params.H0,
        params.alpha_pop,
        params.gamma_Z,
        params.delta_M_step,
        params.delta_beta,
        params.delta_M_W0,
        params.delta_mu_anchor,
        params.delta_mu_crowd,
        params.delta_ZP_inst,
        params.delta_color_inst,
    ])


# =============================================================================
# Prior Sampling and Evaluation
# =============================================================================

def sample_prior(
    priors: JointSystematicsPriors,
    rng: np.random.Generator,
) -> JointSystematicsParameters:
    """
    Draw a sample from the prior distribution.

    Useful for prior predictive checks and initializing MCMC walkers.
    """
    return JointSystematicsParameters(
        H0=rng.uniform(priors.H0_min, priors.H0_max),
        alpha_pop=rng.normal(0.0, priors.sigma_alpha_pop),
        gamma_Z=rng.normal(0.0, priors.sigma_gamma_Z),
        delta_M_step=rng.normal(0.0, priors.sigma_delta_M_step),
        delta_beta=rng.normal(0.0, priors.sigma_delta_beta),
        delta_M_W0=rng.normal(0.0, priors.sigma_delta_M_W0),
        delta_mu_anchor=rng.normal(0.0, priors.sigma_delta_mu_anchor),
        delta_mu_crowd=rng.normal(0.0, priors.sigma_delta_mu_crowd),
        delta_ZP_inst=rng.normal(0.0, priors.sigma_delta_ZP_inst),
        delta_color_inst=rng.normal(0.0, priors.sigma_delta_color_inst),
        delta_M_TRGB=None,
    )


def log_prior(
    params: JointSystematicsParameters,
    priors: JointSystematicsPriors,
) -> float:
    """
    Compute log prior probability.

    H0: Uniform in [H0_min, H0_max]
    All nuisance parameters: Gaussian N(0, sigma^2)

    Returns:
        log p(params)
    """
    # H0: uniform prior
    if not (priors.H0_min <= params.H0 <= priors.H0_max):
        return -np.inf

    def gauss_log_prob(x: float, sigma: float) -> float:
        """Log of unnormalized Gaussian."""
        return -0.5 * (x / sigma) ** 2

    lp = 0.0

    # SN systematics
    lp += gauss_log_prob(params.alpha_pop, priors.sigma_alpha_pop)
    lp += gauss_log_prob(params.gamma_Z, priors.sigma_gamma_Z)
    lp += gauss_log_prob(params.delta_M_step, priors.sigma_delta_M_step)
    lp += gauss_log_prob(params.delta_beta, priors.sigma_delta_beta)

    # Cepheid systematics
    lp += gauss_log_prob(params.delta_M_W0, priors.sigma_delta_M_W0)
    lp += gauss_log_prob(params.delta_mu_anchor, priors.sigma_delta_mu_anchor)
    lp += gauss_log_prob(params.delta_mu_crowd, priors.sigma_delta_mu_crowd)

    # Instrument systematics
    lp += gauss_log_prob(params.delta_ZP_inst, priors.sigma_delta_ZP_inst)
    lp += gauss_log_prob(params.delta_color_inst, priors.sigma_delta_color_inst)

    # Optional TRGB
    if params.delta_M_TRGB is not None:
        lp += gauss_log_prob(params.delta_M_TRGB, priors.sigma_delta_M_TRGB)

    return float(lp)


def get_initial_theta(
    H0_init: float = 70.0,
    priors: Optional[JointSystematicsPriors] = None,
) -> np.ndarray:
    """
    Get initial theta vector for MCMC.

    Starts H0 at specified value, all nuisance parameters at 0.
    """
    return np.array([
        H0_init,  # H0
        0.0,      # alpha_pop
        0.0,      # gamma_Z
        0.0,      # delta_M_step
        0.0,      # delta_beta
        0.0,      # delta_M_W0
        0.0,      # delta_mu_anchor
        0.0,      # delta_mu_crowd
        0.0,      # delta_ZP_inst
        0.0,      # delta_color_inst
    ])


def get_walker_initialization_scales() -> np.ndarray:
    """
    Get scales for scattering walkers around initial point.

    These are small perturbations to initialize walkers in a tight ball.
    """
    return np.array([
        1.0,    # H0: ±1 km/s/Mpc scatter
        0.01,   # alpha_pop
        0.01,   # gamma_Z
        0.005,  # delta_M_step
        0.05,   # delta_beta
        0.005,  # delta_M_W0
        0.005,  # delta_mu_anchor
        0.005,  # delta_mu_crowd
        0.005,  # delta_ZP_inst
        0.005,  # delta_color_inst
    ])


# =============================================================================
# Utility Functions
# =============================================================================

def compute_delta_M_B_from_systematics(
    params: JointSystematicsParameters,
    z_mean: float = 0.03,
    Z_mean: float = 0.0,
    frac_high_mass: float = 0.5,
) -> float:
    """
    Compute the net shift in M_B from SN systematics.

    This helps understand how H0 shifts map to magnitude shifts.

    Args:
        params: Systematic parameters
        z_mean: Mean redshift of calibrator SNe
        Z_mean: Mean metallicity of hosts
        frac_high_mass: Fraction of SNe in high-mass hosts

    Returns:
        Net shift in M_B (positive = fainter, H0 increases)
    """
    delta_M = 0.0

    # Population drift
    z_ref = 0.5
    delta_M += params.alpha_pop * (z_mean / z_ref)

    # Metallicity
    delta_M += params.gamma_Z * Z_mean

    # Host mass step (net effect depends on calibrator vs flow mix)
    # If calibrators have different mass distribution than flow, there's a bias
    delta_M += params.delta_M_step * (frac_high_mass - 0.5)

    return delta_M


def compute_delta_mu_from_cepheid_systematics(
    params: JointSystematicsParameters,
) -> float:
    """
    Compute the net shift in calibrator mu from Cepheid systematics.

    Returns:
        Net shift in mu_calib (positive = farther, M_B more negative, H0 increases)
    """
    delta_mu = 0.0

    # PL zero-point: if fitter assumes brighter M_W0 than true,
    # distances are underestimated (mu_fit < mu_true)
    # delta_M_W0 = M_W0_fit - M_W0_true (our convention)
    # So delta_mu = -delta_M_W0
    delta_mu -= params.delta_M_W0

    # Anchor bias: direct shift
    delta_mu += params.delta_mu_anchor

    # Crowding: typically makes stars appear brighter, underestimating distance
    delta_mu += params.delta_mu_crowd

    return delta_mu


def compute_delta_H0_from_delta_M_B(delta_M_B: float, H0_base: float = 67.5) -> float:
    """
    Convert shift in M_B to shift in H0.

    H0 ~ 10^(M_B / 5) (simplified)
    delta_H0 / H0 ~ 0.2 * ln(10) * delta_M_B ~ 0.46 * delta_M_B

    More precisely: delta_H0 ~ H0 * delta_M_B * 0.2 * ln(10)
    """
    # From distance modulus: mu = 5 log10(D) + 25
    # m = M + mu => D ~ 10^((m - M)/5)
    # H0 ~ 1/D at fixed z => H0 ~ 10^(M/5)
    # d(H0)/H0 = d(M) * ln(10) / 5 ~ 0.46 * d(M)
    return H0_base * delta_M_B * np.log(10) / 5


def summarize_prior_contributions(priors: JointSystematicsPriors) -> dict:
    """
    Summarize the prior 1-sigma contributions to H0 from each systematic.

    Returns dict mapping parameter name to expected H0 shift at 1-sigma.
    """
    H0_base = 70.0

    contributions = {}

    # SN systematics -> M_B shifts -> H0 shifts
    # alpha_pop: at z_mean ~ 0.03, z_ref = 0.5, delta_M ~ sigma * 0.03/0.5 = 0.06*sigma
    contributions["alpha_pop"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_alpha_pop * 0.06, H0_base
    )

    contributions["gamma_Z"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_gamma_Z * 0.1, H0_base  # Assume Z range ~ 0.1 dex
    )

    contributions["delta_M_step"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_M_step, H0_base
    )

    # delta_beta: affects standardization, ~ sigma_c * delta_beta ~ 0.08 * 0.3 ~ 0.024 mag
    contributions["delta_beta"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_beta * 0.08, H0_base
    )

    # Cepheid systematics -> mu shifts -> M_B shifts -> H0 shifts
    contributions["delta_M_W0"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_M_W0, H0_base
    )

    contributions["delta_mu_anchor"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_mu_anchor, H0_base
    )

    contributions["delta_mu_crowd"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_mu_crowd, H0_base
    )

    # Instrument systematics (smaller effect)
    contributions["delta_ZP_inst"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_ZP_inst, H0_base
    )

    contributions["delta_color_inst"] = compute_delta_H0_from_delta_M_B(
        priors.sigma_delta_color_inst * 0.5, H0_base  # Color ~ 0.5 mag range
    )

    return contributions
