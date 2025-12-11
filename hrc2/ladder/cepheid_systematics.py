#!/usr/bin/env python3
"""
Cepheid/TRGB Systematic Parameters for SIM 12.

Consolidated parameter class for controlling systematics in the
full calibration chain:
  Anchors → Cepheid PL → SN host distances → M_B → H0
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from .cepheid_calibration import (
    Anchor,
    CepheidHost,
    CepheidPLParameters,
    TRGBParameters,
    get_default_anchors,
    get_default_cepheid_hosts,
    compute_calibrator_mu_from_chain,
)


@dataclass
class CepheidSystematicParameters:
    """
    Consolidated parameters controlling Cepheid/TRGB calibration systematics.

    These are the differences between true values and what the fitter assumes,
    or biases that affect the measured distances.
    """

    # =========================================================================
    # Global Anchor Biases
    # =========================================================================

    # Global zero-point bias applied to ALL anchors (mag)
    # Positive = all anchors appear farther = H0 biased high
    delta_mu_anchor_global: float = 0.0

    # Individual anchor offsets (mag), on top of global
    delta_mu_N4258: float = 0.0
    delta_mu_LMC: float = 0.0
    delta_mu_MW: float = 0.0

    # =========================================================================
    # PL Relation Biases
    # =========================================================================

    # Zero-point bias: delta_M_W0 = M_W0_fit - M_W0_true
    # Positive = fitter assumes brighter zero-point = distances underestimated
    delta_M_W0: float = 0.0

    # Slope bias: delta_b_W = b_W_fit - b_W_true
    delta_b_W: float = 0.0

    # Metallicity term bias: delta_gamma_W = gamma_W_fit - gamma_W_true
    delta_gamma_W: float = 0.0

    # =========================================================================
    # Crowding/Blending Biases
    # =========================================================================

    # Crowding bias in anchor fields (mag)
    # Positive = anchor Cepheids appear brighter = anchor appears closer
    delta_mu_crowd_anchor: float = 0.0

    # Crowding bias in SN host fields (mag)
    # Positive = host Cepheids appear brighter = hosts appear closer
    delta_mu_crowd_hosts: float = 0.0

    # =========================================================================
    # TRGB Biases
    # =========================================================================

    # Global TRGB zero-point bias (mag)
    delta_mu_TRGB_global: float = 0.0

    # =========================================================================
    # Calibration Mode
    # =========================================================================

    # Which distance indicator to use
    use_cepheids: bool = True
    use_trgb: bool = False


def build_cepheid_pl_params(sys_params: CepheidSystematicParameters) -> CepheidPLParameters:
    """
    Build CepheidPLParameters from systematic offsets.

    The "true" values are fixed, and "fit" values incorporate the biases.
    """
    # Canonical true values (representative of literature)
    M_W0_true = -5.90
    b_W_true = -3.30
    gamma_W_true = -0.20

    return CepheidPLParameters(
        M_W0_true=M_W0_true,
        b_W_true=b_W_true,
        gamma_W_true=gamma_W_true,
        M_W0_fit=M_W0_true + sys_params.delta_M_W0,
        b_W_fit=b_W_true + sys_params.delta_b_W,
        gamma_W_fit=gamma_W_true + sys_params.delta_gamma_W,
        delta_mu_crowd_anchor=sys_params.delta_mu_crowd_anchor,
        delta_mu_crowd_hosts=sys_params.delta_mu_crowd_hosts,
        sigma_int=0.08,
        sigma_meas=0.05,
    )


def build_trgb_params(sys_params: CepheidSystematicParameters) -> TRGBParameters:
    """
    Build TRGBParameters from systematic offsets.
    """
    M_TRGB_true = -4.05

    return TRGBParameters(
        M_TRGB_true=M_TRGB_true,
        M_TRGB_fit=M_TRGB_true,  # No direct TRGB zero-point bias in this model
        gamma_TRGB_true=0.0,
        gamma_TRGB_fit=0.0,
        sigma_TRGB=0.04,
    )


def get_anchor_biases(sys_params: CepheidSystematicParameters) -> Dict[str, float]:
    """
    Get per-anchor bias dictionary from systematic parameters.
    """
    return {
        "NGC4258": sys_params.delta_mu_N4258,
        "LMC": sys_params.delta_mu_LMC,
        "MW": sys_params.delta_mu_MW,
    }


def apply_full_cepheid_chain(
    sys_params: CepheidSystematicParameters,
    cosmo_H0: float = 67.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply the full Cepheid/TRGB calibration chain and return biased calibrator μ.

    This is the main interface for SIM 12:
    1. Get default anchors and hosts
    2. Build PL and TRGB parameters from systematics
    3. Propagate biases through the chain
    4. Return biased μ_calib for SN calibrators

    Args:
        sys_params: CepheidSystematicParameters
        cosmo_H0: True H0 for computing true distances
        rng: Random generator

    Returns:
        Array of biased distance moduli for calibrator hosts
    """
    anchors = get_default_anchors()
    hosts = get_default_cepheid_hosts(cosmo_H0=cosmo_H0)

    pl_params = build_cepheid_pl_params(sys_params)
    trgb_params = build_trgb_params(sys_params)
    anchor_biases = get_anchor_biases(sys_params)

    mu_calib, _ = compute_calibrator_mu_from_chain(
        anchors=anchors,
        hosts=hosts,
        pl_params=pl_params,
        trgb_params=trgb_params,
        anchor_biases=anchor_biases,
        delta_mu_anchor_global=sys_params.delta_mu_anchor_global,
        delta_mu_crowd_anchor=sys_params.delta_mu_crowd_anchor,
        delta_mu_trgb_global=sys_params.delta_mu_TRGB_global,
        use_cepheids=sys_params.use_cepheids,
        use_trgb=sys_params.use_trgb,
        rng=rng,
    )

    return mu_calib


def compute_expected_H0_bias_from_mu_bias(
    mean_mu_bias: float,
    H0_true: float = 67.5,
) -> float:
    """
    Estimate expected H0 bias from mean distance modulus bias.

    If calibrators appear farther by delta_mu (positive bias):
    - M_B inferred is brighter (more negative)
    - Hubble flow distances are overestimated
    - H0 is biased HIGH

    Approximate relation: delta_H0 / H0 ≈ 0.2 * delta_mu (for small delta_mu)
    (from delta_H0 / H0 = ln(10)/5 * delta_mu ≈ 0.46 * delta_mu for distance,
     but the actual ladder response is more complex)

    Args:
        mean_mu_bias: Mean bias in calibrator distance moduli (mag)
        H0_true: True H0

    Returns:
        Expected delta_H0 (km/s/Mpc)
    """
    # H0 ∝ 10^(0.2 * μ), so d(H0)/H0 = ln(10) * 0.2 * dμ ≈ 0.46 * dμ
    # But this is the sensitivity; for the ladder it's approximately:
    # delta_H0 ≈ H0_true * (10^(0.2 * delta_mu) - 1)
    factor = 10**(0.2 * mean_mu_bias) - 1
    return H0_true * factor


# =============================================================================
# Prior Definitions for Analysis
# =============================================================================

def is_cepheid_realistic(r: dict) -> bool:
    """
    Check if Cepheid/TRGB systematics are in 'realistic' range.

    Realistic = very conservative, well-constrained values.
    """
    return (
        abs(r.get("delta_mu_anchor_global", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_N4258", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_LMC", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_MW", 0.0)) <= 0.02 and
        abs(r.get("delta_M_W0", 0.0)) <= 0.03 and
        abs(r.get("delta_b_W", 0.0)) <= 0.03 and
        abs(r.get("delta_gamma_W", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_crowd_anchor", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_crowd_hosts", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_TRGB_global", 0.0)) <= 0.02
    )


def is_cepheid_moderate(r: dict) -> bool:
    """
    Check if Cepheid/TRGB systematics are in 'moderate' range.

    Moderate = upper bound but still plausible.
    """
    return (
        abs(r.get("delta_mu_anchor_global", 0.0)) <= 0.04 and
        abs(r.get("delta_mu_N4258", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_LMC", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_MW", 0.0)) <= 0.03 and
        abs(r.get("delta_M_W0", 0.0)) <= 0.05 and
        abs(r.get("delta_b_W", 0.0)) <= 0.05 and
        abs(r.get("delta_gamma_W", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_crowd_anchor", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_crowd_hosts", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_TRGB_global", 0.0)) <= 0.03
    )
