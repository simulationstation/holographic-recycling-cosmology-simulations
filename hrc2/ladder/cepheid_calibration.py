#!/usr/bin/env python3
"""
Cepheid/TRGB Calibration Chain for SN Ia Distance Ladder.

Models the full SH0ES-like calibration hierarchy:
  Anchors (NGC 4258, LMC, MW) → Cepheid PL relation → SN host distances → M_B → H0

This module defines:
- Anchor objects (geometric distance anchors)
- Cepheid Period-Luminosity (PL) relation parameters
- Cepheid host galaxies
- Functions to simulate biased distance moduli through the chain
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Anchor:
    """
    A geometric distance anchor for the Cepheid PL zero-point.

    Examples: NGC 4258 (maser), LMC (eclipsing binaries), MW (Gaia parallaxes)
    """
    name: str
    mu_true: float          # True distance modulus (mag)
    Z: float                # Metallicity proxy (e.g., 12 + log(O/H) - 8.9)
    sigma_mu_stat: float    # Nominal statistical uncertainty (mag)


@dataclass
class CepheidPLParameters:
    """
    Cepheid Period-Luminosity relation parameters.

    The PL relation in Wesenheit magnitude W:
        M_W = M_W0 + b_W * (log P - 1) + gamma_W * (Z - Z_ref)

    Where:
        M_W0: zero-point at log P = 1 (P = 10 days)
        b_W: slope with log-period
        gamma_W: metallicity dependence (mag/dex)
    """
    # True PL parameters (what nature uses)
    M_W0_true: float = -5.90       # Wesenheit zero-point
    b_W_true: float = -3.30        # Slope with log P
    gamma_W_true: float = -0.20    # Metallicity term (mag/dex)

    # Fitted PL parameters (what the analysis assumes)
    M_W0_fit: float = -5.90
    b_W_fit: float = -3.30
    gamma_W_fit: float = -0.20

    # Reference metallicity for PL relation
    Z_ref: float = 0.0             # Solar metallicity reference

    # Crowding/blending biases (mag)
    delta_mu_crowd_anchor: float = 0.0   # Bias in anchor fields
    delta_mu_crowd_hosts: float = 0.0    # Bias in SN host fields

    # Intrinsic scatter and measurement noise
    sigma_int: float = 0.08        # Intrinsic scatter in PL
    sigma_meas: float = 0.05       # Photometric measurement error


@dataclass
class CepheidHost:
    """
    A galaxy hosting both Cepheids and a calibrator SN Ia.

    The Cepheid-based distance to this host anchors the SN absolute magnitude.
    """
    name: str
    mu_true: float          # True distance modulus
    logM_star: float        # log10(stellar mass / M_sun)
    Z: float                # Metallicity proxy
    anchor_name: str        # Which anchor provides the PL zero-point
    n_cepheids: int = 20    # Number of Cepheids observed in this host

    # Optional: if TRGB distance is also available
    has_trgb: bool = False
    mu_trgb_true: float = 0.0


@dataclass
class TRGBParameters:
    """
    TRGB (Tip of the Red Giant Branch) calibration parameters.

    TRGB provides an alternative distance indicator with different systematics.
    """
    M_TRGB_true: float = -4.05     # True TRGB absolute magnitude (I-band)
    M_TRGB_fit: float = -4.05      # Fitted value

    # Color/metallicity dependence
    gamma_TRGB_true: float = 0.0   # True metallicity term
    gamma_TRGB_fit: float = 0.0    # Fitted value

    sigma_TRGB: float = 0.04       # TRGB measurement scatter


# =============================================================================
# Default Anchors and Hosts
# =============================================================================

def get_default_anchors() -> List[Anchor]:
    """
    Return default set of geometric distance anchors.

    Distance moduli are approximate Planck-consistent values.
    """
    return [
        Anchor(
            name="NGC4258",
            mu_true=29.40,      # ~7.6 Mpc maser distance
            Z=0.0,              # Near-solar metallicity
            sigma_mu_stat=0.03,
        ),
        Anchor(
            name="LMC",
            mu_true=18.49,      # ~50 kpc
            Z=-0.3,             # Sub-solar metallicity
            sigma_mu_stat=0.02,
        ),
        Anchor(
            name="MW",
            mu_true=0.0,        # Local (Gaia parallaxes)
            Z=0.0,              # Solar by definition
            sigma_mu_stat=0.02,
        ),
    ]


def get_default_cepheid_hosts(cosmo_H0: float = 67.5) -> List[CepheidHost]:
    """
    Return default set of Cepheid SN host galaxies.

    These are representative of the SH0ES sample.
    Distances computed for the input H0 (Planck-like default).
    """
    # Approximate redshifts and properties for SH0ES-like hosts
    hosts_data = [
        ("NGC4536", 0.0060, 10.5, 0.1, "NGC4258", 25),
        ("NGC4639", 0.0034, 10.2, 0.0, "NGC4258", 18),
        ("NGC3370", 0.0043, 10.3, 0.05, "LMC", 22),
        ("NGC3982", 0.0037, 10.1, -0.1, "LMC", 15),
        ("NGC1309", 0.0071, 10.4, 0.0, "MW", 30),
    ]

    hosts = []
    for name, z, logM, Z, anchor, n_ceph in hosts_data:
        # Approximate mu from redshift (simplified)
        # mu ≈ 5 log10(c*z / H0) + 25 for low z
        c_km_s = 299792.458
        if z > 0:
            d_Mpc = c_km_s * z / cosmo_H0
            mu_true = 5 * np.log10(d_Mpc) + 25
        else:
            mu_true = 0.0

        hosts.append(CepheidHost(
            name=name,
            mu_true=mu_true,
            logM_star=logM,
            Z=Z,
            anchor_name=anchor,
            n_cepheids=n_ceph,
            has_trgb=(name in ["NGC4536", "NGC1309"]),  # Some have TRGB
            mu_trgb_true=mu_true,
        ))

    return hosts


# =============================================================================
# Simulation Functions
# =============================================================================

def simulate_anchor_mu(
    anchor: Anchor,
    delta_mu_global: float,
    delta_mu_individual: float,
    delta_mu_crowd: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate biased anchor distance modulus.

    Args:
        anchor: Anchor object with true mu
        delta_mu_global: Global zero-point bias applied to all anchors
        delta_mu_individual: Bias specific to this anchor
        delta_mu_crowd: Crowding bias in anchor field
        rng: Random generator for statistical noise

    Returns:
        Biased anchor distance modulus
    """
    mu = anchor.mu_true

    # Apply systematic biases
    mu += delta_mu_global
    mu += delta_mu_individual
    mu += delta_mu_crowd

    # Add statistical noise if requested
    if rng is not None and anchor.sigma_mu_stat > 0:
        mu += rng.normal(0, anchor.sigma_mu_stat)

    return mu


def simulate_cepheid_host_distance(
    host: CepheidHost,
    anchor: Anchor,
    anchor_mu_biased: float,
    pl_params: CepheidPLParameters,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate Cepheid-based distance modulus for a SN host galaxy.

    The bias propagation is:
    1. Anchor bias shifts the PL zero-point calibration
    2. PL parameter mismatches (slope, metallicity) add additional bias
    3. Host-specific crowding adds further bias

    Args:
        host: CepheidHost object
        anchor: Anchor used for this host's PL calibration
        anchor_mu_biased: Biased anchor distance modulus
        pl_params: PL relation parameters (true and fit)
        rng: Random generator

    Returns:
        Biased host distance modulus from Cepheids
    """
    # Start with true distance
    mu = host.mu_true

    # 1. Zero-point bias from anchor
    # If anchor appears farther (mu_biased > mu_true), PL zero-point shifts brighter,
    # making hosts appear farther too
    anchor_bias = anchor_mu_biased - anchor.mu_true
    mu += anchor_bias

    # 2. PL zero-point mismatch
    # delta_M_W0 = M_W0_fit - M_W0_true
    # If fitter assumes brighter zero-point than true, distances are underestimated
    delta_M_W0 = pl_params.M_W0_fit - pl_params.M_W0_true
    mu -= delta_M_W0  # Sign: brighter assumed M means smaller inferred mu

    # 3. Metallicity term mismatch
    # If fitter uses wrong gamma_W, metallicity difference between anchor and host
    # causes additional bias
    delta_gamma = pl_params.gamma_W_fit - pl_params.gamma_W_true
    Z_diff = host.Z - anchor.Z
    mu -= delta_gamma * Z_diff

    # 4. Crowding bias in host (different from anchor crowding)
    mu += pl_params.delta_mu_crowd_hosts

    # 5. Add measurement scatter
    if rng is not None:
        # Uncertainty scales as 1/sqrt(N_cepheids)
        sigma_total = np.sqrt(
            pl_params.sigma_int**2 / host.n_cepheids +
            pl_params.sigma_meas**2
        )
        mu += rng.normal(0, sigma_total)

    return mu


def simulate_trgb_host_distance(
    host: CepheidHost,
    trgb_params: TRGBParameters,
    delta_mu_trgb_global: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate TRGB-based distance modulus for a SN host galaxy.

    Args:
        host: CepheidHost object (must have has_trgb=True)
        trgb_params: TRGB calibration parameters
        delta_mu_trgb_global: Global TRGB zero-point bias
        rng: Random generator

    Returns:
        Biased host distance modulus from TRGB
    """
    if not host.has_trgb:
        return np.nan

    mu = host.mu_trgb_true

    # TRGB zero-point bias
    delta_M_TRGB = trgb_params.M_TRGB_fit - trgb_params.M_TRGB_true
    mu -= delta_M_TRGB

    # Global TRGB calibration bias
    mu += delta_mu_trgb_global

    # Metallicity mismatch
    delta_gamma_TRGB = trgb_params.gamma_TRGB_fit - trgb_params.gamma_TRGB_true
    mu -= delta_gamma_TRGB * host.Z

    # Add measurement noise
    if rng is not None and trgb_params.sigma_TRGB > 0:
        mu += rng.normal(0, trgb_params.sigma_TRGB)

    return mu


def compute_calibrator_mu_from_chain(
    anchors: List[Anchor],
    hosts: List[CepheidHost],
    pl_params: CepheidPLParameters,
    trgb_params: TRGBParameters,
    anchor_biases: Dict[str, float],
    delta_mu_anchor_global: float,
    delta_mu_crowd_anchor: float,
    delta_mu_trgb_global: float,
    use_cepheids: bool = True,
    use_trgb: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute biased calibrator distance moduli through the full chain.

    Args:
        anchors: List of Anchor objects
        hosts: List of CepheidHost objects
        pl_params: Cepheid PL parameters
        trgb_params: TRGB parameters
        anchor_biases: Dict of per-anchor biases (e.g., {"NGC4258": 0.02})
        delta_mu_anchor_global: Global anchor bias
        delta_mu_crowd_anchor: Crowding bias in anchor fields
        delta_mu_trgb_global: Global TRGB zero-point bias
        use_cepheids: Use Cepheid-based distances
        use_trgb: Use TRGB-based distances (can combine with Cepheids)
        rng: Random generator

    Returns:
        (mu_calib_array, mu_host_dict)
        mu_calib_array: Biased mu for each host (same order as hosts)
        mu_host_dict: Dict mapping host name to biased mu
    """
    # Build anchor lookup
    anchor_dict = {a.name: a for a in anchors}

    # Compute biased anchor distances
    biased_anchor_mu = {}
    for anchor in anchors:
        delta_ind = anchor_biases.get(anchor.name, 0.0)
        biased_anchor_mu[anchor.name] = simulate_anchor_mu(
            anchor,
            delta_mu_global=delta_mu_anchor_global,
            delta_mu_individual=delta_ind,
            delta_mu_crowd=delta_mu_crowd_anchor,
            rng=rng,
        )

    # Compute biased host distances
    mu_host_dict = {}
    for host in hosts:
        mu_cepheid = np.nan
        mu_trgb = np.nan

        if use_cepheids and host.anchor_name in anchor_dict:
            anchor = anchor_dict[host.anchor_name]
            mu_cepheid = simulate_cepheid_host_distance(
                host,
                anchor,
                biased_anchor_mu[host.anchor_name],
                pl_params,
                rng,
            )

        if use_trgb and host.has_trgb:
            mu_trgb = simulate_trgb_host_distance(
                host,
                trgb_params,
                delta_mu_trgb_global,
                rng,
            )

        # Combine Cepheid and TRGB if both available
        if use_cepheids and use_trgb:
            if not np.isnan(mu_cepheid) and not np.isnan(mu_trgb):
                # Simple average (could use inverse-variance weighting)
                mu_host_dict[host.name] = (mu_cepheid + mu_trgb) / 2
            elif not np.isnan(mu_cepheid):
                mu_host_dict[host.name] = mu_cepheid
            elif not np.isnan(mu_trgb):
                mu_host_dict[host.name] = mu_trgb
            else:
                mu_host_dict[host.name] = host.mu_true  # Fallback
        elif use_cepheids:
            mu_host_dict[host.name] = mu_cepheid if not np.isnan(mu_cepheid) else host.mu_true
        elif use_trgb:
            mu_host_dict[host.name] = mu_trgb if not np.isnan(mu_trgb) else host.mu_true
        else:
            mu_host_dict[host.name] = host.mu_true

    # Return as array in same order as hosts
    mu_calib_array = np.array([mu_host_dict[h.name] for h in hosts])

    return mu_calib_array, mu_host_dict


def compute_mean_calibrator_bias(
    hosts: List[CepheidHost],
    mu_biased: np.ndarray,
) -> float:
    """
    Compute the mean bias in calibrator distances.

    Args:
        hosts: List of CepheidHost objects
        mu_biased: Biased distance moduli

    Returns:
        Mean (mu_biased - mu_true) across hosts
    """
    mu_true = np.array([h.mu_true for h in hosts])
    return float(np.mean(mu_biased - mu_true))


# =============================================================================
# Multi-Instrument Cepheid Photometry (SIM 13)
# =============================================================================

def simulate_cepheid_magnitudes_and_colors(
    host: CepheidHost,
    pl_params: CepheidPLParameters,
    n_cepheids: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Simulate true (instrument-independent) Cepheid magnitudes and colors.

    Args:
        host: CepheidHost object
        pl_params: PL relation parameters (true values used)
        n_cepheids: Number of Cepheids to simulate
        rng: Random generator

    Returns:
        Dict with logP, color_true, m_true (reference band W magnitudes)
    """
    # Log-period distribution centered at log P = 1 (P = 10 days)
    logP = rng.normal(loc=1.0, scale=0.3, size=n_cepheids)

    # Color distribution (e.g., V-I or H-K)
    color_true = rng.normal(loc=0.8, scale=0.1, size=n_cepheids)

    # True PL relation in Wesenheit magnitude W
    M_W = (pl_params.M_W0_true
           + pl_params.b_W_true * (logP - 1.0)
           + pl_params.gamma_W_true * (host.Z - pl_params.Z_ref))

    # Add intrinsic scatter
    M_W += rng.normal(0, pl_params.sigma_int, size=n_cepheids)

    # Apparent magnitude = absolute + distance modulus
    m_true = M_W + host.mu_true

    return {
        "logP": logP,
        "color_true": color_true,
        "m_true": m_true,
        "M_W_true": M_W,
        "host_Z": np.full(n_cepheids, host.Z),
        "host_logM": np.full(n_cepheids, host.logM_star),
    }


def generate_cepheid_data_for_host(
    host: CepheidHost,
    pl_params: CepheidPLParameters,
    inst_hst,  # InstrumentPhotometrySystematics
    inst_jwst,  # InstrumentPhotometrySystematics
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Generate Cepheid photometry for a single host in both HST and JWST bands.

    Args:
        host: CepheidHost object
        pl_params: PL relation parameters
        inst_hst: HST-like instrument systematics
        inst_jwst: JWST-like instrument systematics
        rng: Random generator

    Returns:
        Dict with:
            logP, color_true, m_true,
            m_hst (HST-observed mags), m_jwst (JWST-observed mags),
            host_Z, host_logM
    """
    # Import here to avoid circular imports
    from .instrument_photometry import apply_instrument_effects

    # Generate true Cepheid properties
    ceph_data = simulate_cepheid_magnitudes_and_colors(
        host, pl_params, host.n_cepheids, rng
    )

    # Apply HST-like instrument effects
    m_hst = apply_instrument_effects(
        m_true=ceph_data["m_true"],
        color_true=ceph_data["color_true"],
        host_logM=ceph_data["host_logM"],
        inst=inst_hst,
    )

    # Apply JWST-like instrument effects
    m_jwst = apply_instrument_effects(
        m_true=ceph_data["m_true"],
        color_true=ceph_data["color_true"],
        host_logM=ceph_data["host_logM"],
        inst=inst_jwst,
    )

    ceph_data["m_hst"] = m_hst
    ceph_data["m_jwst"] = m_jwst

    return ceph_data


def fit_PL_zero_point_from_instrument(
    cepheid_data_by_host: Dict[str, Dict[str, np.ndarray]],
    hosts: List[CepheidHost],
    pl_params: CepheidPLParameters,
    instrument_key: str = "m_hst",
) -> Tuple[float, float]:
    """
    Fit the PL zero-point M_W0 from instrument-biased magnitudes.

    Uses a simple least-squares approach:
        m_obs = M_W0 + b_W_fit * (logP - 1) + gamma_W_fit * (Z - Z_ref) + mu_host

    Solving for M_W0 given known mu_host (from host.mu_true for anchors,
    or iterated for SN hosts).

    For SIM 13, we assume anchors provide the zero-point calibration.

    Args:
        cepheid_data_by_host: Dict mapping host name to Cepheid data
        hosts: List of CepheidHost objects
        pl_params: PL relation parameters (fit values used for b_W, gamma_W)
        instrument_key: Which magnitude to use ("m_hst" or "m_jwst")

    Returns:
        (M_W0_fit, residual_rms)
    """
    # Collect all Cepheids with their predicted model values
    residuals = []

    for host in hosts:
        if host.name not in cepheid_data_by_host:
            continue

        data = cepheid_data_by_host[host.name]
        logP = data["logP"]
        m_obs = data[instrument_key]

        # Model: m_obs = M_W0 + b_W*(logP-1) + gamma_W*(Z-Z_ref) + mu_host
        # Rearrange: M_W0 = m_obs - b_W*(logP-1) - gamma_W*(Z-Z_ref) - mu_host
        M_W0_inferred = (
            m_obs
            - pl_params.b_W_fit * (logP - 1.0)
            - pl_params.gamma_W_fit * (host.Z - pl_params.Z_ref)
            - host.mu_true
        )

        residuals.extend(M_W0_inferred)

    residuals = np.array(residuals)
    M_W0_fit = float(np.mean(residuals))
    residual_rms = float(np.std(residuals))

    return M_W0_fit, residual_rms


def compute_host_mu_from_instrument_cepheids(
    cepheid_data: Dict[str, np.ndarray],
    host: CepheidHost,
    M_W0_fit: float,
    pl_params: CepheidPLParameters,
    instrument_key: str = "m_hst",
) -> float:
    """
    Compute host distance modulus from Cepheid photometry using fitted PL zero-point.

    Args:
        cepheid_data: Cepheid data for this host
        host: CepheidHost object
        M_W0_fit: Fitted PL zero-point
        pl_params: PL parameters (b_W_fit, gamma_W_fit used)
        instrument_key: Which magnitude to use

    Returns:
        Estimated distance modulus for this host
    """
    logP = cepheid_data["logP"]
    m_obs = cepheid_data[instrument_key]

    # Model: m_obs = M_W + mu
    # M_W = M_W0_fit + b_W_fit*(logP-1) + gamma_W_fit*(Z-Z_ref)
    M_W_model = (
        M_W0_fit
        + pl_params.b_W_fit * (logP - 1.0)
        + pl_params.gamma_W_fit * (host.Z - pl_params.Z_ref)
    )

    # mu = m_obs - M_W
    mu_estimates = m_obs - M_W_model

    # Return mean distance modulus
    return float(np.mean(mu_estimates))
