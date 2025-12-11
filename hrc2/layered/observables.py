"""
Observables for Layered Expansion (Bent-Deck) Cosmology

This module computes cosmological observables for a given layered expansion
configuration, and calculates chi-squared values against CMB, BAO, and SN data.

The key observables are:
- H0_eff: effective Hubble constant at z=0
- theta_star: CMB acoustic scale
- D_A(z), D_V(z): angular diameter and volume-averaged distances
- mu(z): SN Ia distance modulus

We use the existing BAO and SN data/chi2 infrastructure from rs_parametric.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
from scipy.integrate import quad

from .layered_expansion import (
    LayeredExpansionHyperparams,
    LayeredExpansionParams,
    LCDMBackground,
    H_of_z_layered,
    get_H0_effective,
    log_smoothness_prior,
)


# Physical constants
C_KM_S = 299792.458  # Speed of light in km/s

# CMB reference values (Planck 2018)
THETA_STAR_REF = 0.0104092  # CMB acoustic scale [radians]
THETA_STAR_SIGMA = 0.0000031  # Uncertainty
Z_STAR = 1089.92  # Recombination redshift
Z_DRAG = 1059.94  # Baryon drag epoch

# Sound horizon in LCDM (approximate, Planck 2018)
RS_LCDM = 147.09  # Mpc


@dataclass
class LayeredObservables:
    """
    Collection of observables for a layered expansion model.

    Attributes
    ----------
    H0_eff : float
        Effective Hubble constant at z=0 [km/s/Mpc]
    theta_star : float
        CMB acoustic scale [radians]
    r_s : float
        Sound horizon at drag epoch [Mpc] (assumed, not recomputed)
    D_A_star : float
        Angular diameter distance to last scattering [Mpc]
    D_C_star : float
        Comoving distance to last scattering [Mpc]
    bao_distances : Dict[float, Dict[str, float]]
        Distances at BAO redshifts: {z: {"D_M": ..., "D_H": ..., "D_V": ...}}
    sn_distances : Dict[float, float]
        Distance moduli at representative SN redshifts: {z: mu}
    """
    H0_eff: float
    theta_star: float
    r_s: float
    D_A_star: float
    D_C_star: float
    bao_distances: Dict[float, Dict[str, float]] = field(default_factory=dict)
    sn_distances: Dict[float, float] = field(default_factory=dict)


@dataclass
class LayeredChi2Result:
    """
    Chi-squared results for layered expansion model.

    Attributes
    ----------
    chi2_total : float
        Total chi-squared (CMB + BAO + SN)
    chi2_cmb : float
        Chi-squared from CMB theta_* constraint
    chi2_bao : float
        Chi-squared from BAO measurements
    chi2_sn : float
        Chi-squared from SN Ia measurements
    delta_chi2 : float
        Difference from baseline LCDM chi-squared
    H0_eff : float
        Effective H0 [km/s/Mpc]
    theta_star : float
        CMB acoustic scale [radians]
    log_prior_smoothness : float
        Log of smoothness prior
    is_physical : bool
        Whether the model is physically valid
    """
    chi2_total: float
    chi2_cmb: float
    chi2_bao: float
    chi2_sn: float
    delta_chi2: float
    H0_eff: float
    theta_star: float
    log_prior_smoothness: float
    is_physical: bool


# =============================================================================
# Distance Calculations for Layered Models
# =============================================================================

def comoving_distance_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Compute comoving distance D_C(z) for layered expansion model.

    D_C = c * integral_0^z dz' / H(z')

    Parameters
    ----------
    z : float
        Redshift
    lcdm : LCDMBackground
        Baseline LCDM (provides H0 for normalization)
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float
        Comoving distance in Mpc
    """
    if z <= 0:
        return 0.0

    def integrand(zp):
        H = H_of_z_layered(zp, lcdm, params, hyp)
        if H <= 0 or not np.isfinite(H):
            return np.nan
        return C_KM_S / H

    limit = 200 if z > 100 else 100
    result, _ = quad(integrand, 0, z, limit=limit, epsrel=1e-8, epsabs=1e-10)

    return result


def angular_diameter_distance_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Angular diameter distance D_A(z) for layered model (flat universe).

    D_A = D_C / (1 + z)
    """
    D_C = comoving_distance_layered(z, lcdm, params, hyp)
    return D_C / (1 + z)


def luminosity_distance_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Luminosity distance D_L(z) for layered model (flat universe).

    D_L = (1 + z) * D_C
    """
    D_C = comoving_distance_layered(z, lcdm, params, hyp)
    return (1 + z) * D_C


def hubble_distance_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Hubble distance D_H(z) = c / H(z) for layered model.
    """
    H = H_of_z_layered(z, lcdm, params, hyp)
    if H <= 0 or not np.isfinite(H):
        return np.nan
    return C_KM_S / H


def volume_average_distance_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Volume-averaged distance D_V(z) for layered model.

    D_V = [z * D_H(z) * D_M(z)^2]^(1/3)
    where D_M = D_C for flat universe
    """
    D_M = comoving_distance_layered(z, lcdm, params, hyp)
    D_H = hubble_distance_layered(z, lcdm, params, hyp)

    if D_M <= 0 or D_H <= 0 or not np.isfinite(D_M) or not np.isfinite(D_H):
        return np.nan

    return (z * D_H * D_M**2)**(1.0/3.0)


# =============================================================================
# Sound Horizon and Theta Star
# =============================================================================

def compute_sound_horizon_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    z_drag: float = Z_DRAG,
    omega_b: float = 0.0224,
) -> float:
    """
    Compute the sound horizon r_s for the layered model.

    r_s = integral_{z_drag}^{infinity} c_s / H(z) dz

    where c_s = c / sqrt(3(1 + R_b)) is the sound speed in the baryon-photon fluid.

    NOTE: This is computationally expensive and only modestly affected by
    late-time modifications. For speed, we use an approximation:
    r_s_layered ≈ r_s_LCDM * correction_factor

    where the correction accounts for high-z H(z) changes.

    For most late-time models with z_max < z_drag, this is a good approximation.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    z_drag : float
        Drag epoch redshift
    omega_b : float
        Physical baryon density

    Returns
    -------
    float
        Sound horizon in Mpc
    """
    # For models that only modify H(z) at z < z_drag,
    # the sound horizon is essentially unchanged from LCDM.

    if hyp.z_max < z_drag:
        # Late-time model: r_s ≈ r_s_LCDM
        return RS_LCDM

    # For models extending to high z, compute correction
    # This is a simplified approximation

    # Full calculation would require:
    # 1. Sound speed c_s(z) including baryon loading
    # 2. Integration from z_drag to high z

    # For now, use LCDM value with a warning
    # A proper implementation would integrate c_s/H from z_drag to z_inf

    return RS_LCDM


def compute_theta_star_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    r_s: Optional[float] = None,
    z_star: float = Z_STAR,
) -> float:
    """
    Compute the CMB acoustic scale theta_* for layered model.

    theta_* = r_s / D_C(z_*)

    where D_C is the comoving distance to last scattering.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    r_s : float, optional
        Sound horizon in Mpc (default: compute from layered model)
    z_star : float
        Recombination redshift

    Returns
    -------
    float
        Acoustic scale theta_* in radians
    """
    if r_s is None:
        r_s = compute_sound_horizon_layered(lcdm, params, hyp)

    D_C_star = comoving_distance_layered(z_star, lcdm, params, hyp)

    if D_C_star <= 0 or not np.isfinite(D_C_star):
        return np.nan

    return r_s / D_C_star


# =============================================================================
# BAO Data and Chi-Squared
# =============================================================================

@dataclass
class BAODataPoint:
    """A single BAO measurement."""
    z_eff: float
    observable: str  # 'DV_rd', 'DM_rd', 'DH_rd'
    value: float
    sigma: float
    survey: str = ""


# BAO data compilation (same as in rs_parametric.py)
BAO_DATA = [
    # SDSS DR7 MGS
    BAODataPoint(z_eff=0.15, observable='DV_rd', value=4.47, sigma=0.17, survey='SDSS_MGS'),
    # BOSS DR12 consensus
    BAODataPoint(z_eff=0.38, observable='DM_rd', value=10.23, sigma=0.17, survey='BOSS_DR12'),
    BAODataPoint(z_eff=0.38, observable='DH_rd', value=25.0, sigma=0.76, survey='BOSS_DR12'),
    BAODataPoint(z_eff=0.51, observable='DM_rd', value=13.36, sigma=0.21, survey='BOSS_DR12'),
    BAODataPoint(z_eff=0.51, observable='DH_rd', value=22.33, sigma=0.58, survey='BOSS_DR12'),
    BAODataPoint(z_eff=0.61, observable='DM_rd', value=15.45, sigma=0.27, survey='BOSS_DR12'),
    BAODataPoint(z_eff=0.61, observable='DH_rd', value=20.43, sigma=0.48, survey='BOSS_DR12'),
    # eBOSS LRG
    BAODataPoint(z_eff=0.70, observable='DM_rd', value=17.86, sigma=0.33, survey='eBOSS_LRG'),
    BAODataPoint(z_eff=0.70, observable='DH_rd', value=19.33, sigma=0.53, survey='eBOSS_LRG'),
    # eBOSS Quasar
    BAODataPoint(z_eff=1.48, observable='DM_rd', value=30.69, sigma=0.80, survey='eBOSS_QSO'),
    BAODataPoint(z_eff=1.48, observable='DH_rd', value=13.26, sigma=0.55, survey='eBOSS_QSO'),
    # eBOSS Lya
    BAODataPoint(z_eff=2.33, observable='DM_rd', value=37.6, sigma=1.9, survey='eBOSS_Lya'),
    BAODataPoint(z_eff=2.33, observable='DH_rd', value=8.93, sigma=0.28, survey='eBOSS_Lya'),
]


def compute_chi2_bao_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    r_d: Optional[float] = None,
    bao_data: Optional[List[BAODataPoint]] = None,
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Compute chi-squared for BAO constraints with layered model.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    r_d : float, optional
        Sound horizon at drag epoch (default: use r_s from layered model * 0.99)
    bao_data : list, optional
        BAO data points (default: BAO_DATA)

    Returns
    -------
    tuple
        (chi2_bao, distances_dict) where distances_dict contains model predictions
    """
    if bao_data is None:
        bao_data = BAO_DATA

    if r_d is None:
        r_s = compute_sound_horizon_layered(lcdm, params, hyp)
        r_d = r_s * 0.99  # r_drag slightly smaller than r_*

    chi2 = 0.0
    distances = {}

    # Cache distances at each unique z
    z_values = list(set(dp.z_eff for dp in bao_data))
    for z in z_values:
        D_M = comoving_distance_layered(z, lcdm, params, hyp)
        D_H = hubble_distance_layered(z, lcdm, params, hyp)
        D_V = volume_average_distance_layered(z, lcdm, params, hyp)
        distances[z] = {"D_M": D_M, "D_H": D_H, "D_V": D_V}

    for dp in bao_data:
        z = dp.z_eff
        d = distances[z]

        if dp.observable == 'DV_rd':
            model = d["D_V"] / r_d
        elif dp.observable == 'DM_rd':
            model = d["D_M"] / r_d
        elif dp.observable == 'DH_rd':
            model = d["D_H"] / r_d
        else:
            continue

        if not np.isfinite(model):
            chi2 += 1e10  # Large penalty for invalid model
            continue

        chi2 += ((model - dp.value) / dp.sigma)**2

    return chi2, distances


# =============================================================================
# SN Ia Constraints
# =============================================================================

def distance_modulus_layered(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
) -> float:
    """
    Compute distance modulus mu(z) for layered model.

    mu = 5 * log10(D_L / 10 pc)

    Parameters
    ----------
    z : float
        Redshift
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float
        Distance modulus
    """
    if z <= 0:
        return 0.0

    D_L = luminosity_distance_layered(z, lcdm, params, hyp)
    if D_L <= 0 or not np.isfinite(D_L):
        return np.nan

    # D_L in Mpc, convert to pc
    D_L_pc = D_L * 1e6
    return 5.0 * np.log10(D_L_pc / 10.0)


def compute_chi2_sn_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    use_shoes_prior: bool = True,
) -> Tuple[float, Dict[float, float]]:
    """
    Compute chi-squared for SN Ia constraints with layered model.

    Uses a compressed constraint based on the SH0ES H0 measurement:
    H0 = 73.04 ± 1.04 km/s/Mpc

    The SN constraint essentially measures H0 at low redshift, so we
    compare the effective H0 to the SH0ES value.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    use_shoes_prior : bool
        If True, use SH0ES H0 constraint. If False, just use SN shape.

    Returns
    -------
    tuple
        (chi2_sn, mu_dict) where mu_dict contains distance moduli
    """
    H0_eff = get_H0_effective(lcdm, params, hyp)

    # SH0ES constraint
    H0_shoes = 73.04
    sigma_H0_shoes = 1.04

    # Compute distance moduli at representative redshifts
    z_sn = [0.01, 0.1, 0.5, 1.0]
    mu_dict = {}
    for z in z_sn:
        mu_dict[z] = distance_modulus_layered(z, lcdm, params, hyp)

    if use_shoes_prior:
        # Chi-squared from H0 constraint
        chi2 = ((H0_eff - H0_shoes) / sigma_H0_shoes)**2
    else:
        # Without SH0ES prior, SN alone don't constrain H0 strongly
        # Just check that the shape is reasonable (implicit in BAO fit)
        chi2 = 0.0

    return chi2, mu_dict


# =============================================================================
# CMB Constraint
# =============================================================================

def compute_chi2_cmb_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    r_s: Optional[float] = None,
    theta_star_ref: float = THETA_STAR_REF,
    theta_star_sigma: float = THETA_STAR_SIGMA,
) -> Tuple[float, float]:
    """
    Compute chi-squared for CMB theta_* constraint.

    The CMB acoustic scale theta_* = r_s / D_C(z_*) is measured very precisely.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM
    params : LayeredExpansionParams
        Layered parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    r_s : float, optional
        Sound horizon (default: compute from layered model)
    theta_star_ref : float
        Reference theta_* value (Planck 2018)
    theta_star_sigma : float
        Uncertainty on theta_*

    Returns
    -------
    tuple
        (chi2_cmb, theta_star)
    """
    theta_star = compute_theta_star_layered(lcdm, params, hyp, r_s=r_s)

    if not np.isfinite(theta_star):
        return 1e10, np.nan

    chi2 = ((theta_star - theta_star_ref) / theta_star_sigma)**2

    return chi2, theta_star


# =============================================================================
# Combined Observable Computation
# =============================================================================

def compute_background_observables_layered(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
) -> LayeredObservables:
    """
    Compute all background observables for a layered expansion model.

    This is the main observable computation function, returning all
    quantities needed for cosmological constraints.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    LayeredObservables
        Collection of observables
    """
    # Effective H0
    H0_eff = get_H0_effective(lcdm, params, hyp)

    # Sound horizon
    r_s = compute_sound_horizon_layered(lcdm, params, hyp)

    # Comoving distance to CMB
    D_C_star = comoving_distance_layered(Z_STAR, lcdm, params, hyp)
    D_A_star = D_C_star / (1 + Z_STAR)

    # Theta star
    theta_star = r_s / D_C_star if D_C_star > 0 else np.nan

    # BAO distances
    _, bao_distances = compute_chi2_bao_layered(lcdm, params, hyp, r_d=r_s*0.99)

    # SN distances
    _, sn_distances = compute_chi2_sn_layered(lcdm, params, hyp, use_shoes_prior=False)

    return LayeredObservables(
        H0_eff=H0_eff,
        theta_star=theta_star,
        r_s=r_s,
        D_A_star=D_A_star,
        D_C_star=D_C_star,
        bao_distances=bao_distances,
        sn_distances=sn_distances,
    )


def compute_chi2_cmb_bao_sn(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    include_shoes: bool = True,
    chi2_baseline: float = 0.0,
) -> LayeredChi2Result:
    """
    Compute total chi-squared for CMB + BAO + SN constraints.

    This is the main chi-squared function for evaluating how well
    a layered expansion model fits the data.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    include_shoes : bool
        Include SH0ES H0 constraint in SN chi-squared
    chi2_baseline : float
        Baseline chi-squared for Δχ² computation

    Returns
    -------
    LayeredChi2Result
        Chi-squared breakdown and diagnostics
    """
    # Check physical validity first
    from .layered_expansion import check_physical_validity
    validity = check_physical_validity(lcdm, params, hyp)
    is_physical = validity["valid"]
    H0_eff = validity["H0_eff"]

    if not is_physical:
        # Return large chi-squared for unphysical models
        return LayeredChi2Result(
            chi2_total=1e10,
            chi2_cmb=1e10,
            chi2_bao=1e10,
            chi2_sn=1e10,
            delta_chi2=1e10,
            H0_eff=H0_eff,
            theta_star=np.nan,
            log_prior_smoothness=-1e10,
            is_physical=False,
        )

    # CMB constraint
    chi2_cmb, theta_star = compute_chi2_cmb_layered(lcdm, params, hyp)

    # BAO constraint
    chi2_bao, _ = compute_chi2_bao_layered(lcdm, params, hyp)

    # SN constraint
    chi2_sn, _ = compute_chi2_sn_layered(lcdm, params, hyp, use_shoes_prior=include_shoes)

    # Total chi-squared
    chi2_total = chi2_cmb + chi2_bao + chi2_sn

    # Delta from baseline
    delta_chi2 = chi2_total - chi2_baseline

    # Smoothness prior
    log_prior = log_smoothness_prior(params, hyp)

    return LayeredChi2Result(
        chi2_total=chi2_total,
        chi2_cmb=chi2_cmb,
        chi2_bao=chi2_bao,
        chi2_sn=chi2_sn,
        delta_chi2=delta_chi2,
        H0_eff=H0_eff,
        theta_star=theta_star,
        log_prior_smoothness=log_prior,
        is_physical=is_physical,
    )


def compute_baseline_chi2(
    lcdm: LCDMBackground,
    hyp: LayeredExpansionHyperparams,
    include_shoes: bool = True,
) -> float:
    """
    Compute chi-squared for baseline LCDM (delta=0).

    This provides the reference chi-squared for computing Δχ².

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    hyp : LayeredExpansionHyperparams
        Hyperparameters (for node structure)
    include_shoes : bool
        Include SH0ES constraint

    Returns
    -------
    float
        Baseline chi-squared
    """
    from .layered_expansion import make_zero_params
    params_zero = make_zero_params(hyp)

    result = compute_chi2_cmb_bao_sn(lcdm, params_zero, hyp, include_shoes=include_shoes)
    return result.chi2_total
