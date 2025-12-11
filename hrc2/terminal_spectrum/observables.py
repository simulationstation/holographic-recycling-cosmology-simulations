"""
Observables for Terminal Spectrum / Multi-Mode Cosmology

This module computes cosmological observables for a given terminal spectrum
configuration, including:
- CMB acoustic scale θ* = r_s / D_A(z_*)
- BAO distance ratios D_V/r_d, D_M/r_d, D_H/r_d
- SN Ia distance moduli
- Effective H0

We also compute chi-squared values against CMB, BAO, and SN data to
evaluate consistency of different mode configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .mode_spectrum import (
    TerminalSpectrumParams,
    SpectrumCosmoConfig,
    compute_modified_H_of_z,
    H_LCDM,
    get_H0_effective,
    check_physical_validity,
)


# Physical constants
C_KM_S = 299792.458  # Speed of light in km/s

# CMB reference values (Planck 2018)
THETA_STAR_REF = 0.0104092  # CMB acoustic scale [radians]
THETA_STAR_SIGMA = 0.0000031  # Uncertainty
Z_STAR = 1089.92  # Recombination redshift
Z_DRAG = 1059.94  # Baryon drag epoch

# Sound horizon reference (Planck 2018 ΛCDM)
RS_LCDM = 147.09  # Mpc at recombination
RD_LCDM = 147.09 * 0.99  # Mpc at drag epoch (slightly smaller)


@dataclass
class SpectrumObservables:
    """
    Collection of cosmological observables for a terminal spectrum model.

    Attributes
    ----------
    H0 : float
        Effective Hubble constant at z=0 [km/s/Mpc]
    rs_drag : float
        Sound horizon at drag epoch [Mpc]
    DA_rec : float
        Angular diameter distance to recombination [Mpc]
    theta_star : float
        CMB acoustic angle θ* [radians]
    H_of_z : NDArray
        H(z) on a redshift grid
    z_grid : NDArray
        Redshift grid for H_of_z
    DL_of_z : NDArray
        Luminosity distances on the z_grid [Mpc]
    """
    H0: float
    rs_drag: float
    DA_rec: float
    theta_star: float
    H_of_z: NDArray[np.floating]
    z_grid: NDArray[np.floating]
    DL_of_z: NDArray[np.floating]


@dataclass
class SpectrumChi2Result:
    """
    Chi-squared results for terminal spectrum model.

    Attributes
    ----------
    chi2_total : float
        Total chi-squared (CMB + BAO + SN)
    chi2_cmb : float
        Chi-squared from CMB θ* constraint
    chi2_bao : float
        Chi-squared from BAO measurements
    chi2_sn : float
        Chi-squared from SN distance shape
    H0_eff : float
        Effective H0 [km/s/Mpc]
    theta_star : float
        CMB acoustic angle [radians]
    theta_star_dev_percent : float
        Fractional deviation of θ* from Planck [%]
    max_bao_dev_percent : float
        Maximum BAO distance deviation [%]
    max_sn_dev_percent : float
        Maximum SN distance deviation [%]
    is_physical : bool
        Whether the model is physically valid
    passes_theta_star : bool
        Whether |Δθ*/θ*| < threshold (default 0.1%)
    passes_bao : bool
        Whether max BAO deviation < threshold (default 2%)
    passes_sn : bool
        Whether max SN deviation < threshold (default 2%)
    """
    chi2_total: float
    chi2_cmb: float
    chi2_bao: float
    chi2_sn: float
    H0_eff: float
    theta_star: float
    theta_star_dev_percent: float
    max_bao_dev_percent: float
    max_sn_dev_percent: float
    is_physical: bool
    passes_theta_star: bool = True
    passes_bao: bool = True
    passes_sn: bool = True


# =============================================================================
# Distance Calculations
# =============================================================================

def comoving_distance(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute comoving distance χ(z) for modified cosmology.

    χ = c * ∫₀ᶻ dz' / H(z')

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Comoving distance in Mpc
    """
    if z <= 0:
        return 0.0

    def integrand(zp):
        H = compute_modified_H_of_z(zp, cosmo, spec)
        if H <= 0 or not np.isfinite(H):
            return np.nan
        return C_KM_S / H

    limit = 200 if z > 100 else 100
    result, _ = quad(integrand, 0, z, limit=limit, epsrel=1e-8, epsabs=1e-10)

    return result


def comoving_distance_LCDM(
    z: float,
    cosmo: SpectrumCosmoConfig
) -> float:
    """
    Compute comoving distance χ(z) for baseline ΛCDM.

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters

    Returns
    -------
    float
        Comoving distance in Mpc
    """
    if z <= 0:
        return 0.0

    def integrand(zp):
        H = H_LCDM(zp, cosmo)
        if H <= 0 or not np.isfinite(H):
            return np.nan
        return C_KM_S / H

    limit = 200 if z > 100 else 100
    result, _ = quad(integrand, 0, z, limit=limit, epsrel=1e-8, epsabs=1e-10)

    return result


def angular_diameter_distance(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute angular diameter distance D_A(z) = χ/(1+z) for flat universe.

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Angular diameter distance in Mpc
    """
    chi = comoving_distance(z, cosmo, spec)
    return chi / (1 + z)


def luminosity_distance(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute luminosity distance D_L(z) = (1+z) * χ for flat universe.

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Luminosity distance in Mpc
    """
    chi = comoving_distance(z, cosmo, spec)
    return (1 + z) * chi


def hubble_distance(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute Hubble distance D_H(z) = c / H(z).

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Hubble distance in Mpc
    """
    H = compute_modified_H_of_z(z, cosmo, spec)
    if H <= 0 or not np.isfinite(H):
        return np.nan
    return C_KM_S / H


def volume_average_distance(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute volume-averaged distance D_V(z) = [z D_H D_M²]^(1/3).

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Volume-averaged distance in Mpc
    """
    D_M = comoving_distance(z, cosmo, spec)  # D_M = χ for flat
    D_H = hubble_distance(z, cosmo, spec)

    if D_M <= 0 or D_H <= 0 or not np.isfinite(D_M) or not np.isfinite(D_H):
        return np.nan

    return (z * D_H * D_M**2)**(1.0/3.0)


def distance_modulus(
    z: float,
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams
) -> float:
    """
    Compute distance modulus μ(z) = 5 log₁₀(D_L / 10 pc).

    Parameters
    ----------
    z : float
        Redshift
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters

    Returns
    -------
    float
        Distance modulus
    """
    if z <= 0:
        return 0.0

    D_L = luminosity_distance(z, cosmo, spec)
    if D_L <= 0 or not np.isfinite(D_L):
        return np.nan

    # D_L in Mpc, convert to pc
    D_L_pc = D_L * 1e6
    return 5.0 * np.log10(D_L_pc / 10.0)


# =============================================================================
# Sound Horizon
# =============================================================================

def compute_rs_drag(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    z_drag: float = Z_DRAG,
    omega_b: float = 0.0224,
) -> float:
    """
    Compute sound horizon at drag epoch r_s(z_drag).

    The sound horizon is:
        r_s = ∫_{z_drag}^∞ c_s / H(z) dz

    where c_s = c / √(3(1 + R_b)) is the sound speed and R_b is the
    baryon-to-photon momentum ratio.

    For late-time modifications (modes at z < z_drag), the sound horizon
    is essentially unchanged from ΛCDM. For early-time modifications,
    we compute the integral with the modified H(z).

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    z_drag : float
        Drag epoch redshift
    omega_b : float
        Physical baryon density ω_b = Ω_b h²

    Returns
    -------
    float
        Sound horizon in Mpc
    """
    # Check if any modes extend to high redshift
    if spec.n_modes == 0:
        return RS_LCDM

    # Check if modes are all at low z (below drag epoch)
    # If so, r_s is unchanged
    all_low_z = True
    for mode in spec.modes:
        # Convert mu_ln_a to z
        z_mode = np.exp(-mode.mu_ln_a) - 1
        # Mode extends roughly ±3σ in ln(a)
        z_high = np.exp(-(mode.mu_ln_a - 3*mode.sigma_ln_a)) - 1
        if z_high > 0.5 * z_drag:  # If mode reaches near drag epoch
            all_low_z = False
            break

    if all_low_z:
        return RS_LCDM

    # Full calculation for high-z modifications
    # Sound speed in baryon-photon fluid
    omega_r = 2.47e-5 * (cosmo.H0 / 100)**2  # Radiation density
    omega_b_h2 = omega_b

    def sound_speed_over_c(z):
        """c_s / c = 1 / sqrt(3(1 + R_b))"""
        # R_b = 3 ρ_b / (4 ρ_γ) ∝ (1+z)^-1
        # Approximate: R_b = 31500 ω_b (T_CMB/2.7K)^-4 / (1+z)
        R_b = 31500 * omega_b_h2 * (2.725/2.7)**(-4) / (1 + z)
        return 1.0 / np.sqrt(3 * (1 + R_b))

    def integrand(z):
        H = compute_modified_H_of_z(z, cosmo, spec)
        if H <= 0 or not np.isfinite(H):
            return 0.0
        cs = sound_speed_over_c(z) * C_KM_S
        return cs / H

    # Integrate from z_drag to high z (use 1e5 as proxy for infinity)
    z_max_int = 1e5
    result, _ = quad(integrand, z_drag, z_max_int, limit=500, epsrel=1e-6)

    return result


# =============================================================================
# CMB Acoustic Scale
# =============================================================================

def compute_theta_star(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    rs: Optional[float] = None,
    z_star: float = Z_STAR,
) -> float:
    """
    Compute CMB acoustic angle θ* = r_s / D_M(z_*).

    The acoustic scale uses the comoving angular diameter distance D_M = χ
    (not the proper angular diameter distance D_A = χ/(1+z)).

    Following Planck 2018: θ_∗ ≡ r_∗/D_M(z_∗)

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    rs : float, optional
        Sound horizon (default: compute from model)
    z_star : float
        Recombination redshift

    Returns
    -------
    float
        Acoustic angle in radians
    """
    if rs is None:
        rs = compute_rs_drag(cosmo, spec)

    # Use comoving distance D_M = χ, not angular diameter distance D_A = χ/(1+z)
    D_M = comoving_distance(z_star, cosmo, spec)

    if D_M <= 0 or not np.isfinite(D_M):
        return np.nan

    return rs / D_M


# =============================================================================
# BAO Constraints
# =============================================================================

@dataclass
class BAODataPoint:
    """A single BAO measurement."""
    z_eff: float
    observable: str  # 'DV_rd', 'DM_rd', 'DH_rd'
    value: float
    sigma: float
    survey: str = ""


# BAO data compilation
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

# Key BAO redshifts for quick checks
BAO_CHECK_REDSHIFTS = [0.35, 0.57, 1.52]


def compute_bao_distances(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    z_bao: List[float],
    r_d: Optional[float] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Compute BAO distance ratios at specified redshifts.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    z_bao : list of float
        BAO redshifts
    r_d : float, optional
        Sound horizon at drag epoch (default: compute from model)

    Returns
    -------
    dict
        {z: {"DM_rd": ..., "DH_rd": ..., "DV_rd": ...}} for each redshift
    """
    if r_d is None:
        r_d = compute_rs_drag(cosmo, spec) * 0.99  # r_d slightly less than r_s

    distances = {}
    for z in z_bao:
        D_M = comoving_distance(z, cosmo, spec)
        D_H = hubble_distance(z, cosmo, spec)
        D_V = volume_average_distance(z, cosmo, spec)

        distances[z] = {
            "DM_rd": D_M / r_d if r_d > 0 else np.nan,
            "DH_rd": D_H / r_d if r_d > 0 else np.nan,
            "DV_rd": D_V / r_d if r_d > 0 else np.nan,
            "D_M": D_M,
            "D_H": D_H,
            "D_V": D_V,
        }

    return distances


def compute_chi2_bao(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    r_d: Optional[float] = None,
    bao_data: Optional[List[BAODataPoint]] = None,
) -> Tuple[float, float, Dict[float, Dict[str, float]]]:
    """
    Compute chi-squared for BAO constraints.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    r_d : float, optional
        Sound horizon at drag epoch
    bao_data : list, optional
        BAO data points (default: BAO_DATA)

    Returns
    -------
    tuple
        (chi2_bao, max_dev_percent, distances_dict)
    """
    if bao_data is None:
        bao_data = BAO_DATA

    if r_d is None:
        r_d = compute_rs_drag(cosmo, spec) * 0.99

    # Get unique redshifts
    z_values = list(set(dp.z_eff for dp in bao_data))

    # Compute model distances
    distances = compute_bao_distances(cosmo, spec, z_values, r_d)

    # Also compute LCDM reference
    spec_zero = TerminalSpectrumParams(modes=[])
    distances_lcdm = compute_bao_distances(cosmo, spec_zero, z_values, RD_LCDM)

    chi2 = 0.0
    max_dev_percent = 0.0

    for dp in bao_data:
        z = dp.z_eff
        d = distances[z]

        if dp.observable == 'DV_rd':
            model = d["DV_rd"]
        elif dp.observable == 'DM_rd':
            model = d["DM_rd"]
        elif dp.observable == 'DH_rd':
            model = d["DH_rd"]
        else:
            continue

        if not np.isfinite(model):
            chi2 += 1e10
            continue

        chi2 += ((model - dp.value) / dp.sigma)**2

        # Track maximum deviation from LCDM
        lcdm_val = distances_lcdm[z].get(dp.observable.replace('_rd', '_rd'), dp.value)
        if dp.observable == 'DV_rd':
            lcdm_ratio = distances_lcdm[z]["DV_rd"]
        elif dp.observable == 'DM_rd':
            lcdm_ratio = distances_lcdm[z]["DM_rd"]
        else:
            lcdm_ratio = distances_lcdm[z]["DH_rd"]

        if lcdm_ratio > 0:
            dev_percent = abs(model / lcdm_ratio - 1.0) * 100
            max_dev_percent = max(max_dev_percent, dev_percent)

    return chi2, max_dev_percent, distances


# =============================================================================
# SN Constraints
# =============================================================================

# Representative SN redshifts for shape comparison
SN_REDSHIFTS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]


def compute_sn_distances(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    z_sn: Optional[List[float]] = None,
) -> Dict[float, float]:
    """
    Compute SN Ia distance moduli at specified redshifts.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    z_sn : list, optional
        SN redshifts (default: SN_REDSHIFTS)

    Returns
    -------
    dict
        {z: mu} for each redshift
    """
    if z_sn is None:
        z_sn = SN_REDSHIFTS

    mu_dict = {}
    for z in z_sn:
        mu_dict[z] = distance_modulus(z, cosmo, spec)

    return mu_dict


def compute_chi2_sn(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    use_shoes_prior: bool = False,
    z_sn: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Compute chi-squared for SN constraints.

    For this simplified analysis, we compare the shape of the distance-redshift
    relation to ΛCDM. The absolute calibration (M_B or H0) is treated separately.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    use_shoes_prior : bool
        If True, add chi2 from SH0ES H0 constraint
    z_sn : list, optional
        SN redshifts

    Returns
    -------
    tuple
        (chi2_sn, max_dev_percent, mu_dict)
    """
    if z_sn is None:
        z_sn = SN_REDSHIFTS

    # Get model distances
    mu_dict = compute_sn_distances(cosmo, spec, z_sn)

    # Get LCDM reference
    spec_zero = TerminalSpectrumParams(modes=[])
    mu_lcdm = compute_sn_distances(cosmo, spec_zero, z_sn)

    # Compute shape comparison (relative to LCDM)
    # Use Δμ = μ_model - μ_LCDM at z > 0.01
    chi2_shape = 0.0
    max_dev_percent = 0.0

    # Typical Pantheon+ uncertainties
    sigma_mu = 0.15  # ~0.15 mag typical uncertainty

    for z in z_sn:
        if z < 0.01:
            continue

        mu_mod = mu_dict[z]
        mu_ref = mu_lcdm[z]

        if not np.isfinite(mu_mod) or not np.isfinite(mu_ref):
            chi2_shape += 1e10
            continue

        # Shape deviation (relative to low-z anchor)
        delta_mu = mu_mod - mu_ref

        chi2_shape += (delta_mu / sigma_mu)**2

        # Track max deviation in distance
        # D_L ratio: 10^(Δμ/5) - 1
        DL_ratio = 10**(delta_mu / 5)
        dev_percent = abs(DL_ratio - 1.0) * 100
        max_dev_percent = max(max_dev_percent, dev_percent)

    # Optional SH0ES prior
    chi2_H0 = 0.0
    if use_shoes_prior:
        H0_eff = get_H0_effective(cosmo, spec)
        H0_shoes = 73.04
        sigma_H0 = 1.04
        chi2_H0 = ((H0_eff - H0_shoes) / sigma_H0)**2

    chi2_total = chi2_shape + chi2_H0

    return chi2_total, max_dev_percent, mu_dict


# =============================================================================
# CMB Constraint
# =============================================================================

def compute_chi2_cmb(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    rs: Optional[float] = None,
    theta_star_ref: float = THETA_STAR_REF,
    theta_star_sigma: float = THETA_STAR_SIGMA,
) -> Tuple[float, float, float]:
    """
    Compute chi-squared for CMB θ* constraint.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    rs : float, optional
        Sound horizon (default: compute from model)
    theta_star_ref : float
        Reference θ* value (Planck 2018)
    theta_star_sigma : float
        Uncertainty on θ*

    Returns
    -------
    tuple
        (chi2_cmb, theta_star, theta_star_dev_percent)
    """
    theta_star = compute_theta_star(cosmo, spec, rs=rs)

    if not np.isfinite(theta_star):
        return 1e10, np.nan, np.nan

    chi2 = ((theta_star - theta_star_ref) / theta_star_sigma)**2

    dev_percent = abs(theta_star / theta_star_ref - 1.0) * 100

    return chi2, theta_star, dev_percent


# =============================================================================
# Combined Observable Computation
# =============================================================================

def compute_spectrum_observables(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    z_sn_grid: Optional[List[float]] = None,
    z_rec: float = Z_STAR,
) -> SpectrumObservables:
    """
    Compute all cosmological observables for a terminal spectrum configuration.

    This is the main observable computation function.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    z_sn_grid : list, optional
        Redshifts for SN distance computation
    z_rec : float
        Recombination redshift

    Returns
    -------
    SpectrumObservables
        Complete set of observables
    """
    if z_sn_grid is None:
        z_sn_grid = SN_REDSHIFTS

    # Effective H0
    H0_eff = get_H0_effective(cosmo, spec)

    # Sound horizon
    rs_drag = compute_rs_drag(cosmo, spec)

    # Comoving distance to recombination (for θ* calculation)
    D_C_rec = comoving_distance(z_rec, cosmo, spec)

    # Angular diameter distance to recombination
    DA_rec = D_C_rec / (1 + z_rec) if D_C_rec > 0 else 0.0

    # CMB acoustic angle (uses comoving distance D_M = χ, not D_A)
    theta_star = rs_drag / D_C_rec if D_C_rec > 0 else np.nan

    # H(z) on grid
    z_grid = np.concatenate([
        np.linspace(0, 2, 50),
        np.linspace(2, 10, 20),
        np.logspace(1, 4, 30)
    ])
    z_grid = np.unique(np.sort(z_grid))

    H_of_z = np.array([compute_modified_H_of_z(z, cosmo, spec) for z in z_grid])

    # Luminosity distances
    DL_of_z = np.array([luminosity_distance(z, cosmo, spec) for z in z_grid])

    return SpectrumObservables(
        H0=H0_eff,
        rs_drag=rs_drag,
        DA_rec=DA_rec,
        theta_star=theta_star,
        H_of_z=H_of_z,
        z_grid=z_grid,
        DL_of_z=DL_of_z,
    )


def compute_full_chi2(
    cosmo: SpectrumCosmoConfig,
    spec: TerminalSpectrumParams,
    theta_star_tol: float = 0.1,  # percent
    bao_tol: float = 2.0,  # percent
    sn_tol: float = 2.0,  # percent
    use_shoes_prior: bool = False,
) -> SpectrumChi2Result:
    """
    Compute total chi-squared and pass/fail flags for all constraints.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    spec : TerminalSpectrumParams
        Terminal spectrum parameters
    theta_star_tol : float
        Tolerance for θ* deviation [%]
    bao_tol : float
        Tolerance for BAO deviation [%]
    sn_tol : float
        Tolerance for SN deviation [%]
    use_shoes_prior : bool
        Include SH0ES H0 prior in SN chi2

    Returns
    -------
    SpectrumChi2Result
        Complete chi2 result with pass/fail flags
    """
    # Check physical validity first
    validity = check_physical_validity(cosmo, spec)
    is_physical = validity["valid"]
    H0_eff = validity["H0_eff"]

    if not is_physical:
        return SpectrumChi2Result(
            chi2_total=1e10,
            chi2_cmb=1e10,
            chi2_bao=1e10,
            chi2_sn=1e10,
            H0_eff=H0_eff,
            theta_star=np.nan,
            theta_star_dev_percent=np.nan,
            max_bao_dev_percent=np.nan,
            max_sn_dev_percent=np.nan,
            is_physical=False,
            passes_theta_star=False,
            passes_bao=False,
            passes_sn=False,
        )

    # Compute sound horizon (shared by CMB and BAO)
    rs = compute_rs_drag(cosmo, spec)

    # CMB constraint
    chi2_cmb, theta_star, theta_dev = compute_chi2_cmb(cosmo, spec, rs=rs)

    # BAO constraint
    chi2_bao, bao_dev, _ = compute_chi2_bao(cosmo, spec, r_d=rs * 0.99)

    # SN constraint
    chi2_sn, sn_dev, _ = compute_chi2_sn(cosmo, spec, use_shoes_prior=use_shoes_prior)

    # Total chi2
    chi2_total = chi2_cmb + chi2_bao + chi2_sn

    # Pass/fail flags
    passes_theta_star = theta_dev < theta_star_tol if np.isfinite(theta_dev) else False
    passes_bao = bao_dev < bao_tol if np.isfinite(bao_dev) else False
    passes_sn = sn_dev < sn_tol if np.isfinite(sn_dev) else False

    return SpectrumChi2Result(
        chi2_total=chi2_total,
        chi2_cmb=chi2_cmb,
        chi2_bao=chi2_bao,
        chi2_sn=chi2_sn,
        H0_eff=H0_eff,
        theta_star=theta_star,
        theta_star_dev_percent=theta_dev,
        max_bao_dev_percent=bao_dev,
        max_sn_dev_percent=sn_dev,
        is_physical=is_physical,
        passes_theta_star=passes_theta_star,
        passes_bao=passes_bao,
        passes_sn=passes_sn,
    )


def compute_baseline_chi2(
    cosmo: SpectrumCosmoConfig,
    use_shoes_prior: bool = False,
) -> float:
    """
    Compute chi-squared for baseline ΛCDM (zero modes).

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    use_shoes_prior : bool
        Include SH0ES prior

    Returns
    -------
    float
        Baseline chi-squared
    """
    spec_zero = TerminalSpectrumParams(modes=[])
    result = compute_full_chi2(cosmo, spec_zero, use_shoes_prior=use_shoes_prior)
    return result.chi2_total
