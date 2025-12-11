"""
Black-Hole Interior Transition Computation Cosmology (BITCC) - Cosmological Mapping

This module provides a simple mapping from H_init (from BITCC) to full
cosmological parameters, and computes distance proxies for data comparison.

The mapping is phenomenological and designed to keep the cosmology roughly
consistent with observations while exploring the effects of the BITCC prior.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.integrate import quad


# Physical constants
C_KM_S = 299792.458  # Speed of light [km/s]

# Reference cosmology (Planck 2018-like)
H0_REF_COSMO = 67.5
OMEGA_M_REF = 0.315
OMEGA_L_REF = 0.685
Z_STAR = 1089.92  # Recombination redshift


@dataclass
class BITCCCosmoParams:
    """
    Cosmological parameters derived from BITCC H_init.

    Attributes
    ----------
    H0 : float
        Hubble constant [km/s/Mpc]

    Omega_m : float
        Matter density parameter (today)

    Omega_L : float
        Dark energy (cosmological constant) density parameter
        Note: Omega_m + Omega_L = 1 (flat universe)
    """
    H0: float
    Omega_m: float
    Omega_L: float

    def __post_init__(self):
        """Validate flatness."""
        if abs(self.Omega_m + self.Omega_L - 1.0) > 0.001:
            raise ValueError(
                f"Cosmology must be flat: Omega_m + Omega_L = "
                f"{self.Omega_m + self.Omega_L:.4f} != 1.0"
            )


def map_H_init_to_cosmo(
    H_init: float,
    Omega_m_ref: float = OMEGA_M_REF,
    Omega_L_ref: float = OMEGA_L_REF,
    k_Omega: float = 0.01,
) -> BITCCCosmoParams:
    """
    Map H_init from BITCC to full cosmological parameters.

    This is a simple phenomenological mapping that:
    - Sets H0 = H_init directly
    - Adjusts Omega_m slightly with H0 to preserve rough distance scales
    - Maintains flatness (Omega_m + Omega_L = 1)

    Parameters
    ----------
    H_init : float
        Initial expansion scale from BITCC [km/s/Mpc]

    Omega_m_ref : float, default=0.315
        Reference matter density (Planck-like)

    Omega_L_ref : float, default=0.685
        Reference dark energy density (Planck-like)

    k_Omega : float, default=0.01
        Coefficient controlling how Omega_m varies with H0.
        Positive k_Omega means higher H0 => slightly higher Omega_m.
        This is a weak effect by design.

    Returns
    -------
    BITCCCosmoParams
        The cosmological parameters.

    Notes
    -----
    The mapping is:
        x = (H_init - H0_ref) / H0_ref
        Omega_m = Omega_m_ref * (1 + k_Omega * x)
        Omega_L = 1 - Omega_m

    This is a toy mapping to keep things roughly self-consistent.
    The small k_Omega ensures Omega_m stays close to observed values.
    """
    # Fractional deviation from reference H0
    x = (H_init - H0_REF_COSMO) / H0_REF_COSMO

    # Adjust Omega_m (weak dependence)
    Omega_m = Omega_m_ref * (1.0 + k_Omega * x)

    # Clip to physical range
    Omega_m = np.clip(Omega_m, 0.1, 0.9)

    # Maintain flatness
    Omega_L = 1.0 - Omega_m

    return BITCCCosmoParams(
        H0=H_init,
        Omega_m=float(Omega_m),
        Omega_L=float(Omega_L),
    )


def _E_of_z(z: float, Omega_m: float, Omega_L: float) -> float:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0 for flat LCDM.

    E^2(z) = Omega_m * (1+z)^3 + Omega_L
    """
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)


def comoving_distance(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """
    Compute comoving distance D_C(z) in Mpc.

    D_C = (c/H0) * integral_0^z dz'/E(z')

    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant [km/s/Mpc]
    Omega_m : float
        Matter density parameter
    Omega_L : float
        Dark energy density parameter

    Returns
    -------
    float
        Comoving distance [Mpc]
    """
    if z <= 0:
        return 0.0

    def integrand(zp):
        return 1.0 / _E_of_z(zp, Omega_m, Omega_L)

    limit = 200 if z > 100 else 100
    result, _ = quad(integrand, 0, z, limit=limit, epsrel=1e-9)

    return (C_KM_S / H0) * result


def angular_diameter_distance(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """
    Angular diameter distance D_A(z) in Mpc.

    For flat universe: D_A = D_C / (1 + z)
    """
    D_C = comoving_distance(z, H0, Omega_m, Omega_L)
    return D_C / (1 + z)


def luminosity_distance(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """
    Luminosity distance D_L(z) in Mpc.

    For flat universe: D_L = (1 + z) * D_C
    """
    D_C = comoving_distance(z, H0, Omega_m, Omega_L)
    return (1 + z) * D_C


def compute_reference_distances() -> Dict[str, float]:
    """
    Compute reference distances for a Planck-like cosmology.

    Returns
    -------
    dict
        Reference distances at standard redshifts.
    """
    H0 = H0_REF_COSMO
    Om = OMEGA_M_REF
    OL = OMEGA_L_REF

    return {
        "D_L_z0p1": luminosity_distance(0.1, H0, Om, OL),
        "D_L_z0p3": luminosity_distance(0.3, H0, Om, OL),
        "D_L_z0p5": luminosity_distance(0.5, H0, Om, OL),
        "D_L_z1p0": luminosity_distance(1.0, H0, Om, OL),
        "D_A_z_star": angular_diameter_distance(Z_STAR, H0, Om, OL),
        "D_C_z_star": comoving_distance(Z_STAR, H0, Om, OL),
    }


# Pre-compute reference distances
_REF_DISTANCES = None


def get_reference_distances() -> Dict[str, float]:
    """Get cached reference distances."""
    global _REF_DISTANCES
    if _REF_DISTANCES is None:
        _REF_DISTANCES = compute_reference_distances()
    return _REF_DISTANCES


def compute_distance_ladder_proxies(
    cosmo: BITCCCosmoParams,
) -> Dict[str, float]:
    """
    Compute distance proxies for comparison with data.

    This function computes:
    - Luminosity distances at z = 0.1, 0.3, 0.5, 1.0 (SN Ia redshifts)
    - Angular diameter distance at z = z_* (CMB)
    - Fractional deviations from Planck-like reference

    Parameters
    ----------
    cosmo : BITCCCosmoParams
        Cosmological parameters from BITCC mapping.

    Returns
    -------
    dict
        Distance proxies and fractional deviations:
        - D_L_z0p1, D_L_z0p3, D_L_z0p5, D_L_z1p0: Luminosity distances [Mpc]
        - D_A_z_star: Angular diameter distance to CMB [Mpc]
        - delta_D_L_z0p1, ...: Fractional deviations from reference
        - delta_D_A_z_star: Fractional deviation of CMB distance
    """
    ref = get_reference_distances()

    H0 = cosmo.H0
    Om = cosmo.Omega_m
    OL = cosmo.Omega_L

    # Compute distances
    D_L_z0p1 = luminosity_distance(0.1, H0, Om, OL)
    D_L_z0p3 = luminosity_distance(0.3, H0, Om, OL)
    D_L_z0p5 = luminosity_distance(0.5, H0, Om, OL)
    D_L_z1p0 = luminosity_distance(1.0, H0, Om, OL)
    D_A_z_star = angular_diameter_distance(Z_STAR, H0, Om, OL)

    # Compute fractional deviations
    delta_D_L_z0p1 = (D_L_z0p1 - ref["D_L_z0p1"]) / ref["D_L_z0p1"]
    delta_D_L_z0p3 = (D_L_z0p3 - ref["D_L_z0p3"]) / ref["D_L_z0p3"]
    delta_D_L_z0p5 = (D_L_z0p5 - ref["D_L_z0p5"]) / ref["D_L_z0p5"]
    delta_D_L_z1p0 = (D_L_z1p0 - ref["D_L_z1p0"]) / ref["D_L_z1p0"]
    delta_D_A_z_star = (D_A_z_star - ref["D_A_z_star"]) / ref["D_A_z_star"]

    return {
        # Absolute distances
        "D_L_z0p1": D_L_z0p1,
        "D_L_z0p3": D_L_z0p3,
        "D_L_z0p5": D_L_z0p5,
        "D_L_z1p0": D_L_z1p0,
        "D_A_z_star": D_A_z_star,
        # Fractional deviations from reference
        "delta_D_L_z0p1": delta_D_L_z0p1,
        "delta_D_L_z0p3": delta_D_L_z0p3,
        "delta_D_L_z0p5": delta_D_L_z0p5,
        "delta_D_L_z1p0": delta_D_L_z1p0,
        "delta_D_A_z_star": delta_D_A_z_star,
    }


def check_data_compatibility(
    proxies: Dict[str, float],
    tol_D_A_z_star: float = 0.005,  # 0.5% tolerance on CMB distance
    tol_D_L_z0p5: float = 0.02,     # 2% tolerance on mid-z SN
    tol_D_L_z1p0: float = 0.03,     # 3% tolerance on high-z SN
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if a cosmology is compatible with observational constraints.

    This is a simplified compatibility check based on distance tolerances.
    A more rigorous analysis would use full likelihoods.

    Parameters
    ----------
    proxies : dict
        Distance proxies from compute_distance_ladder_proxies().

    tol_D_A_z_star : float, default=0.005
        Tolerance on CMB angular diameter distance (0.5%).

    tol_D_L_z0p5 : float, default=0.02
        Tolerance on D_L at z=0.5 (2%).

    tol_D_L_z1p0 : float, default=0.03
        Tolerance on D_L at z=1.0 (3%).

    Returns
    -------
    tuple
        (is_compatible, checks_dict)
        - is_compatible: True if all checks pass
        - checks_dict: Individual check results
    """
    checks = {
        "cmb_ok": abs(proxies["delta_D_A_z_star"]) < tol_D_A_z_star,
        "sn_z0p5_ok": abs(proxies["delta_D_L_z0p5"]) < tol_D_L_z0p5,
        "sn_z1p0_ok": abs(proxies["delta_D_L_z1p0"]) < tol_D_L_z1p0,
    }

    is_compatible = all(checks.values())

    return is_compatible, checks


def compute_approximate_chi2(
    proxies: Dict[str, float],
    sigma_D_A_z_star: float = 0.003,  # ~0.3% CMB precision
    sigma_D_L_z0p5: float = 0.02,     # ~2% SN precision at z=0.5
    sigma_D_L_z1p0: float = 0.03,     # ~3% SN precision at z=1.0
) -> float:
    """
    Compute an approximate chi-squared from distance deviations.

    This is a simplified chi-squared that penalizes deviations from
    the reference Planck-like cosmology.

    Parameters
    ----------
    proxies : dict
        Distance proxies from compute_distance_ladder_proxies().

    sigma_* : float
        Fractional uncertainties on each distance measurement.

    Returns
    -------
    float
        Approximate chi-squared value.
    """
    chi2 = 0.0

    # CMB constraint (tight)
    chi2 += (proxies["delta_D_A_z_star"] / sigma_D_A_z_star) ** 2

    # SN constraints (looser)
    chi2 += (proxies["delta_D_L_z0p5"] / sigma_D_L_z0p5) ** 2
    chi2 += (proxies["delta_D_L_z1p0"] / sigma_D_L_z1p0) ** 2

    return chi2
