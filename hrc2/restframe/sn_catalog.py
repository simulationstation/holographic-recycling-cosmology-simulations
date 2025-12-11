"""
hrc2.restframe.sn_catalog - SN Ia catalog with sky positions and redshifts

This module provides:
1. SNSample dataclass for Hubble flow SN Ia samples
2. Functions to generate isotropic or hemispherical sky distributions
3. Functions to generate realistic redshift distributions
4. Combined catalog generation with kinematic effects
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .frames import (
    RestFrameDefinition,
    compute_kinematic_redshift,
    correct_redshift_to_frame,
    compute_dipole_modulation,
    C_LIGHT,
)


# =============================================================================
# SN Sample Data Structure
# =============================================================================

@dataclass
class SNSample:
    """
    Container for a Hubble flow SN Ia sample.

    All arrays have the same length N_sn.
    """
    # Sky positions (Galactic coordinates)
    l_gal: np.ndarray  # degrees, shape (N_sn,)
    b_gal: np.ndarray  # degrees, shape (N_sn,)

    # Redshifts
    z_cosmo: np.ndarray   # True cosmological redshift
    z_helio: np.ndarray   # Observed heliocentric redshift

    # Distance modulus
    mu_true: np.ndarray   # True distance modulus (from cosmology)
    mu_obs: np.ndarray    # Observed distance modulus (with scatter)

    # Uncertainty
    sigma_mu: np.ndarray  # Distance modulus uncertainty

    @property
    def n_sn(self) -> int:
        return len(self.z_cosmo)


# =============================================================================
# Sky Distribution Generators
# =============================================================================

def generate_isotropic_sky_positions(
    n_sn: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate isotropic (uniform) sky positions in Galactic coordinates.

    Parameters
    ----------
    n_sn : int
        Number of positions to generate
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    l_gal : ndarray
        Galactic longitude [0, 360) degrees
    b_gal : ndarray
        Galactic latitude [-90, 90] degrees
    """
    # Uniform on sphere: l uniform, b from arcsin(uniform(-1,1))
    l_gal = rng.uniform(0, 360, n_sn)
    sin_b = rng.uniform(-1, 1, n_sn)
    b_gal = np.degrees(np.arcsin(sin_b))

    return l_gal, b_gal


def generate_hemispherical_sky_positions(
    n_sn: int,
    apex_l: float,
    apex_b: float,
    rng: np.random.Generator,
    hemisphere: str = "toward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sky positions concentrated in one hemisphere.

    This simulates incomplete sky coverage (e.g., N vs S hemisphere).

    Parameters
    ----------
    n_sn : int
        Number of positions
    apex_l, apex_b : float
        Apex direction defining the hemisphere [degrees]
    rng : np.random.Generator
        Random number generator
    hemisphere : str
        "toward" = toward apex, "away" = away from apex

    Returns
    -------
    l_gal, b_gal : ndarray
        Galactic coordinates [degrees]
    """
    # Generate isotropic positions first
    l_gal, b_gal = generate_isotropic_sky_positions(n_sn * 3, rng)

    # Compute angular distance from apex
    apex_l_rad = np.radians(apex_l)
    apex_b_rad = np.radians(apex_b)
    l_rad = np.radians(l_gal)
    b_rad = np.radians(b_gal)

    cos_theta = (np.sin(apex_b_rad) * np.sin(b_rad) +
                 np.cos(apex_b_rad) * np.cos(b_rad) * np.cos(l_rad - apex_l_rad))

    # Select hemisphere
    if hemisphere == "toward":
        mask = cos_theta > 0
    else:
        mask = cos_theta < 0

    l_selected = l_gal[mask][:n_sn]
    b_selected = b_gal[mask][:n_sn]

    # Ensure we have enough
    while len(l_selected) < n_sn:
        l_extra, b_extra = generate_isotropic_sky_positions(n_sn, rng)
        l_rad_extra = np.radians(l_extra)
        b_rad_extra = np.radians(b_extra)
        cos_theta_extra = (np.sin(apex_b_rad) * np.sin(b_rad_extra) +
                          np.cos(apex_b_rad) * np.cos(b_rad_extra) *
                          np.cos(l_rad_extra - apex_l_rad))
        if hemisphere == "toward":
            mask_extra = cos_theta_extra > 0
        else:
            mask_extra = cos_theta_extra < 0
        l_selected = np.concatenate([l_selected, l_extra[mask_extra]])
        b_selected = np.concatenate([b_selected, b_extra[mask_extra]])

    return l_selected[:n_sn], b_selected[:n_sn]


# =============================================================================
# Redshift Distribution Generators
# =============================================================================

def generate_redshifts_uniform(
    n_sn: int,
    z_min: float,
    z_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate uniformly distributed redshifts."""
    return rng.uniform(z_min, z_max, n_sn)


def generate_redshifts_volume_weighted(
    n_sn: int,
    z_min: float,
    z_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate redshifts weighted by comoving volume element.

    dN/dz ∝ dV_c/dz ∝ D_C^2 / E(z) ≈ z^2 for low z

    At low z, this is approximately z^2 weighting.
    """
    # Use rejection sampling with z^2 envelope
    z_vals = []
    while len(z_vals) < n_sn:
        z_try = rng.uniform(z_min, z_max, n_sn * 2)
        # Approximate acceptance probability ∝ z^2 / z_max^2
        accept_prob = (z_try / z_max) ** 2
        accept = rng.random(len(z_try)) < accept_prob
        z_vals.extend(z_try[accept])

    return np.array(z_vals[:n_sn])


def generate_redshifts_magnitude_limited(
    n_sn: int,
    z_min: float,
    z_max: float,
    m_lim: float,
    M_B: float,
    sigma_int: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate redshifts for a magnitude-limited sample.

    Accounts for Malmquist-type selection effects.

    Parameters
    ----------
    n_sn : int
        Number of SNe to generate
    z_min, z_max : float
        Redshift range
    m_lim : float
        Limiting magnitude
    M_B : float
        Mean absolute magnitude
    sigma_int : float
        Intrinsic scatter in absolute magnitude
    rng : np.random.Generator
        Random number generator
    """
    # For now, use volume-weighted and apply selection later
    return generate_redshifts_volume_weighted(n_sn, z_min, z_max, rng)


# =============================================================================
# Distance Modulus Functions
# =============================================================================

def mu_of_z_approx(z: np.ndarray, H0: float = 70.0) -> np.ndarray:
    """
    Approximate distance modulus for low-z Hubble flow.

    μ = 5 * log10(cz / H0) + 25

    Valid for z << 1 where peculiar velocities are small compared to cz.
    """
    # Avoid log of zero
    z_safe = np.maximum(z, 1e-10)
    return 5.0 * np.log10(C_LIGHT * z_safe / H0) + 25.0


def mu_of_z_flat(z: np.ndarray, H0: float = 70.0, Omega_m: float = 0.3) -> np.ndarray:
    """
    Distance modulus for flat ΛCDM cosmology.

    Uses numerical integration of E(z) = sqrt(Omega_m(1+z)^3 + Omega_Lambda).
    """
    from scipy import integrate

    Omega_L = 1.0 - Omega_m

    def E(z_val):
        return np.sqrt(Omega_m * (1 + z_val)**3 + Omega_L)

    def D_L(z_val):
        # Comoving distance (in Mpc)
        if z_val <= 0:
            return 0.0
        D_c, _ = integrate.quad(lambda zp: 1.0 / E(zp), 0, z_val)
        D_c *= C_LIGHT / H0  # Convert to Mpc
        return D_c * (1 + z_val)  # Luminosity distance

    # Vectorize
    if np.isscalar(z):
        D_L_arr = D_L(z)
    else:
        D_L_arr = np.array([D_L(zi) for zi in z])

    # Distance modulus
    D_L_safe = np.maximum(D_L_arr, 1e-10)
    return 5.0 * np.log10(D_L_safe) + 25.0


# =============================================================================
# Full Catalog Generation
# =============================================================================

def generate_sn_catalog(
    n_sn: int,
    z_min: float,
    z_max: float,
    H0_true: float,
    true_frame: RestFrameDefinition,
    helio_velocity: RestFrameDefinition,
    sigma_mu: float,
    rng: np.random.Generator,
    Omega_m: float = 0.3,
    sky_coverage: str = "isotropic",
) -> SNSample:
    """
    Generate a complete SN Ia Hubble flow catalog.

    Parameters
    ----------
    n_sn : int
        Number of SNe
    z_min, z_max : float
        Redshift range (cosmological)
    H0_true : float
        True Hubble constant [km/s/Mpc]
    true_frame : RestFrameDefinition
        The true cosmic rest frame (e.g., radio dipole)
    helio_velocity : RestFrameDefinition
        Our heliocentric velocity (for z_cosmo -> z_helio conversion)
    sigma_mu : float
        Distance modulus uncertainty (typical ~0.15 mag)
    rng : np.random.Generator
        Random number generator
    Omega_m : float
        Matter density parameter
    sky_coverage : str
        "isotropic", "toward_apex", or "away_from_apex"

    Returns
    -------
    SNSample
        Complete SN catalog with all observables
    """
    # Generate sky positions
    if sky_coverage == "isotropic":
        l_gal, b_gal = generate_isotropic_sky_positions(n_sn, rng)
    elif sky_coverage == "toward_apex":
        l_gal, b_gal = generate_hemispherical_sky_positions(
            n_sn, true_frame.l_apex, true_frame.b_apex, rng, "toward"
        )
    elif sky_coverage == "away_from_apex":
        l_gal, b_gal = generate_hemispherical_sky_positions(
            n_sn, true_frame.l_apex, true_frame.b_apex, rng, "away"
        )
    else:
        raise ValueError(f"Unknown sky_coverage: {sky_coverage}")

    # Generate cosmological redshifts
    z_cosmo = generate_redshifts_volume_weighted(n_sn, z_min, z_max, rng)

    # Compute heliocentric redshifts (what we observe)
    # The "true frame" defines where z_cosmo is measured
    # We observe z_helio which includes our motion
    z_helio = np.array([
        compute_kinematic_redshift(z_c, l, b, helio_velocity)
        for z_c, l, b in zip(z_cosmo, l_gal, b_gal)
    ])

    # True distance modulus (in true cosmology)
    mu_true = mu_of_z_flat(z_cosmo, H0_true, Omega_m)

    # Observed distance modulus (with scatter)
    mu_obs = mu_true + rng.normal(0, sigma_mu, n_sn)

    # Distance modulus uncertainty (constant for simplicity)
    sigma_arr = np.full(n_sn, sigma_mu)

    return SNSample(
        l_gal=l_gal,
        b_gal=b_gal,
        z_cosmo=z_cosmo,
        z_helio=z_helio,
        mu_true=mu_true,
        mu_obs=mu_obs,
        sigma_mu=sigma_arr,
    )


def generate_sn_catalog_with_radio_dipole(
    n_sn: int,
    z_min: float,
    z_max: float,
    H0_true: float,
    v_radio: float,
    sigma_mu: float,
    rng: np.random.Generator,
    Omega_m: float = 0.3,
    sky_coverage: str = "isotropic",
) -> SNSample:
    """
    Convenience function to generate catalog with radio dipole as true frame.

    Parameters
    ----------
    v_radio : float
        Radio dipole velocity [km/s], e.g., 600-1200 km/s
    """
    from .frames import get_radio_dipole_frame, get_heliocentric_velocity

    # The "true" cosmic rest frame has a larger velocity amplitude
    # This is the velocity we should correct for, but don't know about
    true_frame = get_radio_dipole_frame(v_radio)

    # Our heliocentric velocity relative to the TRUE frame
    # This is what we'd need to correct to get z_cosmo from z_helio
    # But we typically use CMB velocity (369 km/s) instead of true (v_radio)
    helio_velocity = RestFrameDefinition(
        name="Helio->True",
        v_mag=v_radio,
        l_apex=270.0,  # Approximate direction
        b_apex=45.0,
    )

    return generate_sn_catalog(
        n_sn=n_sn,
        z_min=z_min,
        z_max=z_max,
        H0_true=H0_true,
        true_frame=true_frame,
        helio_velocity=helio_velocity,
        sigma_mu=sigma_mu,
        rng=rng,
        Omega_m=Omega_m,
        sky_coverage=sky_coverage,
    )
