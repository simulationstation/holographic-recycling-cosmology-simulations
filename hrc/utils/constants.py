"""Physical constants for HRC cosmology.

All constants are provided in both SI and Planck units.
"""

from dataclasses import dataclass
from typing import Final
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants in specified units."""

    # Fundamental constants
    c: float  # Speed of light [m/s or 1]
    G: float  # Newton's constant [m³/(kg·s²) or 1]
    hbar: float  # Reduced Planck constant [J·s or 1]
    k_B: float  # Boltzmann constant [J/K or 1]

    # Planck scales
    M_Planck: float  # Planck mass [kg or 1]
    l_Planck: float  # Planck length [m or 1]
    t_Planck: float  # Planck time [s or 1]
    T_Planck: float  # Planck temperature [K or 1]

    # Cosmological
    H0_fiducial: float  # Fiducial Hubble constant [km/s/Mpc or reduced units]
    Mpc_in_m: float  # Megaparsec in meters
    yr_in_s: float  # Year in seconds

    # Derived
    rho_crit_h2: float  # Critical density × h² [kg/m³]


# SI units
SI_UNITS: Final[PhysicalConstants] = PhysicalConstants(
    c=2.99792458e8,  # m/s
    G=6.67430e-11,  # m³/(kg·s²)
    hbar=1.054571817e-34,  # J·s
    k_B=1.380649e-23,  # J/K
    M_Planck=2.176434e-8,  # kg
    l_Planck=1.616255e-35,  # m
    t_Planck=5.391247e-44,  # s
    T_Planck=1.416784e32,  # K
    H0_fiducial=70.0,  # km/s/Mpc
    Mpc_in_m=3.085677581e22,  # m
    yr_in_s=3.15576e7,  # s
    rho_crit_h2=1.87847e-26,  # kg/m³ (for h=1)
)


# Planck units (c = G = hbar = k_B = 1)
PLANCK_UNITS: Final[PhysicalConstants] = PhysicalConstants(
    c=1.0,
    G=1.0,
    hbar=1.0,
    k_B=1.0,
    M_Planck=1.0,
    l_Planck=1.0,
    t_Planck=1.0,
    T_Planck=1.0,
    H0_fiducial=1.0,  # Dimensionless in Planck units
    Mpc_in_m=1.0,
    yr_in_s=1.0,
    rho_crit_h2=1.0,
)


# Cosmological parameters from Planck 2018
@dataclass(frozen=True)
class Planck2018Values:
    """Planck 2018 cosmological parameters (TT,TE,EE+lowE+lensing)."""

    H0: float = 67.36  # km/s/Mpc
    H0_sigma: float = 0.54

    Omega_b_h2: float = 0.02237
    Omega_b_h2_sigma: float = 0.00015

    Omega_c_h2: float = 0.1200
    Omega_c_h2_sigma: float = 0.0012

    Omega_m: float = 0.3153
    Omega_m_sigma: float = 0.0073

    Omega_Lambda: float = 0.6847

    n_s: float = 0.9649
    n_s_sigma: float = 0.0042

    sigma8: float = 0.8111
    sigma8_sigma: float = 0.0060

    tau: float = 0.0544
    tau_sigma: float = 0.0073

    z_star: float = 1089.92  # Redshift at last scattering
    z_star_sigma: float = 0.25

    r_s: float = 144.43  # Sound horizon at last scattering [Mpc]
    r_s_sigma: float = 0.26

    theta_star: float = 1.04109e-2  # Angular size of sound horizon [rad]
    theta_star_sigma: float = 3.1e-6


PLANCK_2018: Final[Planck2018Values] = Planck2018Values()


# SH0ES measurement
@dataclass(frozen=True)
class SH0ESValues:
    """SH0ES 2024 local H0 measurement."""

    H0: float = 73.04  # km/s/Mpc
    H0_sigma: float = 1.04


SHOES_2024: Final[SH0ESValues] = SH0ESValues()


def convert_H0_to_si(H0_km_s_Mpc: float) -> float:
    """Convert H0 from km/s/Mpc to s⁻¹."""
    return H0_km_s_Mpc * 1e3 / SI_UNITS.Mpc_in_m


def convert_H0_to_natural(H0_km_s_Mpc: float) -> float:
    """Convert H0 to natural units (1/Mpc)."""
    return H0_km_s_Mpc / SI_UNITS.c * 1e3  # 1/Mpc


def hubble_time(H0_km_s_Mpc: float) -> float:
    """Return Hubble time in Gyr."""
    H0_si = convert_H0_to_si(H0_km_s_Mpc)
    return 1.0 / H0_si / SI_UNITS.yr_in_s / 1e9
