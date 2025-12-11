"""
hrc2.restframe.frames - Rest-frame velocity definitions and geometry

This module provides:
1. RestFrameDefinition dataclass for velocity vectors
2. CMB dipole parameters (canonical ~369 km/s)
3. Radio dipole parameters (controversial ~1000 km/s)
4. Kinematic redshift transformations between frames
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

C_LIGHT = 299792.458  # km/s


# =============================================================================
# Rest-Frame Definition
# =============================================================================

@dataclass
class RestFrameDefinition:
    """
    Defines a rest frame as a velocity vector relative to the CMB.

    The CMB defines the "canonical" cosmic rest frame with v=0.
    Alternative rest frames (e.g., radio dipole) are specified by
    their velocity relative to CMB.

    Parameters
    ----------
    name : str
        Identifier for this rest frame (e.g., "CMB", "Radio", "LG")
    v_mag : float
        Magnitude of velocity relative to CMB [km/s]
    l_apex : float
        Galactic longitude of apex [degrees]
    b_apex : float
        Galactic latitude of apex [degrees]
    """
    name: str
    v_mag: float  # km/s
    l_apex: float  # degrees (Galactic longitude)
    b_apex: float  # degrees (Galactic latitude)

    def get_cartesian_velocity(self) -> np.ndarray:
        """
        Convert velocity to Cartesian coordinates (Galactic).

        Returns unit vector * v_mag in (x, y, z) Galactic coordinates.
        """
        l_rad = np.radians(self.l_apex)
        b_rad = np.radians(self.b_apex)

        # Galactic coordinates: x toward GC, y toward rotation, z toward NGP
        vx = self.v_mag * np.cos(b_rad) * np.cos(l_rad)
        vy = self.v_mag * np.cos(b_rad) * np.sin(l_rad)
        vz = self.v_mag * np.sin(b_rad)

        return np.array([vx, vy, vz])


# =============================================================================
# Standard Rest Frames
# =============================================================================

def get_cmb_frame() -> RestFrameDefinition:
    """
    Return the CMB dipole rest frame (Planck 2018).

    The CMB dipole corresponds to our motion relative to the CMB:
    v = 369.82 ± 0.11 km/s toward (l, b) = (264.021°, 48.253°)

    In the CMB frame, v = 0 by definition (this is the reference).
    """
    return RestFrameDefinition(
        name="CMB",
        v_mag=0.0,  # Zero by definition (reference frame)
        l_apex=264.021,
        b_apex=48.253,
    )


def get_heliocentric_velocity() -> RestFrameDefinition:
    """
    Return the Sun's velocity relative to CMB (for heliocentric corrections).

    This is the actual CMB dipole measurement - our heliocentric velocity
    relative to the cosmic rest frame.
    """
    return RestFrameDefinition(
        name="Helio->CMB",
        v_mag=369.82,  # Planck 2018: 369.82 ± 0.11 km/s
        l_apex=264.021,  # l = 264.021° ± 0.011°
        b_apex=48.253,   # b = 48.253° ± 0.005°
    )


def get_radio_dipole_frame(v_mag: float = 1000.0) -> RestFrameDefinition:
    """
    Return the radio dipole rest frame.

    The radio galaxy dipole suggests a larger velocity amplitude.
    Recent analyses (Secrest et al. 2021, 2022) find ~2-3x larger
    dipole than expected from CMB alone, suggesting either:
    1. Large local bulk flow
    2. Different cosmic rest frame
    3. Systematic issues

    Parameters
    ----------
    v_mag : float
        Velocity magnitude [km/s]. Default 1000 km/s (~2.7x CMB dipole).
        Typical range explored: 600-1200 km/s.

    Notes
    -----
    Direction is approximately toward (l, b) ≈ (270°, 45°) - similar to
    CMB dipole direction, suggesting alignment but larger amplitude.
    """
    return RestFrameDefinition(
        name="Radio",
        v_mag=v_mag,
        l_apex=270.0,  # Approximate direction
        b_apex=45.0,
    )


def get_local_group_frame() -> RestFrameDefinition:
    """
    Return the Local Group barycenter velocity relative to CMB.

    The Local Group moves at ~627 km/s relative to CMB toward
    (l, b) ≈ (276°, 30°).
    """
    return RestFrameDefinition(
        name="LG",
        v_mag=627.0,
        l_apex=276.0,
        b_apex=30.0,
    )


# =============================================================================
# Kinematic Redshift Transformations
# =============================================================================

def angular_separation(l1: float, b1: float, l2: float, b2: float) -> float:
    """
    Compute angular separation between two directions in Galactic coordinates.

    Parameters
    ----------
    l1, b1 : float
        First direction [degrees]
    l2, b2 : float
        Second direction [degrees]

    Returns
    -------
    float
        Angular separation [radians]
    """
    l1_rad, b1_rad = np.radians(l1), np.radians(b1)
    l2_rad, b2_rad = np.radians(l2), np.radians(b2)

    cos_sep = (np.sin(b1_rad) * np.sin(b2_rad) +
               np.cos(b1_rad) * np.cos(b2_rad) * np.cos(l1_rad - l2_rad))

    # Clamp to [-1, 1] for numerical stability
    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    return np.arccos(cos_sep)


def compute_los_velocity(
    l_sn: float,
    b_sn: float,
    frame: RestFrameDefinition,
) -> float:
    """
    Compute line-of-sight velocity component toward a sky position.

    Parameters
    ----------
    l_sn, b_sn : float
        Galactic coordinates of the SN [degrees]
    frame : RestFrameDefinition
        Rest frame definition with velocity vector

    Returns
    -------
    float
        Line-of-sight velocity [km/s], positive = recession
    """
    if frame.v_mag == 0.0:
        return 0.0

    # Get velocity vector in Cartesian
    v_cart = frame.get_cartesian_velocity()

    # Get unit vector toward SN
    l_rad = np.radians(l_sn)
    b_rad = np.radians(b_sn)

    n_sn = np.array([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad)
    ])

    # Dot product gives line-of-sight component
    return np.dot(v_cart, n_sn)


def compute_kinematic_redshift(
    z_cosmo: float,
    l_sn: float,
    b_sn: float,
    frame: RestFrameDefinition,
) -> float:
    """
    Compute observed redshift including kinematic correction.

    The observed heliocentric redshift includes both cosmological
    expansion and our peculiar motion relative to the cosmic rest frame.

    Parameters
    ----------
    z_cosmo : float
        Cosmological redshift (in true rest frame)
    l_sn, b_sn : float
        Galactic coordinates of the SN [degrees]
    frame : RestFrameDefinition
        Our velocity relative to the true rest frame

    Returns
    -------
    float
        Observed (heliocentric) redshift

    Notes
    -----
    Uses the relativistic formula:
    (1 + z_obs) = (1 + z_cosmo) * (1 + v_los/c)

    For small v/c, approximately:
    z_obs ≈ z_cosmo + v_los/c
    """
    v_los = compute_los_velocity(l_sn, b_sn, frame)

    # Relativistic Doppler
    z_kin = v_los / C_LIGHT

    # Combine: (1 + z_obs) = (1 + z_cosmo)(1 + z_kin)
    z_obs = (1 + z_cosmo) * (1 + z_kin) - 1

    return z_obs


def correct_redshift_to_frame(
    z_helio: float,
    l_sn: float,
    b_sn: float,
    helio_velocity: RestFrameDefinition,
) -> float:
    """
    Correct heliocentric redshift to a rest frame.

    Parameters
    ----------
    z_helio : float
        Heliocentric (observed) redshift
    l_sn, b_sn : float
        Galactic coordinates of the SN [degrees]
    helio_velocity : RestFrameDefinition
        Our heliocentric velocity relative to the target rest frame

    Returns
    -------
    float
        Redshift in the specified rest frame

    Notes
    -----
    This is the inverse of compute_kinematic_redshift.
    """
    v_los = compute_los_velocity(l_sn, b_sn, helio_velocity)
    z_kin = v_los / C_LIGHT

    # Invert: (1 + z_cosmo) = (1 + z_helio) / (1 + z_kin)
    z_frame = (1 + z_helio) / (1 + z_kin) - 1

    return z_frame


def compute_dipole_modulation(
    l_sn: float,
    b_sn: float,
    frame: RestFrameDefinition,
    z_cosmo: float,
) -> float:
    """
    Compute the fractional modulation in distance modulus due to frame velocity.

    This is the key quantity for understanding H0 bias from wrong rest frame.

    Parameters
    ----------
    l_sn, b_sn : float
        Galactic coordinates [degrees]
    frame : RestFrameDefinition
        Velocity frame
    z_cosmo : float
        Cosmological redshift

    Returns
    -------
    float
        Fractional change in apparent distance modulus [mag]

    Notes
    -----
    At low z, the apparent magnitude shift is approximately:
    Δμ ≈ (5/ln(10)) * (v_los/c) / z_cosmo

    For typical values (v ~ 400 km/s, z ~ 0.03):
    Δμ ~ (5/2.3) * (400/3e5) / 0.03 ~ 0.1 mag

    This can bias H0 by several percent.
    """
    if frame.v_mag == 0.0 or z_cosmo <= 0.0:
        return 0.0

    v_los = compute_los_velocity(l_sn, b_sn, frame)

    # Fractional distance modulus change
    # More accurate: Δμ ≈ (5/ln(10)) * v_los / (c * z)
    delta_mu = (5.0 / np.log(10)) * (v_los / C_LIGHT) / z_cosmo

    return delta_mu
