#!/usr/bin/env python3
"""
Instrument photometry systematics for SIM 13: HST vs JWST Cepheid Recalibration.

Models instrument-dependent effects on Cepheid photometry:
- Zero-point offsets
- Color-term differences (bandpass mismatch)
- Non-linearity / count-rate terms
- Crowding/blending biases (environment-dependent)

These systematics affect the Cepheid PL calibration and propagate to H0.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class InstrumentPhotometrySystematics:
    """
    Instrument-dependent photometric systematics for Cepheid observations.

    Attributes:
        name: Instrument identifier (e.g., "HST", "JWST")
        zp_offset: Zero-point offset (mag). m_obs = m_true + zp_offset
        c_color: Color term coefficient (mag per mag).
                 m_obs += c_color * (color - color_ref)
        c_nonlinearity: Non-linearity coefficient (magnitude bias).
                        Approximates count-rate dependence.
        delta_mu_crowd: Crowding/blending bias (mag) for crowded fields.
                        Applied to high-mass (crowded) hosts.
    """
    name: str
    zp_offset: float = 0.0
    c_color: float = 0.0
    c_nonlinearity: float = 0.0
    delta_mu_crowd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "zp_offset": self.zp_offset,
            "c_color": self.c_color,
            "c_nonlinearity": self.c_nonlinearity,
            "delta_mu_crowd": self.delta_mu_crowd,
        }


def apply_instrument_effects(
    m_true: np.ndarray,
    color_true: np.ndarray,
    host_logM: np.ndarray,
    inst: InstrumentPhotometrySystematics,
    logM_crowd_threshold: float = 10.5,
    flux_ref: float = 1.0,
    color_ref: float = 0.0,
) -> np.ndarray:
    """
    Compute observed magnitudes m_obs for a given instrument.

    m_obs = m_true
          + zp_offset
          + c_color * (color_true - color_ref)
          + c_nonlinearity * (flux / flux_ref)
          + crowding bias for high-mass (crowded) hosts

    Args:
        m_true: True (instrument-independent) magnitudes
        color_true: True color indices for each Cepheid
        host_logM: log10(M_star/M_sun) for each Cepheid's host
        inst: Instrument systematics parameters
        logM_crowd_threshold: Host mass above which crowding bias applies
        flux_ref: Reference flux for non-linearity term
        color_ref: Reference color for color term

    Returns:
        Observed magnitudes with instrument systematics applied
    """
    m = np.array(m_true, dtype=float, copy=True)

    # Zero-point offset
    if inst.zp_offset != 0.0:
        m += inst.zp_offset

    # Color term: bandpass mismatch
    if inst.c_color != 0.0:
        m += inst.c_color * (color_true - color_ref)

    # Non-linearity: approximate flux from magnitude
    if inst.c_nonlinearity != 0.0:
        # flux ∝ 10^{-0.4 * m_true}
        flux = 10.0 ** (-0.4 * m_true)
        m += inst.c_nonlinearity * (flux / flux_ref)

    # Crowding bias in high-mass hosts (more stars = more blending)
    if inst.delta_mu_crowd != 0.0:
        mask_crowd = host_logM > logM_crowd_threshold
        m[mask_crowd] += inst.delta_mu_crowd

    return m


def compute_instrument_difference(
    inst_hst: InstrumentPhotometrySystematics,
    inst_jwst: InstrumentPhotometrySystematics,
) -> Dict[str, float]:
    """
    Compute the systematic differences between two instruments.

    Returns:
        Dictionary with zp_diff, c_color_diff, c_nl_diff, crowd_diff
    """
    return {
        "zp_diff": inst_jwst.zp_offset - inst_hst.zp_offset,
        "c_color_diff": inst_jwst.c_color - inst_hst.c_color,
        "c_nl_diff": inst_jwst.c_nonlinearity - inst_hst.c_nonlinearity,
        "crowd_diff": inst_jwst.delta_mu_crowd - inst_hst.delta_mu_crowd,
    }


def create_hst_baseline() -> InstrumentPhotometrySystematics:
    """
    Create baseline HST-like instrument (WFC3 F160W).

    HST has:
    - Nominal zero-point (reference)
    - Some crowding issues in dense stellar environments
    """
    return InstrumentPhotometrySystematics(
        name="HST",
        zp_offset=0.0,
        c_color=0.0,
        c_nonlinearity=0.0,
        delta_mu_crowd=0.03,  # ~3% distance bias from crowding
    )


def create_jwst_baseline() -> InstrumentPhotometrySystematics:
    """
    Create baseline JWST-like instrument (NIRCam F200W).

    JWST has:
    - Higher spatial resolution → less crowding
    - Different bandpass → potential color terms
    - Different detector → potential non-linearity differences
    """
    return InstrumentPhotometrySystematics(
        name="JWST",
        zp_offset=0.0,
        c_color=0.0,
        c_nonlinearity=0.0,
        delta_mu_crowd=0.00,  # Better resolution, less crowding
    )


def create_jwst_with_systematics(
    zp_diff: float = 0.0,
    c_color_diff: float = 0.0,
    c_nl_diff: float = 0.0,
    base_crowd: float = 0.0,
) -> InstrumentPhotometrySystematics:
    """
    Create JWST-like instrument with specified systematic differences from HST.

    Args:
        zp_diff: Zero-point difference JWST - HST (mag)
        c_color_diff: Color term difference JWST - HST (mag/mag)
        c_nl_diff: Non-linearity difference JWST - HST
        base_crowd: Base crowding bias for JWST (default: 0, better than HST)

    Returns:
        JWST instrument with systematics applied
    """
    return InstrumentPhotometrySystematics(
        name="JWST",
        zp_offset=zp_diff,  # Relative to HST baseline of 0
        c_color=c_color_diff,
        c_nonlinearity=c_nl_diff,
        delta_mu_crowd=base_crowd,
    )
