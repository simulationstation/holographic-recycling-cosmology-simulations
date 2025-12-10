"""
HRC2 Components - Additional cosmological components beyond standard LCDM.

This module provides implementations of various exotic matter/energy components
that can be added to the standard cosmological model.
"""

from .black_hole_fertility import (
    BHFCRealParameters,
    bh_formation_window,
    rho_BH,
    rho_extra_rad,
    BHFCBackgroundCosmology,
)

__all__ = [
    'BHFCRealParameters',
    'bh_formation_window',
    'rho_BH',
    'rho_extra_rad',
    'BHFCBackgroundCosmology',
]
