"""
HRC2 Habitability Module - Links cosmological parameters to habitability constraints.

This module provides the connection between habitability-optimal parameters
and cosmological model parameters, particularly for the BHFC (Black-Hole
Fertility Cosmology) framework.
"""

from .habitability_params import (
    A_HAB_STAR,
    map_A_hab_to_A_eff,
    get_fixed_A_eff,
)

__all__ = [
    'A_HAB_STAR',
    'map_A_hab_to_A_eff',
    'get_fixed_A_eff',
]
