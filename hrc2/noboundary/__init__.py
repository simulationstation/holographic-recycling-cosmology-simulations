"""
HRC2 Theory Module: Hawking-Hartle No-Boundary Cosmology

This module implements the Hawking-Hartle no-boundary framework for
cosmological parameter inference.

Submodules:
    no_boundary_prior: Prior distributions over primordial parameters
    no_boundary_to_cosmo: Mapping from primordial to cosmological parameters
"""

from .no_boundary_prior import (
    NoBoundaryHyperparams,
    NoBoundaryParams,
    NOBOUNDARY_PARAM_NAMES,
    NOBOUNDARY_NDIM,
    log_prior_no_boundary,
    sample_no_boundary_prior,
    get_prior_bounds,
    log_prior_from_array,
)

from .no_boundary_to_cosmo import (
    CosmoParams,
    InflationModel,
    compute_slow_roll_params,
    compute_primordial_spectrum,
    compute_curvature_from_efolds,
    compute_sound_horizon,
    compute_theta_star,
    primordial_to_cosmo,
    compute_early_H0,
    compute_late_H0,
)

__all__ = [
    # Prior module
    "NoBoundaryHyperparams",
    "NoBoundaryParams",
    "NOBOUNDARY_PARAM_NAMES",
    "NOBOUNDARY_NDIM",
    "log_prior_no_boundary",
    "sample_no_boundary_prior",
    "get_prior_bounds",
    "log_prior_from_array",
    # Mapping module
    "CosmoParams",
    "InflationModel",
    "compute_slow_roll_params",
    "compute_primordial_spectrum",
    "compute_curvature_from_efolds",
    "compute_sound_horizon",
    "compute_theta_star",
    "primordial_to_cosmo",
    "compute_early_H0",
    "compute_late_H0",
]
