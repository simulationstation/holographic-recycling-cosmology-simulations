"""
Black-Hole Interior Transition Computation Cosmology (BITCC) Module

SIMULATION 24: This module implements a phenomenological model where black-hole
interiors undergo alternating quiet (computational) and noise (mass-inflation)
phases. The transitional computational window produces an emergent scalar χ_trans
that determines the initial expansion scale H_init for a daughter universe.

Key components:
- BITCCInteriorParams: Configuration of a single BH interior
- BITCCHyperparams: Prior distributions over interior parameters
- compute_chi_trans(): Map interior params to computational residue
- map_chi_to_H_init(): Map χ_trans to initial Hubble scale
- sample_bitcc_prior(): Sample from the BITCC prior over H0
- BITCCCosmoParams: Full cosmological parameters from H_init
- compute_distance_ladder_proxies(): Distance measurements for data comparison

This is NOT claimed to be physically true. We use it to explore whether such
priors naturally prefer certain H0 values, and how they interact with data.
"""

from .bitcc_internal import (
    BITCCInteriorParams,
    BITCCHyperparams,
    BITCCDerivedParams,
    compute_chi_trans,
    map_chi_to_H_init,
    sample_interiors,
    compute_derived_params,
)

from .bitcc_prior_predictive import (
    BITCCPriorSample,
    sample_bitcc_prior,
    compute_H0_distribution_from_bitcc,
    compute_chi_trans_statistics,
    run_bitcc_prior_predictive,
    extract_arrays_from_samples,
)

from .bitcc_cosmo_mapping import (
    BITCCCosmoParams,
    map_H_init_to_cosmo,
    compute_distance_ladder_proxies,
    check_data_compatibility,
    compute_approximate_chi2,
    comoving_distance,
    angular_diameter_distance,
    luminosity_distance,
)

__all__ = [
    # Internal model
    "BITCCInteriorParams",
    "BITCCHyperparams",
    "BITCCDerivedParams",
    "compute_chi_trans",
    "map_chi_to_H_init",
    "sample_interiors",
    "compute_derived_params",
    # Prior predictive
    "BITCCPriorSample",
    "sample_bitcc_prior",
    "compute_H0_distribution_from_bitcc",
    "compute_chi_trans_statistics",
    "run_bitcc_prior_predictive",
    "extract_arrays_from_samples",
    # Cosmological mapping
    "BITCCCosmoParams",
    "map_H_init_to_cosmo",
    "compute_distance_ladder_proxies",
    "check_data_compatibility",
    "compute_approximate_chi2",
    "comoving_distance",
    "angular_diameter_distance",
    "luminosity_distance",
]
