"""
Terminal Spectrum / Multi-Mode Knob Cosmology

SIMULATION 25: A phenomenological framework for exploring how a "terminal
core release" (e.g., from a parent black hole interior quiet state) could
leave a spectral imprint on the expansion history H(z).

The model parameterizes deviations from ΛCDM as a sum of localized modes
in ln(a) space:

    δH/H(a) = Σ_i A_i * f_i(ln a; μ_i, σ_i, φ_i)

This is a "multi-mode knob" that can be tuned to explore what perturbations
(if any) to the expansion history are allowed by CMB + BAO + SN data while
potentially shifting H0 toward higher values.

IMPORTANT: This is phenomenological parameterization, NOT derived from a
specific quantum gravity model. It provides a systematic way to map out
the allowed space of expansion history modifications.

Modules
-------
mode_spectrum : Core mode definitions and H(z) computation
observables : Cosmological observables (θ*, distances, chi²)
"""

from .mode_spectrum import (
    # Dataclasses
    TerminalMode,
    TerminalSpectrumParams,
    SpectrumCosmoConfig,
    # Profile functions
    mode_profile_ln_a,
    delta_H_over_H_ln_a,
    delta_H_over_H_of_z,
    # Background functions
    E_squared_LCDM,
    H_LCDM,
    # Modified H(z)
    compute_modified_H_of_z,
    E_modified,
    get_H0_effective,
    # Validity checks
    check_physical_validity,
    # Convenience constructors
    make_3mode_template,
    make_single_mode,
    make_zero_spectrum,
)

from .observables import (
    # Dataclasses
    SpectrumObservables,
    SpectrumChi2Result,
    BAODataPoint,
    # Distance functions
    comoving_distance,
    comoving_distance_LCDM,
    angular_diameter_distance,
    luminosity_distance,
    hubble_distance,
    volume_average_distance,
    distance_modulus,
    # Sound horizon
    compute_rs_drag,
    compute_theta_star,
    # Constraint functions
    compute_bao_distances,
    compute_chi2_bao,
    compute_sn_distances,
    compute_chi2_sn,
    compute_chi2_cmb,
    # Combined
    compute_spectrum_observables,
    compute_full_chi2,
    compute_baseline_chi2,
    # Constants
    THETA_STAR_REF,
    THETA_STAR_SIGMA,
    Z_STAR,
    Z_DRAG,
    RS_LCDM,
    BAO_DATA,
    SN_REDSHIFTS,
)

__all__ = [
    # Mode spectrum dataclasses
    "TerminalMode",
    "TerminalSpectrumParams",
    "SpectrumCosmoConfig",
    # Profile functions
    "mode_profile_ln_a",
    "delta_H_over_H_ln_a",
    "delta_H_over_H_of_z",
    # Background functions
    "E_squared_LCDM",
    "H_LCDM",
    # Modified H(z)
    "compute_modified_H_of_z",
    "E_modified",
    "get_H0_effective",
    # Validity checks
    "check_physical_validity",
    # Convenience constructors
    "make_3mode_template",
    "make_single_mode",
    "make_zero_spectrum",
    # Observable dataclasses
    "SpectrumObservables",
    "SpectrumChi2Result",
    "BAODataPoint",
    # Distance functions
    "comoving_distance",
    "comoving_distance_LCDM",
    "angular_diameter_distance",
    "luminosity_distance",
    "hubble_distance",
    "volume_average_distance",
    "distance_modulus",
    # Sound horizon
    "compute_rs_drag",
    "compute_theta_star",
    # Constraint functions
    "compute_bao_distances",
    "compute_chi2_bao",
    "compute_sn_distances",
    "compute_chi2_sn",
    "compute_chi2_cmb",
    # Combined
    "compute_spectrum_observables",
    "compute_full_chi2",
    "compute_baseline_chi2",
    # Constants
    "THETA_STAR_REF",
    "THETA_STAR_SIGMA",
    "Z_STAR",
    "Z_DRAG",
    "RS_LCDM",
    "BAO_DATA",
    "SN_REDSHIFTS",
]
