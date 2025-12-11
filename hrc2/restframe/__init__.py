"""
hrc2.restframe - Rest-Frame Misalignment Bias Module

SIMULATION 14: Tests how using the wrong cosmic rest frame (CMB dipole vs
radio dipole) can bias the inferred Hubble constant.

Key concepts:
- The CMB defines the canonical cosmic rest frame with v ~ 369 km/s
- Radio galaxy surveys suggest a larger dipole (~1000 km/s)
- If the true rest frame differs from assumed, H0 fits can be biased
- Effect depends on sky coverage and sample redshift distribution
"""

from .frames import (
    RestFrameDefinition,
    get_cmb_frame,
    get_heliocentric_velocity,
    get_radio_dipole_frame,
    get_local_group_frame,
    angular_separation,
    compute_los_velocity,
    compute_kinematic_redshift,
    correct_redshift_to_frame,
    compute_dipole_modulation,
    C_LIGHT,
)

from .sn_catalog import (
    SNSample,
    generate_isotropic_sky_positions,
    generate_hemispherical_sky_positions,
    generate_redshifts_uniform,
    generate_redshifts_volume_weighted,
    generate_sn_catalog,
    generate_sn_catalog_with_radio_dipole,
    mu_of_z_approx,
    mu_of_z_flat,
)

from .h0_fitter import (
    H0FitResult,
    fit_H0_simple,
    fit_H0_with_frame_correction,
    fit_H0_true_frame,
    compute_H0_bias_from_frame_mismatch,
    estimate_H0_bias_analytical,
    estimate_H0_scatter_from_dipole,
)

__all__ = [
    # frames.py
    'RestFrameDefinition',
    'get_cmb_frame',
    'get_heliocentric_velocity',
    'get_radio_dipole_frame',
    'get_local_group_frame',
    'angular_separation',
    'compute_los_velocity',
    'compute_kinematic_redshift',
    'correct_redshift_to_frame',
    'compute_dipole_modulation',
    'C_LIGHT',
    # sn_catalog.py
    'SNSample',
    'generate_isotropic_sky_positions',
    'generate_hemispherical_sky_positions',
    'generate_redshifts_uniform',
    'generate_redshifts_volume_weighted',
    'generate_sn_catalog',
    'generate_sn_catalog_with_radio_dipole',
    'mu_of_z_approx',
    'mu_of_z_flat',
    # h0_fitter.py
    'H0FitResult',
    'fit_H0_simple',
    'fit_H0_with_frame_correction',
    'fit_H0_true_frame',
    'compute_H0_bias_from_frame_mismatch',
    'estimate_H0_bias_analytical',
    'estimate_H0_scatter_from_dipole',
]
