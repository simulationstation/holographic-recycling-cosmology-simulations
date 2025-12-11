"""
hrc2.ladder - SN Ia Distance Ladder Systematics Module

SIMULATION 11: Tests how SN Ia systematics (population drift, metallicity,
dust miscalibration, Malmquist bias) affect the inferred H0.

SIMULATION 11B: SH0ES-like two-step ladder with SALT2 model.

SIMULATION 11C: Combined calibrator + SN systematics.

SIMULATION 12: Full Cepheid/TRGB calibration chain.

SIMULATION 13: HST vs JWST Cepheid recalibration test.

SIMULATION 15: Joint hierarchical systematics + H0 inference.
"""

from .cosmology_baseline import TrueCosmology, mu_of_z
from .snia_population import SNSystematicParameters, simulate_snia_sample
from .naive_fitter import NaiveFitResult, fit_naive_H0_M_B
from .host_population import HostGalaxy, HostPopulationParams, sample_hosts
from .snia_salt2 import (
    SNSystematicParameters11B,
    simulate_snia_with_hosts,
    apply_magnitude_limit,
)
from .ladder_pipeline import (
    LadderFitResult,
    run_ladder,
    calibrate_M_B_from_mu,
    run_two_step_ladder_with_cepheid_chain,
)
from .calibrator_bias import CalibratorBiasParameters, apply_calibrator_biases
from .cepheid_calibration import (
    Anchor,
    CepheidPLParameters,
    CepheidHost,
    TRGBParameters,
    get_default_anchors,
    get_default_cepheid_hosts,
    compute_calibrator_mu_from_chain,
    simulate_cepheid_magnitudes_and_colors,
    generate_cepheid_data_for_host,
    fit_PL_zero_point_from_instrument,
    compute_host_mu_from_instrument_cepheids,
)
from .cepheid_systematics import (
    CepheidSystematicParameters,
    build_cepheid_pl_params,
    build_trgb_params,
    apply_full_cepheid_chain,
    is_cepheid_realistic,
    is_cepheid_moderate,
)
from .instrument_photometry import (
    InstrumentPhotometrySystematics,
    apply_instrument_effects,
    compute_instrument_difference,
    create_hst_baseline,
    create_jwst_baseline,
    create_jwst_with_systematics,
)
from .joint_systematics_model import (
    JointSystematicsPriors,
    JointSystematicsParameters,
    theta_to_params,
    params_to_theta,
    log_prior as joint_log_prior,
    sample_prior as joint_sample_prior,
    PARAM_NAMES as JOINT_PARAM_NAMES,
    NDIM as JOINT_NDIM,
)
from .joint_systematics_likelihood import (
    SyntheticLadderData,
    generate_synthetic_ladder_data,
    compute_log_likelihood as joint_compute_log_likelihood,
    log_posterior as joint_log_posterior,
    test_log_posterior,
    find_best_fit_H0_no_systematics,
)

__all__ = [
    'TrueCosmology',
    'mu_of_z',
    'SNSystematicParameters',
    'simulate_snia_sample',
    'NaiveFitResult',
    'fit_naive_H0_M_B',
    'HostGalaxy',
    'HostPopulationParams',
    'sample_hosts',
    'SNSystematicParameters11B',
    'simulate_snia_with_hosts',
    'apply_magnitude_limit',
    'LadderFitResult',
    'run_ladder',
    'calibrate_M_B_from_mu',
    'run_two_step_ladder_with_cepheid_chain',
    'CalibratorBiasParameters',
    'apply_calibrator_biases',
    # SIM 12: Cepheid/TRGB calibration
    'Anchor',
    'CepheidPLParameters',
    'CepheidHost',
    'TRGBParameters',
    'get_default_anchors',
    'get_default_cepheid_hosts',
    'compute_calibrator_mu_from_chain',
    'CepheidSystematicParameters',
    'build_cepheid_pl_params',
    'build_trgb_params',
    'apply_full_cepheid_chain',
    'is_cepheid_realistic',
    'is_cepheid_moderate',
    # SIM 13: Multi-instrument Cepheid photometry
    'simulate_cepheid_magnitudes_and_colors',
    'generate_cepheid_data_for_host',
    'fit_PL_zero_point_from_instrument',
    'compute_host_mu_from_instrument_cepheids',
    'InstrumentPhotometrySystematics',
    'apply_instrument_effects',
    'compute_instrument_difference',
    'create_hst_baseline',
    'create_jwst_baseline',
    'create_jwst_with_systematics',
    # SIM 15: Joint hierarchical systematics
    'JointSystematicsPriors',
    'JointSystematicsParameters',
    'theta_to_params',
    'params_to_theta',
    'joint_log_prior',
    'joint_sample_prior',
    'JOINT_PARAM_NAMES',
    'JOINT_NDIM',
    'SyntheticLadderData',
    'generate_synthetic_ladder_data',
    'joint_compute_log_likelihood',
    'joint_log_posterior',
    'test_log_posterior',
    'find_best_fit_H0_no_systematics',
]
