"""
hrc2.ladder - SN Ia Distance Ladder Systematics Module

SIMULATION 11: Tests how SN Ia systematics (population drift, metallicity,
dust miscalibration, Malmquist bias) affect the inferred H0.

SIMULATION 11B: SH0ES-like two-step ladder with SALT2 model.
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
from .ladder_pipeline import LadderFitResult, run_ladder, calibrate_M_B_from_mu
from .calibrator_bias import CalibratorBiasParameters, apply_calibrator_biases

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
    'CalibratorBiasParameters',
    'apply_calibrator_biases',
]
