"""
Layered Expansion (Bent-Deck) Cosmology Module

SIMULATION 24: Tests whether any smooth, layered expansion history H(z)
can reconcile CMB + BAO + SN with H0 ~ 73 km/s/Mpc.

The "bent deck" metaphor: think of the expansion history as a stack of cards
at different redshifts. Each card can tilt slightly, but sharp kinks are
penalized by a smoothness prior.
"""

from .layered_expansion import (
    LayeredExpansionHyperparams,
    LayeredExpansionParams,
    LCDMBackground,
    make_default_nodes,
    log_smoothness_prior,
    H_of_z_layered,
    E_of_z_layered,
    get_H0_effective,
    check_physical_validity,
    make_zero_params,
    make_random_params,
)

from .observables import (
    LayeredObservables,
    LayeredChi2Result,
    BAODataPoint,
    BAO_DATA,
    comoving_distance_layered,
    angular_diameter_distance_layered,
    luminosity_distance_layered,
    compute_theta_star_layered,
    compute_background_observables_layered,
    compute_chi2_cmb_bao_sn,
    compute_chi2_bao_layered,
    compute_chi2_sn_layered,
    compute_chi2_cmb_layered,
    compute_baseline_chi2,
)

__all__ = [
    # Core expansion
    "LayeredExpansionHyperparams",
    "LayeredExpansionParams",
    "LCDMBackground",
    "make_default_nodes",
    "log_smoothness_prior",
    "H_of_z_layered",
    "E_of_z_layered",
    "get_H0_effective",
    "check_physical_validity",
    "make_zero_params",
    "make_random_params",
    # Observables
    "LayeredObservables",
    "LayeredChi2Result",
    "BAODataPoint",
    "BAO_DATA",
    "comoving_distance_layered",
    "angular_diameter_distance_layered",
    "luminosity_distance_layered",
    "compute_theta_star_layered",
    "compute_background_observables_layered",
    "compute_chi2_cmb_bao_sn",
    "compute_chi2_bao_layered",
    "compute_chi2_sn_layered",
    "compute_chi2_cmb_layered",
    "compute_baseline_chi2",
]
