"""
Holographic Recycling Cosmology (HRC) Package

A comprehensive framework for modeling black hole evaporation, Planck-mass remnant
formation, and their cosmological consequences. HRC provides an alternative to ΛCDM
that naturally resolves the Hubble tension through epoch-dependent gravitational
coupling.

Modules
-------
theory : Theoretical foundations
    Field equations, action principles, Friedmann equations with scalar field

dynamics : Black hole and recycling dynamics
    Hawking evaporation, remnant formation, mass recycling rates

observations : Observational constraints
    MCMC fitting to BAO, SNe, CMB data; parameter estimation

signatures : Unique predictions
    CMB signatures, expansion history, GW echoes, dark matter properties

Key Classes
-----------
HRCTheory : Main theoretical framework
HRCPredictions : Cosmological predictions
CMBSignatures : CMB modifications from HRC
ExpansionSignatures : H(z) and w(z) predictions
GWSignatures : Gravitational wave signatures
DarkMatterSignatures : Remnant dark matter properties

Example
-------
>>> from hrc import HRCTheory, summarize_signatures
>>> theory = HRCTheory(xi=0.03, phi_0=0.2)
>>> summary = summarize_signatures()
>>> print(f"Hubble tension resolved: {summary['summary']['tension_resolved']}")

References
----------
[1] Rovelli (2018), arXiv:1805.03872 - White hole remnants
[2] Planck Collaboration (2020), A&A 641, A6
[3] DESI Collaboration (2024), arXiv:2404.03002

Author: HRC Collaboration
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "HRC Collaboration"

# Import from parent directory modules
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Theory module
from hrc_theory import (
    HRCTheory,
    FieldEquations as HRCFieldEquations,
    ActionComponents as HRCAction,
)

# Dynamics module
from hrc_dynamics import (
    BlackHolePopulation,
    RemnantProperties,
    RecyclingDynamics,
    HRCCosmology as CosmicRecyclingHistory,
    PhysicalConstants as DynamicsConstants,
    CONSTANTS as DYNAMICS_CONSTANTS,
    Units,
    UNITS,
)

# Observations module
from hrc_observations import (
    LCDMCosmology,
    HRCPredictions,
    ObservationalData,
    CosmologyConstants,
    COSMO_CONST,
)

# Signatures module
from hrc_signatures import (
    CMBSignatures,
    ExpansionSignatures,
    GWSignatures,
    DarkMatterSignatures,
    summarize_signatures,
    prioritized_tests,
    create_signature_table,
    SignatureConstants,
    SIG_CONST,
)

# Convenience aliases
Theory = HRCTheory
Predictions = HRCPredictions
CMB = CMBSignatures
Expansion = ExpansionSignatures
GW = GWSignatures
DarkMatter = DarkMatterSignatures

# Default HRC parameters that resolve the Hubble tension
DEFAULT_PARAMS = {
    'H0_true': 70.0,          # True Hubble constant [km/s/Mpc]
    'Omega_m': 0.315,         # Total matter density
    'Omega_b': 0.049,         # Baryon density
    'Omega_rem': 0.05,        # Remnant density
    'xi': 0.03,               # Non-minimal coupling
    'alpha': 0.01,            # Scalar field evolution exponent
    'phi_0': 0.2,             # Scalar field value today [Planck units]
    'f_remnant': 0.2,         # Fraction of DM in remnants
}

def quick_summary():
    """
    Print a quick summary of HRC predictions with default parameters.

    Returns
    -------
    dict
        Summary dictionary with key predictions
    """
    summary = summarize_signatures(DEFAULT_PARAMS)

    print("=" * 70)
    print("HOLOGRAPHIC RECYCLING COSMOLOGY - QUICK SUMMARY")
    print("=" * 70)
    print(f"\nDefault parameters: ξ={DEFAULT_PARAMS['xi']}, φ₀={DEFAULT_PARAMS['phi_0']}")
    print(f"\nKey predictions:")
    print(f"  • Hubble tension resolved: {summary['summary']['tension_resolved']}")
    print(f"  • H₀ (local): ~76 km/s/Mpc")
    print(f"  • H₀ (CMB): ~70 km/s/Mpc")
    print(f"  • GW echo time (30 M☉): ~27 ms")
    print(f"  • Remnant mass: M_Planck ≈ 2.18×10⁻⁸ kg")
    print("\nTop observational tests:")
    tests = prioritized_tests()
    for test in tests[:3]:
        print(f"  {test['rank']}. {test['test']} ({test['timeline']})")
    print("=" * 70)

    return summary


def run_full_analysis(params=None, verbose=True):
    """
    Run the complete HRC analysis pipeline.

    Parameters
    ----------
    params : dict, optional
        HRC model parameters. Uses DEFAULT_PARAMS if not provided.
    verbose : bool
        Whether to print progress updates

    Returns
    -------
    dict
        Complete analysis results including:
        - cmb: CMB signature predictions
        - expansion: H(z) and w(z) predictions
        - gw: Gravitational wave signatures
        - dm: Dark matter properties
        - summary: Overall summary
        - tests: Prioritized observational tests
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    if verbose:
        print("Running HRC full analysis pipeline...")
        print(f"Parameters: ξ={params.get('xi', 0.03)}, φ₀={params.get('phi_0', 0.2)}")

    results = {}

    # CMB signatures
    if verbose:
        print("  Computing CMB signatures...")
    cmb = CMBSignatures(params)
    results['cmb'] = cmb.cmb_summary()

    # Expansion history
    if verbose:
        print("  Computing expansion history...")
    exp = ExpansionSignatures(params)
    results['expansion'] = {
        'hubble_tension': exp.hubble_tension_vs_z(),
        'w_fit': exp.w0_wa_fit(),
    }

    # GW signatures
    if verbose:
        print("  Computing GW signatures...")
    gw = GWSignatures(params)
    results['gw'] = gw.gw_summary()

    # Dark matter
    if verbose:
        print("  Computing DM signatures...")
    dm = DarkMatterSignatures(params)
    results['dm'] = dm.dm_summary()

    # Summary
    if verbose:
        print("  Generating summary...")
    results['summary'] = summarize_signatures(params)
    results['tests'] = prioritized_tests()

    if verbose:
        print("Analysis complete!")

    return results


# Module-level documentation
__all__ = [
    # Theory
    'HRCTheory',
    'HRCFieldEquations',
    'HRCAction',

    # Dynamics
    'BlackHolePopulation',
    'RemnantProperties',
    'RecyclingDynamics',
    'CosmicRecyclingHistory',
    'DynamicsConstants',
    'DYNAMICS_CONSTANTS',
    'Units',
    'UNITS',

    # Observations
    'LCDMCosmology',
    'HRCPredictions',
    'ObservationalData',
    'CosmologyConstants',
    'COSMO_CONST',

    # Signatures
    'CMBSignatures',
    'ExpansionSignatures',
    'GWSignatures',
    'DarkMatterSignatures',
    'summarize_signatures',
    'prioritized_tests',
    'create_signature_table',
    'SignatureConstants',
    'SIG_CONST',

    # Convenience
    'Theory',
    'Predictions',
    'CMB',
    'Expansion',
    'GW',
    'DarkMatter',
    'DEFAULT_PARAMS',
    'quick_summary',
    'run_full_analysis',
]
