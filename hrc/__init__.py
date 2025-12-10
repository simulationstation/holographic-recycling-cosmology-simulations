"""Holographic Recycling Cosmology (HRC) - A rigorous cosmology framework.

This package implements the HRC model where:
- Black hole evaporation produces stable Planck-mass remnants
- A scalar "recycling field" φ couples non-minimally to curvature
- The effective gravitational constant varies as G_eff = G/(1 - 8πGξφ)
- The Hubble tension arises naturally from epoch-dependent G_eff

Key modules:
    background: Background cosmology evolution with ODE integration
    scalar_field: Scalar field dynamics and evolution
    effective_gravity: G_eff computation and evolution
    remnants: Planck-mass remnant physics
    perturbations: Stability checks and CLASS interface
    constraints: BBN, PPN, stellar, structure growth constraints
    observables: H₀, BAO, SNe, standard siren likelihoods
    sampling: MCMC parameter inference
    plots: Publication-quality figures

Example usage:
    >>> from hrc import HRCParameters, BackgroundCosmology
    >>> params = HRCParameters(xi=0.03, phi_0=0.2)
    >>> cosmo = BackgroundCosmology(params)
    >>> solution = cosmo.solve(z_max=1100)
    >>> print(f"G_eff/G at z=0: {solution.G_eff_at(0):.3f}")
"""

__version__ = "2.0.0"
__author__ = "HRC Collaboration"

# Core configuration
from .utils.config import HRCParameters, HRCConfig, PotentialConfig
from .utils.constants import (
    PhysicalConstants,
    PLANCK_UNITS,
    SI_UNITS,
    PLANCK_2018,
    SHOES_2024,
)

# Background cosmology
from .background import (
    BackgroundCosmology,
    BackgroundSolution,
    BackgroundState,
)

# Scalar field
from .scalar_field import (
    ScalarFieldSolver,
    ScalarFieldSolution,
    compute_slow_roll_parameters,
    is_slow_roll_valid,
)

# Potentials
from .potentials import (
    Potential,
    QuadraticPotential,
    PlateauPotential,
    SymmetronPotential,
    ExponentialPotential,
    DoubleExponentialPotential,
    InverseExponentialPotential,
    PotentialParams,
    get_potential,
    POTENTIAL_REGISTRY,
)

# Effective gravity
from .effective_gravity import (
    EffectiveGravity,
    GeffEvolution,
    GeffResult,
    check_G_eff_constraints,
    compute_hubble_tension,
)

# Remnants
from .remnants import (
    RemnantProperties,
    RemnantPopulation,
    HawkingEvaporation,
    compute_remnant_omega,
    remnant_summary,
)

# Perturbations and stability
from .perturbations import (
    StabilityChecker,
    StabilityResult,
    check_all_stability,
    check_no_ghost,
    check_gradient_stability,
    CLASSInterface,
    CLASSOutput,
)

# Constraints
from .constraints import (
    check_bbn_constraint,
    check_ppn_constraints,
    check_stellar_constraints,
    check_growth_constraints,
    BBNConstraint,
    PPNConstraint,
    StellarConstraint,
    GrowthConstraint,
)

# Observables
from .observables import (
    DistanceCalculator,
    CosmologicalDistances,
    SH0ESLikelihood,
    TRGBLikelihood,
    CMBDistanceLikelihood,
    BAOLikelihood,
    PantheonPlusLikelihood,
    StandardSirenLikelihood,
)

# Sampling
from .sampling import (
    MCMCSampler,
    MCMCResult,
    PriorSet,
    UniformPrior,
    GaussianPrior,
    LogUniformPrior,
    run_mcmc,
)

# Analysis
from .analysis import (
    scan_parameter_space,
    ParameterScanResult,
    PointClassification,
    compare_potentials,
    print_layman_summary,
    generate_report,
    quick_potential_check,
    PotentialSummary,
)


def quick_summary(params: HRCParameters = None, check_constraints: bool = True) -> dict:
    """Print quick summary of HRC predictions with given parameters.

    Args:
        params: HRC parameters (default: fiducial values)
        check_constraints: Whether to run constraint checks (default: True)

    Returns:
        Dictionary with key predictions including:
        - valid: Overall validity (G_eff + constraints)
        - geff_valid: Whether G_eff stays finite
        - constraints_passed: Which constraints pass/fail
        - phi_critical: Critical scalar field value
        - tension: Hubble tension results
        - remnants: Remnant properties
    """
    from .utils.numerics import compute_critical_phi, check_geff_validity

    if params is None:
        params = HRCParameters(xi=0.03, phi_0=0.2, h=0.7)

    print("=" * 60)
    print("HOLOGRAPHIC RECYCLING COSMOLOGY - QUICK SUMMARY")
    print("=" * 60)

    # Validate parameters
    valid, errors = params.validate()
    if not valid:
        print(f"\n Parameter validation failed: {errors}")
        return {"valid": False, "errors": errors, "constraints_passed": {}}

    print(f"\nParameters:")
    print(f"  xi (coupling)   = {params.xi}")
    print(f"  phi_0 (field)   = {params.phi_0}")
    print(f"  h               = {params.h}")
    print(f"  Omega_m         = {params.Omega_m:.3f}")

    # Compute critical phi and check validity
    phi_c = compute_critical_phi(params.xi)
    validity = check_geff_validity(params.phi_0, params.xi)

    print(f"\nCritical Field Value:")
    print(f"  phi_c           = {phi_c:.4f}")
    print(f"  phi_0/phi_c     = {params.phi_0/phi_c:.4f}" if phi_c != float('inf') else "  phi_0/phi_c     = N/A (xi <= 0)")

    # Compute G_eff at z=0
    eff_grav = EffectiveGravity(params)
    G_eff_0 = eff_grav.G_eff_ratio(params.phi_0)

    print(f"\nEffective Gravity (z=0):")
    print(f"  G_eff/G         = {G_eff_0.G_eff_ratio:.4f}")
    print(f"  Physical?       = {G_eff_0.is_physical}")
    if not validity.valid:
        print(f"  WARNING: {validity.message}")

    # Solve background (quick, low resolution)
    tension = None
    remnants = None
    geff_valid = True
    constraints_passed = {}
    constraints_failed = []
    G_eff_cmb = None
    solution = None

    try:
        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=1100, z_points=100)

        if solution.success and solution.geff_valid:
            G_eff_cmb = solution.G_eff_at(1089)

            # Hubble tension
            tension = compute_hubble_tension(solution, params)

            print(f"\nHubble Tension Resolution:")
            print(f"  G_eff/G (z=1089) = {G_eff_cmb:.4f}")
            if tension["valid"]:
                print(f"  H0 local        = {tension['H0_local']:.1f} km/s/Mpc")
                print(f"  H0 CMB          = {tension['H0_cmb']:.1f} km/s/Mpc")
                print(f"  Delta_H0        = {tension['Delta_H0']:.1f} km/s/Mpc")
                print(f"  Tension resolved? {tension['resolves_tension']}")

            # Remnant properties
            remnants = remnant_summary(params)
            print(f"\nRemnant Properties:")
            print(f"  f_rem           = {remnants['f_rem']:.2f}")
            print(f"  n_rem           = {remnants['n_rem_m3']:.2e} m^-3")
            print(f"  Omega_rem       = {remnants['Omega_rem']:.4f}")

            # Check constraints if requested
            if check_constraints:
                print(f"\nConstraint Checks:")

                # BBN constraint
                bbn = check_bbn_constraint(solution)
                constraints_passed["BBN"] = bbn.passed
                status = "PASS" if bbn.passed else "FAIL"
                print(f"  BBN (|DG/G| < 0.1):        {status}")
                if not bbn.passed:
                    constraints_failed.append(f"BBN: {bbn.message}")

                # PPN constraints
                ppn_passed, ppn_results = check_ppn_constraints(solution, params=params)
                constraints_passed["PPN"] = ppn_passed
                status = "PASS" if ppn_passed else "FAIL"
                print(f"  PPN (solar system):        {status}")
                if not ppn_passed:
                    for r in ppn_results:
                        if not r.passed:
                            constraints_failed.append(f"PPN ({r.name}): {r.message}")

                # Stellar constraints
                stellar_passed, stellar_results = check_stellar_constraints(solution)
                constraints_passed["Stellar"] = stellar_passed
                status = "PASS" if stellar_passed else "FAIL"
                print(f"  Stellar (Gdot bounds):     {status}")
                if not stellar_passed:
                    for r in stellar_results:
                        if not r.passed:
                            constraints_failed.append(f"Stellar ({r.name}): {r.message}")

                # Stability
                checker = StabilityChecker(params)
                stable, _, stab_messages = checker.check_solution(solution)
                constraints_passed["Stability"] = stable
                status = "PASS" if stable else "FAIL"
                print(f"  Stability (no-ghost):      {status}")
                if not stable:
                    constraints_failed.append(f"Stability: {stab_messages}")

                # Summary of failed constraints
                if constraints_failed:
                    print(f"\n  Failed constraints:")
                    for fail in constraints_failed:
                        print(f"    - {fail}")

        elif not solution.geff_valid:
            geff_valid = False
            print(f"\n" + "=" * 60)
            print("PARAMETER SET INVALID")
            print("=" * 60)
            print(f"Scalar field approaches critical value and G_eff diverges")
            if solution.geff_divergence_z is not None:
                print(f"Divergence occurs at z = {solution.geff_divergence_z:.2f}")
                if solution.geff_divergence_z < 1089:
                    print("This happens BEFORE recombination - model excluded!")
            print(f"phi_critical = {solution.phi_critical:.6f}")
            constraints_passed["G_eff_finite"] = False
            constraints_failed.append(f"G_eff diverges at z={solution.geff_divergence_z:.2f}")

        else:
            print(f"\n Background integration failed: {solution.message}")
            geff_valid = False
            constraints_passed["integration"] = False
            constraints_failed.append(f"Integration failed: {solution.message}")

    except Exception as e:
        print(f"\n Computation error: {e}")
        return {"valid": False, "error": str(e), "constraints_passed": {}}

    print("\n" + "=" * 60)

    # Determine overall validity
    all_constraints_pass = all(constraints_passed.values()) if constraints_passed else True
    overall_valid = geff_valid and validity.valid and all_constraints_pass

    return {
        "valid": overall_valid,
        "geff_valid": geff_valid,
        "constraints_passed": constraints_passed,
        "constraints_failed": constraints_failed,
        "phi_critical": phi_c,
        "phi_over_phi_c": params.phi_0 / phi_c if phi_c != float('inf') else 0.0,
        "G_eff_0": G_eff_0.G_eff_ratio if G_eff_0.is_physical else None,
        "G_eff_cmb": G_eff_cmb,
        "tension": tension,
        "remnants": remnants,
    }


def run_full_analysis(
    params: HRCParameters = None,
    z_max: float = 1100.0,
    z_points: int = 500,
    check_constraints: bool = True,
    verbose: bool = True,
) -> dict:
    """Run full HRC analysis pipeline.

    Args:
        params: HRC parameters
        z_max: Maximum redshift for integration
        z_points: Number of redshift points
        check_constraints: Run all constraint checks
        verbose: Print progress

    Returns:
        Dictionary with complete analysis results
    """
    if params is None:
        params = HRCParameters(xi=0.03, phi_0=0.2, h=0.7)

    results = {
        "params": params,
        "valid": True,
    }

    # 1. Background evolution
    if verbose:
        print("Computing background evolution...")

    cosmo = BackgroundCosmology(params)
    solution = cosmo.solve(z_max=z_max, z_points=z_points)
    results["background"] = solution

    if not solution.success:
        results["valid"] = False
        results["error"] = solution.message
        return results

    # 2. G_eff evolution
    if verbose:
        print("Computing G_eff evolution...")

    eff_grav = EffectiveGravity(params)
    G_eff_evolution = eff_grav.compute_evolution(solution)
    results["G_eff"] = G_eff_evolution

    # 3. Hubble tension
    if verbose:
        print("Computing Hubble tension predictions...")

    tension = compute_hubble_tension(solution, params)
    results["hubble_tension"] = tension

    # 4. Stability checks
    if verbose:
        print("Running stability checks...")

    checker = StabilityChecker(params)
    stable, stable_mask, messages = checker.check_solution(solution)
    results["stability"] = {
        "all_stable": stable,
        "messages": messages,
        "summary": checker.get_stability_summary(),
    }

    # 5. Constraint checks
    if check_constraints:
        if verbose:
            print("Checking observational constraints...")

        # BBN
        bbn = check_bbn_constraint(solution)
        results["bbn_constraint"] = bbn

        # PPN
        ppn_passed, ppn_results = check_ppn_constraints(solution, params=params)
        results["ppn_constraints"] = {
            "passed": ppn_passed,
            "results": ppn_results,
        }

        # Stellar
        stellar_passed, stellar_results = check_stellar_constraints(solution)
        results["stellar_constraints"] = {
            "passed": stellar_passed,
            "results": stellar_results,
        }

    # 6. Remnant properties
    if verbose:
        print("Computing remnant properties...")

    remnants = remnant_summary(params)
    results["remnants"] = remnants

    if verbose:
        print("Analysis complete!")

    return results


__all__ = [
    # Version
    "__version__",
    # Config
    "HRCParameters",
    "HRCConfig",
    "PotentialConfig",
    # Constants
    "PhysicalConstants",
    "PLANCK_UNITS",
    "SI_UNITS",
    "PLANCK_2018",
    "SHOES_2024",
    # Background
    "BackgroundCosmology",
    "BackgroundSolution",
    "BackgroundState",
    # Scalar field
    "ScalarFieldSolver",
    "ScalarFieldSolution",
    # Potentials
    "Potential",
    "QuadraticPotential",
    "PlateauPotential",
    "SymmetronPotential",
    "ExponentialPotential",
    "DoubleExponentialPotential",
    "InverseExponentialPotential",
    "PotentialParams",
    "get_potential",
    "POTENTIAL_REGISTRY",
    # Effective gravity
    "EffectiveGravity",
    "GeffEvolution",
    "check_G_eff_constraints",
    "compute_hubble_tension",
    # Remnants
    "RemnantProperties",
    "HawkingEvaporation",
    "remnant_summary",
    # Stability
    "StabilityChecker",
    "check_all_stability",
    # Constraints
    "check_bbn_constraint",
    "check_ppn_constraints",
    "check_stellar_constraints",
    "check_growth_constraints",
    # Observables
    "DistanceCalculator",
    "SH0ESLikelihood",
    "BAOLikelihood",
    "PantheonPlusLikelihood",
    "StandardSirenLikelihood",
    # Sampling
    "MCMCSampler",
    "MCMCResult",
    "run_mcmc",
    # Analysis
    "scan_parameter_space",
    "ParameterScanResult",
    "PointClassification",
    "compare_potentials",
    "print_layman_summary",
    "generate_report",
    "quick_potential_check",
    "PotentialSummary",
    # Convenience
    "quick_summary",
    "run_full_analysis",
]
