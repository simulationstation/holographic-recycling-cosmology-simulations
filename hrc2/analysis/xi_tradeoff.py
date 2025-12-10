"""Xi-tradeoff analysis for HRC 2.0.

This module implements parameter space scans for general scalar-tensor
models, exploring the tradeoff between:
- Large coupling strength (large effect on G_eff)
- Stability and observational constraints

For each coupling family (linear, quadratic, exponential), we scan
over coupling strength and initial field value to find:
- Maximum achievable |Delta G/G| while satisfying all constraints
- Critical coupling strength where stability breaks down
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import numpy as np
from numpy.typing import NDArray

from ..theory import (
    CouplingFamily,
    PotentialType,
    HRC2Parameters,
    ScalarTensorModel,
    create_model,
)
from ..background import BackgroundCosmology, BackgroundSolution
from ..constraints.stability import check_stability_along_trajectory, StabilityResult
from ..constraints.observational import (
    check_all_constraints_hrc2,
    check_bbn_constraint_hrc2,
    check_ppn_constraints_hrc2,
    check_stellar_constraints_hrc2,
    HRC2ConstraintResult,
    estimate_delta_H0,
)
from ..utils.config import PerformanceConfig


@dataclass
class XiTradeoffResultHRC2:
    """Result of xi-tradeoff scan for HRC 2.0.

    Attributes:
        coupling_family: Type of F(phi) coupling scanned
        potential_type: Type of V(phi) potential used
        xi_values: Array of coupling strength values scanned
        phi0_values: Array of initial field values scanned

        # 2D arrays [n_xi, n_phi0]
        stable_mask: True if dynamically stable
        obs_allowed_mask: True if passes all constraints
        delta_G_over_G: Fractional G_eff change (NaN if invalid)
        G_eff_0: G_eff/G_N at z=0
        G_eff_zmax: G_eff/G_N at z_max

        # Per-xi statistics
        stable_fraction: Fraction of phi0 values that are stable
        obs_allowed_fraction: Fraction that pass all constraints
        max_delta_G_stable: Max |Delta G/G| among stable points
        max_delta_G_allowed: Max |Delta G/G| among allowed points

        # Metadata
        z_max: Maximum redshift
        constraint_level: BBN constraint level used
    """
    coupling_family: CouplingFamily
    potential_type: PotentialType
    xi_values: NDArray[np.floating]
    phi0_values: NDArray[np.floating]

    stable_mask: NDArray[np.bool_]
    obs_allowed_mask: NDArray[np.bool_]
    delta_G_over_G: NDArray[np.floating]
    G_eff_0: NDArray[np.floating]
    G_eff_zmax: NDArray[np.floating]

    stable_fraction: NDArray[np.floating]
    obs_allowed_fraction: NDArray[np.floating]
    max_delta_G_stable: NDArray[np.floating]
    max_delta_G_allowed: NDArray[np.floating]

    z_max: float
    constraint_level: str


@dataclass
class SinglePointResult:
    """Result of evaluating a single (xi, phi0) point.

    Attributes:
        xi: Coupling strength parameter
        phi0: Initial field value
        dynamically_stable: Whether the model is dynamically stable
        bbn_allowed: Passes BBN constraint
        ppn_allowed: Passes PPN constraints
        stellar_allowed: Passes stellar constraints
        all_constraints_allowed: Passes all observational constraints
        delta_G_over_G: Fractional G_eff change
        delta_H0: Estimated Hubble tension contribution
    """
    xi: float
    phi0: float
    dynamically_stable: bool
    bbn_allowed: bool
    ppn_allowed: bool
    stellar_allowed: bool
    all_constraints_allowed: bool
    delta_G_over_G: float
    delta_H0: float


def evaluate_model_point(
    xi: float,
    phi0: float,
    coupling_family: CouplingFamily,
    potential_type: PotentialType,
    perf: PerformanceConfig,
    z_max: float = 1100.0,
    z_points: int = 300,
    constraint_level: str = "conservative",
) -> SinglePointResult:
    """Evaluate a single model point in the parameter space.

    This function is designed to be called in parallel workers.

    Args:
        xi: Coupling strength parameter
        phi0: Initial field value
        coupling_family: Type of F(phi) coupling
        potential_type: Type of V(phi) potential
        perf: Performance configuration
        z_max: Maximum redshift
        z_points: Number of redshift points
        constraint_level: BBN constraint level

    Returns:
        SinglePointResult with evaluation results
    """
    # Timeout enforcement
    start_time = time.time()
    timeout = perf.max_time_per_model

    def _invalid_result():
        return SinglePointResult(
            xi=xi, phi0=phi0,
            dynamically_stable=False,
            bbn_allowed=False, ppn_allowed=False, stellar_allowed=False,
            all_constraints_allowed=False,
            delta_G_over_G=np.nan, delta_H0=np.nan
        )

    try:
        # Create parameters for this point
        params = HRC2Parameters(
            coupling_family=coupling_family,
            potential_type=potential_type,
            xi=xi,
            alpha=xi,
            beta=xi,
            phi_0=phi0,
            phi_dot_0=0.0,
        )

        # Create model and solve
        model = create_model(params)
        cosmo = BackgroundCosmology(params, model)

        try:
            # Pass timeout to ODE solver for real enforcement during integration
            solution = cosmo.solve(
                z_max=z_max,
                z_points=z_points,
                rtol=perf.rtol,
                atol=perf.atol,
                timeout=timeout,
            )
        except RuntimeError as e:
            # Early exit from ODE solver (F invalid, G_eff out of range, timeout, etc.)
            return _invalid_result()
        except Exception:
            return _invalid_result()

        if not solution.success or not solution.geff_valid:
            return SinglePointResult(
                xi=xi, phi0=phi0,
                dynamically_stable=False,
                bbn_allowed=False, ppn_allowed=False, stellar_allowed=False,
                all_constraints_allowed=False,
                delta_G_over_G=np.nan, delta_H0=np.nan
            )

        # Check stability
        stability = check_stability_along_trajectory(solution, model)
        if not stability.is_stable:
            return SinglePointResult(
                xi=xi, phi0=phi0,
                dynamically_stable=False,
                bbn_allowed=False, ppn_allowed=False, stellar_allowed=False,
                all_constraints_allowed=False,
                delta_G_over_G=np.nan, delta_H0=np.nan
            )

        # Point is dynamically stable - compute metrics
        dG = abs(solution.G_eff_ratio[0] - solution.G_eff_ratio[-1])
        dH0 = estimate_delta_H0(dG)

        # Check individual constraints
        bbn_ok, _ = check_bbn_constraint_hrc2(solution, model, constraint_level)
        ppn_ok, _, _ = check_ppn_constraints_hrc2(solution, model, params)
        stellar_ok, _ = check_stellar_constraints_hrc2(solution, model)
        all_ok = bbn_ok and ppn_ok and stellar_ok

        return SinglePointResult(
            xi=xi, phi0=phi0,
            dynamically_stable=True,
            bbn_allowed=bbn_ok,
            ppn_allowed=ppn_ok,
            stellar_allowed=stellar_ok,
            all_constraints_allowed=all_ok,
            delta_G_over_G=dG,
            delta_H0=dH0
        )

    except Exception:
        return SinglePointResult(
            xi=xi, phi0=phi0,
            dynamically_stable=False,
            bbn_allowed=False, ppn_allowed=False, stellar_allowed=False,
            all_constraints_allowed=False,
            delta_G_over_G=np.nan, delta_H0=np.nan
        )


def rebuild_xi_tradeoff_result(
    results: List[SinglePointResult],
    xi_values: NDArray[np.floating],
    phi0_values: NDArray[np.floating],
    coupling_family: CouplingFamily,
    potential_type: PotentialType,
    z_max: float,
    constraint_level: str,
) -> XiTradeoffResultHRC2:
    """Rebuild XiTradeoffResultHRC2 from list of SinglePointResult.

    Args:
        results: List of SinglePointResult from parallel evaluation
        xi_values: Array of xi values scanned
        phi0_values: Array of phi0 values scanned
        coupling_family: Coupling family used
        potential_type: Potential type used
        z_max: Maximum redshift
        constraint_level: Constraint level used

    Returns:
        XiTradeoffResultHRC2 with aggregated results
    """
    n_xi = len(xi_values)
    n_phi0 = len(phi0_values)

    # Initialize result arrays
    stable_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    obs_allowed_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    delta_G_over_G = np.full((n_xi, n_phi0), np.nan)
    G_eff_0 = np.full((n_xi, n_phi0), np.nan)
    G_eff_zmax = np.full((n_xi, n_phi0), np.nan)

    # Create lookup for results
    result_map = {(r.xi, r.phi0): r for r in results}

    for i, xi in enumerate(xi_values):
        for j, phi0 in enumerate(phi0_values):
            key = (xi, phi0)
            if key not in result_map:
                continue

            r = result_map[key]
            stable_mask[i, j] = r.dynamically_stable
            obs_allowed_mask[i, j] = r.all_constraints_allowed
            delta_G_over_G[i, j] = r.delta_G_over_G

    # Compute per-xi statistics
    stable_fraction = np.zeros(n_xi)
    obs_allowed_fraction = np.zeros(n_xi)
    max_delta_G_stable = np.full(n_xi, np.nan)
    max_delta_G_allowed = np.full(n_xi, np.nan)

    for i in range(n_xi):
        n_stable = stable_mask[i, :].sum()
        n_allowed = obs_allowed_mask[i, :].sum()

        stable_fraction[i] = n_stable / n_phi0
        obs_allowed_fraction[i] = n_allowed / n_phi0

        if n_stable > 0:
            max_delta_G_stable[i] = np.nanmax(delta_G_over_G[i, stable_mask[i, :]])

        if n_allowed > 0:
            max_delta_G_allowed[i] = np.nanmax(delta_G_over_G[i, obs_allowed_mask[i, :]])

    return XiTradeoffResultHRC2(
        coupling_family=coupling_family,
        potential_type=potential_type,
        xi_values=xi_values,
        phi0_values=phi0_values,
        stable_mask=stable_mask,
        obs_allowed_mask=obs_allowed_mask,
        delta_G_over_G=delta_G_over_G,
        G_eff_0=G_eff_0,
        G_eff_zmax=G_eff_zmax,
        stable_fraction=stable_fraction,
        obs_allowed_fraction=obs_allowed_fraction,
        max_delta_G_stable=max_delta_G_stable,
        max_delta_G_allowed=max_delta_G_allowed,
        z_max=z_max,
        constraint_level=constraint_level,
    )


def save_partial_results(
    results: List[SinglePointResult],
    xi_values: NDArray[np.floating],
    phi0_values: NDArray[np.floating],
    path: str,
) -> None:
    """Save partial results to disk in a simple, robust format.

    Uses atomic write (write to .tmp then rename) for crash safety.

    Args:
        results: List of SinglePointResult evaluated so far
        xi_values: Full array of xi values in scan
        phi0_values: Full array of phi0 values in scan
        path: Output .npz file path (must end in .npz)
    """
    if not results:
        return  # Nothing to save

    data = {
        "xi": np.array([r.xi for r in results]),
        "phi0": np.array([r.phi0 for r in results]),
        "dynamically_stable": np.array([r.dynamically_stable for r in results]),
        "bbn_allowed": np.array([r.bbn_allowed for r in results]),
        "ppn_allowed": np.array([r.ppn_allowed for r in results]),
        "stellar_allowed": np.array([r.stellar_allowed for r in results]),
        "all_constraints_allowed": np.array([r.all_constraints_allowed for r in results]),
        "delta_G_over_G": np.array([r.delta_G_over_G for r in results]),
        "delta_H0": np.array([r.delta_H0 for r in results]),
        "xi_grid": np.array(xi_values),
        "phi0_grid": np.array(phi0_values),
        "n_completed": len(results),
    }
    # np.savez adds .npz automatically, so strip it for tmp file
    base_path = path[:-4] if path.endswith('.npz') else path
    tmp_path = base_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **data)
    os.replace(tmp_path, path)


def run_xi_tradeoff_parallel(
    xi_values: NDArray[np.floating],
    phi0_values: NDArray[np.floating],
    coupling_family: CouplingFamily,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    perf: Optional[PerformanceConfig] = None,
    z_max: float = 1100.0,
    z_points: int = 300,
    constraint_level: str = "conservative",
    verbose: bool = True,
) -> XiTradeoffResultHRC2:
    """Run parallel xi-tradeoff scan using ProcessPoolExecutor.

    Args:
        xi_values: Array of xi values to scan
        phi0_values: Array of phi0 values to scan
        coupling_family: Type of F(phi) coupling
        potential_type: Type of V(phi) potential
        perf: Performance configuration (uses defaults if None)
        z_max: Maximum redshift
        z_points: Number of redshift points
        constraint_level: BBN constraint level
        verbose: Print progress

    Returns:
        XiTradeoffResultHRC2 with scan results
    """
    if perf is None:
        perf = PerformanceConfig()

    tasks = [(xi, phi0) for xi in xi_values for phi0 in phi0_values]
    total = len(tasks)

    if verbose:
        print(f"Starting parallel scan: {len(xi_values)} xi x {len(phi0_values)} phi0 = {total} points")
        print(f"Using {perf.n_workers} workers")

    results: List[SinglePointResult] = []

    # Partial save path
    partial_path = "results/hrc2_scan/hrc2_partial_scan.npz"
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)

    with ProcessPoolExecutor(max_workers=perf.n_workers) as executor:
        future_map = {
            executor.submit(
                evaluate_model_point,
                xi, phi0,
                coupling_family, potential_type, perf,
                z_max, z_points, constraint_level
            ): (xi, phi0)
            for (xi, phi0) in tasks
        }

        for i, future in enumerate(as_completed(future_map)):
            xi, phi0 = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                res = SinglePointResult(
                    xi=xi, phi0=phi0,
                    dynamically_stable=False,
                    bbn_allowed=False, ppn_allowed=False, stellar_allowed=False,
                    all_constraints_allowed=False,
                    delta_G_over_G=np.nan, delta_H0=np.nan
                )
            results.append(res)

            # Per-completion logging (always, for debugging)
            if verbose:
                print(
                    f"[{i+1}/{total}] xi={xi:.3e}, phi0={phi0:.3f}, "
                    f"stable={res.dynamically_stable}, allowed={res.all_constraints_allowed}, "
                    f"dG={res.delta_G_over_G:.4f}" if not np.isnan(res.delta_G_over_G) else
                    f"[{i+1}/{total}] xi={xi:.3e}, phi0={phi0:.3f}, "
                    f"stable={res.dynamically_stable}, allowed={res.all_constraints_allowed}, dG=NaN"
                )

            # Save after EVERY completion for visibility
            save_partial_results(results, xi_values, phi0_values, partial_path)

    # Final save
    if verbose:
        print(f"Scan complete. Final save: {len(results)} results.")
    save_partial_results(results, xi_values, phi0_values, partial_path)

    if verbose:
        print(f"Parallel scan complete. Rebuilding result structure...")

    return rebuild_xi_tradeoff_result(
        results, xi_values, phi0_values,
        coupling_family, potential_type, z_max, constraint_level
    )


def run_xi_tradeoff_serial(
    xi_values: NDArray[np.floating],
    phi0_values: NDArray[np.floating],
    coupling_family: CouplingFamily,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    perf: Optional[PerformanceConfig] = None,
    z_max: float = 1100.0,
    z_points: int = 300,
    constraint_level: str = "conservative",
    verbose: bool = True,
) -> XiTradeoffResultHRC2:
    """Run serial xi-tradeoff scan (for testing/comparison).

    Same interface as run_xi_tradeoff_parallel but runs sequentially.
    """
    if perf is None:
        perf = PerformanceConfig()

    results: List[SinglePointResult] = []
    total = len(xi_values) * len(phi0_values)
    count = 0

    if verbose:
        print(f"Starting serial scan: {len(xi_values)} xi x {len(phi0_values)} phi0 = {total} points")

    for xi in xi_values:
        for phi0 in phi0_values:
            count += 1
            res = evaluate_model_point(
                xi, phi0,
                coupling_family, potential_type, perf,
                z_max, z_points, constraint_level
            )
            results.append(res)

            if verbose and count % max(1, total // 10) == 0:
                print(f"[serial] {count}/{total} ({100*count/total:.0f}%)")

    return rebuild_xi_tradeoff_result(
        results, xi_values, phi0_values,
        coupling_family, potential_type, z_max, constraint_level
    )


def scan_xi_tradeoff_hrc2(
    coupling_family: CouplingFamily,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    z_points: int = 300,
    constraint_level: str = "conservative",
    verbose: bool = True,
) -> XiTradeoffResultHRC2:
    """Scan xi-phi0 parameter space for a given coupling family.

    For each (xi, phi0) pair:
    1. Create model and integrate background evolution
    2. Check dynamical stability (F > 0, ghost-free, etc.)
    3. Check observational constraints (BBN, PPN, stellar)
    4. Compute Delta G/G between z=0 and z_max

    Args:
        coupling_family: LINEAR, QUADRATIC, or EXPONENTIAL
        potential_type: Potential type for V(phi)
        xi_values: Array of coupling strengths to scan
        phi0_values: Array of initial field values to scan
        z_max: Maximum redshift for integration
        z_points: Number of redshift points
        constraint_level: 'conservative', 'moderate', or 'strict'
        verbose: Print progress

    Returns:
        XiTradeoffResultHRC2 with scan results
    """
    # Default parameter grids
    if xi_values is None:
        xi_values = np.logspace(-4, 0, 15)  # 1e-4 to 1
    if phi0_values is None:
        phi0_values = np.linspace(0.01, 0.5, 15)

    n_xi = len(xi_values)
    n_phi0 = len(phi0_values)

    if verbose:
        print(f"Scanning {coupling_family.value} coupling:")
        print(f"  xi range: [{xi_values.min():.1e}, {xi_values.max():.1e}]")
        print(f"  phi0 range: [{phi0_values.min():.3f}, {phi0_values.max():.3f}]")
        print(f"  Total points: {n_xi * n_phi0}")

    # Initialize result arrays
    stable_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    obs_allowed_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    delta_G_over_G = np.full((n_xi, n_phi0), np.nan)
    G_eff_0 = np.full((n_xi, n_phi0), np.nan)
    G_eff_zmax = np.full((n_xi, n_phi0), np.nan)

    total = n_xi * n_phi0
    count = 0

    for i, xi in enumerate(xi_values):
        for j, phi0 in enumerate(phi0_values):
            count += 1
            if verbose and count % max(1, total // 10) == 0:
                print(f"  Progress: {100*count/total:.0f}%")

            # Create parameters for this point
            params = HRC2Parameters(
                coupling_family=coupling_family,
                potential_type=potential_type,
                xi=xi,
                alpha=xi,  # For linear coupling, use same value
                beta=xi,   # For exponential, use same value
                phi_0=phi0,
                phi_dot_0=0.0,
            )

            try:
                # Create model and solve
                model = create_model(params)
                cosmo = BackgroundCosmology(params, model)
                solution = cosmo.solve(z_max=z_max, z_points=z_points)

                if not solution.success or not solution.geff_valid:
                    continue

                # Check stability
                stability = check_stability_along_trajectory(solution, model)
                if not stability.is_stable:
                    continue

                # Point is dynamically stable
                stable_mask[i, j] = True

                # Store G_eff values
                G_eff_0[i, j] = solution.G_eff_ratio[0]
                G_eff_zmax[i, j] = solution.G_eff_ratio[-1]

                # Compute Delta G/G
                dG = abs(solution.G_eff_ratio[0] - solution.G_eff_ratio[-1])
                delta_G_over_G[i, j] = dG

                # Check observational constraints
                constraints = check_all_constraints_hrc2(
                    solution, model, params, constraint_level
                )

                if constraints.all_allowed:
                    obs_allowed_mask[i, j] = True

            except Exception as e:
                # Integration or model creation failed
                continue

    # Compute per-xi statistics
    stable_fraction = np.zeros(n_xi)
    obs_allowed_fraction = np.zeros(n_xi)
    max_delta_G_stable = np.full(n_xi, np.nan)
    max_delta_G_allowed = np.full(n_xi, np.nan)

    for i in range(n_xi):
        n_stable = stable_mask[i, :].sum()
        n_allowed = obs_allowed_mask[i, :].sum()

        stable_fraction[i] = n_stable / n_phi0
        obs_allowed_fraction[i] = n_allowed / n_phi0

        if n_stable > 0:
            max_delta_G_stable[i] = np.nanmax(delta_G_over_G[i, stable_mask[i, :]])

        if n_allowed > 0:
            max_delta_G_allowed[i] = np.nanmax(delta_G_over_G[i, obs_allowed_mask[i, :]])

    return XiTradeoffResultHRC2(
        coupling_family=coupling_family,
        potential_type=potential_type,
        xi_values=xi_values,
        phi0_values=phi0_values,
        stable_mask=stable_mask,
        obs_allowed_mask=obs_allowed_mask,
        delta_G_over_G=delta_G_over_G,
        G_eff_0=G_eff_0,
        G_eff_zmax=G_eff_zmax,
        stable_fraction=stable_fraction,
        obs_allowed_fraction=obs_allowed_fraction,
        max_delta_G_stable=max_delta_G_stable,
        max_delta_G_allowed=max_delta_G_allowed,
        z_max=z_max,
        constraint_level=constraint_level,
    )


def find_critical_xi_hrc2(
    result: XiTradeoffResultHRC2,
    use_constraints: bool = True,
) -> Tuple[float, float]:
    """Find critical xi and maximum achievable |Delta G/G|.

    Args:
        result: XiTradeoffResultHRC2 from scan
        use_constraints: If True, use constrained values; else stable-only

    Returns:
        Tuple of (xi_critical, max_delta_G)
    """
    if use_constraints:
        fraction = result.obs_allowed_fraction
        max_delta = result.max_delta_G_allowed
    else:
        fraction = result.stable_fraction
        max_delta = result.max_delta_G_stable

    # Find largest xi with any solutions
    has_solutions = fraction > 0

    if not np.any(has_solutions):
        return 0.0, 0.0

    xi_crit = result.xi_values[has_solutions].max()
    max_delta_G = np.nanmax(max_delta[has_solutions])

    return xi_crit, max_delta_G


def compare_coupling_families(
    coupling_families: Optional[List[CouplingFamily]] = None,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    constraint_level: str = "conservative",
    verbose: bool = True,
) -> Dict[CouplingFamily, XiTradeoffResultHRC2]:
    """Compare xi-tradeoff across multiple coupling families.

    Args:
        coupling_families: List of families to compare (default: all three)
        potential_type: Potential type for V(phi)
        xi_values: Array of coupling strengths
        phi0_values: Array of initial field values
        z_max: Maximum redshift
        constraint_level: BBN constraint level
        verbose: Print progress

    Returns:
        Dictionary mapping CouplingFamily -> XiTradeoffResultHRC2
    """
    if coupling_families is None:
        coupling_families = [
            CouplingFamily.LINEAR,
            CouplingFamily.QUADRATIC,
            CouplingFamily.EXPONENTIAL,
        ]

    results = {}

    for family in coupling_families:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scanning {family.value.upper()} coupling family")
            print(f"{'='*60}")

        result = scan_xi_tradeoff_hrc2(
            coupling_family=family,
            potential_type=potential_type,
            xi_values=xi_values,
            phi0_values=phi0_values,
            z_max=z_max,
            constraint_level=constraint_level,
            verbose=verbose,
        )

        results[family] = result

    return results


def print_xi_tradeoff_summary_hrc2(
    result: XiTradeoffResultHRC2,
    show_table: bool = True,
) -> str:
    """Print summary of xi-tradeoff scan results.

    Args:
        result: XiTradeoffResultHRC2 from scan
        show_table: Whether to print detailed table

    Returns:
        Summary text
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"HRC 2.0 XI-TRADEOFF ANALYSIS: {result.coupling_family.value.upper()} COUPLING")
    lines.append("=" * 80)

    lines.append(f"Potential type: {result.potential_type.value}")
    lines.append(f"z_max = {result.z_max:.0f}")
    lines.append(f"Constraint level: {result.constraint_level}")
    lines.append(f"xi range: [{result.xi_values.min():.1e}, {result.xi_values.max():.1e}]")
    lines.append(f"phi0 range: [{result.phi0_values.min():.3f}, {result.phi0_values.max():.3f}]")

    if show_table:
        lines.append("")
        lines.append("RESULTS BY XI VALUE:")
        lines.append("-" * 80)
        lines.append(f"{'xi':>12} | {'Stable%':>8} | {'Allowed%':>9} | "
                    f"{'MaxΔG(stab)':>12} | {'MaxΔG(ok)':>12}")
        lines.append("-" * 80)

        for i, xi in enumerate(result.xi_values):
            stable_pct = 100 * result.stable_fraction[i]
            allowed_pct = 100 * result.obs_allowed_fraction[i]
            max_dg_s = result.max_delta_G_stable[i]
            max_dg_a = result.max_delta_G_allowed[i]

            max_dg_s_str = f"{max_dg_s:.5f}" if not np.isnan(max_dg_s) else "---"
            max_dg_a_str = f"{max_dg_a:.5f}" if not np.isnan(max_dg_a) else "---"

            lines.append(f"{xi:>12.2e} | {stable_pct:>7.1f}% | {allowed_pct:>8.1f}% | "
                        f"{max_dg_s_str:>12} | {max_dg_a_str:>12}")

        lines.append("-" * 80)

    # Key findings
    xi_crit_stable, max_dg_stable = find_critical_xi_hrc2(result, use_constraints=False)
    xi_crit_allowed, max_dg_allowed = find_critical_xi_hrc2(result, use_constraints=True)

    lines.append("")
    lines.append("KEY FINDINGS:")
    lines.append("-" * 80)

    lines.append("")
    lines.append("DYNAMICAL STABILITY ONLY:")
    if xi_crit_stable > 0:
        dH0_stable = estimate_delta_H0(max_dg_stable)
        lines.append(f"  ξ_crit = {xi_crit_stable:.2e}")
        lines.append(f"  Max |ΔG/G| = {max_dg_stable:.4f}")
        lines.append(f"  → ΔH₀ ~ {dH0_stable:.1f} km/s/Mpc")
    else:
        lines.append("  No stable solutions found!")

    lines.append("")
    lines.append(f"WITH OBSERVATIONAL CONSTRAINTS ({result.constraint_level} BBN):")
    if xi_crit_allowed > 0:
        dH0_allowed = estimate_delta_H0(max_dg_allowed)
        lines.append(f"  ξ_crit = {xi_crit_allowed:.2e}")
        lines.append(f"  Max |ΔG/G| = {max_dg_allowed:.4f}")
        lines.append(f"  → ΔH₀ ~ {dH0_allowed:.1f} km/s/Mpc")

        if dH0_allowed >= 5.0:
            lines.append("  STATUS: VIABLE - can address Hubble tension")
        elif dH0_allowed >= 2.0:
            lines.append("  STATUS: MARGINAL - partial resolution")
        else:
            lines.append("  STATUS: INSUFFICIENT - effect too small")
    else:
        lines.append("  No observationally allowed solutions found!")

    lines.append("")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)
    return text


def print_comparison_summary(
    results: Dict[CouplingFamily, XiTradeoffResultHRC2],
) -> str:
    """Print comparison summary across coupling families.

    Args:
        results: Dictionary of results from compare_coupling_families

    Returns:
        Summary text
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("HRC 2.0 COUPLING FAMILY COMPARISON")
    lines.append("=" * 80)

    # Summary table
    lines.append("")
    lines.append(f"{'Coupling':>15} | {'ξ_crit (stable)':>15} | {'ξ_crit (constr)':>15} | "
                f"{'MaxΔG/G':>10} | {'ΔH₀ (km/s/Mpc)':>15}")
    lines.append("-" * 80)

    best_family = None
    best_dH0 = 0.0

    for family, result in results.items():
        xi_crit_s, max_dg_s = find_critical_xi_hrc2(result, use_constraints=False)
        xi_crit_c, max_dg_c = find_critical_xi_hrc2(result, use_constraints=True)

        xi_s_str = f"{xi_crit_s:.2e}" if xi_crit_s > 0 else "N/A"
        xi_c_str = f"{xi_crit_c:.2e}" if xi_crit_c > 0 else "N/A"
        max_dg_str = f"{max_dg_c:.4f}" if max_dg_c > 0 else "N/A"

        dH0 = estimate_delta_H0(max_dg_c) if max_dg_c > 0 else 0.0
        dH0_str = f"{dH0:.1f}" if dH0 > 0 else "N/A"

        lines.append(f"{family.value:>15} | {xi_s_str:>15} | {xi_c_str:>15} | "
                    f"{max_dg_str:>10} | {dH0_str:>15}")

        if dH0 > best_dH0:
            best_dH0 = dH0
            best_family = family

    lines.append("-" * 80)

    # Conclusions
    lines.append("")
    lines.append("CONCLUSIONS:")
    lines.append("-" * 80)

    if best_family is not None and best_dH0 > 0:
        lines.append(f"Best performing coupling: {best_family.value.upper()}")
        lines.append(f"Maximum ΔH₀ contribution: {best_dH0:.1f} km/s/Mpc")
        lines.append("")

        if best_dH0 >= 5.0:
            lines.append("RESULT: HRC 2.0 CAN potentially resolve the Hubble tension")
            lines.append("        with appropriate coupling family choice!")
        elif best_dH0 >= 3.5:
            lines.append("RESULT: HRC 2.0 shows IMPROVEMENT over HRC 1.x (~3.5 km/s/Mpc)")
            lines.append("        but may still need additional mechanisms.")
        else:
            lines.append("RESULT: No significant improvement over HRC 1.x linear coupling.")
            lines.append("        The stability-effect tradeoff remains severe.")
    else:
        lines.append("RESULT: No viable constrained solutions found for any coupling family.")

    lines.append("")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)
    return text
