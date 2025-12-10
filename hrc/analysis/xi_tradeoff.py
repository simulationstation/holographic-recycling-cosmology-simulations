"""Analysis of the xi-stability-effect tradeoff in HRC.

This module quantifies the fundamental tradeoff:
- Smaller xi -> more stable (field doesn't reach critical value)
- Smaller xi -> smaller G_eff variation (less Hubble tension resolution)

The key question: For xi small enough to be stable to z~1100,
how large can |ΔG/G| be after imposing observational constraints?
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..utils.numerics import compute_critical_phi
from ..background import BackgroundCosmology
from ..potentials import (
    Potential,
    QuadraticPotential,
    PlateauPotential,
    POTENTIAL_REGISTRY,
)
from ..constraints import (
    check_bbn_constraint,
    check_ppn_constraints,
    check_stellar_constraints,
    BBNConstraint,
)


# Safety margin for G_eff: require |1 - 8πξφ| > GEFF_SAFETY_MARGIN
GEFF_SAFETY_MARGIN = 0.05  # 5% margin from divergence


@dataclass
class XiTradeoffResult:
    """Result of xi-tradeoff scan with constraint information."""

    xi_values: NDArray[np.floating]
    phi0_values: NDArray[np.floating]

    # 2D arrays [n_xi, n_phi0]
    stable_mask: NDArray[np.bool_]  # True if dynamically stable to z_max
    obs_allowed_mask: NDArray[np.bool_]  # True if passes observational constraints
    delta_G_over_G: NDArray[np.floating]  # NaN for invalid points
    G_eff_0: NDArray[np.floating]  # G_eff/G at z=0
    G_eff_zrec: NDArray[np.floating]  # G_eff/G at z_rec
    divergence_z: NDArray[np.floating]  # z where divergence occurs (NaN if stable)

    # Constraint results (2D arrays)
    bbn_allowed: NDArray[np.bool_]
    ppn_allowed: NDArray[np.bool_]
    stellar_allowed: NDArray[np.bool_]

    # Summary statistics per xi (stable only)
    stable_fraction: NDArray[np.floating]
    max_delta_G_stable: NDArray[np.floating]

    # Summary statistics per xi (stable + observationally allowed)
    obs_allowed_fraction: NDArray[np.floating]
    max_delta_G_allowed: NDArray[np.floating]

    # Metadata
    z_max: float
    z_rec: float
    potential_name: str
    constraint_level: str  # 'conservative', 'moderate', or 'strict'


def _check_geff_safety_margin(
    G_eff_values: NDArray[np.floating],
    margin: float = GEFF_SAFETY_MARGIN,
) -> bool:
    """Check if G_eff values stay safely away from divergence.

    Args:
        G_eff_values: Array of G_eff/G values
        margin: Safety margin (require G_eff > margin and G_eff < 1/margin)

    Returns:
        True if all values are within safe bounds
    """
    if np.any(np.isnan(G_eff_values)):
        return False

    # G_eff should be positive and not too large
    min_geff = np.min(G_eff_values)
    max_geff = np.max(G_eff_values)

    # Require G_eff to be positive and within reasonable bounds
    # Near divergence, G_eff -> infinity, so we cap it
    if min_geff < margin or max_geff > 1.0 / margin:
        return False

    return True


def scan_xi_tradeoff(
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    z_rec: float = 1089.0,
    z_points: int = 500,
    potential: Optional[Potential] = None,
    constraint_level: str = "conservative",
    check_ppn: bool = True,
    check_stellar: bool = True,
    geff_safety_margin: float = GEFF_SAFETY_MARGIN,
    verbose: bool = True,
) -> XiTradeoffResult:
    """Scan the xi-phi0 parameter space with constraint checking.

    For each (xi, phi0) pair:
      1. Integrate background cosmology to z_max
      2. Check dynamical stability (no G_eff divergence, safety margin)
      3. If stable, check observational constraints (BBN, PPN, stellar)
      4. Compute ΔG/G = (G_eff(z=0) - G_eff(z_rec)) / G

    Args:
        xi_values: Array of xi values (log-spaced recommended)
        phi0_values: Array of phi0 values
        z_max: Maximum redshift for integration
        z_rec: Recombination redshift for ΔG/G calculation
        z_points: Number of redshift points for integration
        potential: Scalar field potential (default: QuadraticPotential)
        constraint_level: 'conservative', 'moderate', or 'strict' for BBN
        check_ppn: Whether to check PPN constraints
        check_stellar: Whether to check stellar constraints
        geff_safety_margin: Safety margin for G_eff (avoid near-divergence)
        verbose: Print progress

    Returns:
        XiTradeoffResult with full scan results including constraints
    """
    # Default parameter grids
    if xi_values is None:
        xi_values = np.logspace(-4, -1, 15)
    if phi0_values is None:
        phi0_values = np.linspace(0.01, 0.5, 20)

    # Default potential
    if potential is None:
        potential = QuadraticPotential(V0=0.7, m=1.0)

    potential_name = getattr(potential, 'name', 'unknown')

    n_xi = len(xi_values)
    n_phi0 = len(phi0_values)

    # Initialize result arrays
    stable_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    obs_allowed_mask = np.zeros((n_xi, n_phi0), dtype=bool)
    delta_G_over_G = np.full((n_xi, n_phi0), np.nan)
    G_eff_0 = np.full((n_xi, n_phi0), np.nan)
    G_eff_zrec = np.full((n_xi, n_phi0), np.nan)
    divergence_z = np.full((n_xi, n_phi0), np.nan)

    # Constraint arrays
    bbn_allowed = np.zeros((n_xi, n_phi0), dtype=bool)
    ppn_allowed = np.zeros((n_xi, n_phi0), dtype=bool)
    stellar_allowed = np.zeros((n_xi, n_phi0), dtype=bool)

    total = n_xi * n_phi0
    count = 0

    for i, xi in enumerate(xi_values):
        phi_crit = compute_critical_phi(xi)

        for j, phi0 in enumerate(phi0_values):
            count += 1

            if verbose and count % max(1, total // 10) == 0:
                print(f"  Progress: {100*count/total:.0f}% ({count}/{total})")

            # Skip if phi0 already exceeds critical value (with margin)
            if phi_crit != float('inf') and abs(phi0) >= phi_crit * (1 - geff_safety_margin):
                stable_mask[i, j] = False
                divergence_z[i, j] = 0.0
                continue

            try:
                params = HRCParameters(xi=xi, phi_0=phi0)
                cosmo = BackgroundCosmology(params, potential=potential)
                sol = cosmo.solve(z_max=z_max, z_points=z_points)

                if not sol.geff_valid or not sol.success:
                    stable_mask[i, j] = False
                    if sol.geff_divergence_z is not None:
                        divergence_z[i, j] = sol.geff_divergence_z
                    continue

                # Check safety margin on G_eff values
                if not _check_geff_safety_margin(sol.G_eff_ratio, geff_safety_margin):
                    stable_mask[i, j] = False
                    continue

                # Dynamically stable!
                stable_mask[i, j] = True

                # Compute G_eff values
                g0 = sol.G_eff_at(0.0)
                grec = sol.G_eff_at(z_rec) if z_rec <= z_max else sol.G_eff_at(z_max)

                G_eff_0[i, j] = g0
                G_eff_zrec[i, j] = grec

                # ΔG/G = G_eff(0) - G_eff(z_rec)
                delta_G_over_G[i, j] = g0 - grec

                # Check observational constraints
                # BBN constraint
                bbn_result = check_bbn_constraint(
                    solution=sol,
                    constraint_level=constraint_level,
                )
                bbn_allowed[i, j] = bbn_result.allowed

                # PPN constraints
                if check_ppn:
                    ppn_passed, _ = check_ppn_constraints(
                        solution=sol,
                        params=params,
                    )
                    ppn_allowed[i, j] = ppn_passed
                else:
                    ppn_allowed[i, j] = True

                # Stellar constraints
                if check_stellar:
                    stellar_passed, _ = check_stellar_constraints(solution=sol)
                    stellar_allowed[i, j] = stellar_passed
                else:
                    stellar_allowed[i, j] = True

                # Combined observationally allowed
                obs_allowed_mask[i, j] = (
                    bbn_allowed[i, j] and
                    ppn_allowed[i, j] and
                    stellar_allowed[i, j]
                )

            except Exception as e:
                stable_mask[i, j] = False

    # Compute summary statistics per xi
    stable_fraction = np.zeros(n_xi)
    max_delta_G_stable = np.full(n_xi, np.nan)
    obs_allowed_fraction = np.zeros(n_xi)
    max_delta_G_allowed = np.full(n_xi, np.nan)

    for i in range(n_xi):
        # Stable-only statistics
        stable_count = stable_mask[i, :].sum()
        stable_fraction[i] = stable_count / n_phi0

        if stable_count > 0:
            valid_deltas = delta_G_over_G[i, stable_mask[i, :]]
            max_delta_G_stable[i] = np.nanmax(np.abs(valid_deltas))

        # Stable + observationally allowed statistics
        allowed_count = obs_allowed_mask[i, :].sum()
        obs_allowed_fraction[i] = allowed_count / n_phi0

        if allowed_count > 0:
            valid_deltas = delta_G_over_G[i, obs_allowed_mask[i, :]]
            max_delta_G_allowed[i] = np.nanmax(np.abs(valid_deltas))

    return XiTradeoffResult(
        xi_values=xi_values,
        phi0_values=phi0_values,
        stable_mask=stable_mask,
        obs_allowed_mask=obs_allowed_mask,
        delta_G_over_G=delta_G_over_G,
        G_eff_0=G_eff_0,
        G_eff_zrec=G_eff_zrec,
        divergence_z=divergence_z,
        bbn_allowed=bbn_allowed,
        ppn_allowed=ppn_allowed,
        stellar_allowed=stellar_allowed,
        stable_fraction=stable_fraction,
        max_delta_G_stable=max_delta_G_stable,
        obs_allowed_fraction=obs_allowed_fraction,
        max_delta_G_allowed=max_delta_G_allowed,
        z_max=z_max,
        z_rec=z_rec,
        potential_name=potential_name,
        constraint_level=constraint_level,
    )


def find_critical_xi(
    result: XiTradeoffResult,
    use_constraints: bool = True,
) -> Tuple[float, float]:
    """Find the critical xi value where stability breaks down.

    Args:
        result: XiTradeoffResult from scan_xi_tradeoff
        use_constraints: If True, use observationally allowed points;
                        if False, use dynamically stable points only

    Returns:
        Tuple of (xi_crit, max_delta_G):
        - xi_crit: Largest xi with any stable/allowed solutions
        - max_delta_G: Maximum |ΔG/G| achievable
    """
    if use_constraints:
        fraction = result.obs_allowed_fraction
        max_delta = result.max_delta_G_allowed
    else:
        fraction = result.stable_fraction
        max_delta = result.max_delta_G_stable

    # Find largest xi with fraction > 0
    has_solutions = fraction > 0

    if not np.any(has_solutions):
        return 0.0, 0.0

    xi_crit = result.xi_values[has_solutions].max()
    max_delta_G = np.nanmax(max_delta[has_solutions])

    return xi_crit, max_delta_G


def print_xi_tradeoff_summary(
    result: XiTradeoffResult,
    show_constraints: bool = True,
) -> str:
    """Print a plain-language summary of the xi-tradeoff analysis.

    Args:
        result: XiTradeoffResult from scan_xi_tradeoff
        show_constraints: Whether to show constraint information

    Returns:
        Summary text
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"XI-STABILITY-EFFECT TRADEOFF ANALYSIS ({result.potential_name} potential)")
    lines.append("=" * 80)

    lines.append("")
    lines.append(f"Integration up to z_max = {result.z_max:.0f}")
    lines.append(f"Recombination at z_rec = {result.z_rec:.0f}")
    lines.append(f"xi range: [{result.xi_values.min():.1e}, {result.xi_values.max():.1e}]")
    lines.append(f"phi0 range: [{result.phi0_values.min():.3f}, {result.phi0_values.max():.3f}]")
    lines.append(f"BBN constraint level: {result.constraint_level}")

    lines.append("")
    lines.append("RESULTS BY XI VALUE:")
    lines.append("-" * 80)

    if show_constraints:
        header = (f"{'xi':>12} | {'Stable%':>8} | {'Allowed%':>8} | "
                 f"{'Max|ΔG/G|':>10} | {'Max(constr)':>11} | {'Status':<15}")
    else:
        header = f"{'xi':>12} | {'Stable %':>10} | {'Max |ΔG/G|':>12} | {'Status':<20}"

    lines.append(header)
    lines.append("-" * 80)

    for i, xi in enumerate(result.xi_values):
        stable_pct = 100 * result.stable_fraction[i]
        allowed_pct = 100 * result.obs_allowed_fraction[i]
        max_dg_stable = result.max_delta_G_stable[i]
        max_dg_allowed = result.max_delta_G_allowed[i]

        if stable_pct > 0:
            max_dg_str = f"{max_dg_stable:.4f}"
        else:
            max_dg_str = "N/A"

        if allowed_pct > 0:
            max_dg_constr_str = f"{max_dg_allowed:.4f}"
            status = "ALLOWED"
        elif stable_pct > 0:
            max_dg_constr_str = "N/A"
            status = "stable only"
        else:
            max_dg_constr_str = "N/A"
            status = "UNSTABLE"

        if show_constraints:
            lines.append(f"{xi:>12.2e} | {stable_pct:>7.1f}% | {allowed_pct:>7.1f}% | "
                        f"{max_dg_str:>10} | {max_dg_constr_str:>11} | {status:<15}")
        else:
            lines.append(f"{xi:>12.2e} | {stable_pct:>9.1f}% | {max_dg_str:>12} | {status:<20}")

    lines.append("-" * 80)

    # Find critical values
    xi_crit_stable, max_dg_stable = find_critical_xi(result, use_constraints=False)
    xi_crit_allowed, max_dg_allowed = find_critical_xi(result, use_constraints=True)

    lines.append("")
    lines.append("KEY FINDINGS:")
    lines.append("-" * 80)

    # Stability-only results
    lines.append("")
    lines.append("DYNAMICAL STABILITY ONLY:")
    if xi_crit_stable > 0:
        lines.append(f"  Critical xi: {xi_crit_stable:.2e}")
        lines.append(f"  Max |ΔG/G| (stable): {max_dg_stable:.4f}")
        delta_H0_stable = 35 * max_dg_stable
        lines.append(f"  Approx ΔH0 contribution: ~{delta_H0_stable:.1f} km/s/Mpc")
    else:
        lines.append("  No stable solutions found!")

    # With constraints
    if show_constraints:
        lines.append("")
        lines.append(f"WITH OBSERVATIONAL CONSTRAINTS ({result.constraint_level} BBN):")
        if xi_crit_allowed > 0:
            lines.append(f"  Critical xi: {xi_crit_allowed:.2e}")
            lines.append(f"  Max |ΔG/G| (allowed): {max_dg_allowed:.4f}")
            delta_H0_allowed = 35 * max_dg_allowed
            lines.append(f"  Approx ΔH0 contribution: ~{delta_H0_allowed:.1f} km/s/Mpc")
        else:
            lines.append("  No observationally allowed solutions found!")

    # Conclusions
    lines.append("")
    lines.append("CONCLUSIONS:")
    lines.append("-" * 80)

    if xi_crit_allowed > 0 and max_dg_allowed > 0.1:
        lines.append(f"The model can produce |ΔG/G| ~ {max_dg_allowed:.2f} while satisfying")
        lines.append(f"all constraints up to z = {result.z_max:.0f}.")
        lines.append(f"This corresponds to ΔH0 ~ {35*max_dg_allowed:.0f} km/s/Mpc,")
        if max_dg_allowed > 0.15:
            lines.append("which is sufficient to address the Hubble tension.")
        else:
            lines.append("which may partially address the Hubble tension.")
    elif xi_crit_allowed > 0:
        lines.append(f"Stable + allowed solutions exist but with small effect:")
        lines.append(f"  |ΔG/G| ≤ {max_dg_allowed:.4f} (ΔH0 ≤ {35*max_dg_allowed:.1f} km/s/Mpc)")
        lines.append("This is insufficient to resolve the Hubble tension.")
    elif xi_crit_stable > 0:
        lines.append("Dynamically stable solutions exist, but they violate")
        lines.append("observational constraints (BBN/PPN/stellar).")
        lines.append(f"The unconstrained max |ΔG/G| = {max_dg_stable:.4f} is ruled out.")
    else:
        lines.append("No stable solutions found in the scanned parameter range.")
        lines.append("The scalar field evolves to divergence for all (xi, phi0) tested.")

    lines.append("")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)
    return text


def compare_potentials_xi_tradeoff(
    potentials: Optional[dict] = None,
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    constraint_level: str = "conservative",
    verbose: bool = True,
) -> dict:
    """Compare xi-tradeoff for multiple potentials.

    Args:
        potentials: Dict of name -> Potential instances
        xi_values: Array of xi values
        phi0_values: Array of phi0 values
        z_max: Maximum redshift
        constraint_level: BBN constraint level
        verbose: Print progress

    Returns:
        Dict of potential_name -> XiTradeoffResult
    """
    if potentials is None:
        potentials = {
            "quadratic": QuadraticPotential(V0=0.7, m=1.0),
            "plateau": PlateauPotential(V0=0.7, M=0.5, n=2.0),
        }

    results = {}

    for name, pot in potentials.items():
        if verbose:
            print(f"\nScanning {name} potential...")

        result = scan_xi_tradeoff(
            xi_values=xi_values,
            phi0_values=phi0_values,
            z_max=z_max,
            potential=pot,
            constraint_level=constraint_level,
            verbose=verbose,
        )
        results[name] = result

    return results
