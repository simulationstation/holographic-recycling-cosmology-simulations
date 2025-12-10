"""Analysis of the xi-stability-effect tradeoff in HRC.

This module quantifies the fundamental tradeoff:
- Smaller xi -> more stable (field doesn't reach critical value)
- Smaller xi -> smaller G_eff variation (less Hubble tension resolution)

The key question: For xi small enough to be stable to z~1100,
how large can |ΔG/G| be?
"""

from dataclasses import dataclass
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


@dataclass
class XiTradeoffResult:
    """Result of xi-tradeoff scan."""

    xi_values: NDArray[np.floating]
    phi0_values: NDArray[np.floating]

    # 2D arrays [n_xi, n_phi0]
    stable_mask: NDArray[np.bool_]  # True if stable to z_max
    delta_G_over_G: NDArray[np.floating]  # NaN for unstable points
    G_eff_0: NDArray[np.floating]  # G_eff/G at z=0
    G_eff_zrec: NDArray[np.floating]  # G_eff/G at z_rec
    divergence_z: NDArray[np.floating]  # z where divergence occurs (NaN if stable)

    # Summary statistics per xi
    stable_fraction: NDArray[np.floating]  # Fraction of phi0 values that are stable
    max_delta_G: NDArray[np.floating]  # Max |ΔG/G| among stable points per xi

    # Metadata
    z_max: float
    z_rec: float
    potential_name: str


def scan_xi_tradeoff(
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    z_rec: float = 1089.0,
    z_points: int = 500,
    potential: Optional[Potential] = None,
    verbose: bool = True,
) -> XiTradeoffResult:
    """Scan the xi-phi0 parameter space to quantify the stability-effect tradeoff.

    For each (xi, phi0) pair:
      - Integrate background cosmology to z_max
      - Check stability (no G_eff divergence)
      - If stable, compute ΔG/G = (G_eff(z=0) - G_eff(z_rec)) / G

    Args:
        xi_values: Array of xi values (log-spaced recommended)
                  Default: 15 points from 1e-4 to 1e-1
        phi0_values: Array of phi0 values
                    Default: 20 points from 0.01 to 0.5
        z_max: Maximum redshift for integration
        z_rec: Recombination redshift for ΔG/G calculation
        z_points: Number of redshift points for integration
        potential: Scalar field potential (default: QuadraticPotential)
        verbose: Print progress

    Returns:
        XiTradeoffResult with full scan results
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
    delta_G_over_G = np.full((n_xi, n_phi0), np.nan)
    G_eff_0 = np.full((n_xi, n_phi0), np.nan)
    G_eff_zrec = np.full((n_xi, n_phi0), np.nan)
    divergence_z = np.full((n_xi, n_phi0), np.nan)

    total = n_xi * n_phi0
    count = 0

    for i, xi in enumerate(xi_values):
        phi_crit = compute_critical_phi(xi)

        for j, phi0 in enumerate(phi0_values):
            count += 1

            if verbose and count % max(1, total // 10) == 0:
                print(f"  Progress: {100*count/total:.0f}% ({count}/{total})")

            # Skip if phi0 already exceeds critical value
            if phi_crit != float('inf') and abs(phi0) >= phi_crit * 0.99:
                stable_mask[i, j] = False
                divergence_z[i, j] = 0.0
                continue

            try:
                params = HRCParameters(xi=xi, phi_0=phi0)
                cosmo = BackgroundCosmology(params, potential=potential)
                sol = cosmo.solve(z_max=z_max, z_points=z_points)

                if sol.geff_valid and sol.success:
                    stable_mask[i, j] = True

                    # Compute G_eff values
                    g0 = sol.G_eff_at(0.0)
                    grec = sol.G_eff_at(z_rec) if z_rec <= z_max else sol.G_eff_at(z_max)

                    G_eff_0[i, j] = g0
                    G_eff_zrec[i, j] = grec

                    # ΔG/G = (G_eff(0) - G_eff(z_rec)) / G
                    # Since G_eff is already G_eff/G, this is just the difference
                    delta_G_over_G[i, j] = g0 - grec

                else:
                    stable_mask[i, j] = False
                    if sol.geff_divergence_z is not None:
                        divergence_z[i, j] = sol.geff_divergence_z

            except Exception as e:
                stable_mask[i, j] = False

    # Compute summary statistics per xi
    stable_fraction = np.zeros(n_xi)
    max_delta_G = np.full(n_xi, np.nan)

    for i in range(n_xi):
        stable_count = stable_mask[i, :].sum()
        stable_fraction[i] = stable_count / n_phi0

        if stable_count > 0:
            valid_deltas = delta_G_over_G[i, stable_mask[i, :]]
            max_delta_G[i] = np.nanmax(np.abs(valid_deltas))

    return XiTradeoffResult(
        xi_values=xi_values,
        phi0_values=phi0_values,
        stable_mask=stable_mask,
        delta_G_over_G=delta_G_over_G,
        G_eff_0=G_eff_0,
        G_eff_zrec=G_eff_zrec,
        divergence_z=divergence_z,
        stable_fraction=stable_fraction,
        max_delta_G=max_delta_G,
        z_max=z_max,
        z_rec=z_rec,
        potential_name=potential_name,
    )


def find_critical_xi(result: XiTradeoffResult) -> Tuple[float, float]:
    """Find the critical xi value where stability breaks down.

    Args:
        result: XiTradeoffResult from scan_xi_tradeoff

    Returns:
        Tuple of (xi_crit, max_delta_G_stable):
        - xi_crit: Largest xi with any stable solutions
        - max_delta_G_stable: Maximum |ΔG/G| achievable with stable solutions
    """
    # Find largest xi with stable_fraction > 0
    stable_xi_mask = result.stable_fraction > 0

    if not np.any(stable_xi_mask):
        return 0.0, 0.0

    xi_crit = result.xi_values[stable_xi_mask].max()

    # Max |ΔG/G| among all stable points
    max_delta_G = np.nanmax(result.max_delta_G[stable_xi_mask])

    return xi_crit, max_delta_G


def print_xi_tradeoff_summary(result: XiTradeoffResult) -> str:
    """Print a plain-language summary of the xi-tradeoff analysis.

    Args:
        result: XiTradeoffResult from scan_xi_tradeoff

    Returns:
        Summary text
    """
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"XI-STABILITY-EFFECT TRADEOFF ANALYSIS ({result.potential_name} potential)")
    lines.append("=" * 72)

    lines.append("")
    lines.append(f"Integration up to z_max = {result.z_max:.0f}")
    lines.append(f"Recombination at z_rec = {result.z_rec:.0f}")
    lines.append(f"xi range: [{result.xi_values.min():.1e}, {result.xi_values.max():.1e}]")
    lines.append(f"phi0 range: [{result.phi0_values.min():.3f}, {result.phi0_values.max():.3f}]")

    lines.append("")
    lines.append("RESULTS BY XI VALUE:")
    lines.append("-" * 72)
    lines.append(f"{'xi':>12} | {'Stable %':>10} | {'Max |ΔG/G|':>12} | {'Status':<20}")
    lines.append("-" * 72)

    for i, xi in enumerate(result.xi_values):
        stable_pct = 100 * result.stable_fraction[i]
        max_dg = result.max_delta_G[i]

        if stable_pct > 0:
            status = "has stable solutions"
            max_dg_str = f"{max_dg:.4f}"
        else:
            status = "ALL UNSTABLE"
            max_dg_str = "N/A"

        lines.append(f"{xi:>12.2e} | {stable_pct:>9.1f}% | {max_dg_str:>12} | {status:<20}")

    lines.append("-" * 72)

    # Find critical xi
    xi_crit, max_delta_G_stable = find_critical_xi(result)

    lines.append("")
    lines.append("KEY FINDINGS:")
    lines.append("-" * 72)

    if xi_crit > 0:
        lines.append(f"Critical xi value: xi_crit ≈ {xi_crit:.2e}")
        lines.append(f"  - For xi < {xi_crit:.2e}, stable solutions exist up to z = {result.z_max:.0f}")
        lines.append(f"  - Maximum achievable |ΔG/G| among stable solutions: {max_delta_G_stable:.4f}")

        # Convert to approximate ΔH0
        # H0 ~ sqrt(G_eff), so ΔH0/H0 ~ (1/2) ΔG_eff/G_eff
        # For H0 ~ 70 km/s/Mpc, ΔH0 ~ 35 * |ΔG/G| km/s/Mpc
        delta_H0_approx = 35 * max_delta_G_stable
        lines.append(f"  - Approximate ΔH0 contribution: ~{delta_H0_approx:.1f} km/s/Mpc")

        if max_delta_G_stable < 0.01:
            lines.append("")
            lines.append("CONCLUSION: The maximum |ΔG/G| for stable solutions is very small.")
            lines.append("This suggests the non-minimal coupling cannot produce large")
            lines.append("enough G_eff variation to resolve the Hubble tension while")
            lines.append("remaining stable up to recombination.")
        elif max_delta_G_stable < 0.05:
            lines.append("")
            lines.append("CONCLUSION: Modest |ΔG/G| achievable with stable solutions.")
            lines.append("May contribute partially to Hubble tension resolution.")
        else:
            lines.append("")
            lines.append("CONCLUSION: Significant |ΔG/G| achievable with stable solutions.")
            lines.append("The model may be viable for Hubble tension resolution.")
    else:
        lines.append("No stable solutions found for any xi value in the scanned range.")
        lines.append("The scalar field reaches the critical value before z = {result.z_max:.0f}")
        lines.append("for all tested parameter combinations.")
        lines.append("")
        lines.append("CONCLUSION: The non-minimal coupling drives phi to divergence")
        lines.append("regardless of initial conditions in this parameter range.")

    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text


def compare_potentials_xi_tradeoff(
    potentials: Optional[dict] = None,
    xi_values: Optional[NDArray[np.floating]] = None,
    phi0_values: Optional[NDArray[np.floating]] = None,
    z_max: float = 1100.0,
    verbose: bool = True,
) -> dict:
    """Compare xi-tradeoff for multiple potentials.

    Args:
        potentials: Dict of name -> Potential instances
        xi_values: Array of xi values
        phi0_values: Array of phi0 values
        z_max: Maximum redshift
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
            verbose=verbose,
        )
        results[name] = result

    return results
