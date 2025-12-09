"""Stellar evolution constraints on variable G.

Stellar structure and evolution provide constraints on variations of
Newton's constant over cosmic time. Key constraints come from:

1. Helioseismology: Constrains G at solar formation (4.5 Gyr ago)
2. White dwarf cooling: Constrains G evolution over Gyr timescales
3. Globular cluster ages: Constrains G at early times
4. Binary pulsar orbits: Constrains Ġ/G

These constraints are generally weaker than solar system tests but
probe G over longer timescales.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from ..utils.config import HRCParameters
from ..background import BackgroundSolution


@dataclass
class StellarConstraint:
    """Result of a stellar evolution constraint check."""

    name: str
    allowed: bool
    value: float
    bound: float
    sigma_margin: float
    z_range: Tuple[float, float]  # Redshift range probed
    message: str


def _z_from_lookback_time(t_Gyr: float, H0: float = 70.0) -> float:
    """Convert lookback time in Gyr to redshift.

    Using approximate relation for flat ΛCDM with Ω_m = 0.3.

    Args:
        t_Gyr: Lookback time in Gyr
        H0: Hubble constant in km/s/Mpc

    Returns:
        Approximate redshift
    """
    # Hubble time: t_H = 1/H0 ≈ 14 Gyr for H0 = 70
    t_H = 9.78 / (H0 / 100)  # Gyr

    # For flat ΛCDM with Ω_m = 0.3:
    # t(z) ≈ t_H * 2/3 * [1 - (1+z)^(-3/2)] / Ω_m^(1/2)
    # Inverting approximately:
    if t_Gyr >= t_H:
        return np.inf

    x = t_Gyr / t_H
    z = (1 - x * 1.5) ** (-2 / 3) - 1

    return max(0.0, z)


def _helioseismology_constraint() -> Tuple[float, float, float]:
    """Return helioseismology constraint on ΔG/G.

    Solar oscillation frequencies constrain the solar structure,
    which depends on G at the time of solar formation (z ≈ 0.35,
    t ≈ 4.5 Gyr lookback).

    Constraint: |G(t_solar)/G(today) - 1| < 0.02

    Returns:
        Tuple of (bound, sigma, z_formation)
    """
    return 0.02, 0.01, 0.35


def _white_dwarf_cooling_constraint() -> Tuple[float, float, float]:
    """Return white dwarf cooling constraint on ΔG/G.

    White dwarf cooling rates depend on G. Comparing observed
    cooling sequences with models constrains G variation.

    Constraint: |ΔG/G| < 0.05 over last 10 Gyr (z ≈ 2)

    Returns:
        Tuple of (bound, sigma, z_max)
    """
    return 0.05, 0.02, 2.0


def _globular_cluster_constraint() -> Tuple[float, float, float]:
    """Return globular cluster age constraint on ΔG/G.

    Globular cluster ages (determined from main sequence turnoff)
    depend on stellar evolution timescales ∝ G^(-5/2).

    Constraint: |ΔG/G| < 0.1 at z ≈ 3-5 (cluster formation)

    Returns:
        Tuple of (bound, sigma, z_formation)
    """
    return 0.10, 0.05, 4.0


def _binary_pulsar_constraint() -> Tuple[float, float]:
    """Return binary pulsar Ġ/G constraint.

    Binary pulsars (especially PSR J0737-3039) provide very precise
    tests of GR. The orbital period derivative constrains Ġ/G.

    Constraint: |Ġ/G| < 2 × 10⁻¹² yr⁻¹ (comparable to LLR)

    Returns:
        Tuple of (bound in H₀ units, sigma)
    """
    # 2e-12 yr^-1 = 6.3e-20 s^-1
    # H0 = 70 km/s/Mpc = 2.27e-18 s^-1
    bound_H0 = 6.3e-20 / 2.27e-18
    return bound_H0, bound_H0 * 0.5


def check_helioseismology_constraint(
    solution: BackgroundSolution,
) -> StellarConstraint:
    """Check helioseismology constraint.

    Args:
        solution: Background cosmology solution

    Returns:
        StellarConstraint result
    """
    bound, sigma, z_solar = _helioseismology_constraint()

    G_eff_today = solution.G_eff_at(0.0)
    G_eff_solar = solution.G_eff_at(z_solar)

    if np.isnan(G_eff_today) or np.isnan(G_eff_solar):
        return StellarConstraint(
            name="helioseismology",
            allowed=False,
            value=np.inf,
            bound=bound,
            sigma_margin=-np.inf,
            z_range=(0.0, z_solar),
            message="G_eff is NaN",
        )

    value = abs(G_eff_solar / G_eff_today - 1.0)
    allowed = value < bound
    sigma_margin = (bound - value) / sigma

    if allowed:
        message = f"|ΔG/G|_solar = {value:.3f} < {bound:.2f} ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|ΔG/G|_solar = {value:.3f} > {bound:.2f} VIOLATED"

    return StellarConstraint(
        name="helioseismology",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        z_range=(0.0, z_solar),
        message=message,
    )


def check_white_dwarf_constraint(
    solution: BackgroundSolution,
) -> StellarConstraint:
    """Check white dwarf cooling constraint.

    Args:
        solution: Background cosmology solution

    Returns:
        StellarConstraint result
    """
    bound, sigma, z_max = _white_dwarf_cooling_constraint()

    G_eff_today = solution.G_eff_at(0.0)

    # Use minimum of solution range and z_max
    z_probe = min(z_max, solution.z[-1])
    G_eff_wd = solution.G_eff_at(z_probe)

    if np.isnan(G_eff_today) or np.isnan(G_eff_wd):
        return StellarConstraint(
            name="white_dwarf",
            allowed=False,
            value=np.inf,
            bound=bound,
            sigma_margin=-np.inf,
            z_range=(0.0, z_probe),
            message="G_eff is NaN",
        )

    value = abs(G_eff_wd / G_eff_today - 1.0)
    allowed = value < bound
    sigma_margin = (bound - value) / sigma

    if allowed:
        message = f"|ΔG/G|_WD = {value:.3f} < {bound:.2f} ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|ΔG/G|_WD = {value:.3f} > {bound:.2f} VIOLATED"

    return StellarConstraint(
        name="white_dwarf",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        z_range=(0.0, z_probe),
        message=message,
    )


def check_globular_cluster_constraint(
    solution: BackgroundSolution,
) -> StellarConstraint:
    """Check globular cluster age constraint.

    Args:
        solution: Background cosmology solution

    Returns:
        StellarConstraint result
    """
    bound, sigma, z_gc = _globular_cluster_constraint()

    G_eff_today = solution.G_eff_at(0.0)

    # Use minimum of solution range and z_gc
    z_probe = min(z_gc, solution.z[-1])
    G_eff_gc = solution.G_eff_at(z_probe)

    if np.isnan(G_eff_today) or np.isnan(G_eff_gc):
        return StellarConstraint(
            name="globular_cluster",
            allowed=False,
            value=np.inf,
            bound=bound,
            sigma_margin=-np.inf,
            z_range=(0.0, z_probe),
            message="G_eff is NaN",
        )

    value = abs(G_eff_gc / G_eff_today - 1.0)
    allowed = value < bound
    sigma_margin = (bound - value) / sigma

    if allowed:
        message = f"|ΔG/G|_GC = {value:.3f} < {bound:.2f} ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|ΔG/G|_GC = {value:.3f} > {bound:.2f} VIOLATED"

    return StellarConstraint(
        name="globular_cluster",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        z_range=(0.0, z_probe),
        message=message,
    )


def check_stellar_constraints(
    solution: BackgroundSolution,
    verbose: bool = False,
) -> Tuple[bool, List[StellarConstraint]]:
    """Check all stellar evolution constraints.

    Args:
        solution: Background cosmology solution
        verbose: Print results

    Returns:
        Tuple of (all_passed, list of StellarConstraint)
    """
    results = [
        check_helioseismology_constraint(solution),
        check_white_dwarf_constraint(solution),
        check_globular_cluster_constraint(solution),
    ]

    all_passed = all(r.allowed for r in results)

    if verbose:
        print("\n=== Stellar Evolution Constraint Checks ===")
        for r in results:
            status = "✓" if r.allowed else "✗"
            print(f"{status} {r.name}: {r.message}")

    return all_passed, results
