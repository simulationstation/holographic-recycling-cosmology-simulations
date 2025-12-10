"""Summary and reporting functions for HRC model analysis.

This module provides functions to generate human-readable summaries
of HRC analysis results, including a layman-friendly explanation of
what different potentials mean for the model.
"""

from typing import Dict, Optional, List, Union
from dataclasses import dataclass

from ..potentials import (
    Potential,
    QuadraticPotential,
    PlateauPotential,
    SymmetronPotential,
    ExponentialPotential,
    POTENTIAL_REGISTRY,
)
from .parameter_scan import ParameterScanResult, scan_parameter_space


@dataclass
class PotentialSummary:
    """Summary of a potential's performance."""

    name: str
    valid_fraction: float
    resolves_fraction: float
    invalid_fraction: float
    description: str
    recommendation: str


# Layman-friendly descriptions of potentials
POTENTIAL_DESCRIPTIONS = {
    "quadratic": (
        "The simplest potential, like a ball in a curved bowl. "
        "The field tends to roll toward larger values as the universe expands, "
        "which can cause problems at early times."
    ),
    "plateau": (
        "A potential that flattens out at large field values, like a tabletop. "
        "This can help stabilize the field and prevent it from running away "
        "to dangerous values."
    ),
    "symmetron": (
        "A 'Mexican hat' shaped potential with two stable points. "
        "The field can settle into one of these stable spots, potentially "
        "avoiding the problematic divergence region."
    ),
    "exponential": (
        "A potential that decreases exponentially as the field grows. "
        "This allows the field to 'track' the expansion of the universe "
        "in a controlled way."
    ),
}


def compare_potentials(
    potentials: Optional[Dict[str, Potential]] = None,
    xi_range: tuple = (0.01, 0.1),
    phi_0_range: tuple = (0.05, 0.5),
    n_xi: int = 15,
    n_phi_0: int = 15,
    z_max: float = 1100.0,
    verbose: bool = True,
) -> Dict[str, PotentialSummary]:
    """Compare multiple potentials and return summaries.

    Args:
        potentials: Dictionary of potential name -> Potential instance.
                   If None, uses default set of potentials.
        xi_range: Range of xi values to scan
        phi_0_range: Range of phi_0 values to scan
        n_xi: Number of xi grid points
        n_phi_0: Number of phi_0 grid points
        z_max: Maximum redshift for integration
        verbose: Print progress

    Returns:
        Dictionary of potential name -> PotentialSummary
    """
    if potentials is None:
        potentials = {
            "quadratic": QuadraticPotential(V0=0.7, m=1.0),
            "plateau": PlateauPotential(V0=0.7, M=0.5, n=2.0),
            "symmetron": SymmetronPotential(V0=0.7, mu2=1.0, lambda_=2.0),
            "exponential": ExponentialPotential(V0=0.7, lambda_=0.5, M=1.0),
        }

    summaries = {}

    for name, pot in potentials.items():
        if verbose:
            print(f"Analyzing {name} potential...")

        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
            potential=pot,
        )

        total = n_xi * n_phi_0
        valid_frac = result.geff_valid.sum() / total
        resolves_frac = result.n_valid_resolves / total
        invalid_frac = result.n_invalid / total

        # Generate recommendation based on results
        if valid_frac > 0.5:
            if resolves_frac > 0.2:
                recommendation = "Promising - large viable parameter space with tension resolution"
            else:
                recommendation = "Stable but limited tension resolution"
        elif valid_frac > 0.1:
            recommendation = "Partially viable - careful parameter tuning needed"
        else:
            recommendation = "Mostly excluded - field evolves to problematic region"

        description = POTENTIAL_DESCRIPTIONS.get(
            name,
            "A scalar field potential that shapes the field's evolution."
        )

        summaries[name] = PotentialSummary(
            name=name,
            valid_fraction=valid_frac,
            resolves_fraction=resolves_frac,
            invalid_fraction=invalid_frac,
            description=description,
            recommendation=recommendation,
        )

    return summaries


def print_layman_summary(
    summaries: Optional[Dict[str, PotentialSummary]] = None,
    z_max: float = 1100.0,
) -> str:
    """Print a layman-friendly summary of potential comparison results.

    Args:
        summaries: Pre-computed summaries (will compute if None)
        z_max: Maximum redshift (for context)

    Returns:
        Summary text as a string
    """
    if summaries is None:
        print("Computing potential comparison (this may take a moment)...")
        summaries = compare_potentials(z_max=z_max, verbose=True)

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("HOLOGRAPHIC RECYCLING COSMOLOGY - POTENTIAL COMPARISON SUMMARY")
    lines.append("=" * 72)

    lines.append("")
    lines.append("BACKGROUND:")
    lines.append("-" * 72)
    lines.append("In HRC theory, a scalar field (phi) modifies how gravity behaves.")
    lines.append("The 'potential' V(phi) determines how this field evolves over time.")
    lines.append("")
    lines.append("The PROBLEM: If phi grows too large, gravity becomes infinitely strong")
    lines.append("at a critical value, which is unphysical. This happens before the CMB")
    lines.append(f"epoch (z ~ {z_max:.0f}) with the simplest potentials.")
    lines.append("")
    lines.append("The GOAL: Find potentials that keep phi below the critical value")
    lines.append("while still allowing enough gravity variation to resolve the")
    lines.append("Hubble tension (disagreement in measurements of the expansion rate).")

    lines.append("")
    lines.append("RESULTS BY POTENTIAL:")
    lines.append("-" * 72)

    # Sort by valid fraction
    sorted_summaries = sorted(
        summaries.values(),
        key=lambda s: s.valid_fraction,
        reverse=True
    )

    for s in sorted_summaries:
        lines.append("")
        lines.append(f"{s.name.upper()}")
        lines.append(f"  What it is: {s.description}")
        lines.append(f"  Viable parameter space: {100*s.valid_fraction:.0f}%")
        lines.append(f"  Can resolve Hubble tension: {100*s.resolves_fraction:.0f}%")
        lines.append(f"  Assessment: {s.recommendation}")

    # Overall conclusions
    lines.append("")
    lines.append("CONCLUSIONS:")
    lines.append("-" * 72)

    best = sorted_summaries[0]
    if best.valid_fraction > 0.3:
        lines.append(f"Best performing potential: {best.name.upper()}")
        lines.append(f"  - {100*best.valid_fraction:.0f}% of parameter space remains physically viable")
        lines.append(f"  - {100*best.resolves_fraction:.0f}% can potentially resolve the Hubble tension")
        lines.append("")
        if best.name != "quadratic":
            lines.append("The non-quadratic potential stabilizes the scalar field,")
            lines.append("preventing it from reaching the problematic critical value.")
        else:
            lines.append("Even the simple quadratic potential works in some parameter regions.")
    else:
        lines.append("All tested potentials show limited viable parameter space.")
        lines.append("Further theoretical development may be needed to find stable")
        lines.append("configurations that can resolve the Hubble tension.")

    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text


def generate_report(
    potentials: Optional[Dict[str, Potential]] = None,
    xi_range: tuple = (0.01, 0.1),
    phi_0_range: tuple = (0.05, 0.5),
    n_xi: int = 15,
    n_phi_0: int = 15,
    z_max: float = 1100.0,
    output_path: Optional[str] = None,
) -> str:
    """Generate a full report comparing potentials.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi: Number of xi grid points
        n_phi_0: Number of phi_0 grid points
        z_max: Maximum redshift
        output_path: Path to save report (optional)

    Returns:
        Report text
    """
    summaries = compare_potentials(
        potentials=potentials,
        xi_range=xi_range,
        phi_0_range=phi_0_range,
        n_xi=n_xi,
        n_phi_0=n_phi_0,
        z_max=z_max,
        verbose=True,
    )

    report = print_layman_summary(summaries, z_max=z_max)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")

    return report


def quick_potential_check(
    potential: Union[Potential, str],
    xi: float = 0.03,
    phi_0: float = 0.15,
    z_max: float = 1100.0,
) -> Dict:
    """Quick check of a single potential with specific parameters.

    Args:
        potential: Potential instance or name string
        xi: Non-minimal coupling
        phi_0: Initial scalar field value
        z_max: Maximum redshift

    Returns:
        Dictionary with results
    """
    from ..utils.config import HRCParameters
    from ..background import BackgroundCosmology

    # Resolve potential
    if isinstance(potential, str):
        if potential not in POTENTIAL_REGISTRY:
            raise ValueError(f"Unknown potential: {potential}")
        pot = POTENTIAL_REGISTRY[potential]()
        pot_name = potential
    else:
        pot = potential
        pot_name = getattr(pot, 'name', 'custom')

    params = HRCParameters(xi=xi, phi_0=phi_0)
    cosmo = BackgroundCosmology(params, potential=pot)
    sol = cosmo.solve(z_max=z_max, z_points=500)

    result = {
        "potential": pot_name,
        "xi": xi,
        "phi_0": phi_0,
        "z_max": z_max,
        "valid": sol.geff_valid,
        "divergence_z": sol.geff_divergence_z,
        "phi_critical": sol.phi_critical,
    }

    if sol.geff_valid:
        result["G_eff_0"] = sol.G_eff_at(0)
        result["G_eff_cmb"] = sol.G_eff_at(1089) if z_max >= 1089 else None
        result["phi_final"] = sol.phi[-1]

        # Compute Delta H0
        from ..effective_gravity import compute_hubble_tension
        tension = compute_hubble_tension(sol, params)
        if tension["valid"]:
            result["Delta_H0"] = tension["Delta_H0"]
            result["resolves_tension"] = tension["Delta_H0"] >= 3.0

    return result
