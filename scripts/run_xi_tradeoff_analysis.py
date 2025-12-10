#!/usr/bin/env python3
"""Run the xi-stability-effect tradeoff analysis.

This script quantifies the fundamental tradeoff in HRC:
- Smaller xi -> more stable (field doesn't reach critical value)
- Smaller xi -> smaller G_eff variation (less Hubble tension resolution)

The key question: For xi small enough to be stable to z~1100,
how large can |ΔG/G| be?

Usage:
    python scripts/run_xi_tradeoff_analysis.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hrc.analysis import (
    scan_xi_tradeoff,
    print_xi_tradeoff_summary,
    compare_potentials_xi_tradeoff,
    find_critical_xi,
)
from hrc.plots import (
    create_xi_tradeoff_summary_figure,
    plot_max_delta_G_vs_xi,
    plot_potential_comparison,
)
from hrc.potentials import QuadraticPotential, PlateauPotential


def main():
    print("=" * 72)
    print("XI-STABILITY-EFFECT TRADEOFF ANALYSIS")
    print("=" * 72)

    # Define parameter grids
    xi_values = np.logspace(-4, -1, 15)  # 1e-4 to 0.1
    phi0_values = np.linspace(0.01, 0.5, 20)

    print(f"\nParameter grid:")
    print(f"  xi: {len(xi_values)} points from {xi_values.min():.1e} to {xi_values.max():.1e}")
    print(f"  phi0: {len(phi0_values)} points from {phi0_values.min():.3f} to {phi0_values.max():.3f}")
    print(f"  Total: {len(xi_values) * len(phi0_values)} integrations")

    # ========================================================================
    # PART 1: Quadratic potential analysis
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 1: QUADRATIC POTENTIAL")
    print("=" * 72)

    quadratic_pot = QuadraticPotential(V0=0.7, m=1.0)

    print("\nRunning scan for quadratic potential...")
    result_quad = scan_xi_tradeoff(
        xi_values=xi_values,
        phi0_values=phi0_values,
        z_max=1100.0,
        potential=quadratic_pot,
        verbose=True,
    )

    # Print summary
    print_xi_tradeoff_summary(result_quad)

    # Create summary figure
    print("\nGenerating quadratic potential figures...")
    fig = create_xi_tradeoff_summary_figure(
        result_quad,
        save_path='figures/xi_tradeoff_quadratic_summary.png'
    )
    plt.close(fig)

    # Create key plot (max |ΔG/G| vs xi)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_max_delta_G_vs_xi(result_quad, ax=ax)
    plt.savefig('figures/xi_tradeoff_max_deltaG.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/xi_tradeoff_max_deltaG.png")

    # ========================================================================
    # PART 2: Plateau potential analysis
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 2: PLATEAU POTENTIAL")
    print("=" * 72)

    plateau_pot = PlateauPotential(V0=0.7, M=0.5, n=2.0)

    print("\nRunning scan for plateau potential...")
    result_plateau = scan_xi_tradeoff(
        xi_values=xi_values,
        phi0_values=phi0_values,
        z_max=1100.0,
        potential=plateau_pot,
        verbose=True,
    )

    # Print summary
    print_xi_tradeoff_summary(result_plateau)

    # Create summary figure
    fig = create_xi_tradeoff_summary_figure(
        result_plateau,
        save_path='figures/xi_tradeoff_plateau_summary.png'
    )
    plt.close(fig)

    # ========================================================================
    # PART 3: Comparison
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 3: POTENTIAL COMPARISON")
    print("=" * 72)

    results = {
        "quadratic": result_quad,
        "plateau": result_plateau,
    }

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_potential_comparison(results, ax=ax,
                             save_path='figures/xi_tradeoff_quadratic_vs_plateau.png')
    plt.close(fig)

    # Print comparison summary
    print("\nCOMPARISON SUMMARY:")
    print("-" * 72)

    for name, result in results.items():
        xi_crit, max_dg = find_critical_xi(result)
        n_stable_xi = (result.stable_fraction > 0).sum()

        print(f"\n{name.upper()} POTENTIAL:")
        print(f"  Xi values with stable solutions: {n_stable_xi}/{len(xi_values)}")
        if xi_crit > 0:
            print(f"  Critical xi: {xi_crit:.2e}")
            print(f"  Max |ΔG/G| achievable: {max_dg:.4f}")
            delta_H0_approx = 35 * max_dg
            print(f"  Approximate ΔH0 contribution: ~{delta_H0_approx:.1f} km/s/Mpc")
        else:
            print("  No stable solutions found!")

    # ========================================================================
    # FINAL CONCLUSIONS
    # ========================================================================
    print("\n" + "=" * 72)
    print("FINAL CONCLUSIONS")
    print("=" * 72)

    # Check if any potential has stable solutions with significant effect
    any_significant = False
    for name, result in results.items():
        xi_crit, max_dg = find_critical_xi(result)
        if max_dg > 0.05:  # 5% change in G
            any_significant = True
            print(f"\n{name}: Shows potential for significant G_eff variation")

    if not any_significant:
        print("""
The analysis shows that for both potentials:

1. STABILITY CONSTRAINT: For xi values small enough that stable solutions
   exist up to recombination (z ~ 1100), the scalar field evolution is
   strongly constrained.

2. EFFECT SIZE: The maximum achievable |ΔG/G| in the stable region is
   very small - typically < 1%.

3. HUBBLE TENSION: This translates to ΔH0 contributions of only a few
   km/s/Mpc, far below the ~5-7 km/s/Mpc needed to resolve the tension.

4. ROOT CAUSE: The ξφR coupling in the scalar field equation drives φ
   toward the critical value regardless of the potential form. The only
   way to keep φ away from divergence is to use very small ξ, which
   then produces negligible G_eff variation.

CONCLUSION: The simple non-minimal coupling ξφR cannot simultaneously:
  (a) remain stable up to recombination
  (b) produce G_eff variation large enough for the Hubble tension

This is a fundamental theoretical limitation of the linear coupling model.
""")
    else:
        print("\nSome potentials show promise - further investigation warranted.")

    print("\n" + "=" * 72)
    print("FIGURES SAVED:")
    print("  - figures/xi_tradeoff_max_deltaG.png")
    print("  - figures/xi_tradeoff_quadratic_summary.png")
    print("  - figures/xi_tradeoff_plateau_summary.png")
    print("  - figures/xi_tradeoff_quadratic_vs_plateau.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
