#!/usr/bin/env python3
"""Run the xi-stability-effect tradeoff analysis with observational constraints.

This script quantifies the fundamental tradeoff in HRC:
- Smaller xi -> more stable (field doesn't reach critical value)
- Smaller xi -> smaller G_eff variation (less Hubble tension resolution)

The key question: For xi small enough to be stable to z~1100 AND pass
BBN/PPN/stellar constraints, how large can |ΔG/G| be?

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
    plot_max_delta_G_with_constraints,
    plot_potential_comparison,
    plot_constraint_breakdown,
)
from hrc.potentials import QuadraticPotential, PlateauPotential


def print_numeric_summary(result, potential_name):
    """Print concise numeric summary for a potential.

    Shows for each xi:
    - Fraction stable, fraction BBN-allowed, fraction all-allowed
    - Max |ΔG/G| for stable-only, Max |ΔG/G| for allowed
    """
    print("")
    print(f"NUMERIC SUMMARY ({potential_name}):")
    print("-" * 90)
    print(f"{'xi':>12} | {'Stable':>8} | {'BBN':>8} | {'PPN':>8} | {'All':>8} | "
          f"{'MaxΔG(stab)':>12} | {'MaxΔG(ok)':>12}")
    print("-" * 90)

    n_phi0 = len(result.phi0_values)

    for i, xi in enumerate(result.xi_values):
        stable_frac = result.stable_fraction[i]

        # Compute individual constraint fractions among stable points
        stable_pts = result.stable_mask[i, :]
        n_stable = stable_pts.sum()

        if n_stable > 0:
            bbn_frac = (result.bbn_allowed[i, :] & stable_pts).sum() / n_phi0
            ppn_frac = (result.ppn_allowed[i, :] & stable_pts).sum() / n_phi0
        else:
            bbn_frac = 0.0
            ppn_frac = 0.0

        allowed_frac = result.obs_allowed_fraction[i]

        max_dg_stable = result.max_delta_G_stable[i]
        max_dg_allowed = result.max_delta_G_allowed[i]

        max_dg_s_str = f"{max_dg_stable:.5f}" if not np.isnan(max_dg_stable) else "---"
        max_dg_a_str = f"{max_dg_allowed:.5f}" if not np.isnan(max_dg_allowed) else "---"

        print(f"{xi:>12.2e} | {100*stable_frac:>7.1f}% | {100*bbn_frac:>7.1f}% | "
              f"{100*ppn_frac:>7.1f}% | {100*allowed_frac:>7.1f}% | "
              f"{max_dg_s_str:>12} | {max_dg_a_str:>12}")

    print("-" * 90)


def print_final_summary(results):
    """Print the final summary with conclusions for all potentials."""
    print("")
    print("=" * 72)
    print("FINAL CONSTRAINT-AWARE SUMMARY")
    print("=" * 72)

    for name, result in results.items():
        xi_crit_stable, max_dg_stable = find_critical_xi(result, use_constraints=False)
        xi_crit_allowed, max_dg_allowed = find_critical_xi(result, use_constraints=True)

        print(f"\n{name.upper()} POTENTIAL:")
        print("-" * 50)

        # Stability-only
        if xi_crit_stable > 0:
            delta_H0_stable = 35 * max_dg_stable
            print(f"  Dynamically stable region:")
            print(f"    ξ_crit = {xi_crit_stable:.2e}")
            print(f"    Max |ΔG/G| = {max_dg_stable:.4f}")
            print(f"    -> ΔH₀ ~ {delta_H0_stable:.1f} km/s/Mpc")
        else:
            print("  No dynamically stable solutions found.")

        # With constraints
        if xi_crit_allowed > 0:
            delta_H0_allowed = 35 * max_dg_allowed
            print(f"  With BBN + PPN + stellar constraints:")
            print(f"    ξ_crit = {xi_crit_allowed:.2e}")
            print(f"    Max |ΔG/G| = {max_dg_allowed:.4f}")
            print(f"    -> ΔH₀ ~ {delta_H0_allowed:.1f} km/s/Mpc")

            # Conclusions based on effect size
            if delta_H0_allowed >= 5.0:
                print(f"    STATUS: VIABLE - can address Hubble tension")
            elif delta_H0_allowed >= 2.0:
                print(f"    STATUS: MARGINAL - partial tension resolution")
            else:
                print(f"    STATUS: INSUFFICIENT - effect too small")
        elif xi_crit_stable > 0:
            print("  With constraints: NO allowed configurations!")
            print("    All stable solutions violate BBN/PPN/stellar bounds.")

    # Overall conclusion
    print("")
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)

    # Check best case across all potentials
    best_max_dg = 0.0
    best_potential = None
    for name, result in results.items():
        _, max_dg = find_critical_xi(result, use_constraints=True)
        if max_dg > best_max_dg:
            best_max_dg = max_dg
            best_potential = name

    if best_max_dg > 0:
        best_delta_H0 = 35 * best_max_dg
        print(f"""
For the {best_potential} potential with {result.constraint_level} BBN constraints:
  - Max |ΔG/G| = {best_max_dg:.4f} (after all constraints)
  - This corresponds to ΔH₀ ~ {best_delta_H0:.1f} km/s/Mpc

The Hubble tension is ΔH₀ ~ 5-7 km/s/Mpc between local and CMB measurements.
""")
        if best_delta_H0 >= 5.0:
            print("RESULT: The HRC model CAN potentially resolve the Hubble tension")
            print("        while satisfying all observational constraints.")
        elif best_delta_H0 >= 2.0:
            print("RESULT: The HRC model can PARTIALLY address the Hubble tension.")
            print("        Additional mechanisms may be needed.")
        else:
            print("RESULT: The HRC model CANNOT resolve the Hubble tension.")
            print("        The stability-effect tradeoff is too severe.")
    else:
        print("""
No configurations pass all constraints (dynamical stability + BBN + PPN + stellar).

The fundamental tension:
  - Large ξ needed for significant G_eff variation
  - Large ξ causes instability (G_eff divergence) or constraint violations

RESULT: The linear ξφR coupling CANNOT simultaneously:
  (a) remain stable up to recombination
  (b) satisfy observational constraints
  (c) produce G_eff variation large enough for Hubble tension
""")


def main():
    print("=" * 72)
    print("XI-STABILITY-EFFECT TRADEOFF ANALYSIS WITH CONSTRAINTS")
    print("=" * 72)

    # Define parameter grids
    xi_values = np.logspace(-4, -1, 15)  # 1e-4 to 0.1
    phi0_values = np.linspace(0.01, 0.5, 20)

    print(f"\nParameter grid:")
    print(f"  xi: {len(xi_values)} points from {xi_values.min():.1e} to {xi_values.max():.1e}")
    print(f"  phi0: {len(phi0_values)} points from {phi0_values.min():.3f} to {phi0_values.max():.3f}")
    print(f"  Total: {len(xi_values) * len(phi0_values)} integrations")
    print(f"\nConstraints: BBN (conservative) + PPN + Stellar")
    print(f"Safety margin: 5% from G_eff divergence")

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
        constraint_level="conservative",
        check_ppn=True,
        check_stellar=True,
        verbose=True,
    )

    # Print detailed summary
    print_xi_tradeoff_summary(result_quad)

    # Print numeric summary table
    print_numeric_summary(result_quad, "quadratic")

    # Create summary figure (4-panel with constraints)
    print("\nGenerating quadratic potential figures...")
    fig = create_xi_tradeoff_summary_figure(
        result_quad,
        save_path='figures/xi_tradeoff_quadratic_summary.png'
    )
    plt.close(fig)

    # Create key plot (max |ΔG/G| vs xi with constraints)
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_max_delta_G_with_constraints(result_quad, ax=ax,
                                      save_path='figures/xi_tradeoff_max_deltaG_with_constraints.png')
    plt.close(fig)

    # Create constraint breakdown plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_constraint_breakdown(result_quad, ax=ax)
    plt.savefig('figures/xi_tradeoff_constraint_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/xi_tradeoff_constraint_breakdown.png")

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
        constraint_level="conservative",
        check_ppn=True,
        check_stellar=True,
        verbose=True,
    )

    # Print detailed summary
    print_xi_tradeoff_summary(result_plateau)

    # Print numeric summary table
    print_numeric_summary(result_plateau, "plateau")

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

    # Create comparison plot (stable only)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_potential_comparison(results, ax=ax, use_constraints=False,
                             save_path='figures/xi_tradeoff_quadratic_vs_plateau.png')
    plt.close(fig)

    # Create comparison plot (with constraints)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_potential_comparison(results, ax=ax, use_constraints=True,
                             save_path='figures/xi_tradeoff_comparison_constrained.png')
    plt.close(fig)

    # ========================================================================
    # FINAL SUMMARY WITH CONSTRAINTS
    # ========================================================================
    print_final_summary(results)

    print("\n" + "=" * 72)
    print("FIGURES SAVED:")
    print("  - figures/xi_tradeoff_max_deltaG_with_constraints.png")
    print("  - figures/xi_tradeoff_constraint_breakdown.png")
    print("  - figures/xi_tradeoff_quadratic_summary.png")
    print("  - figures/xi_tradeoff_plateau_summary.png")
    print("  - figures/xi_tradeoff_quadratic_vs_plateau.png")
    print("  - figures/xi_tradeoff_comparison_constrained.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
