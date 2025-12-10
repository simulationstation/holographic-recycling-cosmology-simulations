#!/usr/bin/env python3
"""Run HRC 2.0 exploration: scan coupling families and evaluate Hubble tension potential.

This script:
1. Scans three coupling families (linear, quadratic, exponential)
2. For each family, explores (xi, phi0) parameter space
3. Checks dynamical stability and observational constraints
4. Computes max |Delta G/G| achievable under each scenario
5. Compares across families to find if any can exceed HRC 1.x limits

Key question: Can advanced coupling families break the ~3.5 km/s/Mpc ceiling
found for linear coupling, without violating stability or constraints?

Usage:
    python scripts/run_hrc2_exploration.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hrc2.theory import CouplingFamily, PotentialType
from hrc2.analysis import (
    scan_xi_tradeoff_hrc2,
    find_critical_xi_hrc2,
    compare_coupling_families,
    print_xi_tradeoff_summary_hrc2,
)
from hrc2.analysis.xi_tradeoff import print_comparison_summary
from hrc2.plots import (
    plot_xi_tradeoff_hrc2,
    plot_coupling_comparison,
    create_hrc2_summary_figure,
)
from hrc2.constraints.observational import estimate_delta_H0


def main():
    print("=" * 80)
    print("HRC 2.0: GENERAL SCALAR-TENSOR EXPLORATION")
    print("=" * 80)
    print()
    print("Scanning coupling families: LINEAR, QUADRATIC, EXPONENTIAL")
    print("Goal: Find if any coupling can exceed HRC 1.x (~3.5 km/s/Mpc) ceiling")
    print()

    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Parameter grids
    xi_values = np.logspace(-4, 0, 12)  # 1e-4 to 1
    phi0_values = np.linspace(0.01, 0.5, 12)
    z_max = 1100.0
    constraint_level = "conservative"

    print(f"Parameter grid:")
    print(f"  xi: {len(xi_values)} points from {xi_values.min():.1e} to {xi_values.max():.1e}")
    print(f"  phi0: {len(phi0_values)} points from {phi0_values.min():.3f} to {phi0_values.max():.3f}")
    print(f"  Total per family: {len(xi_values) * len(phi0_values)} integrations")
    print(f"  Constraint level: {constraint_level}")
    print()

    # =========================================================================
    # Scan all coupling families
    # =========================================================================
    coupling_families = [
        CouplingFamily.LINEAR,
        CouplingFamily.QUADRATIC,
        CouplingFamily.EXPONENTIAL,
    ]

    results = compare_coupling_families(
        coupling_families=coupling_families,
        potential_type=PotentialType.QUADRATIC,
        xi_values=xi_values,
        phi0_values=phi0_values,
        z_max=z_max,
        constraint_level=constraint_level,
        verbose=True,
    )

    # =========================================================================
    # Print detailed summaries for each family
    # =========================================================================
    for family, result in results.items():
        print_xi_tradeoff_summary_hrc2(result)

    # =========================================================================
    # Generate individual plots for each family
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    for family, result in results.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_xi_tradeoff_hrc2(result, ax=ax,
                              save_path=f'figures/hrc2_xi_tradeoff_{family.value}.png')
        plt.close(fig)

    # =========================================================================
    # Comparison plots
    # =========================================================================

    # Constrained comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_coupling_comparison(results, ax=ax, use_constraints=True,
                            save_path='figures/hrc2_xi_tradeoff_comparison.png')
    plt.close(fig)

    # Stable-only comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_coupling_comparison(results, ax=ax, use_constraints=False,
                            save_path='figures/hrc2_xi_tradeoff_comparison_stable.png')
    plt.close(fig)

    # Summary figure
    fig = create_hrc2_summary_figure(results,
                                     save_path='figures/hrc2_summary.png')
    plt.close(fig)

    # =========================================================================
    # Save results to npz files
    # =========================================================================
    for family, result in results.items():
        np.savez(
            f'results/hrc2_xi_tradeoff_{family.value}.npz',
            coupling_family=family.value,
            potential_type=result.potential_type.value,
            xi_values=result.xi_values,
            phi0_values=result.phi0_values,
            stable_mask=result.stable_mask,
            obs_allowed_mask=result.obs_allowed_mask,
            delta_G_over_G=result.delta_G_over_G,
            stable_fraction=result.stable_fraction,
            obs_allowed_fraction=result.obs_allowed_fraction,
            max_delta_G_stable=result.max_delta_G_stable,
            max_delta_G_allowed=result.max_delta_G_allowed,
            z_max=result.z_max,
            constraint_level=result.constraint_level,
        )
        print(f"Saved results/hrc2_xi_tradeoff_{family.value}.npz")

    # =========================================================================
    # Print comparison summary
    # =========================================================================
    print_comparison_summary(results)

    # =========================================================================
    # Final "new dork" evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION: CAN HRC 2.0 BREAK THE ~3.5 km/s/Mpc CEILING?")
    print("=" * 80)

    hrc1_ceiling = 3.5  # km/s/Mpc from HRC 1.x analysis

    best_family = None
    best_dH0 = 0.0
    best_max_dg = 0.0

    for family, result in results.items():
        _, max_dg = find_critical_xi_hrc2(result, use_constraints=True)
        dH0 = estimate_delta_H0(max_dg) if max_dg > 0 else 0.0

        if dH0 > best_dH0:
            best_dH0 = dH0
            best_family = family
            best_max_dg = max_dg

    print()
    print(f"HRC 1.x (linear coupling) ceiling: ~{hrc1_ceiling:.1f} km/s/Mpc")
    print()

    if best_family is not None:
        print(f"HRC 2.0 best result:")
        print(f"  Coupling family: {best_family.value.upper()}")
        print(f"  Max |ΔG/G| (constrained): {best_max_dg:.4f}")
        print(f"  Estimated ΔH₀: {best_dH0:.1f} km/s/Mpc")
        print()

        improvement = best_dH0 - hrc1_ceiling
        if improvement > 0.5:
            print(f"  IMPROVEMENT: +{improvement:.1f} km/s/Mpc over HRC 1.x")
        elif improvement > -0.5:
            print("  NO SIGNIFICANT IMPROVEMENT over HRC 1.x")
        else:
            print(f"  REGRESSION: {improvement:.1f} km/s/Mpc compared to HRC 1.x")

        print()
        if best_dH0 >= 5.0:
            print("  VERDICT: HRC 2.0 CAN potentially resolve the Hubble tension!")
        elif best_dH0 >= 3.5:
            print("  VERDICT: HRC 2.0 matches but doesn't exceed HRC 1.x limits.")
            print("           The stability-effect tradeoff persists across coupling families.")
        else:
            print("  VERDICT: HRC 2.0 CANNOT resolve the Hubble tension.")
            print("           More fundamental modifications may be needed.")
    else:
        print("No valid constrained solutions found for any coupling family!")
        print("The stability requirements may be too stringent.")

    print()
    print("=" * 80)
    print("FIGURES SAVED:")
    print("  - figures/hrc2_xi_tradeoff_linear.png")
    print("  - figures/hrc2_xi_tradeoff_quadratic.png")
    print("  - figures/hrc2_xi_tradeoff_exponential.png")
    print("  - figures/hrc2_xi_tradeoff_comparison.png")
    print("  - figures/hrc2_xi_tradeoff_comparison_stable.png")
    print("  - figures/hrc2_summary.png")
    print()
    print("RESULTS SAVED:")
    print("  - results/hrc2_xi_tradeoff_linear.npz")
    print("  - results/hrc2_xi_tradeoff_quadratic.npz")
    print("  - results/hrc2_xi_tradeoff_exponential.npz")
    print("=" * 80)


if __name__ == "__main__":
    main()
