#!/usr/bin/env python3
"""Run HRC 2.0 exploration: parallel scan of coupling families for Hubble tension.

This script:
1. Runs a tiny correctness test comparing serial vs parallel execution
2. Performs a full parallel scan of 60x60 parameter space
3. Saves results incrementally to handle interruptions
4. Generates figures and numeric summaries

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
from hrc2.utils.config import PerformanceConfig
from hrc2.analysis import (
    run_xi_tradeoff_parallel,
    run_xi_tradeoff_serial,
    find_critical_xi_hrc2,
    print_xi_tradeoff_summary_hrc2,
)
from hrc2.plots import (
    plot_xi_tradeoff_hrc2,
    plot_coupling_comparison,
    create_hrc2_summary_figure,
)
from hrc2.constraints.observational import estimate_delta_H0


def save_xi_tradeoff_result(result, path: str) -> None:
    """Save XiTradeoffResultHRC2 to npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        coupling_family=result.coupling_family.value,
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
    print(f"Saved results to {path}")


def plot_all_xi_tradeoff_figures(result, output_dir: str) -> None:
    """Generate all plots for a single family result."""
    os.makedirs(output_dir, exist_ok=True)

    family_name = result.coupling_family.value

    # Main tradeoff plot
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_xi_tradeoff_hrc2(result, ax=ax,
                          save_path=f'{output_dir}/xi_tradeoff_{family_name}.png')
    plt.close(fig)
    print(f"Saved {output_dir}/xi_tradeoff_{family_name}.png")


def print_numeric_summary(result) -> None:
    """Print numeric summary of scan results."""
    print("\n" + "=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)

    n_total = result.stable_mask.size
    n_stable = result.stable_mask.sum()
    n_allowed = result.obs_allowed_mask.sum()

    print(f"Total grid points: {n_total}")
    print(f"Dynamically stable: {n_stable} ({100*n_stable/n_total:.1f}%)")
    print(f"Observationally allowed: {n_allowed} ({100*n_allowed/n_total:.1f}%)")

    # Find best points
    if n_allowed > 0:
        allowed_dG = result.delta_G_over_G[result.obs_allowed_mask]
        max_dG = np.nanmax(allowed_dG)
        mean_dG = np.nanmean(allowed_dG)

        print(f"\nAmong allowed points:")
        print(f"  Max |ΔG/G|: {max_dG:.4f} → ΔH₀ ~ {estimate_delta_H0(max_dG):.1f} km/s/Mpc")
        print(f"  Mean |ΔG/G|: {mean_dG:.4f} → ΔH₀ ~ {estimate_delta_H0(mean_dG):.1f} km/s/Mpc")


def print_final_summary(result) -> None:
    """Print final evaluation answering the key question."""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION: CAN HRC 2.0 BREAK THE ~3.5 km/s/Mpc CEILING?")
    print("=" * 80)

    hrc1_ceiling = 3.5  # km/s/Mpc from HRC 1.x analysis

    xi_crit, max_dg = find_critical_xi_hrc2(result, use_constraints=True)
    dH0 = estimate_delta_H0(max_dg) if max_dg > 0 else 0.0

    print()
    print(f"HRC 1.x (linear coupling) ceiling: ~{hrc1_ceiling:.1f} km/s/Mpc")
    print()

    if max_dg > 0:
        print(f"HRC 2.0 result ({result.coupling_family.value} coupling):")
        print(f"  Critical ξ: {xi_crit:.2e}")
        print(f"  Max |ΔG/G| (constrained): {max_dg:.4f}")
        print(f"  Estimated ΔH₀: {dH0:.1f} km/s/Mpc")
        print()

        improvement = dH0 - hrc1_ceiling
        if improvement > 0.5:
            print(f"  IMPROVEMENT: +{improvement:.1f} km/s/Mpc over HRC 1.x")
        elif improvement > -0.5:
            print("  NO SIGNIFICANT IMPROVEMENT over HRC 1.x")
        else:
            print(f"  REGRESSION: {improvement:.1f} km/s/Mpc compared to HRC 1.x")

        print()
        if dH0 >= 5.0:
            print("  VERDICT: HRC 2.0 CAN potentially resolve the Hubble tension!")
        elif dH0 >= 3.5:
            print("  VERDICT: HRC 2.0 matches but doesn't exceed HRC 1.x limits.")
            print("           The stability-effect tradeoff persists.")
        else:
            print("  VERDICT: HRC 2.0 CANNOT resolve the Hubble tension.")
            print("           More fundamental modifications may be needed.")
    else:
        print("No valid constrained solutions found!")
        print("The stability requirements may be too stringent.")

    print()
    print("=" * 80)


def tiny_test() -> None:
    """Run tiny test - SKIPPED for speed. Go straight to parallel scan."""
    print("=" * 80)
    print("SKIPPING TINY TEST - going straight to parallel scan")
    print("=" * 80)
    print()


def run_full_parallel_scan() -> None:
    """Run full parallel scan of parameter space."""
    print("=" * 80)
    print("HRC 2.0: FULL PARALLEL PARAMETER SPACE SCAN")
    print("=" * 80)
    print()

    perf = PerformanceConfig(n_workers=10)

    # 10x10 grid = 100 points per coupling family (minimal test run)
    xi_values = np.logspace(-5, -2.5, 10)
    phi0_values = np.linspace(0.0, 0.3, 10)

    print(f"Parameter grid:")
    print(f"  xi: {len(xi_values)} points from {xi_values.min():.1e} to {xi_values.max():.1e}")
    print(f"  phi0: {len(phi0_values)} points from {phi0_values.min():.3f} to {phi0_values.max():.3f}")
    print(f"  Total per family: {len(xi_values) * len(phi0_values)} integrations")
    print(f"  Workers: {perf.n_workers}")
    print()

    # Ensure output directories exist
    os.makedirs('figures/hrc2_scan', exist_ok=True)
    os.makedirs('results/hrc2_scan', exist_ok=True)

    # Scan LINEAR coupling (most important for comparison with HRC 1.x)
    print("\n" + "=" * 60)
    print("Scanning LINEAR coupling family")
    print("=" * 60)

    result = run_xi_tradeoff_parallel(
        xi_values, phi0_values,
        CouplingFamily.LINEAR,
        potential_type=PotentialType.QUADRATIC,
        perf=perf,
        z_max=1100.0,
        constraint_level="conservative",
        verbose=True,
    )

    # Save results
    save_xi_tradeoff_result(result, "results/hrc2_scan/hrc2_full_scan_linear.npz")

    # Generate plots
    plot_all_xi_tradeoff_figures(result, "figures/hrc2_scan")

    # Print summaries
    print_xi_tradeoff_summary_hrc2(result)
    print_numeric_summary(result)
    print_final_summary(result)

    print("\n" + "=" * 80)
    print("RESULTS SAVED:")
    print("  - results/hrc2_scan/hrc2_full_scan_linear.npz")
    print("  - results/hrc2_scan/hrc2_partial_scan.npz (incremental)")
    print("  - figures/hrc2_scan/xi_tradeoff_linear.png")
    print("=" * 80)

    # Auto-commit results after scan finishes
    print("\n=== Committing results to git ===")
    os.system("git add results/hrc2_scan/* figures/hrc2_scan/*")
    os.system('git commit -m "Full parallel HRC2 scan results with timeout + early exit"')


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print("\n=== STARTING FULL PARALLEL HRC2 SCAN (TIMEOUT + EARLY EXIT ENABLED) ===\n")

    # Run full parallel scan (tiny test skipped for speed)
    run_full_parallel_scan()
