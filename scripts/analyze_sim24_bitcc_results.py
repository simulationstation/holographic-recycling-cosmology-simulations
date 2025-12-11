#!/usr/bin/env python3
"""
SIMULATION 24: BITCC Results Analysis

This script analyzes the results from run_sim24_bitcc_prior_and_data.py
and produces publication-quality plots.

Outputs:
- H0_prior_vs_post.png: Histogram of prior vs posterior-like H0 distribution
- chi_trans_vs_H0.png: Scatter plot of χ_trans vs H0
- BH_mass_vs_H0.png: Scatter plot of BH mass vs H0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load BITCC results from files."""
    results = {}

    # Load summary
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results["summary"] = json.load(f)

    # Load arrays
    npz_file = results_dir / "samples.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        results["arrays"] = {k: data[k] for k in data.files}

    return results


def print_text_summary(results: Dict[str, Any]) -> None:
    """Print text summary to stdout."""
    print("\n" + "=" * 70)
    print("SIMULATION 24: BITCC - TEXT SUMMARY")
    print("=" * 70)

    if "summary" not in results:
        print("No summary data found")
        return

    summary = results["summary"]

    print(f"\nSamples:")
    print(f"  Total:      {summary['n_samples']}")
    print(f"  Compatible: {summary['n_compatible']} ({100*summary['fraction_compatible']:.1f}%)")

    print(f"\nPrior H0 distribution:")
    prior = summary["H0_prior"]
    print(f"  Mean:       {prior['mean']:.2f} km/s/Mpc")
    print(f"  Std:        {prior['std']:.2f} km/s/Mpc")
    print(f"  68% CI:     [{prior['q16']:.2f}, {prior['q84']:.2f}]")
    print(f"  P(H0>=73):  {100*prior['P_H0_ge_73']:.2f}%")
    print(f"  P(H0>=70):  {100*prior['P_H0_ge_70']:.2f}%")
    print(f"  P(H0<=65):  {100*prior['P_H0_le_65']:.2f}%")

    print(f"\nPosterior-like H0 distribution:")
    post = summary["H0_post"]
    if post.get("mean") is not None:
        print(f"  Mean:       {post['mean']:.2f} km/s/Mpc")
        print(f"  Std:        {post['std']:.2f} km/s/Mpc")
        if "q16" in post and post["q16"] is not None:
            print(f"  68% CI:     [{post['q16']:.2f}, {post['q84']:.2f}]")
        print(f"  P(H0>=73):  {100*post['P_H0_ge_73']:.2f}%")
        print(f"  P(H0>=70):  {100*post['P_H0_ge_70']:.2f}%")
    else:
        print("  No compatible samples")

    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)

    prior_p73 = prior["P_H0_ge_73"]
    post_p73 = post.get("P_H0_ge_73", 0.0)

    if post_p73 is None:
        post_p73 = 0.0

    print(f"\n  Prior P(H0 >= 73) =     {100*prior_p73:.2f}%")
    print(f"  Posterior P(H0 >= 73) = {100*post_p73:.2f}%")

    if prior_p73 > 0.1:
        print("\n  The BITCC prior gives SIGNIFICANT weight to H0 >= 73.")
    elif prior_p73 > 0.01:
        print("\n  The BITCC prior gives MODEST weight to H0 >= 73.")
    else:
        print("\n  The BITCC prior gives NEGLIGIBLE weight to H0 >= 73.")

    if post_p73 > 0.01:
        print(f"  After data cuts, some support remains.")
    else:
        print(f"  After data cuts, negligible support for high H0.")

    # Does BITCC favor high or low H0?
    prior_mean = prior["mean"]
    if prior_mean > 68:
        print(f"\n  BITCC prior PUSHES H0 UP (mean = {prior_mean:.2f})")
    elif prior_mean < 67:
        print(f"\n  BITCC prior PUSHES H0 DOWN (mean = {prior_mean:.2f})")
    else:
        print(f"\n  BITCC prior is NEUTRAL (mean = {prior_mean:.2f})")

    print("\n" + "=" * 70)


def plot_H0_prior_vs_post(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot H0 prior vs posterior distribution."""
    if not HAS_MATPLOTLIB:
        return

    if "arrays" not in results:
        print("No array data for plotting")
        return

    arrays = results["arrays"]
    H0_prior = arrays.get("H0_prior", np.array([]))
    H0_post = arrays.get("H0_post", np.array([]))

    if len(H0_prior) == 0:
        print("No H0_prior data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot prior
    ax.hist(H0_prior, bins=50, density=True, alpha=0.6, color="steelblue",
            edgecolor="white", label=f"Prior (n={len(H0_prior)})")

    # Plot posterior if available
    if len(H0_post) > 0:
        ax.hist(H0_post, bins=30, density=True, alpha=0.6, color="coral",
                edgecolor="white", label=f"Posterior-like (n={len(H0_post)})")

    # Reference lines
    ax.axvline(67.5, color="green", linestyle="--", lw=2, label="Planck (67.5)")
    ax.axvline(73.0, color="red", linestyle="--", lw=2, label="SH0ES (73.0)")

    ax.set_xlabel("H0 [km/s/Mpc]", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title("SIM 24: BITCC Prior vs Posterior-like H0 Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(50, 80)

    plt.tight_layout()
    output_file = output_dir / "H0_prior_vs_post.png"
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.close()


def plot_chi_trans_vs_H0(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot chi_trans vs H0."""
    if not HAS_MATPLOTLIB:
        return

    if "arrays" not in results:
        return

    arrays = results["arrays"]
    H0_prior = arrays.get("H0_prior", np.array([]))
    chi_trans = arrays.get("chi_trans", np.array([]))

    if len(H0_prior) == 0 or len(chi_trans) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with alpha for density visualization
    ax.scatter(chi_trans, H0_prior, alpha=0.1, s=5, c="steelblue")

    # Reference lines
    ax.axhline(67.5, color="green", linestyle="--", lw=1.5, alpha=0.7, label="Planck (67.5)")
    ax.axhline(73.0, color="red", linestyle="--", lw=1.5, alpha=0.7, label="SH0ES (73.0)")

    ax.set_xlabel("χ_trans (computational residue)", fontsize=12)
    ax.set_ylabel("H0 [km/s/Mpc]", fontsize=12)
    ax.set_title("SIM 24: BITCC χ_trans vs H0", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / "chi_trans_vs_H0.png"
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.close()


def plot_BH_mass_vs_H0(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot BH mass vs H0."""
    if not HAS_MATPLOTLIB:
        return

    if "arrays" not in results:
        return

    arrays = results["arrays"]
    H0_prior = arrays.get("H0_prior", np.array([]))
    m_bh = arrays.get("m_bh", np.array([]))

    if len(H0_prior) == 0 or len(m_bh) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter with log x-axis
    ax.scatter(m_bh, H0_prior, alpha=0.1, s=5, c="steelblue")

    ax.set_xscale("log")

    # Reference lines
    ax.axhline(67.5, color="green", linestyle="--", lw=1.5, alpha=0.7, label="Planck (67.5)")
    ax.axhline(73.0, color="red", linestyle="--", lw=1.5, alpha=0.7, label="SH0ES (73.0)")

    ax.set_xlabel("BH mass (M_BH / 10⁶ M_☉)", fontsize=12)
    ax.set_ylabel("H0 [km/s/Mpc]", fontsize=12)
    ax.set_title("SIM 24: BITCC BH Mass vs H0", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / "BH_mass_vs_H0.png"
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.close()


def plot_interior_params_corner(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot corner-like visualization of interior parameters vs H0."""
    if not HAS_MATPLOTLIB:
        return

    if "arrays" not in results:
        return

    arrays = results["arrays"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    H0 = arrays.get("H0_prior", np.array([]))
    N_q = arrays.get("N_q", np.array([]))
    N_n = arrays.get("N_n", np.array([]))
    chi_trans = arrays.get("chi_trans", np.array([]))

    if len(H0) == 0:
        return

    # N_q vs H0
    ax = axes[0, 0]
    ax.scatter(N_q, H0, alpha=0.05, s=3, c="steelblue")
    ax.set_xlabel("N_q (quiet e-folds)")
    ax.set_ylabel("H0 [km/s/Mpc]")
    ax.axhline(67.5, color="green", linestyle="--", alpha=0.5)
    ax.axhline(73.0, color="red", linestyle="--", alpha=0.5)

    # N_n vs H0
    ax = axes[0, 1]
    ax.scatter(N_n, H0, alpha=0.05, s=3, c="steelblue")
    ax.set_xlabel("N_n (noise e-folds)")
    ax.set_ylabel("H0 [km/s/Mpc]")
    ax.axhline(67.5, color="green", linestyle="--", alpha=0.5)
    ax.axhline(73.0, color="red", linestyle="--", alpha=0.5)

    # N_q vs N_n colored by H0
    ax = axes[1, 0]
    sc = ax.scatter(N_q, N_n, c=H0, alpha=0.3, s=3, cmap="coolwarm", vmin=60, vmax=75)
    ax.set_xlabel("N_q (quiet e-folds)")
    ax.set_ylabel("N_n (noise e-folds)")
    plt.colorbar(sc, ax=ax, label="H0 [km/s/Mpc]")

    # chi_trans histogram
    ax = axes[1, 1]
    ax.hist(chi_trans, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("χ_trans")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of χ_trans")

    fig.suptitle("SIM 24: BITCC Interior Parameters", fontsize=14)
    plt.tight_layout()

    output_file = output_dir / "interior_params.png"
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze SIM 24 BITCC results"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/simulation_24_bitcc",
        help="Results directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: figures/simulation_24_bitcc/)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("figures/simulation_24_bitcc")

    print("\n" + "=" * 70)
    print("SIMULATION 24: BITCC ANALYSIS")
    print("=" * 70)

    # Load results
    print(f"\nLoading results from {results_dir}...")
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Print text summary
    print_text_summary(results)

    # Create plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots in {output_dir}...")

        plot_H0_prior_vs_post(results, output_dir)
        plot_chi_trans_vs_H0(results, output_dir)
        plot_BH_mass_vs_H0(results, output_dir)
        plot_interior_params_corner(results, output_dir)

        print(f"\nPlots saved to {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
