#!/usr/bin/env python3
"""
SIMULATION 24: Layered Expansion (Bent-Deck) Cosmology - Analysis

This script analyzes results from both the grid scan (SIM 24A) and
MCMC (SIM 24B) to provide a comprehensive verdict on whether smooth
layered expansion histories can reconcile the Hubble tension.

Outputs:
- Summary statistics and probabilities
- Plots: H0 vs chi2, posterior distributions, example "bent deck" profiles
- Final verdict on the viability of layered expansion solutions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
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


def load_grid_results(grid_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load grid scan results."""
    summary_file = grid_dir / "summary.json"
    npz_file = grid_dir / "scan_results.npz"
    json_file = grid_dir / "scan_results.json"

    summary = {}
    arrays = {}

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

    if npz_file.exists():
        data = np.load(npz_file)
        arrays = {k: data[k] for k in data.files}

    if json_file.exists():
        with open(json_file) as f:
            results_list = json.load(f)
        summary["results_list"] = results_list

    return summary, arrays


def load_mcmc_results(mcmc_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load MCMC results."""
    summary_file = mcmc_dir / "summary.json"
    chains_file = mcmc_dir / "chains.npz"

    summary = {}
    arrays = {}

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

    if chains_file.exists():
        data = np.load(chains_file)
        arrays = {k: data[k] for k in data.files}

    return summary, arrays


def print_grid_summary(summary: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """Print grid scan summary."""
    print("\n" + "=" * 70)
    print("GRID SCAN (SIM 24A) RESULTS")
    print("=" * 70)

    if not summary:
        print("  No grid scan results found")
        return

    print(f"\nTotal samples:          {summary.get('total_samples', 'N/A')}")
    print(f"Physical models:        {summary.get('physical_models', 'N/A')}")
    print(f"Passing models (chi2):  {summary.get('passing_models', 'N/A')}")

    print(f"\nModels with H0 >= 73:   {summary.get('models_with_H0_ge_73', 'N/A')}")
    print(f"Passing + H0 >= 73:     {summary.get('passing_with_H0_ge_73', 'N/A')}")

    if summary.get("H0_distribution_passing"):
        h0_pass = summary["H0_distribution_passing"]
        if h0_pass.get("max") is not None:
            print(f"\nMax H0 among passing:   {summary.get('max_H0_among_passing', 'N/A'):.2f} km/s/Mpc")

    if summary.get("best_high_h0_model"):
        m = summary["best_high_h0_model"]
        print(f"\nBest model with H0 >= 73:")
        print(f"  H0_eff = {m['H0_eff']:.2f}, delta_chi2 = {m['delta_chi2']:.2f}")


def print_mcmc_summary(summary: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """Print MCMC summary."""
    print("\n" + "=" * 70)
    print("MCMC (SIM 24B) RESULTS")
    print("=" * 70)

    if not summary or "result" not in summary:
        print("  No MCMC results found")
        return

    result = summary["result"]

    print(f"\nH0_eff posterior:")
    print(f"  Mean:   {result['H0_eff_mean']:.2f} +/- {result['H0_eff_std']:.2f} km/s/Mpc")
    print(f"  Median: {result['H0_eff_median']:.2f} km/s/Mpc")
    print(f"  95% CI: [{result['H0_eff_q2p5']:.2f}, {result['H0_eff_q97p5']:.2f}] km/s/Mpc")

    print(f"\nProbabilities:")
    print(f"  P(H0 >= 70) = {100*result['prob_H0_ge_70']:.2f}%")
    print(f"  P(H0 >= 71) = {100*result['prob_H0_ge_71']:.2f}%")
    print(f"  P(H0 >= 73) = {100*result['prob_H0_ge_73']:.2f}%")

    print(f"\nBest-fit:")
    print(f"  H0_eff = {result['best_fit_H0_eff']:.2f} km/s/Mpc")
    print(f"  chi2 = {result['best_fit_chi2']:.1f}")

    print(f"\nConvergence:")
    print(f"  Acceptance: {result['acceptance_fraction']:.3f}")
    print(f"  N_eff: {result['n_effective_samples']}")


def plot_grid_results(
    summary: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """Create plots from grid scan results."""
    if not HAS_MATPLOTLIB:
        return

    if not arrays:
        print("No array data for plotting")
        return

    # H0 vs delta_chi2 scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    H0_eff = arrays.get("H0_eff", np.array([]))
    delta_chi2 = arrays.get("delta_chi2", np.array([]))
    is_physical = arrays.get("is_physical", np.ones(len(H0_eff), dtype=bool))

    if len(H0_eff) == 0:
        print("No H0_eff data for plotting")
        return

    # Physical models
    physical_mask = is_physical.astype(bool)
    ax.scatter(
        H0_eff[physical_mask], delta_chi2[physical_mask],
        c="steelblue", alpha=0.5, s=20, label="Physical models"
    )

    # Unphysical models (if any)
    if np.sum(~physical_mask) > 0:
        ax.scatter(
            H0_eff[~physical_mask], delta_chi2[~physical_mask],
            c="lightgray", alpha=0.3, s=10, label="Unphysical"
        )

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline LCDM")
    ax.axhline(10, color="orange", linestyle=":", alpha=0.7, label="delta_chi2 = 10")
    ax.axvline(73, color="red", linestyle="--", alpha=0.7, label="H0 = 73 (SH0ES)")
    ax.axvline(67.5, color="green", linestyle="--", alpha=0.7, label="H0 = 67.5 (Planck)")

    ax.set_xlabel("H0_eff [km/s/Mpc]", fontsize=12)
    ax.set_ylabel("Delta chi-squared from baseline", fontsize=12)
    ax.set_title("SIM 24A: Layered Expansion Grid Scan", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)

    # Set reasonable y limits
    valid_chi2 = delta_chi2[is_physical & np.isfinite(delta_chi2)]
    if len(valid_chi2) > 0:
        ymax = min(np.percentile(valid_chi2, 99), 500)
        ax.set_ylim(-10, ymax)

    plt.tight_layout()
    fig.savefig(output_dir / "grid_H0_vs_chi2.png", dpi=150)
    print(f"Saved {output_dir / 'grid_H0_vs_chi2.png'}")
    plt.close()


def plot_mcmc_results(
    summary: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """Create plots from MCMC results."""
    if not HAS_MATPLOTLIB:
        return

    H0_samples = arrays.get("H0_eff_samples", np.array([]))

    if len(H0_samples) == 0:
        print("No H0 samples for plotting")
        return

    # H0 posterior histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(H0_samples, bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", label="Posterior")

    # Add reference lines
    ax.axvline(67.5, color="green", linestyle="--", lw=2, label="Planck LCDM (67.5)")
    ax.axvline(73.04, color="red", linestyle="--", lw=2, label="SH0ES (73.04)")

    if summary and "result" in summary:
        result = summary["result"]
        ax.axvline(result["H0_eff_median"], color="blue", linestyle="-", lw=2,
                   label=f"Median ({result['H0_eff_median']:.2f})")

    ax.set_xlabel("H0_eff [km/s/Mpc]", fontsize=12)
    ax.set_ylabel("Posterior density", fontsize=12)
    ax.set_title("SIM 24B: H0 Posterior under Layered Expansion Model", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "mcmc_H0_posterior.png", dpi=150)
    print(f"Saved {output_dir / 'mcmc_H0_posterior.png'}")
    plt.close()


def plot_example_bent_deck(
    grid_summary: Dict[str, Any],
    mcmc_summary: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot example bent deck profiles."""
    if not HAS_MATPLOTLIB:
        return

    # Get best model from grid scan
    best_grid = None
    if grid_summary.get("best_high_h0_model"):
        best_grid = grid_summary["best_high_h0_model"]

    # Get best model from MCMC
    best_mcmc = None
    if mcmc_summary and "result" in mcmc_summary:
        result = mcmc_summary["result"]
        if result.get("best_fit_delta_nodes"):
            best_mcmc = {
                "delta_nodes": result["best_fit_delta_nodes"],
                "H0_eff": result["best_fit_H0_eff"],
            }

    # Need at least one model to plot
    if best_grid is None and best_mcmc is None:
        print("No model data for bent deck plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Default z nodes (log spacing from 0 to 6)
    from hrc2.layered import LayeredExpansionHyperparams, make_default_nodes
    hyp = LayeredExpansionHyperparams(n_layers=6)
    z_nodes_default = make_default_nodes(hyp)

    # Plot baseline (zero deviations)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Baseline LCDM")

    # Plot best grid model
    if best_grid and "delta_nodes" in best_grid:
        delta = np.array(best_grid["delta_nodes"])
        z_nodes = z_nodes_default[:len(delta)]
        ax.plot(z_nodes, delta, "o-", color="red", lw=2, markersize=8,
                label=f"Best grid (H0={best_grid['H0_eff']:.1f})")

    # Plot best MCMC model
    if best_mcmc and "delta_nodes" in best_mcmc:
        delta = np.array(best_mcmc["delta_nodes"])
        z_nodes = z_nodes_default[:len(delta)]
        ax.plot(z_nodes, delta, "s-", color="blue", lw=2, markersize=8,
                label=f"Best MCMC (H0={best_mcmc['H0_eff']:.1f})")

    ax.set_xlabel("Redshift z", fontsize=12)
    ax.set_ylabel("Delta (fractional H modification)", fontsize=12)
    ax.set_title("Example 'Bent Deck' Profiles: delta(z)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "bent_deck_examples.png", dpi=150)
    print(f"Saved {output_dir / 'bent_deck_examples.png'}")
    plt.close()


def print_final_verdict(
    grid_summary: Dict[str, Any],
    mcmc_summary: Dict[str, Any]
) -> None:
    """Print the final verdict."""
    print("\n" + "=" * 70)
    print("FINAL VERDICT: SIMULATION 24 - LAYERED EXPANSION COSMOLOGY")
    print("=" * 70)

    # Grid scan verdict
    grid_can_reach_73 = False
    if grid_summary:
        passing_h0_73 = grid_summary.get("passing_with_H0_ge_73", 0)
        grid_can_reach_73 = passing_h0_73 > 0

    # MCMC verdict
    mcmc_prob_h0_73 = 0.0
    if mcmc_summary and "result" in mcmc_summary:
        mcmc_prob_h0_73 = mcmc_summary["result"].get("prob_H0_ge_73", 0)

    print("\nKey Results:")
    print("-" * 40)

    if grid_summary:
        max_h0 = grid_summary.get("max_H0_among_passing")
        if max_h0 is not None:
            print(f"Grid scan: Max H0 among good-fit models = {max_h0:.2f} km/s/Mpc")
        print(f"Grid scan: Models with H0>=73 and acceptable chi2 = {grid_summary.get('passing_with_H0_ge_73', 0)}")

    if mcmc_summary and "result" in mcmc_summary:
        result = mcmc_summary["result"]
        print(f"MCMC: H0_eff = {result['H0_eff_mean']:.2f} +/- {result['H0_eff_std']:.2f} km/s/Mpc")
        print(f"MCMC: P(H0 >= 73 | data, layered model) = {100*mcmc_prob_h0_73:.2f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)

    if grid_can_reach_73:
        print("\n  The grid scan found some models that achieve H0 >= 73")
        print("  with acceptable chi-squared. However:")
    else:
        print("\n  The grid scan found NO models that achieve H0 >= 73")
        print("  with acceptable chi-squared.")

    if mcmc_prob_h0_73 > 0.05:
        print(f"\n  The MCMC posterior gives P(H0 >= 73) = {100*mcmc_prob_h0_73:.1f}%,")
        print("  suggesting some (but limited) posterior support for high H0.")
    elif mcmc_prob_h0_73 > 0.01:
        print(f"\n  The MCMC posterior gives P(H0 >= 73) = {100*mcmc_prob_h0_73:.2f}%,")
        print("  indicating marginal posterior support for high H0.")
    else:
        print(f"\n  The MCMC posterior gives P(H0 >= 73) = {100*mcmc_prob_h0_73:.3f}%,")
        print("  indicating NEGLIGIBLE posterior support for H0 = 73.")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    if not grid_can_reach_73 and mcmc_prob_h0_73 < 0.01:
        print("""
  Even with the generous freedom of the "bent deck" parameterization,
  allowing arbitrary smooth modifications to the expansion history H(z),
  the data (CMB + BAO + SN) strongly constrain the effective H0.

  The layered expansion model CANNOT reconcile the Hubble tension.

  This is a strong negative result: if a very flexible expansion-history
  reconstruction fails to reach H0 ~ 73 km/s/Mpc, then "late-time wiggles"
  in the expansion are not the solution to the Hubble tension.

  Possible implications:
  - The tension may require early-universe physics (before recombination)
  - Systematic errors in one or more datasets may be responsible
  - New physics affecting the distance ladder itself may be needed
""")
    else:
        print("""
  The layered expansion model shows some ability to shift H0 upward,
  but the constraints from CMB + BAO remain tight.

  Further investigation with more samples and/or different
  parameterizations may be warranted.
""")

    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze SIM 24 Layered Expansion results"
    )
    parser.add_argument(
        "--grid-dir", type=str, default="results/simulation_24_layered_grid",
        help="Grid scan results directory"
    )
    parser.add_argument(
        "--mcmc-dir", type=str, default="results/simulation_24_layered_mcmc",
        help="MCMC results directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: use grid-dir)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    mcmc_dir = Path(args.mcmc_dir)
    output_dir = Path(args.output_dir) if args.output_dir else grid_dir

    print("\n" + "=" * 70)
    print("SIMULATION 24: LAYERED EXPANSION ANALYSIS")
    print("=" * 70)

    # Load results
    print(f"\nLoading grid results from {grid_dir}...")
    grid_summary, grid_arrays = load_grid_results(grid_dir)

    print(f"Loading MCMC results from {mcmc_dir}...")
    mcmc_summary, mcmc_arrays = load_mcmc_results(mcmc_dir)

    # Print summaries
    print_grid_summary(grid_summary, grid_arrays)
    print_mcmc_summary(mcmc_summary, mcmc_arrays)

    # Create plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots in {output_dir}...")
        plot_grid_results(grid_summary, grid_arrays, output_dir)
        plot_mcmc_results(mcmc_summary, mcmc_arrays, output_dir)
        plot_example_bent_deck(grid_summary, mcmc_summary, output_dir)

    # Final verdict
    print_final_verdict(grid_summary, mcmc_summary)


if __name__ == "__main__":
    main()
