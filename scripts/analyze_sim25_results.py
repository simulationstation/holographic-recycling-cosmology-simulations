#!/usr/bin/env python3
"""
SIMULATION 25: Analysis and Visualization Script

This script loads the outputs of SIM 25A (forward scan) and SIM 25B (inverse fit)
and produces diagnostic plots and a human-readable summary.

Plots generated:
- 2D slices of (A1, A2) colored by H0_eff
- 2D slices of (A1, A2) colored by max BAO deviation
- 3D scatter of allowed points colored by H0_eff
- H0 distribution histogram for allowed configurations

Usage:
    python analyze_sim25_results.py [--scan-dir DIR] [--fit-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scan_results(scan_dir: str) -> Optional[Dict]:
    """Load results from SIM 25A scan."""
    npz_path = os.path.join(scan_dir, "scan_results.npz")
    json_path = os.path.join(scan_dir, "scan_summary.json")

    if not os.path.exists(npz_path):
        print(f"Warning: Scan results not found at {npz_path}")
        return None

    data = dict(np.load(npz_path, allow_pickle=True))

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data["summary"] = json.load(f)

    return data


def load_fit_results(fit_dir: str) -> Optional[Dict]:
    """Load results from SIM 25B inverse fit."""
    npz_path = os.path.join(fit_dir, "points.npz")
    json_path = os.path.join(fit_dir, "summary.json")

    if not os.path.exists(npz_path):
        print(f"Warning: Fit results not found at {npz_path}")
        return None

    data = dict(np.load(npz_path, allow_pickle=True))

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data["summary"] = json.load(f)

    return data


def create_plots(
    scan_data: Optional[Dict],
    fit_data: Optional[Dict],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """
    Create all diagnostic plots.

    Parameters
    ----------
    scan_data : dict or None
        Data from SIM 25A scan
    fit_data : dict or None
        Data from SIM 25B inverse fit
    output_dir : str
        Directory to save plots
    verbose : bool
        Print progress
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Use data from whichever source is available
    data = scan_data if scan_data is not None else fit_data

    if data is None:
        print("No data available for plotting")
        return

    # Get the key arrays
    A1 = data.get("A1", np.array([]))
    A2 = data.get("A2", np.array([]))
    A3 = data.get("A3", np.array([]))
    H0_eff = data.get("H0_eff", np.array([]))
    passes = data.get("passes_all", data.get("pass_flags", np.array([])))
    max_bao_dev = data.get("max_bao_dev_percent", data.get("max_bao_dev", np.array([])))

    if len(A1) == 0:
        print("No data points to plot")
        return

    # Convert to numpy if needed
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)
    A3 = np.asarray(A3)
    H0_eff = np.asarray(H0_eff)
    passes = np.asarray(passes).astype(bool)
    max_bao_dev = np.asarray(max_bao_dev)

    # Handle NaN/inf values
    valid = np.isfinite(H0_eff) & np.isfinite(max_bao_dev)

    # =========================================================================
    # Plot 1: 2D slice (A1, A2) at A3=0, colored by H0_eff
    # =========================================================================
    if verbose:
        print("Creating 2D slice plots...")

    # Find points near A3 = 0
    A3_unique = np.unique(A3)
    A3_center = A3_unique[len(A3_unique)//2] if len(A3_unique) > 0 else 0
    A3_tol = 0.001 if len(A3_unique) == 1 else np.abs(A3_unique[1] - A3_unique[0]) / 2 + 0.001

    slice_mask = np.abs(A3 - A3_center) < A3_tol
    if np.sum(slice_mask) < 3:
        # Try with larger tolerance
        A3_tol = 0.02
        slice_mask = np.abs(A3 - A3_center) < A3_tol

    if np.sum(slice_mask) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))

        A1_slice = A1[slice_mask & valid]
        A2_slice = A2[slice_mask & valid]
        H0_slice = H0_eff[slice_mask & valid]
        passes_slice = passes[slice_mask & valid]

        # Plot all points
        sc = ax.scatter(
            A1_slice, A2_slice, c=H0_slice, cmap='viridis',
            s=20, alpha=0.7, edgecolors='none'
        )

        # Highlight allowed points
        if np.sum(passes_slice) > 0:
            ax.scatter(
                A1_slice[passes_slice], A2_slice[passes_slice],
                c='none', edgecolors='red', s=50, linewidths=1.5,
                label='Allowed'
            )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('H0_eff [km/s/Mpc]')

        ax.set_xlabel('A1 (z~3000 mode amplitude)')
        ax.set_ylabel('A2 (z~100 mode amplitude)')
        ax.set_title(f'SIM 25: Mode Amplitude Space (A3 ≈ {A3_center:.3f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        if np.sum(passes_slice) > 0:
            ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "A1_A2_H0_slice.png"), dpi=150)
        plt.close()

    # =========================================================================
    # Plot 2: 2D slice colored by max BAO deviation
    # =========================================================================
    if np.sum(slice_mask) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))

        A1_slice = A1[slice_mask & valid]
        A2_slice = A2[slice_mask & valid]
        bao_slice = max_bao_dev[slice_mask & valid]
        passes_slice = passes[slice_mask & valid]

        # Clip extreme values for visualization
        bao_clipped = np.clip(bao_slice, 0, 10)

        sc = ax.scatter(
            A1_slice, A2_slice, c=bao_clipped, cmap='Reds',
            s=20, alpha=0.7, edgecolors='none'
        )

        # Highlight allowed points
        if np.sum(passes_slice) > 0:
            ax.scatter(
                A1_slice[passes_slice], A2_slice[passes_slice],
                c='none', edgecolors='blue', s=50, linewidths=1.5,
                label='Allowed'
            )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Max BAO deviation [%]')

        ax.set_xlabel('A1 (z~3000 mode amplitude)')
        ax.set_ylabel('A2 (z~100 mode amplitude)')
        ax.set_title(f'SIM 25: BAO Constraint (A3 ≈ {A3_center:.3f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        if np.sum(passes_slice) > 0:
            ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "A1_A2_BAO_slice.png"), dpi=150)
        plt.close()

    # =========================================================================
    # Plot 3: 3D scatter of allowed points
    # =========================================================================
    if verbose:
        print("Creating 3D scatter plot...")

    allowed_mask = passes & valid

    if np.sum(allowed_mask) >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        A1_allowed = A1[allowed_mask]
        A2_allowed = A2[allowed_mask]
        A3_allowed = A3[allowed_mask]
        H0_allowed = H0_eff[allowed_mask]

        sc = ax.scatter(
            A1_allowed, A2_allowed, A3_allowed,
            c=H0_allowed, cmap='viridis', s=30, alpha=0.8
        )

        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('H0_eff [km/s/Mpc]')

        ax.set_xlabel('A1 (z~3000)')
        ax.set_ylabel('A2 (z~100)')
        ax.set_zlabel('A3 (z~1)')
        ax.set_title('SIM 25: Allowed Mode Configurations')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "allowed_3D_scatter.png"), dpi=150)
        plt.close()
    else:
        if verbose:
            print("  Not enough allowed points for 3D plot")

    # =========================================================================
    # Plot 4: H0 distribution histogram
    # =========================================================================
    if verbose:
        print("Creating H0 distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    # All valid points
    H0_valid = H0_eff[valid]
    ax.hist(H0_valid, bins=50, alpha=0.5, label='All configurations', color='blue')

    # Allowed points
    if np.sum(allowed_mask) > 0:
        H0_allowed = H0_eff[allowed_mask]
        ax.hist(H0_allowed, bins=30, alpha=0.7, label='Allowed configurations', color='green')

    # Reference lines
    ax.axvline(67.5, color='green', linestyle='--', linewidth=2, label='Planck (67.5)')
    ax.axvline(73.0, color='red', linestyle='--', linewidth=2, label='SH0ES (73.0)')

    ax.set_xlabel('H0_eff [km/s/Mpc]')
    ax.set_ylabel('Count')
    ax.set_title('SIM 25: Distribution of Effective H0')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "H0_distribution.png"), dpi=150)
    plt.close()

    # =========================================================================
    # Plot 5: Chi-squared landscape
    # =========================================================================
    if "chi2_total" in data:
        if verbose:
            print("Creating chi-squared landscape plot...")

        chi2 = np.asarray(data["chi2_total"])
        chi2_valid = chi2[valid]
        chi2_clipped = np.clip(chi2_valid, 0, 100)  # Clip extreme values

        if np.sum(slice_mask & valid) >= 3:
            fig, ax = plt.subplots(figsize=(8, 6))

            A1_slice = A1[slice_mask & valid]
            A2_slice = A2[slice_mask & valid]
            chi2_slice = np.clip(chi2[slice_mask & valid], 0, 100)

            sc = ax.scatter(
                A1_slice, A2_slice, c=chi2_slice, cmap='hot_r',
                s=20, alpha=0.7, edgecolors='none',
                norm=Normalize(vmin=0, vmax=50)
            )

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('χ² (capped at 50)')

            ax.set_xlabel('A1 (z~3000 mode amplitude)')
            ax.set_ylabel('A2 (z~100 mode amplitude)')
            ax.set_title(f'SIM 25: Chi-squared Landscape (A3 ≈ {A3_center:.3f})')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "chi2_landscape.png"), dpi=150)
            plt.close()

    if verbose:
        print(f"Plots saved to {output_dir}/")


def print_summary(
    scan_data: Optional[Dict],
    fit_data: Optional[Dict],
    verbose: bool = True,
) -> None:
    """
    Print human-readable summary of results.
    """
    print("\n" + "="*70)
    print("SIMULATION 25: MULTI-MODE TERMINAL SPECTRUM ANALYSIS")
    print("="*70)

    # Use whichever data is available
    for name, data in [("Forward Scan (25A)", scan_data), ("Inverse Fit (25B)", fit_data)]:
        if data is None:
            continue

        print(f"\n--- {name} Results ---")

        if "summary" in data:
            summary = data["summary"]
            if isinstance(summary, dict):
                if "summary" in summary:
                    s = summary["summary"]
                else:
                    s = summary

                n_allowed = s.get("n_allowed", 0)
                n_total = s.get("n_total", s.get("n_total_samples", 0))

                print(f"  Total configurations: {n_total}")
                print(f"  Allowed configurations: {n_allowed} ({100*n_allowed/n_total:.2f}%)")

                # H0 statistics
                if "H0_stats" in s:
                    h0 = s["H0_stats"]
                elif "H0_allowed_mean" in s:
                    h0 = {
                        "mean": s.get("H0_allowed_mean"),
                        "std": s.get("H0_allowed_std"),
                        "min": s.get("H0_allowed_min"),
                        "max": s.get("H0_allowed_max"),
                        "n_ge_71": s.get("n_H0_ge_71", 0),
                        "n_ge_73": s.get("n_H0_ge_73", 0),
                    }
                else:
                    h0 = None

                if h0 and n_allowed > 0:
                    print(f"  H0 among allowed:")
                    print(f"    Mean ± Std: {h0['mean']:.2f} ± {h0['std']:.2f} km/s/Mpc")
                    print(f"    Range: [{h0['min']:.2f}, {h0['max']:.2f}] km/s/Mpc")
                    print(f"    N(H0 ≥ 71): {h0['n_ge_71']}")
                    print(f"    N(H0 ≥ 73): {h0['n_ge_73']}")
        else:
            # Compute from raw data
            passes = data.get("passes_all", data.get("pass_flags", np.array([])))
            H0_eff = data.get("H0_eff", np.array([]))

            if len(passes) > 0 and len(H0_eff) > 0:
                passes = np.asarray(passes).astype(bool)
                H0_eff = np.asarray(H0_eff)

                n_allowed = np.sum(passes)
                n_total = len(passes)

                print(f"  Total configurations: {n_total}")
                print(f"  Allowed configurations: {n_allowed} ({100*n_allowed/n_total:.2f}%)")

                if n_allowed > 0:
                    H0_allowed = H0_eff[passes]
                    print(f"  H0 among allowed:")
                    print(f"    Mean ± Std: {np.mean(H0_allowed):.2f} ± {np.std(H0_allowed):.2f} km/s/Mpc")
                    print(f"    Range: [{np.min(H0_allowed):.2f}, {np.max(H0_allowed):.2f}] km/s/Mpc")
                    print(f"    N(H0 ≥ 71): {np.sum(H0_allowed >= 71)}")
                    print(f"    N(H0 ≥ 73): {np.sum(H0_allowed >= 73)}")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    # Collect all allowed H0 values
    all_H0_allowed = []
    for data in [scan_data, fit_data]:
        if data is None:
            continue
        passes = data.get("passes_all", data.get("pass_flags", np.array([])))
        H0_eff = data.get("H0_eff", np.array([]))
        if len(passes) > 0 and len(H0_eff) > 0:
            passes = np.asarray(passes).astype(bool)
            H0_eff = np.asarray(H0_eff)
            all_H0_allowed.extend(H0_eff[passes].tolist())

    if len(all_H0_allowed) == 0:
        print("NEGATIVE: No mode configurations pass all constraints.")
        print("          The multi-mode terminal spectrum model cannot reconcile")
        print("          CMB + BAO + SN data within the explored parameter range.")
    else:
        H0_max = max(all_H0_allowed)
        n_ge_73 = sum(1 for h in all_H0_allowed if h >= 73)
        n_ge_71 = sum(1 for h in all_H0_allowed if h >= 71)

        if n_ge_73 > 0:
            print(f"POSITIVE: Found {n_ge_73} configurations with H0 ≥ 73 km/s/Mpc")
            print("          while remaining consistent with θ*, BAO, and SN constraints.")
            print("          The Hubble tension CAN potentially be addressed by this model.")
        elif n_ge_71 > 0:
            print(f"PARTIAL: Found {n_ge_71} configurations with H0 ≥ 71 km/s/Mpc")
            print(f"         but none reaching H0 ≥ 73 km/s/Mpc (max = {H0_max:.2f}).")
            print("         Some tension relief is possible but not full resolution.")
        else:
            print(f"LIMITED: Allowed configurations have max H0 = {H0_max:.2f} km/s/Mpc")
            print("         This is insufficient to address the Hubble tension.")

    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SIM 25: Analyze multi-mode terminal spectrum results"
    )
    parser.add_argument(
        "--scan-dir", type=str, default="results/simulation_25_mode_spectrum_scan",
        help="Directory containing SIM 25A scan results"
    )
    parser.add_argument(
        "--fit-dir", type=str, default="results/simulation_25_inverse_fit",
        help="Directory containing SIM 25B fit results"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures/simulation_25",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Load data
    if verbose:
        print("Loading results...")

    scan_data = load_scan_results(args.scan_dir)
    fit_data = load_fit_results(args.fit_dir)

    if scan_data is None and fit_data is None:
        print("ERROR: No data found. Run sim25a and/or sim25b first.")
        return 1

    # Create plots
    create_plots(scan_data, fit_data, args.output_dir, verbose=verbose)

    # Print summary
    print_summary(scan_data, fit_data, verbose=verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
