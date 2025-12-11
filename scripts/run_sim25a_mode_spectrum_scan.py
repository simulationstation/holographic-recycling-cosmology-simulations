#!/usr/bin/env python3
"""
SIMULATION 25A: Forward Scan over 3-Mode Terminal Spectra

This script performs a grid scan over simple 3-mode spectral configurations
to map out how different mode amplitude combinations affect:
- θ* (CMB acoustic scale) deviation
- BAO distance deviations
- SN distance deviations
- Effective H0

The goal is to understand what regions of mode-amplitude space are
allowed by current data, and whether any configurations can shift H0
toward higher values while remaining consistent with constraints.

Usage:
    python run_sim25a_mode_spectrum_scan.py [--n-samples N] [--output-dir DIR] [--seed SEED]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.terminal_spectrum import (
    TerminalMode,
    TerminalSpectrumParams,
    SpectrumCosmoConfig,
    compute_full_chi2,
    compute_baseline_chi2,
    get_H0_effective,
    make_3mode_template,
)


def create_output_dirs(output_dir: str) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace("results", "figures"), exist_ok=True)


def run_3mode_scan(
    cosmo: SpectrumCosmoConfig,
    z_centers: Tuple[float, float, float],
    sigma_ln_a: float,
    amplitude_grid: List[float],
    theta_star_tol: float = 0.1,
    bao_tol: float = 2.0,
    sn_tol: float = 2.0,
    verbose: bool = True,
) -> Dict:
    """
    Run a grid scan over 3-mode amplitude space.

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    z_centers : tuple
        Redshift centers for the 3 modes
    sigma_ln_a : float
        Width of each mode in ln(a) space
    amplitude_grid : list
        Amplitude values to scan for each mode
    theta_star_tol : float
        Tolerance for θ* constraint [%]
    bao_tol : float
        Tolerance for BAO constraint [%]
    sn_tol : float
        Tolerance for SN constraint [%]
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Scan results with arrays for each parameter and observable
    """
    n_amp = len(amplitude_grid)
    n_total = n_amp ** 3

    if verbose:
        print(f"Running 3-mode scan: {n_amp}³ = {n_total} configurations")
        print(f"Mode centers: z = {z_centers}")
        print(f"Mode width: σ_ln_a = {sigma_ln_a}")
        print(f"Amplitude range: {amplitude_grid[0]:.3f} to {amplitude_grid[-1]:.3f}")

    # Result arrays
    A1_arr = []
    A2_arr = []
    A3_arr = []
    H0_eff_arr = []
    theta_star_arr = []
    theta_star_dev_arr = []
    chi2_cmb_arr = []
    chi2_bao_arr = []
    chi2_sn_arr = []
    chi2_total_arr = []
    max_bao_dev_arr = []
    max_sn_dev_arr = []
    is_physical_arr = []
    passes_all_arr = []

    # Compute baseline
    if verbose:
        print("Computing baseline LCDM chi2...")
    chi2_baseline = compute_baseline_chi2(cosmo, use_shoes_prior=False)
    if verbose:
        print(f"Baseline chi2 = {chi2_baseline:.2f}")

    # Scan loop
    count = 0
    for A1, A2, A3 in product(amplitude_grid, repeat=3):
        count += 1
        if verbose and count % 50 == 0:
            print(f"  Progress: {count}/{n_total} ({100*count/n_total:.1f}%)")

        # Create spectrum
        spec = make_3mode_template(
            z_centers=z_centers,
            sigma_ln_a=sigma_ln_a,
            amplitudes=(A1, A2, A3)
        )

        # Compute chi2 and observables
        result = compute_full_chi2(
            cosmo, spec,
            theta_star_tol=theta_star_tol,
            bao_tol=bao_tol,
            sn_tol=sn_tol,
            use_shoes_prior=False,
        )

        # Store results
        A1_arr.append(A1)
        A2_arr.append(A2)
        A3_arr.append(A3)
        H0_eff_arr.append(result.H0_eff)
        theta_star_arr.append(result.theta_star)
        theta_star_dev_arr.append(result.theta_star_dev_percent)
        chi2_cmb_arr.append(result.chi2_cmb)
        chi2_bao_arr.append(result.chi2_bao)
        chi2_sn_arr.append(result.chi2_sn)
        chi2_total_arr.append(result.chi2_total)
        max_bao_dev_arr.append(result.max_bao_dev_percent)
        max_sn_dev_arr.append(result.max_sn_dev_percent)
        is_physical_arr.append(result.is_physical)
        passes_all_arr.append(
            result.is_physical and
            result.passes_theta_star and
            result.passes_bao and
            result.passes_sn
        )

    results = {
        "A1": np.array(A1_arr),
        "A2": np.array(A2_arr),
        "A3": np.array(A3_arr),
        "H0_eff": np.array(H0_eff_arr),
        "theta_star": np.array(theta_star_arr),
        "theta_star_dev_percent": np.array(theta_star_dev_arr),
        "chi2_cmb": np.array(chi2_cmb_arr),
        "chi2_bao": np.array(chi2_bao_arr),
        "chi2_sn": np.array(chi2_sn_arr),
        "chi2_total": np.array(chi2_total_arr),
        "max_bao_dev_percent": np.array(max_bao_dev_arr),
        "max_sn_dev_percent": np.array(max_sn_dev_arr),
        "is_physical": np.array(is_physical_arr),
        "passes_all": np.array(passes_all_arr),
        "chi2_baseline": chi2_baseline,
    }

    return results


def analyze_scan_results(results: Dict, verbose: bool = True) -> Dict:
    """
    Analyze scan results and compute summary statistics.

    Parameters
    ----------
    results : dict
        Scan results from run_3mode_scan
    verbose : bool
        Print summary

    Returns
    -------
    dict
        Summary statistics
    """
    n_total = len(results["A1"])
    n_physical = np.sum(results["is_physical"])
    n_allowed = np.sum(results["passes_all"])

    # Filter to allowed configurations
    mask_allowed = results["passes_all"]
    H0_allowed = results["H0_eff"][mask_allowed]

    # Statistics for allowed configurations
    if len(H0_allowed) > 0:
        H0_mean = np.mean(H0_allowed)
        H0_std = np.std(H0_allowed)
        H0_min = np.min(H0_allowed)
        H0_max = np.max(H0_allowed)

        # Check for high H0
        n_H0_ge_71 = np.sum(H0_allowed >= 71)
        n_H0_ge_73 = np.sum(H0_allowed >= 73)
    else:
        H0_mean = H0_std = H0_min = H0_max = np.nan
        n_H0_ge_71 = n_H0_ge_73 = 0

    # Best chi2 configuration
    chi2_arr = results["chi2_total"]
    chi2_arr_safe = np.where(results["is_physical"], chi2_arr, 1e10)
    best_idx = np.argmin(chi2_arr_safe)

    best_config = {
        "A1": float(results["A1"][best_idx]),
        "A2": float(results["A2"][best_idx]),
        "A3": float(results["A3"][best_idx]),
        "H0_eff": float(results["H0_eff"][best_idx]),
        "chi2_total": float(chi2_arr[best_idx]),
        "theta_star_dev_percent": float(results["theta_star_dev_percent"][best_idx]),
        "max_bao_dev_percent": float(results["max_bao_dev_percent"][best_idx]),
    }

    summary = {
        "n_total": int(n_total),
        "n_physical": int(n_physical),
        "n_allowed": int(n_allowed),
        "fraction_allowed": float(n_allowed / n_total) if n_total > 0 else 0,
        "H0_allowed_mean": float(H0_mean),
        "H0_allowed_std": float(H0_std),
        "H0_allowed_min": float(H0_min),
        "H0_allowed_max": float(H0_max),
        "n_H0_ge_71": int(n_H0_ge_71),
        "n_H0_ge_73": int(n_H0_ge_73),
        "best_config": best_config,
    }

    if verbose:
        print("\n" + "="*60)
        print("SCAN RESULTS SUMMARY")
        print("="*60)
        print(f"Total configurations scanned: {n_total}")
        print(f"Physically valid: {n_physical} ({100*n_physical/n_total:.1f}%)")
        print(f"Passing all constraints: {n_allowed} ({100*n_allowed/n_total:.1f}%)")
        print()

        if n_allowed > 0:
            print(f"H0 among allowed configurations:")
            print(f"  Mean ± Std: {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
            print(f"  Range: [{H0_min:.2f}, {H0_max:.2f}] km/s/Mpc")
            print(f"  N(H0 ≥ 71): {n_H0_ge_71}")
            print(f"  N(H0 ≥ 73): {n_H0_ge_73}")
        else:
            print("No configurations pass all constraints!")

        print()
        print("Best chi2 configuration:")
        print(f"  (A1, A2, A3) = ({best_config['A1']:.3f}, {best_config['A2']:.3f}, {best_config['A3']:.3f})")
        print(f"  H0_eff = {best_config['H0_eff']:.2f} km/s/Mpc")
        print(f"  chi2_total = {best_config['chi2_total']:.2f}")
        print(f"  θ* deviation = {best_config['theta_star_dev_percent']:.3f}%")
        print(f"  Max BAO dev = {best_config['max_bao_dev_percent']:.2f}%")
        print("="*60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SIM 25A: Forward scan over 3-mode terminal spectra"
    )
    parser.add_argument(
        "--n-grid", type=int, default=9,
        help="Number of amplitude grid points per mode (default: 9, giving 729 configs)"
    )
    parser.add_argument(
        "--amp-min", type=float, default=-0.05,
        help="Minimum amplitude (default: -0.05)"
    )
    parser.add_argument(
        "--amp-max", type=float, default=0.05,
        help="Maximum amplitude (default: 0.05)"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.3,
        help="Mode width in ln(a) space (default: 0.3)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/simulation_25_mode_spectrum_scan",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed (not used in grid scan, but for reproducibility)"
    )
    parser.add_argument(
        "--theta-tol", type=float, default=0.1,
        help="Tolerance for θ* deviation [%%] (default: 0.1)"
    )
    parser.add_argument(
        "--bao-tol", type=float, default=2.0,
        help="Tolerance for BAO deviation [%%] (default: 2.0)"
    )
    parser.add_argument(
        "--sn-tol", type=float, default=2.0,
        help="Tolerance for SN deviation [%%] (default: 2.0)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    verbose = not args.quiet

    if verbose:
        print("="*60)
        print("SIMULATION 25A: Multi-Mode Terminal Spectrum Scan")
        print("="*60)
        print()

    # Create output directory
    create_output_dirs(args.output_dir)

    # Baseline cosmology (Planck-like)
    cosmo = SpectrumCosmoConfig(
        H0=67.5,
        Omega_m=0.315,
        Omega_L=0.685,
        Omega_r=5e-5,
        Omega_k=0.0,
    )

    # Mode centers:
    # - Mode 0: z ~ 3000 (very early, near recombination)
    # - Mode 1: z ~ 100 (around equality)
    # - Mode 2: z ~ 1 (late-time, SN/BAO regime)
    z_centers = (3000, 100, 1)

    # Amplitude grid
    amplitude_grid = np.linspace(args.amp_min, args.amp_max, args.n_grid).tolist()

    if verbose:
        print(f"Configuration:")
        print(f"  Baseline H0: {cosmo.H0} km/s/Mpc")
        print(f"  Baseline Omega_m: {cosmo.Omega_m}")
        print(f"  Mode centers (z): {z_centers}")
        print(f"  Mode width: σ_ln_a = {args.sigma}")
        print(f"  Amplitude grid: {args.n_grid} points from {args.amp_min} to {args.amp_max}")
        print(f"  Tolerance θ*: {args.theta_tol}%")
        print(f"  Tolerance BAO: {args.bao_tol}%")
        print(f"  Tolerance SN: {args.sn_tol}%")
        print()

    # Run scan
    results = run_3mode_scan(
        cosmo=cosmo,
        z_centers=z_centers,
        sigma_ln_a=args.sigma,
        amplitude_grid=amplitude_grid,
        theta_star_tol=args.theta_tol,
        bao_tol=args.bao_tol,
        sn_tol=args.sn_tol,
        verbose=verbose,
    )

    # Analyze results
    summary = analyze_scan_results(results, verbose=verbose)

    # Save results
    np.savez(
        os.path.join(args.output_dir, "scan_results.npz"),
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
        chi2_baseline=results["chi2_baseline"],
    )

    # Save summary JSON
    summary_full = {
        "summary": summary,
        "config": {
            "n_grid": args.n_grid,
            "amp_min": args.amp_min,
            "amp_max": args.amp_max,
            "sigma_ln_a": args.sigma,
            "z_centers": z_centers,
            "theta_star_tol": args.theta_tol,
            "bao_tol": args.bao_tol,
            "sn_tol": args.sn_tol,
            "cosmo": {
                "H0": cosmo.H0,
                "Omega_m": cosmo.Omega_m,
                "Omega_L": cosmo.Omega_L,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "scan_summary.json"), "w") as f:
        json.dump(summary_full, f, indent=2)

    if verbose:
        print(f"\nResults saved to {args.output_dir}/")
        print(f"  - scan_results.npz (full data)")
        print(f"  - scan_summary.json (summary)")

    # Print verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if summary["n_H0_ge_73"] > 0:
        print(f"FOUND: {summary['n_H0_ge_73']} configurations with H0 ≥ 73 km/s/Mpc")
        print("       while satisfying θ*, BAO, and SN constraints.")
    elif summary["n_H0_ge_71"] > 0:
        print(f"PARTIAL: {summary['n_H0_ge_71']} configurations with H0 ≥ 71 km/s/Mpc")
        print("         but none reaching H0 ≥ 73 km/s/Mpc.")
        print(f"         Maximum H0 achieved: {summary['H0_allowed_max']:.2f} km/s/Mpc")
    elif summary["n_allowed"] > 0:
        print("LIMITED: Some configurations pass all constraints,")
        print(f"         but maximum H0 = {summary['H0_allowed_max']:.2f} km/s/Mpc")
        print("         (far from resolving the Hubble tension)")
    else:
        print("NEGATIVE: No 3-mode configurations in the scanned amplitude range")
        print("          pass all constraints (θ*, BAO, SN).")
        print("          The model may be too constrained or amplitudes too large.")

    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
