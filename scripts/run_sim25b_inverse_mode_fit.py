#!/usr/bin/env python3
"""
SIMULATION 25B: Inverse Search for Mode Patterns Consistent with Data

This script treats the 3-mode amplitudes (A1, A2, A3) as free parameters
and searches for configurations that are allowed by:
- CMB θ* constraint (|Δθ*/θ*| < threshold)
- BAO distance constraints (< threshold)
- SN distance constraints (< threshold)

The goal is to identify the "allowed region" in mode-amplitude space and
determine whether any configurations can achieve H0 ≥ 73 km/s/Mpc while
remaining consistent with all data.

Usage:
    python run_sim25b_inverse_mode_fit.py [--n-samples N] [--output-dir DIR] [--seed SEED]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.terminal_spectrum import (
    TerminalMode,
    TerminalSpectrumParams,
    SpectrumCosmoConfig,
    compute_full_chi2,
    get_H0_effective,
    make_3mode_template,
)


def create_output_dirs(output_dir: str) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace("results", "figures"), exist_ok=True)


def random_sample_3mode(
    rng: np.random.Generator,
    amp_range: Tuple[float, float],
    n_samples: int,
) -> np.ndarray:
    """
    Generate random samples of (A1, A2, A3) amplitudes.

    Parameters
    ----------
    rng : Generator
        Random number generator
    amp_range : tuple
        (min, max) amplitude range
    n_samples : int
        Number of samples

    Returns
    -------
    ndarray
        Shape (n_samples, 3) array of amplitudes
    """
    return rng.uniform(amp_range[0], amp_range[1], size=(n_samples, 3))


def refine_around_region(
    rng: np.random.Generator,
    center: np.ndarray,
    scale: float,
    amp_range: Tuple[float, float],
    n_samples: int,
) -> np.ndarray:
    """
    Generate samples concentrated around a region of interest.

    Uses Gaussian sampling centered on 'center' with std 'scale',
    clipped to amp_range.

    Parameters
    ----------
    rng : Generator
        Random number generator
    center : ndarray
        Center point (A1, A2, A3)
    scale : float
        Standard deviation for Gaussian sampling
    amp_range : tuple
        (min, max) amplitude range
    n_samples : int
        Number of samples

    Returns
    -------
    ndarray
        Shape (n_samples, 3) array of amplitudes
    """
    samples = rng.normal(center, scale, size=(n_samples, 3))
    samples = np.clip(samples, amp_range[0], amp_range[1])
    return samples


def run_inverse_search(
    cosmo: SpectrumCosmoConfig,
    z_centers: Tuple[float, float, float],
    sigma_ln_a: float,
    amp_range: Tuple[float, float],
    n_samples: int,
    theta_star_tol: float = 0.1,
    bao_tol: float = 2.0,
    sn_tol: float = 2.0,
    refine_iterations: int = 2,
    refine_n_samples: int = 200,
    seed: int = 12345,
    verbose: bool = True,
) -> Dict:
    """
    Run inverse search for allowed mode configurations.

    Strategy:
    1. Coarse random sampling over full amplitude range
    2. Identify promising regions (low chi2, passing constraints)
    3. Refine with denser sampling around promising regions
    4. Compile final allowed region

    Parameters
    ----------
    cosmo : SpectrumCosmoConfig
        Baseline cosmological parameters
    z_centers : tuple
        Redshift centers for the 3 modes
    sigma_ln_a : float
        Width of each mode
    amp_range : tuple
        (min, max) amplitude range
    n_samples : int
        Number of initial random samples
    theta_star_tol : float
        θ* tolerance [%]
    bao_tol : float
        BAO tolerance [%]
    sn_tol : float
        SN tolerance [%]
    refine_iterations : int
        Number of refinement iterations
    refine_n_samples : int
        Samples per refinement iteration
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Search results with all sampled points and their properties
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Running inverse search with {n_samples} initial samples")
        print(f"Amplitude range: [{amp_range[0]}, {amp_range[1]}]")
        print(f"Tolerances: θ*={theta_star_tol}%, BAO={bao_tol}%, SN={sn_tol}%")

    # Result storage
    all_A1 = []
    all_A2 = []
    all_A3 = []
    all_H0_eff = []
    all_theta_dev = []
    all_bao_dev = []
    all_sn_dev = []
    all_chi2 = []
    all_passes = []

    # Phase 1: Coarse random sampling
    if verbose:
        print("\nPhase 1: Coarse random sampling...")

    samples = random_sample_3mode(rng, amp_range, n_samples)

    for i, (A1, A2, A3) in enumerate(samples):
        if verbose and (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        spec = make_3mode_template(
            z_centers=z_centers,
            sigma_ln_a=sigma_ln_a,
            amplitudes=(A1, A2, A3)
        )

        result = compute_full_chi2(
            cosmo, spec,
            theta_star_tol=theta_star_tol,
            bao_tol=bao_tol,
            sn_tol=sn_tol,
            use_shoes_prior=False,
        )

        all_A1.append(A1)
        all_A2.append(A2)
        all_A3.append(A3)
        all_H0_eff.append(result.H0_eff)
        all_theta_dev.append(result.theta_star_dev_percent)
        all_bao_dev.append(result.max_bao_dev_percent)
        all_sn_dev.append(result.max_sn_dev_percent)
        all_chi2.append(result.chi2_total)
        all_passes.append(
            result.is_physical and
            result.passes_theta_star and
            result.passes_bao and
            result.passes_sn
        )

    # Phase 2: Refinement around promising regions
    if verbose:
        print("\nPhase 2: Refinement around promising regions...")

    for iteration in range(refine_iterations):
        # Find best points so far (lowest chi2 among physical)
        chi2_arr = np.array(all_chi2)
        passes_arr = np.array(all_passes)
        physical_mask = np.isfinite(chi2_arr) & (chi2_arr < 1e9)

        if np.sum(physical_mask) < 5:
            if verbose:
                print(f"  Iteration {iteration+1}: Not enough physical points for refinement")
            continue

        # Get indices of best points
        chi2_physical = np.where(physical_mask, chi2_arr, 1e10)
        best_indices = np.argsort(chi2_physical)[:10]

        # Also include points that pass all constraints
        passing_indices = np.where(passes_arr)[0]
        if len(passing_indices) > 0:
            # Add high-H0 passing points
            H0_arr = np.array(all_H0_eff)
            H0_passing = H0_arr[passing_indices]
            high_H0_order = np.argsort(H0_passing)[-5:]
            best_indices = np.unique(np.concatenate([best_indices, passing_indices[high_H0_order]]))

        # Refine around each best point
        for idx in best_indices[:5]:  # Top 5
            center = np.array([all_A1[idx], all_A2[idx], all_A3[idx]])
            scale = 0.01 * (1.0 / (iteration + 1))  # Tighter in later iterations

            refinement_samples = refine_around_region(
                rng, center, scale, amp_range, refine_n_samples // 5
            )

            for A1, A2, A3 in refinement_samples:
                spec = make_3mode_template(
                    z_centers=z_centers,
                    sigma_ln_a=sigma_ln_a,
                    amplitudes=(A1, A2, A3)
                )

                result = compute_full_chi2(
                    cosmo, spec,
                    theta_star_tol=theta_star_tol,
                    bao_tol=bao_tol,
                    sn_tol=sn_tol,
                    use_shoes_prior=False,
                )

                all_A1.append(A1)
                all_A2.append(A2)
                all_A3.append(A3)
                all_H0_eff.append(result.H0_eff)
                all_theta_dev.append(result.theta_star_dev_percent)
                all_bao_dev.append(result.max_bao_dev_percent)
                all_sn_dev.append(result.max_sn_dev_percent)
                all_chi2.append(result.chi2_total)
                all_passes.append(
                    result.is_physical and
                    result.passes_theta_star and
                    result.passes_bao and
                    result.passes_sn
                )

        if verbose:
            n_passing = np.sum(all_passes)
            print(f"  Iteration {iteration+1}: {len(all_A1)} total samples, {n_passing} passing")

    results = {
        "A1": np.array(all_A1),
        "A2": np.array(all_A2),
        "A3": np.array(all_A3),
        "H0_eff": np.array(all_H0_eff),
        "theta_star_dev": np.array(all_theta_dev),
        "max_bao_dev": np.array(all_bao_dev),
        "max_sn_dev": np.array(all_sn_dev),
        "chi2_total": np.array(all_chi2),
        "pass_flags": np.array(all_passes),
    }

    return results


def analyze_inverse_results(results: Dict, verbose: bool = True) -> Dict:
    """
    Analyze inverse search results.

    Parameters
    ----------
    results : dict
        Results from run_inverse_search
    verbose : bool
        Print summary

    Returns
    -------
    dict
        Summary statistics
    """
    n_total = len(results["A1"])
    passes = results["pass_flags"]
    n_allowed = np.sum(passes)

    # Allowed region statistics
    if n_allowed > 0:
        mask = passes
        A1_allowed = results["A1"][mask]
        A2_allowed = results["A2"][mask]
        A3_allowed = results["A3"][mask]
        H0_allowed = results["H0_eff"][mask]

        # Amplitude statistics
        A1_mean, A1_std = np.mean(A1_allowed), np.std(A1_allowed)
        A2_mean, A2_std = np.mean(A2_allowed), np.std(A2_allowed)
        A3_mean, A3_std = np.mean(A3_allowed), np.std(A3_allowed)

        # H0 statistics
        H0_mean = np.mean(H0_allowed)
        H0_std = np.std(H0_allowed)
        H0_min = np.min(H0_allowed)
        H0_max = np.max(H0_allowed)

        n_H0_ge_71 = np.sum(H0_allowed >= 71)
        n_H0_ge_73 = np.sum(H0_allowed >= 73)

        # Configuration with highest H0 among allowed
        best_H0_idx = np.argmax(H0_allowed)
        best_H0_config = {
            "A1": float(A1_allowed[best_H0_idx]),
            "A2": float(A2_allowed[best_H0_idx]),
            "A3": float(A3_allowed[best_H0_idx]),
            "H0_eff": float(H0_allowed[best_H0_idx]),
        }
    else:
        A1_mean = A1_std = A2_mean = A2_std = A3_mean = A3_std = np.nan
        H0_mean = H0_std = H0_min = H0_max = np.nan
        n_H0_ge_71 = n_H0_ge_73 = 0
        best_H0_config = None

    summary = {
        "n_total_samples": int(n_total),
        "n_allowed": int(n_allowed),
        "fraction_allowed": float(n_allowed / n_total) if n_total > 0 else 0,
        "amplitude_stats": {
            "A1": {"mean": float(A1_mean), "std": float(A1_std)},
            "A2": {"mean": float(A2_mean), "std": float(A2_std)},
            "A3": {"mean": float(A3_mean), "std": float(A3_std)},
        },
        "H0_stats": {
            "mean": float(H0_mean),
            "std": float(H0_std),
            "min": float(H0_min),
            "max": float(H0_max),
            "n_ge_71": int(n_H0_ge_71),
            "n_ge_73": int(n_H0_ge_73),
        },
        "best_H0_config": best_H0_config,
    }

    if verbose:
        print("\n" + "="*60)
        print("INVERSE SEARCH RESULTS")
        print("="*60)
        print(f"Total samples: {n_total}")
        print(f"Allowed configurations: {n_allowed} ({100*n_allowed/n_total:.2f}%)")
        print()

        if n_allowed > 0:
            print("Allowed amplitude region:")
            print(f"  A1: {A1_mean:.4f} ± {A1_std:.4f}")
            print(f"  A2: {A2_mean:.4f} ± {A2_std:.4f}")
            print(f"  A3: {A3_mean:.4f} ± {A3_std:.4f}")
            print()
            print("H0 distribution among allowed:")
            print(f"  Mean ± Std: {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
            print(f"  Range: [{H0_min:.2f}, {H0_max:.2f}] km/s/Mpc")
            print(f"  N(H0 ≥ 71): {n_H0_ge_71}")
            print(f"  N(H0 ≥ 73): {n_H0_ge_73}")
            print()
            print("Best H0 configuration:")
            print(f"  (A1, A2, A3) = ({best_H0_config['A1']:.4f}, {best_H0_config['A2']:.4f}, {best_H0_config['A3']:.4f})")
            print(f"  H0_eff = {best_H0_config['H0_eff']:.2f} km/s/Mpc")
        else:
            print("No configurations pass all constraints!")

        print("="*60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SIM 25B: Inverse search for allowed mode configurations"
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Number of initial random samples (default: 2000)"
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
        "--output-dir", type=str, default="results/simulation_25_inverse_fit",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed"
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
        "--refine-iter", type=int, default=2,
        help="Number of refinement iterations (default: 2)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("="*60)
        print("SIMULATION 25B: Inverse Mode Search")
        print("="*60)
        print()

    # Create output directory
    create_output_dirs(args.output_dir)

    # Baseline cosmology
    cosmo = SpectrumCosmoConfig(
        H0=67.5,
        Omega_m=0.315,
        Omega_L=0.685,
        Omega_r=5e-5,
        Omega_k=0.0,
    )

    # Mode centers
    z_centers = (3000, 100, 1)

    if verbose:
        print(f"Configuration:")
        print(f"  Baseline H0: {cosmo.H0} km/s/Mpc")
        print(f"  Mode centers (z): {z_centers}")
        print(f"  Mode width: σ_ln_a = {args.sigma}")
        print(f"  Amplitude range: [{args.amp_min}, {args.amp_max}]")
        print(f"  Initial samples: {args.n_samples}")
        print(f"  Refinement iterations: {args.refine_iter}")
        print()

    # Run inverse search
    results = run_inverse_search(
        cosmo=cosmo,
        z_centers=z_centers,
        sigma_ln_a=args.sigma,
        amp_range=(args.amp_min, args.amp_max),
        n_samples=args.n_samples,
        theta_star_tol=args.theta_tol,
        bao_tol=args.bao_tol,
        sn_tol=args.sn_tol,
        refine_iterations=args.refine_iter,
        seed=args.seed,
        verbose=verbose,
    )

    # Analyze results
    summary = analyze_inverse_results(results, verbose=verbose)

    # Save results
    np.savez(
        os.path.join(args.output_dir, "points.npz"),
        **results
    )

    # Save summary
    summary_full = {
        "summary": summary,
        "config": {
            "n_samples": args.n_samples,
            "amp_range": [args.amp_min, args.amp_max],
            "sigma_ln_a": args.sigma,
            "z_centers": z_centers,
            "theta_star_tol": args.theta_tol,
            "bao_tol": args.bao_tol,
            "sn_tol": args.sn_tol,
            "refine_iterations": args.refine_iter,
            "seed": args.seed,
            "cosmo": {
                "H0": cosmo.H0,
                "Omega_m": cosmo.Omega_m,
                "Omega_L": cosmo.Omega_L,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary_full, f, indent=2)

    if verbose:
        print(f"\nResults saved to {args.output_dir}/")
        print(f"  - points.npz (all sampled points)")
        print(f"  - summary.json (analysis summary)")

    # Print verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    H0_stats = summary["H0_stats"]
    if H0_stats["n_ge_73"] > 0:
        print(f"POSITIVE: Found {H0_stats['n_ge_73']} mode configurations with H0 ≥ 73 km/s/Mpc")
        print(f"          while satisfying all constraints (θ*, BAO, SN).")
        if summary["best_H0_config"]:
            cfg = summary["best_H0_config"]
            print(f"          Best config: (A1, A2, A3) = ({cfg['A1']:.4f}, {cfg['A2']:.4f}, {cfg['A3']:.4f})")
            print(f"          H0_eff = {cfg['H0_eff']:.2f} km/s/Mpc")
    elif H0_stats["n_ge_71"] > 0:
        print(f"PARTIAL: Found {H0_stats['n_ge_71']} configurations with H0 ≥ 71 km/s/Mpc")
        print(f"         but none reaching H0 ≥ 73 km/s/Mpc.")
        print(f"         Maximum H0 achieved: {H0_stats['max']:.2f} km/s/Mpc")
    elif summary["n_allowed"] > 0:
        print("LIMITED: Some configurations pass all constraints,")
        print(f"         but maximum H0 = {H0_stats['max']:.2f} km/s/Mpc")
        print("         (far from resolving the Hubble tension)")
    else:
        print("NEGATIVE: No mode configurations in the explored region")
        print("          pass all constraints simultaneously.")

    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
