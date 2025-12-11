#!/usr/bin/env python3
"""
SIMULATION 24: Black-Hole Interior Transition Computation Cosmology (BITCC)

This script runs the full BITCC analysis:
1. Prior predictive sampling: What H0 values are typical under the BITCC prior?
2. Data compatibility: Which samples are consistent with CMB+BAO+SN constraints?
3. Posterior-like analysis: What is P(H0 >= 73) after imposing data cuts?

The goal is to explore whether the BITCC prior naturally prefers high-H0 values
and how strongly observational constraints push back toward Planck-like values.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.bitcc import (
    BITCCHyperparams,
    BITCCPriorSample,
    sample_bitcc_prior,
    compute_H0_distribution_from_bitcc,
    compute_chi_trans_statistics,
    extract_arrays_from_samples,
    map_H_init_to_cosmo,
    compute_distance_ladder_proxies,
    check_data_compatibility,
    compute_approximate_chi2,
)


def run_prior_predictive(
    hyper: BITCCHyperparams,
    n_samples: int,
    seed: int,
    H0_ref: float,
    gamma_H: float,
    verbose: bool = True,
) -> Tuple[List[BITCCPriorSample], Dict[str, Any]]:
    """
    Run prior predictive sampling from BITCC.

    Parameters
    ----------
    hyper : BITCCHyperparams
        Hyperparameters for the BITCC prior.
    n_samples : int
        Number of samples to draw.
    seed : int
        Random seed.
    H0_ref : float
        Reference H0 for mapping.
    gamma_H : float
        Sensitivity parameter for H_init mapping.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (samples, prior_stats)
    """
    if verbose:
        print("=" * 70)
        print("STEP 1: Prior Predictive Sampling")
        print("=" * 70)
        print(f"\nDrawing {n_samples} samples from BITCC prior...")

    rng = np.random.default_rng(seed)

    samples = sample_bitcc_prior(
        hyper=hyper,
        n_samples=n_samples,
        rng=rng,
        H0_ref=H0_ref,
        gamma_H=gamma_H,
    )

    prior_stats = compute_H0_distribution_from_bitcc(samples)
    chi_trans_stats = compute_chi_trans_statistics(samples)

    if verbose:
        print(f"\nPrior H0 distribution:")
        print(f"  Mean:     {prior_stats['mean']:.2f} km/s/Mpc")
        print(f"  Std:      {prior_stats['std']:.2f} km/s/Mpc")
        print(f"  Median:   {prior_stats['median']:.2f} km/s/Mpc")
        print(f"  Range:    [{prior_stats['min']:.2f}, {prior_stats['max']:.2f}]")
        print(f"  P(H0>=73): {100*prior_stats['P_H0_ge_73']:.2f}%")
        print(f"  P(H0>=70): {100*prior_stats['P_H0_ge_70']:.2f}%")
        print(f"  P(H0<=65): {100*prior_stats['P_H0_le_65']:.2f}%")

        print(f"\nχ_trans distribution:")
        print(f"  Mean:   {chi_trans_stats['mean']:.4f}")
        print(f"  Std:    {chi_trans_stats['std']:.4f}")
        print(f"  Range:  [{chi_trans_stats['min']:.4f}, {chi_trans_stats['max']:.4f}]")

    return samples, prior_stats


def apply_data_compatibility(
    samples: List[BITCCPriorSample],
    tol_D_A_z_star: float = 0.005,
    tol_D_L_z0p5: float = 0.02,
    tol_D_L_z1p0: float = 0.03,
    verbose: bool = True,
) -> Tuple[List[BITCCPriorSample], List[Dict[str, float]], Dict[str, Any]]:
    """
    Apply data compatibility cuts to BITCC samples.

    Parameters
    ----------
    samples : list
        BITCC prior samples.
    tol_* : float
        Tolerance levels for distance constraints.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (compatible_samples, all_proxies, post_stats)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Apply Data Compatibility Cuts")
        print("=" * 70)
        print(f"\nConstraints:")
        print(f"  |ΔD_A(z_*)/D_A| < {100*tol_D_A_z_star:.1f}% (CMB)")
        print(f"  |ΔD_L(z=0.5)/D_L| < {100*tol_D_L_z0p5:.1f}% (SN)")
        print(f"  |ΔD_L(z=1.0)/D_L| < {100*tol_D_L_z1p0:.1f}% (SN)")

    compatible_samples = []
    compatible_indices = []
    all_proxies = []
    all_chi2 = []

    for i, sample in enumerate(samples):
        # Map to cosmology
        cosmo = map_H_init_to_cosmo(sample.derived.H_init)

        # Compute distance proxies
        proxies = compute_distance_ladder_proxies(cosmo)
        all_proxies.append(proxies)

        # Compute chi2
        chi2 = compute_approximate_chi2(proxies)
        all_chi2.append(chi2)

        # Check compatibility
        is_compat, _ = check_data_compatibility(
            proxies,
            tol_D_A_z_star=tol_D_A_z_star,
            tol_D_L_z0p5=tol_D_L_z0p5,
            tol_D_L_z1p0=tol_D_L_z1p0,
        )

        if is_compat:
            compatible_samples.append(sample)
            compatible_indices.append(i)

    # Compute posterior-like statistics
    if compatible_samples:
        post_stats = compute_H0_distribution_from_bitcc(compatible_samples)
    else:
        post_stats = {
            "mean": None,
            "std": None,
            "median": None,
            "P_H0_ge_73": 0.0,
            "P_H0_ge_70": 0.0,
            "n_samples": 0,
        }

    fraction_compatible = len(compatible_samples) / len(samples)

    if verbose:
        print(f"\nCompatibility results:")
        print(f"  Total samples:     {len(samples)}")
        print(f"  Compatible:        {len(compatible_samples)} ({100*fraction_compatible:.1f}%)")

        if compatible_samples:
            print(f"\nPosterior-like H0 distribution:")
            print(f"  Mean:     {post_stats['mean']:.2f} km/s/Mpc")
            print(f"  Std:      {post_stats['std']:.2f} km/s/Mpc")
            print(f"  Median:   {post_stats['median']:.2f} km/s/Mpc")
            print(f"  P(H0>=73): {100*post_stats['P_H0_ge_73']:.2f}%")
            print(f"  P(H0>=70): {100*post_stats['P_H0_ge_70']:.2f}%")
        else:
            print("\n  No compatible samples found!")

    return compatible_samples, all_proxies, post_stats


def save_results(
    samples: List[BITCCPriorSample],
    compatible_samples: List[BITCCPriorSample],
    prior_stats: Dict[str, Any],
    post_stats: Dict[str, Any],
    hyper: BITCCHyperparams,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract arrays for npz
    arrays = extract_arrays_from_samples(samples)
    H0_prior = arrays["H_init"]
    chi_trans = arrays["chi_trans"]
    m_bh = arrays["m_bh"]
    N_q = arrays["N_q"]
    N_n = arrays["N_n"]

    # Posterior H0 values
    if compatible_samples:
        H0_post = np.array([s.derived.H_init for s in compatible_samples])
    else:
        H0_post = np.array([])

    # Save arrays
    np.savez(
        output_dir / "samples.npz",
        H0_prior=H0_prior,
        H0_post=H0_post,
        chi_trans=chi_trans,
        m_bh=m_bh,
        N_q=N_q,
        N_n=N_n,
    )
    if verbose:
        print(f"Saved samples to {output_dir / 'samples.npz'}")

    # Build summary
    summary = {
        "n_samples": len(samples),
        "n_compatible": len(compatible_samples),
        "fraction_compatible": len(compatible_samples) / len(samples) if samples else 0,
        "H0_prior": prior_stats,
        "H0_post": post_stats,
        "hyperparams": hyper.to_dict(),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Saved summary to {output_dir / 'summary.json'}")


def print_final_verdict(
    prior_stats: Dict[str, Any],
    post_stats: Dict[str, Any],
    fraction_compatible: float,
) -> None:
    """Print final verdict."""
    print("\n" + "=" * 70)
    print("FINAL VERDICT: SIMULATION 24 - BITCC")
    print("=" * 70)

    print("\nKey Results:")
    print("-" * 40)
    print(f"Prior P(H0 >= 73):        {100*prior_stats['P_H0_ge_73']:.2f}%")
    print(f"Prior P(H0 >= 70):        {100*prior_stats['P_H0_ge_70']:.2f}%")
    print(f"Prior mean H0:            {prior_stats['mean']:.2f} km/s/Mpc")

    print(f"\nData-compatible fraction: {100*fraction_compatible:.1f}%")

    if post_stats["mean"] is not None:
        print(f"\nPosterior-like P(H0 >= 73): {100*post_stats['P_H0_ge_73']:.2f}%")
        print(f"Posterior-like P(H0 >= 70): {100*post_stats['P_H0_ge_70']:.2f}%")
        print(f"Posterior-like mean H0:     {post_stats['mean']:.2f} km/s/Mpc")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    # Does BITCC push H0 up or down?
    prior_mean = prior_stats["mean"]
    h0_ref = 67.5

    if prior_mean > h0_ref + 1:
        direction = "UPWARD"
        print(f"\nThe BITCC prior pushes H0 {direction} relative to Planck.")
        print(f"Prior mean ({prior_mean:.2f}) > Planck reference ({h0_ref:.2f})")
    elif prior_mean < h0_ref - 1:
        direction = "DOWNWARD"
        print(f"\nThe BITCC prior pushes H0 {direction} relative to Planck.")
        print(f"Prior mean ({prior_mean:.2f}) < Planck reference ({h0_ref:.2f})")
    else:
        print(f"\nThe BITCC prior is roughly centered around Planck H0.")
        print(f"Prior mean ({prior_mean:.2f}) ≈ Planck reference ({h0_ref:.2f})")

    if prior_stats["P_H0_ge_73"] > 0.1:
        print(f"\nThe BITCC prior gives significant weight to H0 >= 73:")
        print(f"  P(H0 >= 73) = {100*prior_stats['P_H0_ge_73']:.1f}% (prior)")
    else:
        print(f"\nThe BITCC prior gives modest weight to H0 >= 73:")
        print(f"  P(H0 >= 73) = {100*prior_stats['P_H0_ge_73']:.1f}% (prior)")

    if post_stats["mean"] is not None:
        post_p73 = post_stats["P_H0_ge_73"]
        if post_p73 > 0.01:
            print(f"\nAfter data cuts, some support for H0 >= 73 remains:")
            print(f"  P(H0 >= 73) = {100*post_p73:.2f}% (posterior-like)")
        else:
            print(f"\nAfter data cuts, negligible support for H0 >= 73:")
            print(f"  P(H0 >= 73) = {100*post_p73:.2f}% (posterior-like)")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SIM 24: BITCC Prior and Data Analysis"
    )
    parser.add_argument(
        "--n-samples", type=int, default=20000,
        help="Number of samples (default: 20000)"
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed (default: 1234)"
    )
    parser.add_argument(
        "--H0-ref", type=float, default=67.5,
        help="Reference H0 for mapping (default: 67.5)"
    )
    parser.add_argument(
        "--gamma-H", type=float, default=0.25,
        help="Sensitivity parameter for H_init mapping (default: 0.25)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/simulation_24_bitcc/)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer samples"
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_samples = 1000
    else:
        n_samples = args.n_samples

    # Define hyperparameters
    hyper = BITCCHyperparams(
        N_q_mean=8.0,
        N_q_sigma=3.0,
        N_n_mean=3.0,
        N_n_sigma=1.0,
        k_trans_mean=1.0,
        k_trans_sigma=0.5,
        s_q_alpha=2.0,
        s_q_beta=2.0,
        log10_m_bh_min=5.0,
        log10_m_bh_max=9.0,
    )

    config = {
        "n_samples": n_samples,
        "seed": args.seed,
        "H0_ref": args.H0_ref,
        "gamma_H": args.gamma_H,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/simulation_24_bitcc")

    # Print header
    print("\n" + "=" * 70)
    print("SIMULATION 24: Black-Hole Interior Transition Computation Cosmology")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_samples: {n_samples}")
    print(f"  seed: {args.seed}")
    print(f"  H0_ref: {args.H0_ref}")
    print(f"  gamma_H: {args.gamma_H}")
    print(f"  output: {output_dir}")

    print(f"\nBITCC Hyperparameters:")
    print(f"  N_q ~ N({hyper.N_q_mean}, {hyper.N_q_sigma})")
    print(f"  N_n ~ N({hyper.N_n_mean}, {hyper.N_n_sigma})")
    print(f"  k_trans ~ N({hyper.k_trans_mean}, {hyper.k_trans_sigma})")
    print(f"  s_q ~ Beta({hyper.s_q_alpha}, {hyper.s_q_beta})")
    print(f"  log10(m_bh) ~ U({hyper.log10_m_bh_min}, {hyper.log10_m_bh_max})")

    # Run analysis
    start_time = time.time()

    # Step 1: Prior predictive
    samples, prior_stats = run_prior_predictive(
        hyper=hyper,
        n_samples=n_samples,
        seed=args.seed,
        H0_ref=args.H0_ref,
        gamma_H=args.gamma_H,
    )

    # Step 2: Apply data compatibility
    compatible_samples, all_proxies, post_stats = apply_data_compatibility(
        samples=samples,
    )

    fraction_compatible = len(compatible_samples) / len(samples)

    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.1f} seconds")

    # Save results
    save_results(
        samples=samples,
        compatible_samples=compatible_samples,
        prior_stats=prior_stats,
        post_stats=post_stats,
        hyper=hyper,
        config=config,
        output_dir=output_dir,
    )

    # Print verdict
    print_final_verdict(prior_stats, post_stats, fraction_compatible)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
