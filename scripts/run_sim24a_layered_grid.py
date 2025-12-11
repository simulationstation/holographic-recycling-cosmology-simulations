#!/usr/bin/env python3
"""
SIMULATION 24A: Layered Expansion (Bent-Deck) Grid Scan

This script performs a grid scan over layered expansion parameters to
explore whether any smooth H(z) modification can reconcile CMB+BAO+SN
with H0 ~ 73 km/s/Mpc.

The scan explores:
- n_layers: number of redshift nodes (4, 6)
- smooth_sigma: stiffness of smoothness prior (0.02, 0.05)
- sigma_delta: amplitude of random deviations (0.02, 0.05, 0.1)

For each configuration, random samples are drawn and evaluated.

Key question: Does any smooth layered expansion achieve H0 >= 73
with acceptable chi-squared?
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.layered import (
    LayeredExpansionHyperparams,
    LayeredExpansionParams,
    LCDMBackground,
    make_default_nodes,
    make_random_params,
    make_zero_params,
    log_smoothness_prior,
    H_of_z_layered,
    get_H0_effective,
    check_physical_validity,
    compute_chi2_cmb_bao_sn,
    compute_baseline_chi2,
)


@dataclass
class GridScanConfig:
    """Configuration for the grid scan."""
    n_layers_list: List[int]
    smooth_sigma_list: List[float]
    sigma_delta_list: List[float]
    n_samples_per_config: int
    mode: str  # "delta_H" or "delta_w"
    include_shoes: bool  # Include SH0ES H0 prior in SN constraint
    seed: int


@dataclass
class ScanResult:
    """Result for a single parameter configuration."""
    # Configuration
    n_layers: int
    smooth_sigma: float
    sigma_delta: float
    sample_idx: int

    # Delta values at nodes
    delta_nodes: List[float]
    z_nodes: List[float]

    # Results
    H0_eff: float
    chi2_total: float
    chi2_cmb: float
    chi2_bao: float
    chi2_sn: float
    delta_chi2: float
    log_prior_smoothness: float
    is_physical: bool

    # Diagnostics
    theta_star: float
    passes_fit: bool  # delta_chi2 < threshold


def run_single_sample(
    lcdm: LCDMBackground,
    hyp: LayeredExpansionHyperparams,
    sigma_delta: float,
    rng: np.random.Generator,
    chi2_baseline: float,
    include_shoes: bool,
    delta_chi2_threshold: float = 10.0,
) -> ScanResult:
    """
    Run a single sample in the grid scan.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    hyp : LayeredExpansionHyperparams
        Hyperparameters for the layered model
    sigma_delta : float
        Standard deviation for random delta values
    rng : Generator
        Random number generator
    chi2_baseline : float
        Baseline chi-squared for delta computation
    include_shoes : bool
        Include SH0ES prior
    delta_chi2_threshold : float
        Threshold for "good fit" classification

    Returns
    -------
    ScanResult
        Result for this sample
    """
    # Generate random parameters
    params = make_random_params(hyp, sigma_delta=sigma_delta, rng=rng)

    # Compute chi-squared
    result = compute_chi2_cmb_bao_sn(
        lcdm, params, hyp,
        include_shoes=include_shoes,
        chi2_baseline=chi2_baseline
    )

    # Check if it passes the fit threshold
    passes_fit = result.is_physical and (result.delta_chi2 < delta_chi2_threshold)

    return ScanResult(
        n_layers=hyp.n_layers,
        smooth_sigma=hyp.smooth_sigma,
        sigma_delta=sigma_delta,
        sample_idx=0,  # Will be set by caller
        delta_nodes=params.delta_nodes.tolist(),
        z_nodes=params.z_nodes.tolist(),
        H0_eff=result.H0_eff,
        chi2_total=result.chi2_total,
        chi2_cmb=result.chi2_cmb,
        chi2_bao=result.chi2_bao,
        chi2_sn=result.chi2_sn,
        delta_chi2=result.delta_chi2,
        log_prior_smoothness=result.log_prior_smoothness,
        is_physical=result.is_physical,
        theta_star=result.theta_star,
        passes_fit=passes_fit,
    )


def run_grid_scan(
    config: GridScanConfig,
    verbose: bool = True,
) -> Tuple[List[ScanResult], Dict[str, Any]]:
    """
    Run the full grid scan.

    Parameters
    ----------
    config : GridScanConfig
        Scan configuration
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (results_list, summary_dict)
    """
    rng = np.random.default_rng(config.seed)

    # Setup baseline LCDM
    lcdm = LCDMBackground(H0=67.5, Omega_m=0.315)

    # Compute baseline chi-squared (with any hyperparams - they don't affect baseline)
    hyp_baseline = LayeredExpansionHyperparams(n_layers=6, mode=config.mode)
    chi2_baseline = compute_baseline_chi2(lcdm, hyp_baseline, include_shoes=config.include_shoes)

    if verbose:
        print(f"Baseline LCDM chi-squared: {chi2_baseline:.2f}")
        print(f"Baseline H0: {lcdm.H0} km/s/Mpc")
        print()

    results = []
    total_configs = (
        len(config.n_layers_list) *
        len(config.smooth_sigma_list) *
        len(config.sigma_delta_list) *
        config.n_samples_per_config
    )

    if verbose:
        print(f"Running {total_configs} total configurations...")
        print()

    sample_idx = 0
    for n_layers in config.n_layers_list:
        for smooth_sigma in config.smooth_sigma_list:
            for sigma_delta in config.sigma_delta_list:
                # Create hyperparameters for this configuration
                hyp = LayeredExpansionHyperparams(
                    n_layers=n_layers,
                    smooth_sigma=smooth_sigma,
                    mode=config.mode,
                    z_max=6.0,
                    spacing="log",
                )

                for i in range(config.n_samples_per_config):
                    result = run_single_sample(
                        lcdm=lcdm,
                        hyp=hyp,
                        sigma_delta=sigma_delta,
                        rng=rng,
                        chi2_baseline=chi2_baseline,
                        include_shoes=config.include_shoes,
                    )
                    result.sample_idx = sample_idx
                    results.append(result)
                    sample_idx += 1

                    if verbose and (sample_idx % 100 == 0 or sample_idx == total_configs):
                        print(f"  Completed {sample_idx}/{total_configs} samples")

    # Compute summary statistics
    summary = compute_summary(results, chi2_baseline)

    return results, summary


def compute_summary(results: List[ScanResult], chi2_baseline: float) -> Dict[str, Any]:
    """
    Compute summary statistics from scan results.

    Parameters
    ----------
    results : list
        List of ScanResult objects
    chi2_baseline : float
        Baseline chi-squared

    Returns
    -------
    dict
        Summary statistics
    """
    # Filter to physical models only
    physical = [r for r in results if r.is_physical]

    # Models passing the fit threshold
    passing = [r for r in physical if r.passes_fit]

    # H0 values
    H0_all = np.array([r.H0_eff for r in physical])
    H0_passing = np.array([r.H0_eff for r in passing])

    # Models with H0 >= 73
    high_h0 = [r for r in physical if r.H0_eff >= 73.0]
    high_h0_passing = [r for r in passing if r.H0_eff >= 73.0]

    # Best models by chi2 among those with high H0
    best_high_h0 = None
    if high_h0:
        best_high_h0 = min(high_h0, key=lambda r: r.delta_chi2)

    # Maximum H0 among passing models
    max_H0_passing = max(H0_passing) if len(H0_passing) > 0 else None

    summary = {
        "total_samples": len(results),
        "physical_models": len(physical),
        "passing_models": len(passing),
        "fraction_passing": len(passing) / len(results) if results else 0,

        "models_with_H0_ge_73": len(high_h0),
        "passing_with_H0_ge_73": len(high_h0_passing),

        "H0_distribution_all": {
            "min": float(np.min(H0_all)) if len(H0_all) > 0 else None,
            "max": float(np.max(H0_all)) if len(H0_all) > 0 else None,
            "mean": float(np.mean(H0_all)) if len(H0_all) > 0 else None,
            "std": float(np.std(H0_all)) if len(H0_all) > 0 else None,
            "median": float(np.median(H0_all)) if len(H0_all) > 0 else None,
        },

        "H0_distribution_passing": {
            "min": float(np.min(H0_passing)) if len(H0_passing) > 0 else None,
            "max": float(np.max(H0_passing)) if len(H0_passing) > 0 else None,
            "mean": float(np.mean(H0_passing)) if len(H0_passing) > 0 else None,
            "std": float(np.std(H0_passing)) if len(H0_passing) > 0 else None,
            "median": float(np.median(H0_passing)) if len(H0_passing) > 0 else None,
        },

        "max_H0_among_passing": float(max_H0_passing) if max_H0_passing is not None else None,

        "chi2_baseline": chi2_baseline,
    }

    # Best model with H0 >= 73
    if best_high_h0:
        summary["best_high_h0_model"] = {
            "H0_eff": best_high_h0.H0_eff,
            "delta_chi2": best_high_h0.delta_chi2,
            "chi2_total": best_high_h0.chi2_total,
            "chi2_cmb": best_high_h0.chi2_cmb,
            "chi2_bao": best_high_h0.chi2_bao,
            "chi2_sn": best_high_h0.chi2_sn,
            "n_layers": best_high_h0.n_layers,
            "smooth_sigma": best_high_h0.smooth_sigma,
            "sigma_delta": best_high_h0.sigma_delta,
            "delta_nodes": best_high_h0.delta_nodes,
        }
    else:
        summary["best_high_h0_model"] = None

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary."""
    print("\n" + "=" * 70)
    print("SIMULATION 24A: Layered Expansion Grid Scan - SUMMARY")
    print("=" * 70)

    print(f"\nTotal samples:        {summary['total_samples']}")
    print(f"Physical models:      {summary['physical_models']}")
    print(f"Passing models:       {summary['passing_models']} "
          f"({100*summary['fraction_passing']:.1f}%)")

    print(f"\nModels with H0 >= 73: {summary['models_with_H0_ge_73']}")
    print(f"Passing + H0 >= 73:   {summary['passing_with_H0_ge_73']}")

    print("\nH0 distribution (all physical models):")
    h0_all = summary["H0_distribution_all"]
    if h0_all["mean"] is not None:
        print(f"  min = {h0_all['min']:.2f}, max = {h0_all['max']:.2f}")
        print(f"  mean = {h0_all['mean']:.2f} +/- {h0_all['std']:.2f}")
        print(f"  median = {h0_all['median']:.2f}")

    print("\nH0 distribution (passing models):")
    h0_pass = summary["H0_distribution_passing"]
    if h0_pass["mean"] is not None:
        print(f"  min = {h0_pass['min']:.2f}, max = {h0_pass['max']:.2f}")
        print(f"  mean = {h0_pass['mean']:.2f} +/- {h0_pass['std']:.2f}")
        print(f"  median = {h0_pass['median']:.2f}")
    else:
        print("  No passing models")

    print(f"\nMaximum H0 among passing models: ", end="")
    if summary["max_H0_among_passing"] is not None:
        print(f"{summary['max_H0_among_passing']:.2f} km/s/Mpc")
    else:
        print("N/A")

    if summary["best_high_h0_model"]:
        print("\nBest model with H0 >= 73:")
        m = summary["best_high_h0_model"]
        print(f"  H0_eff = {m['H0_eff']:.2f} km/s/Mpc")
        print(f"  delta_chi2 = {m['delta_chi2']:.2f}")
        print(f"  chi2 breakdown: CMB={m['chi2_cmb']:.1f}, "
              f"BAO={m['chi2_bao']:.1f}, SN={m['chi2_sn']:.1f}")
        print(f"  Configuration: n_layers={m['n_layers']}, "
              f"smooth_sigma={m['smooth_sigma']}, sigma_delta={m['sigma_delta']}")
    else:
        print("\nNo models achieved H0 >= 73")

    print("\n" + "=" * 70)
    print("VERDICT:")
    if summary["passing_with_H0_ge_73"] > 0:
        print(f"  {summary['passing_with_H0_ge_73']} models achieve H0 >= 73 "
              f"with acceptable chi-squared.")
        print("  Smooth layered expansion *can* potentially reconcile the tension.")
    else:
        print("  NO models achieve H0 >= 73 with acceptable chi-squared.")
        print("  Smooth layered expansion alone CANNOT rescue H0 = 73.")
    print("=" * 70)


def save_results(
    results: List[ScanResult],
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

    # Save all results as JSON
    results_file = output_dir / "scan_results.json"
    results_dicts = [asdict(r) for r in results]
    with open(results_file, "w") as f:
        json.dump(results_dicts, f, indent=1)
    print(f"Saved full results to {results_file}")

    # Save as numpy arrays for easier analysis
    npz_file = output_dir / "scan_results.npz"
    H0_values = np.array([r.H0_eff for r in results])
    chi2_values = np.array([r.chi2_total for r in results])
    delta_chi2_values = np.array([r.delta_chi2 for r in results])
    is_physical = np.array([r.is_physical for r in results])
    passes_fit = np.array([r.passes_fit for r in results])

    np.savez(
        npz_file,
        H0_eff=H0_values,
        chi2_total=chi2_values,
        delta_chi2=delta_chi2_values,
        is_physical=is_physical,
        passes_fit=passes_fit,
    )
    print(f"Saved numpy arrays to {npz_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SIM 24A: Layered Expansion Grid Scan"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of samples per configuration (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-shoes", action="store_true",
        help="Exclude SH0ES H0 prior from SN constraint"
    )
    parser.add_argument(
        "--mode", type=str, default="delta_H", choices=["delta_H", "delta_w"],
        help="Mode: 'delta_H' (direct H modification) or 'delta_w' (w(z) modification)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/simulation_24_layered_grid/)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer samples (for testing)"
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_samples = 10
        n_layers_list = [4]
        smooth_sigma_list = [0.05]
        sigma_delta_list = [0.05]
    else:
        n_samples = args.n_samples
        n_layers_list = [4, 6]
        smooth_sigma_list = [0.02, 0.05]
        sigma_delta_list = [0.02, 0.05, 0.1]

    config = GridScanConfig(
        n_layers_list=n_layers_list,
        smooth_sigma_list=smooth_sigma_list,
        sigma_delta_list=sigma_delta_list,
        n_samples_per_config=n_samples,
        mode=args.mode,
        include_shoes=not args.no_shoes,
        seed=args.seed,
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/simulation_24_layered_grid")

    # Print configuration
    print("\n" + "=" * 70)
    print("SIMULATION 24A: Layered Expansion (Bent-Deck) Grid Scan")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_layers: {n_layers_list}")
    print(f"  smooth_sigma: {smooth_sigma_list}")
    print(f"  sigma_delta: {sigma_delta_list}")
    print(f"  samples per config: {n_samples}")
    print(f"  mode: {config.mode}")
    print(f"  include SH0ES: {config.include_shoes}")
    print(f"  seed: {config.seed}")
    print(f"  output: {output_dir}")
    print()

    # Run the scan
    start_time = time.time()
    results, summary = run_grid_scan(config, verbose=True)
    elapsed = time.time() - start_time

    print(f"\nScan completed in {elapsed:.1f} seconds")

    # Print summary
    print_summary(summary)

    # Save results
    save_results(results, summary, output_dir)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
