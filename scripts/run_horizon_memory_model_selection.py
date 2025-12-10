#!/usr/bin/env python3
"""Horizon-Memory Refinement Model Selection.

This script runs all four refinement pathways (T06A-D) with 2D parameter scans,
evaluates full viability, and identifies the best models for addressing the
CMB distance tension while maintaining Hubble tension relief.

Usage:
    python scripts/run_horizon_memory_model_selection.py [--quick] [--resume]

Arguments:
    --quick: Run smaller 10x10 grids instead of 20x20
    --resume: Resume from partial results if available
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.horizon_models import (
    HorizonMemoryParameters,
    RefinementType,
    AdaptiveMemoryKernel,
    TwoComponentMemory,
    EarlyTimeSuppression,
    DynamicalEoSModifier,
)
from hrc2.horizon_models.refinement_a import create_adaptive_kernel_model
from hrc2.horizon_models.refinement_b import create_two_component_model
from hrc2.horizon_models.refinement_c import create_early_suppression_model
from hrc2.horizon_models.refinement_d import create_dynamical_eos_model
from hrc2.analysis.horizon_memory_comparator import (
    HorizonMemoryComparator,
    ModelViability,
    compare_horizon_memory_refinements,
)


# Default scan configurations
SCAN_CONFIGS = {
    "T06A": {
        "param1_name": "tau0",
        "param1_range": (0.01, 0.5),
        "param2_name": "p_hor",
        "param2_range": (-1.0, 3.0),
        "fixed": {"lambda_hor": 0.2},
    },
    "T06B": {
        "param1_name": "tau1",
        "param1_range": (0.01, 0.2),
        "param2_name": "tau2",
        "param2_range": (0.1, 1.0),
        "fixed": {"lambda1": 0.15, "lambda2": 0.05},
    },
    "T06C": {
        "param1_name": "a_supp",
        "param1_range": (0.001, 0.1),
        "param1_log": True,  # Use log spacing
        "param2_name": "n_supp",
        "param2_range": (1.0, 5.0),
        "fixed": {"lambda_hor": 0.2, "tau_hor": 0.1},
    },
    "T06D": {
        "param1_name": "delta_w",
        "param1_range": (-0.5, 0.1),
        "param2_name": "a_w",
        "param2_range": (0.1, 0.5),
        "fixed": {"m_eos": 2.0, "lambda_hor": 0.2, "tau_hor": 0.1},
    },
}


def evaluate_model_point(args: Tuple) -> Dict[str, Any]:
    """Evaluate a single model point in parameter space.

    Args:
        args: Tuple of (refinement, param1, param2, param1_name, param2_name, fixed_params, z_max)

    Returns:
        Dictionary with model results
    """
    refinement, p1, p2, p1_name, p2_name, fixed, z_max = args

    try:
        # Create model based on refinement type
        params = {p1_name: p1, p2_name: p2, **fixed}

        if refinement == "T06A":
            model = create_adaptive_kernel_model(**params)
        elif refinement == "T06B":
            model = create_two_component_model(**params)
        elif refinement == "T06C":
            model = create_early_suppression_model(**params)
        elif refinement == "T06D":
            model = create_dynamical_eos_model(**params)
        else:
            return {"success": False, "error": f"Unknown refinement: {refinement}"}

        # Solve model
        result = model.solve(z_max=z_max)

        if not result.success:
            return {
                "success": False,
                "error": result.message,
                p1_name: p1,
                p2_name: p2,
            }

        return {
            "success": True,
            p1_name: p1,
            p2_name: p2,
            "delta_H0_percent": result.delta_H0_frac * 100,  # Always ~0 with self-consistent Lambda
            "late_time_H_effect": result.late_time_H_effect,  # H deviation at z=0.5 (%)
            "sn_distance_deviation": result.sn_distance_deviation,  # D_L deviation at z=0.5 (%)
            "cmb_deviation_percent": result.cmb_distance_deviation,  # D_A deviation at z*
            "Omega_hor0": result.Omega_hor0,  # Horizon memory density fraction
            "Omega_L0_eff": result.Omega_L0_eff,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            p1_name: p1,
            p2_name: p2,
        }


def run_refinement_scan(
    refinement: str,
    n_grid: int = 20,
    z_max: float = 1200.0,
    n_workers: int = None,
    output_dir: str = "results/tests",
    resume: bool = False,
) -> Dict[str, Any]:
    """Run 2D parameter scan for a single refinement.

    Args:
        refinement: Refinement name (T06A, T06B, T06C, T06D)
        n_grid: Grid size for each parameter
        z_max: Maximum redshift for integration
        n_workers: Number of parallel workers (default: CPU count - 1)
        output_dir: Directory for output files
        resume: Whether to resume from partial results

    Returns:
        Dictionary with scan results
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    config = SCAN_CONFIGS[refinement]

    # Setup parameter grids
    p1_name = config["param1_name"]
    p2_name = config["param2_name"]
    p1_range = config["param1_range"]
    p2_range = config["param2_range"]
    fixed = config["fixed"]

    # Use log spacing for certain parameters
    if config.get("param1_log", False):
        p1_vals = np.logspace(np.log10(p1_range[0]), np.log10(p1_range[1]), n_grid)
    else:
        p1_vals = np.linspace(p1_range[0], p1_range[1], n_grid)

    p2_vals = np.linspace(p2_range[0], p2_range[1], n_grid)

    # Output path
    scan_dir = os.path.join(output_dir, f"{refinement}_scan")
    os.makedirs(scan_dir, exist_ok=True)

    # Check for partial results
    partial_path = os.path.join(scan_dir, "partial_results.npz")
    if resume and os.path.exists(partial_path):
        print(f"  Resuming from {partial_path}")
        partial = np.load(partial_path)
        delta_H0 = partial["delta_H0_percent"].copy()
        cmb_dev = partial["cmb_deviation_percent"].copy()
        computed = partial["computed"].copy()
    else:
        delta_H0 = np.full((n_grid, n_grid), np.nan)
        omega_hor0 = np.full((n_grid, n_grid), np.nan)
        sn_dev = np.full((n_grid, n_grid), np.nan)
        cmb_dev = np.full((n_grid, n_grid), np.nan)
        computed = np.zeros((n_grid, n_grid), dtype=bool)

    # Build task list
    tasks = []
    for i, p1 in enumerate(p1_vals):
        for j, p2 in enumerate(p2_vals):
            if computed[i, j]:
                continue
            tasks.append((refinement, p1, p2, p1_name, p2_name, fixed, z_max))

    if len(tasks) == 0:
        print(f"  {refinement}: All points already computed")
    else:
        print(f"  {refinement}: Computing {len(tasks)} points with {n_workers} workers...")

        # Run in parallel with simple progress indicator
        completed = 0
        with Pool(n_workers) as pool:
            results_iter = pool.imap_unordered(evaluate_model_point, tasks)

            for result in results_iter:
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"\r    {refinement}: {completed}/{len(tasks)} ", end="", flush=True)

                p1 = result[p1_name]
                p2 = result[p2_name]

                # Find indices
                i = np.argmin(np.abs(p1_vals - p1))
                j = np.argmin(np.abs(p2_vals - p2))

                if result["success"]:
                    delta_H0[i, j] = result["delta_H0_percent"]
                    omega_hor0[i, j] = result["Omega_hor0"] * 100  # Convert to percent
                    sn_dev[i, j] = result["sn_distance_deviation"]
                    cmb_dev[i, j] = result["cmb_deviation_percent"]

                computed[i, j] = True

                # Save partial results periodically
                if np.sum(computed) % 50 == 0:
                    np.savez(
                        partial_path,
                        delta_H0_percent=delta_H0,
                        cmb_deviation_percent=cmb_dev,
                        computed=computed,
                        **{p1_name + "_vals": p1_vals, p2_name + "_vals": p2_vals},
                    )

    # Save final results
    final_path = os.path.join(scan_dir, "scan.npz")
    np.savez(
        final_path,
        refinement=refinement,
        delta_H0_percent=delta_H0,
        omega_hor0_percent=omega_hor0,
        sn_deviation_percent=sn_dev,
        cmb_deviation_percent=cmb_dev,
        success_mask=~np.isnan(cmb_dev),
        **{p1_name + "_vals": p1_vals, p2_name + "_vals": p2_vals},
        **fixed,
    )

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Find best model
    valid = ~np.isnan(cmb_dev) & ~np.isnan(omega_hor0)

    if not np.any(valid):
        best = None
    else:
        # Prioritize low CMB deviation while requiring meaningful Omega_hor0
        omega_min_threshold = 5.0  # Minimum 5% horizon memory contribution

        # Mask: valid AND Omega_hor0 > threshold
        viable = valid & (omega_hor0 >= omega_min_threshold)

        if np.any(viable):
            # Find minimum CMB deviation among viable
            cmb_masked = np.where(viable, cmb_dev, np.inf)
            best_idx = np.unravel_index(np.argmin(cmb_masked), cmb_masked.shape)
        else:
            # Fallback: find best CMB among all valid
            cmb_masked = np.where(valid, cmb_dev, np.inf)
            best_idx = np.unravel_index(np.argmin(cmb_masked), cmb_masked.shape)

        best = {
            "model_id": f"{refinement}_{best_idx[0]}_{best_idx[1]}",
            p1_name: float(p1_vals[best_idx[0]]),
            p2_name: float(p2_vals[best_idx[1]]),
            "omega_hor0_percent": float(omega_hor0[best_idx]),
            "sn_deviation_percent": float(sn_dev[best_idx]),
            "cmb_deviation_percent": float(cmb_dev[best_idx]),
        }

    # Save summary JSON
    summary = {
        "refinement": refinement,
        "n_grid": n_grid,
        "n_valid": int(np.sum(valid)),
        "n_total": n_grid * n_grid,
        p1_name + "_range": list(p1_range),
        p2_name + "_range": list(p2_range),
        "fixed_params": fixed,
        "best_model": best,
        "statistics": {
            "omega_hor0_min": float(np.nanmin(omega_hor0)) if np.any(valid) else None,
            "omega_hor0_max": float(np.nanmax(omega_hor0)) if np.any(valid) else None,
            "sn_dev_min": float(np.nanmin(sn_dev)) if np.any(valid) else None,
            "sn_dev_max": float(np.nanmax(sn_dev)) if np.any(valid) else None,
            "cmb_dev_min": float(np.nanmin(cmb_dev)) if np.any(valid) else None,
            "cmb_dev_max": float(np.nanmax(cmb_dev)) if np.any(valid) else None,
        }
    }

    summary_path = os.path.join(scan_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_scan_plots(
    refinement: str,
    output_dir: str = "results/tests",
    figures_dir: str = "figures/tests",
) -> None:
    """Generate diagnostic plots for a refinement scan.

    Args:
        refinement: Refinement name
        output_dir: Directory with scan results
        figures_dir: Directory for figures
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scan_path = os.path.join(output_dir, f"{refinement}_scan", "scan.npz")
    if not os.path.exists(scan_path):
        print(f"  No scan found for {refinement}")
        return

    data = np.load(scan_path)
    config = SCAN_CONFIGS[refinement]

    p1_name = config["param1_name"]
    p2_name = config["param2_name"]
    p1_vals = data[p1_name + "_vals"]
    p2_vals = data[p2_name + "_vals"]
    delta_H0 = data["delta_H0_percent"]
    cmb_dev = data["cmb_deviation_percent"]

    fig_dir = os.path.join(figures_dir, f"{refinement}_scan")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: Delta H0 map
    fig, ax = plt.subplots(figsize=(10, 8))

    if config.get("param1_log", False):
        P1, P2 = np.meshgrid(p1_vals, p2_vals, indexing='ij')
        im = ax.pcolormesh(P1, P2, delta_H0, shading='auto', cmap='RdYlBu_r')
        ax.set_xscale('log')
    else:
        im = ax.imshow(
            delta_H0.T, origin='lower', aspect='auto',
            extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]],
            cmap='RdYlBu_r'
        )

    plt.colorbar(im, ax=ax, label='ΔH0 (%)')
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_title(f'{refinement}: H0 Effect (% shift)')

    # Mark best point
    valid = ~np.isnan(cmb_dev)
    if np.any(valid):
        # Find best CMB point with reasonable H0 effect
        viable = valid & (delta_H0 >= 2.0)
        if np.any(viable):
            cmb_masked = np.where(viable, cmb_dev, np.inf)
            best_idx = np.unravel_index(np.argmin(cmb_masked), cmb_masked.shape)
            ax.plot(p1_vals[best_idx[0]], p2_vals[best_idx[1]], 'k*', markersize=15, label='Best')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "delta_H0_map.png"), dpi=150)
    plt.close()

    # Plot 2: CMB deviation map
    fig, ax = plt.subplots(figsize=(10, 8))

    if config.get("param1_log", False):
        im = ax.pcolormesh(P1, P2, cmb_dev, shading='auto', cmap='RdYlGn_r', vmin=0, vmax=2)
        ax.set_xscale('log')
    else:
        im = ax.imshow(
            cmb_dev.T, origin='lower', aspect='auto',
            extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]],
            cmap='RdYlGn_r', vmin=0, vmax=2
        )

    plt.colorbar(im, ax=ax, label='CMB D_A deviation (%)')
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_title(f'{refinement}: CMB Distance Deviation (%)')

    # Mark target region (< 0.3%)
    ax.contour(
        p1_vals, p2_vals, cmb_dev.T,
        levels=[0.3, 0.5, 1.0],
        colors=['green', 'yellow', 'orange'],
        linewidths=2
    )

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cmb_deviation_map.png"), dpi=150)
    plt.close()

    # Plot 3: Combined viability map
    fig, ax = plt.subplots(figsize=(10, 8))

    # Viability score: low CMB deviation + high H0 effect
    # Score = H0_effect / (1 + CMB_dev)
    viability = delta_H0 / (1.0 + cmb_dev)

    if config.get("param1_log", False):
        im = ax.pcolormesh(P1, P2, viability, shading='auto', cmap='viridis')
        ax.set_xscale('log')
    else:
        im = ax.imshow(
            viability.T, origin='lower', aspect='auto',
            extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]],
            cmap='viridis'
        )

    plt.colorbar(im, ax=ax, label='Viability Score')
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_title(f'{refinement}: Viability Score (ΔH0 / (1 + CMB_dev))')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "viability_map.png"), dpi=150)
    plt.close()

    print(f"  {refinement}: Plots saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run horizon-memory refinement model selection"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use smaller 10x10 grids for quick testing"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial results if available"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--refinements", type=str, nargs="+",
        default=["T06A", "T06B", "T06C", "T06D"],
        help="Which refinements to run"
    )

    args = parser.parse_args()

    n_grid = 10 if args.quick else 20
    z_max = 1200.0
    output_dir = "results/tests"
    figures_dir = "figures/tests"

    print("=" * 80)
    print("HORIZON-MEMORY REFINEMENT MODEL SELECTION")
    print("=" * 80)
    print(f"\nGrid size: {n_grid}x{n_grid}")
    print(f"Refinements: {args.refinements}")
    print(f"Resume: {args.resume}")
    print(f"z_max: {z_max}")
    print()

    start_time = time.time()

    # Run scans for each refinement
    all_summaries = {}

    for refinement in args.refinements:
        print(f"\n{'='*60}")
        print(f"Running {refinement} scan...")
        print(f"{'='*60}")

        summary = run_refinement_scan(
            refinement=refinement,
            n_grid=n_grid,
            z_max=z_max,
            n_workers=args.workers,
            output_dir=output_dir,
            resume=args.resume,
        )
        all_summaries[refinement] = summary

        # Generate plots
        print(f"\n  Generating plots for {refinement}...")
        generate_scan_plots(refinement, output_dir, figures_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("SCAN RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Refinement':<12} {'Valid':<10} {'Ω_hor0 (%)':<12} {'SN D_L (%)':<12} {'CMB D_A (%)':<12} {'Best Model'}")
    print("-" * 90)

    for ref in args.refinements:
        summary = all_summaries[ref]
        best = summary.get("best_model", {})
        if best:
            omega = best.get('omega_hor0_percent', 0)
            sn = best.get('sn_deviation_percent', 0)
            cmb = best.get('cmb_deviation_percent', 0)
            print(f"{ref:<12} {summary['n_valid']:>5}/{summary['n_total']:<4} "
                  f"{omega:>10.2f}   "
                  f"{sn:>10.4f}   "
                  f"{cmb:>10.4f}   "
                  f"{best['model_id']}")
        else:
            print(f"{ref:<12} {summary['n_valid']:>5}/{summary['n_total']:<4}   No valid models")

    # Save combined summary
    combined_path = os.path.join(output_dir, "T06_refinement_selection", "combined_summary.json")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)

    combined = {
        "timestamp": datetime.now().isoformat(),
        "n_grid": n_grid,
        "z_max": z_max,
        "refinements": all_summaries,
    }

    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined summary saved to {combined_path}")

    # Find global best
    best_models = []
    for ref, summary in all_summaries.items():
        if summary.get("best_model"):
            best_models.append((ref, summary["best_model"]))

    if best_models:
        # Sort by CMB deviation (lower is better) among models with Omega_hor0 > 5%
        viable = [
            (ref, m) for ref, m in best_models
            if m.get("omega_hor0_percent", 0) >= 5.0
        ]

        if viable:
            viable.sort(key=lambda x: x[1]["cmb_deviation_percent"])
            best_ref, best = viable[0]
        else:
            # Fallback: sort by CMB deviation
            best_models.sort(key=lambda x: x[1]["cmb_deviation_percent"])
            best_ref, best = best_models[0]

        print("\n" + "=" * 80)
        print("GLOBAL BEST MODEL")
        print("=" * 80)
        print(f"  Refinement: {best_ref}")
        print(f"  Model ID: {best['model_id']}")
        for k, v in best.items():
            if k != "model_id":
                print(f"  {k}: {v}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
