#!/usr/bin/env python3
"""
Analyze SIMULATION 14: Rest-Frame Misalignment Bias Results

Reads the scan_results.json from SIM 14 and produces:
1. Summary statistics by (v_true, sky_coverage) combination
2. Distribution of H0 biases
3. Identification of worst-case scenarios
4. Theoretical vs simulation comparison
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
        }

    arr = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "abs_max": float(np.max(np.abs(arr))),
    }


def main():
    """Analyze SIM 14 results and print summary."""
    input_file = Path("results/simulation_14_restframe_misalignment/scan_results.json")

    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run SIM 14 first.")
        return

    with open(input_file, "r") as f:
        results = json.load(f)

    # Get unique parameter values
    v_true_values = sorted(set(r["v_true"] for r in results))
    sky_coverage_values = sorted(set(r["sky_coverage"] for r in results))

    print()
    print("=" * 70)
    print("         SIMULATION 14 SUMMARY: Rest-Frame Misalignment Bias")
    print("=" * 70)
    print()
    print(f"True H0: 67.5 km/s/Mpc")
    print(f"CMB velocity (assumed): 369.82 km/s")
    print(f"Total realizations: {len(results)}")
    print(f"True velocity values: {v_true_values} km/s")
    print(f"Sky coverage scenarios: {sky_coverage_values}")
    print()

    # Compute stats by scenario
    print("-" * 70)
    print("STATISTICS BY SCENARIO (ΔH0 = H0_fit - H0_true)")
    print("-" * 70)
    print(f"{'v_true':<10} {'Sky Coverage':<15} {'N':>5} {'Mean ΔH0':>10} {'Std':>8} {'Max |ΔH0|':>12}")
    print("-" * 70)

    stats_by_scenario = {}

    for v_true in v_true_values:
        for sky in sky_coverage_values:
            subset = [r for r in results
                      if r["v_true"] == v_true and r["sky_coverage"] == sky]
            delta_H0_cmb = [r["delta_H0_cmb"] for r in subset]
            stats = compute_stats(delta_H0_cmb)
            stats_by_scenario[(v_true, sky)] = stats

            print(f"{v_true:<10.0f} {sky:<15} {stats['count']:>5} "
                  f"{stats['mean']:>10.2f} {stats['std']:>8.2f} {stats['abs_max']:>12.2f}")

    print()

    # Frame mismatch bias (difference between CMB fit and true frame fit)
    print("-" * 70)
    print("FRAME MISMATCH BIAS (H0_cmb_fit - H0_true_fit)")
    print("-" * 70)
    print(f"{'v_true':<10} {'Sky Coverage':<15} {'Mean Bias':>12} {'Std':>8} {'Max |Bias|':>12}")
    print("-" * 70)

    for v_true in v_true_values:
        for sky in sky_coverage_values:
            subset = [r for r in results
                      if r["v_true"] == v_true and r["sky_coverage"] == sky]
            bias_values = [r["bias_from_frame"] for r in subset]
            stats = compute_stats(bias_values)

            print(f"{v_true:<10.0f} {sky:<15} {stats['mean']:>12.3f} "
                  f"{stats['std']:>8.3f} {stats['abs_max']:>12.3f}")

    print()

    # Overall statistics
    print("-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)

    all_delta_H0 = [r["delta_H0_cmb"] for r in results]
    all_bias = [r["bias_from_frame"] for r in results]

    overall_stats = compute_stats(all_delta_H0)
    bias_stats = compute_stats(all_bias)

    print(f"Total realizations: {len(results)}")
    print()
    print("ΔH0_cmb (H0_fit - H0_true using CMB correction):")
    print(f"  Mean: {overall_stats['mean']:.2f} km/s/Mpc")
    print(f"  Std:  {overall_stats['std']:.2f} km/s/Mpc")
    print(f"  Max |ΔH0_cmb|: {overall_stats['abs_max']:.2f} km/s/Mpc")
    print()
    print("Frame mismatch bias (H0_cmb - H0_true_frame):")
    print(f"  Mean: {bias_stats['mean']:.3f} km/s/Mpc")
    print(f"  Std:  {bias_stats['std']:.3f} km/s/Mpc")
    print(f"  Max |bias|: {bias_stats['abs_max']:.3f} km/s/Mpc")
    print()

    # Sky coverage dependence
    print("-" * 70)
    print("SKY COVERAGE DEPENDENCE (averaged over v_true)")
    print("-" * 70)

    for sky in sky_coverage_values:
        subset = [r for r in results if r["sky_coverage"] == sky]
        bias_values = [r["bias_from_frame"] for r in subset]
        delta_values = [r["delta_H0_cmb"] for r in subset]

        bias_stats = compute_stats(bias_values)
        delta_stats = compute_stats(delta_values)

        print(f"{sky}:")
        print(f"  Mean frame mismatch bias: {bias_stats['mean']:+.3f} km/s/Mpc")
        print(f"  Mean ΔH0_cmb: {delta_stats['mean']:+.2f} km/s/Mpc")
        print(f"  Max |ΔH0_cmb|: {delta_stats['abs_max']:.2f} km/s/Mpc")
        print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Get isotropic results
    iso_results = [r for r in results if r["sky_coverage"] == "isotropic"]
    iso_bias = [r["bias_from_frame"] for r in iso_results]
    iso_bias_stats = compute_stats(iso_bias)

    # Get hemispherical results
    toward_results = [r for r in results if r["sky_coverage"] == "toward_apex"]
    away_results = [r for r in results if r["sky_coverage"] == "away_from_apex"]

    toward_bias = [r["bias_from_frame"] for r in toward_results]
    away_bias = [r["bias_from_frame"] for r in away_results]

    toward_stats = compute_stats(toward_bias)
    away_stats = compute_stats(away_bias)

    print("Key findings:")
    print()
    print("1. ISOTROPIC SKY COVERAGE:")
    print(f"   Mean frame bias: {iso_bias_stats['mean']:+.3f} km/s/Mpc")
    if abs(iso_bias_stats['mean']) < 0.5:
        print("   -> Dipole averages out; frame mismatch has NEGLIGIBLE effect")
    else:
        print("   -> Unexpectedly large bias; check simulation")
    print()

    print("2. HEMISPHERICAL COVERAGE:")
    print(f"   Toward apex: {toward_stats['mean']:+.2f} km/s/Mpc mean bias")
    print(f"   Away from apex: {away_stats['mean']:+.2f} km/s/Mpc mean bias")

    if abs(toward_stats['mean'] - away_stats['mean']) > 0.5:
        print("   -> Significant asymmetry detected from incomplete sky coverage")
    else:
        print("   -> Modest asymmetry from incomplete sky coverage")
    print()

    print("3. OVERALL CONCLUSION:")
    max_frame_bias = max(abs(r["bias_from_frame"]) for r in results)
    max_iso_bias = max(abs(b) for b in iso_bias) if iso_bias else 0

    if max_iso_bias < 0.5:
        print("   For ISOTROPIC samples, rest-frame misalignment bias is <0.5 km/s/Mpc")
        print("   This is MUCH SMALLER than the ~5 km/s/Mpc Hubble tension")
    else:
        print(f"   Maximum isotropic bias: {max_iso_bias:.2f} km/s/Mpc")

    if max_frame_bias > 1.0:
        print(f"   WARNING: Hemispherical samples can have biases up to {max_frame_bias:.2f} km/s/Mpc")
    print()

    print("=" * 70)

    # Save machine-readable summary
    summary = {
        "stats_by_scenario": {
            f"v{v}_sky{s}": stats_by_scenario[(v, s)]
            for v, s in stats_by_scenario
        },
        "overall": {
            "delta_H0_cmb": overall_stats,
            "frame_bias": bias_stats,
        },
        "by_sky_coverage": {
            "isotropic": iso_bias_stats,
            "toward_apex": toward_stats,
            "away_from_apex": away_stats,
        },
        "total_realizations": len(results),
        "v_true_values": v_true_values,
        "sky_coverage_values": sky_coverage_values,
    }

    summary_file = Path("results/simulation_14_restframe_misalignment/summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    main()
