#!/usr/bin/env python3
"""
Analyze SIMULATION 13: HST vs JWST Cepheid Recalibration Results

Reads the scan_results.json from SIM 13 and produces:
1. Summary statistics for instrument-induced H0 shifts
2. Classification by realistic vs moderate prior regions
3. Identification of which instrument systematics drive H0 shifts
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


# =============================================================================
# Prior Definitions
# =============================================================================

def is_inst_realistic(r: Dict[str, Any]) -> bool:
    """
    Realistic (conservative) instrument systematic region.

    Based on actual HST vs JWST calibration precision.
    """
    return (
        abs(r["zp_diff"]) <= 0.02 and        # ≤ 2% zero-point (≤1% distance)
        abs(r["c_color_diff"]) <= 0.03 and   # ≤ 0.03 mag/mag color term
        abs(r["c_nl_diff"]) <= 0.01          # ≤ 0.01 non-linearity
    )


def is_inst_moderate(r: Dict[str, Any]) -> bool:
    """
    Moderate (upper bound) instrument systematic region.

    Allows larger but still plausible systematics.
    """
    return (
        abs(r["zp_diff"]) <= 0.04 and        # ≤ 4% zero-point (≤2% distance)
        abs(r["c_color_diff"]) <= 0.06 and   # ≤ 0.06 mag/mag color term
        abs(r["c_nl_diff"]) <= 0.02          # ≤ 0.02 non-linearity
    )


def compute_stats(results: List[Dict], filter_fn=None, key: str = "delta_H0_inst"):
    """Compute statistics for a filtered subset of results."""
    if filter_fn:
        filtered = [r for r in results if filter_fn(r)]
    else:
        filtered = results

    if not filtered:
        return {
            "count": 0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    values = [abs(r[key]) for r in filtered]
    return {
        "count": len(filtered),
        "max": float(max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def count_above_threshold(results: List[Dict], threshold: float, filter_fn=None):
    """Count results with |delta_H0_inst| >= threshold."""
    if filter_fn:
        filtered = [r for r in results if filter_fn(r)]
    else:
        filtered = results
    return sum(1 for r in filtered if abs(r["delta_H0_inst"]) >= threshold)


def main():
    """Analyze SIM 13 results and print summary."""
    input_file = Path("results/simulation_13_jwst_hst_recalibration/scan_results.json")

    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run SIM 13 first.")
        return

    with open(input_file, "r") as f:
        results = json.load(f)

    # Compute statistics for different subsets
    stats = {
        "all": {
            "name": "All scenarios",
            **compute_stats(results),
        },
        "realistic": {
            "name": "Realistic (conservative)",
            **compute_stats(results, is_inst_realistic),
        },
        "moderate": {
            "name": "Moderate (upper bound)",
            **compute_stats(results, is_inst_moderate),
        },
    }

    # Count thresholds
    counts = {}
    for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:
        counts[f"all_ge_{threshold}"] = count_above_threshold(results, threshold)
        counts[f"realistic_ge_{threshold}"] = count_above_threshold(results, threshold, is_inst_realistic)
        counts[f"moderate_ge_{threshold}"] = count_above_threshold(results, threshold, is_inst_moderate)

    # Find top 5 largest |delta_H0_inst| (positive direction, JWST higher)
    sorted_positive = sorted(
        [r for r in results if r["delta_H0_inst"] > 0],
        key=lambda r: r["delta_H0_inst"],
        reverse=True
    )[:5]

    # Find top 5 largest negative (HST higher)
    sorted_negative = sorted(
        [r for r in results if r["delta_H0_inst"] < 0],
        key=lambda r: r["delta_H0_inst"],
    )[:5]

    # Print summary
    print()
    print("=" * 70)
    print("         SIMULATION 13 SUMMARY: HST vs JWST Recalibration")
    print("=" * 70)
    print()
    print(f"True H0: 67.5 km/s/Mpc")
    print(f"Total instrument scenarios: {len(results)}")
    print()
    print("-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"{'Category':<30} {'Count':>8} {'Max |dH0|':>10} {'Mean |dH0|':>10} {'Median':>10}")
    print("-" * 70)
    for key, s in stats.items():
        print(f"{s['name']:<30} {s['count']:>8} {s['max']:>10.2f} {s['mean']:>10.2f} {s['median']:>10.2f}")
    print()

    print("-" * 70)
    print("THRESHOLD COUNTS: |ΔH0_inst| ≥ X km/s/Mpc")
    print("-" * 70)
    print(f"{'Threshold':<15} {'All':>10} {'Realistic':>12} {'Moderate':>10}")
    print("-" * 70)
    for thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        all_c = counts[f"all_ge_{thresh}"]
        real_c = counts[f"realistic_ge_{thresh}"]
        mod_c = counts[f"moderate_ge_{thresh}"]
        print(f"≥ {thresh:.1f} km/s/Mpc  {all_c:>10} {real_c:>12} {mod_c:>10}")
    print()

    print("-" * 70)
    print("TOP 5 POSITIVE ΔH0_inst (JWST gives HIGHER H0 than HST)")
    print("-" * 70)
    for i, r in enumerate(sorted_positive, 1):
        print(f"  {i}. ΔH0_inst = +{r['delta_H0_inst']:.2f} km/s/Mpc")
        print(f"     zp_diff={r['zp_diff']:.3f}, c_color={r['c_color_diff']:.3f}, c_nl={r['c_nl_diff']:.3f}")
        print(f"     H0_HST={r['H0_fit_hst']:.1f}, H0_JWST={r['H0_fit_jwst']:.1f}")
    print()

    print("-" * 70)
    print("TOP 5 NEGATIVE ΔH0_inst (JWST gives LOWER H0 than HST)")
    print("-" * 70)
    for i, r in enumerate(sorted_negative, 1):
        print(f"  {i}. ΔH0_inst = {r['delta_H0_inst']:.2f} km/s/Mpc")
        print(f"     zp_diff={r['zp_diff']:.3f}, c_color={r['c_color_diff']:.3f}, c_nl={r['c_nl_diff']:.3f}")
        print(f"     H0_HST={r['H0_fit_hst']:.1f}, H0_JWST={r['H0_fit_jwst']:.1f}")
    print()

    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    max_realistic = stats["realistic"]["max"]
    max_moderate = stats["moderate"]["max"]
    max_all = stats["all"]["max"]

    print(f"Maximum |ΔH0_inst| in realistic region: {max_realistic:.2f} km/s/Mpc")
    print(f"Maximum |ΔH0_inst| in moderate region:  {max_moderate:.2f} km/s/Mpc")
    print(f"Maximum |ΔH0_inst| overall:             {max_all:.2f} km/s/Mpc")
    print()
    print("Key findings:")
    if max_realistic >= 2.0:
        print(f"  - Realistic HST-JWST systematics CAN produce ≥2 km/s/Mpc H0 shift")
    else:
        print(f"  - Realistic HST-JWST systematics produce <2 km/s/Mpc H0 shift")

    if max_moderate >= 3.0:
        print(f"  - Moderate systematics CAN produce ≥3 km/s/Mpc H0 shift")
    else:
        print(f"  - Even moderate systematics cannot reach 3 km/s/Mpc shift")

    if counts["realistic_ge_1.0"] > 0:
        print(f"  - {counts['realistic_ge_1.0']} realistic scenarios show ≥1 km/s/Mpc shift")
    print()
    print("=" * 70)

    # Save machine-readable summary
    summary = {
        "stats": stats,
        "counts": counts,
        "top_5_positive": sorted_positive,
        "top_5_negative": sorted_negative,
        "total_scenarios": len(results),
    }

    summary_file = Path("results/simulation_13_jwst_hst_recalibration/summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    main()
