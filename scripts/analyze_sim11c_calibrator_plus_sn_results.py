#!/usr/bin/env python3
"""
Analyze SIMULATION 11C results: Combined Calibrator + SN Ia Systematics.

Defines "realistic" and "moderate" priors for both SN systematics and
calibrator biases, then computes summary statistics.
"""

import json
import numpy as np
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load scan results from JSON file."""
    results_file = results_dir / "scan_results.json"
    with open(results_file, "r") as f:
        return json.load(f)


# =============================================================================
# Classification functions
# =============================================================================

def is_sn_realistic(r: dict) -> bool:
    """Check if SN systematics are in 'realistic' range."""
    return (
        abs(r.get("alpha_pop", 0.0)) <= 0.05 and
        abs(r.get("gamma_Z", 0.0)) <= 0.05 and
        abs(r.get("delta_step_true", 0.0)) <= 0.05 and
        abs(r.get("delta_beta", 0.0)) <= 0.3
    )


def is_sn_moderate(r: dict) -> bool:
    """Check if SN systematics are in 'moderate' range."""
    return (
        abs(r.get("alpha_pop", 0.0)) <= 0.10 and
        abs(r.get("gamma_Z", 0.0)) <= 0.10 and
        abs(r.get("delta_step_true", 0.0)) <= 0.10 and
        abs(r.get("delta_beta", 0.0)) <= 0.5
    )


def is_calib_realistic(r: dict) -> bool:
    """Check if calibrator biases are in 'realistic' range."""
    return (
        abs(r.get("delta_mu_global", 0.0)) <= 0.02 and  # <= 1% distance
        abs(r.get("k_mu_Z", 0.0)) <= 0.03 and          # modest metallicity
        abs(r.get("delta_mu_crowd", 0.0)) <= 0.03      # small crowding
    )


def is_calib_moderate(r: dict) -> bool:
    """Check if calibrator biases are in 'moderate' range."""
    return (
        abs(r.get("delta_mu_global", 0.0)) <= 0.04 and  # <= 2% distance
        abs(r.get("k_mu_Z", 0.0)) <= 0.06 and
        abs(r.get("delta_mu_crowd", 0.0)) <= 0.06
    )


def is_realistic(r: dict) -> bool:
    """Combined realistic region: both SN and calibrator systematics modest."""
    return is_sn_realistic(r) and is_calib_realistic(r)


def is_moderate(r: dict) -> bool:
    """Combined moderate region."""
    return is_sn_moderate(r) and is_calib_moderate(r)


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(results: dict) -> dict:
    """Analyze simulation results and compute summary statistics."""
    scenarios = results.get("results", [])

    # Extract delta_H0 arrays for different subsets
    all_delta_H0 = []
    realistic_delta_H0 = []
    moderate_delta_H0 = []
    sn_only_realistic = []
    calib_only_realistic = []

    for r in scenarios:
        delta_H0 = abs(r["delta_H0"])
        all_delta_H0.append(delta_H0)

        if is_realistic(r):
            realistic_delta_H0.append(delta_H0)

        if is_moderate(r):
            moderate_delta_H0.append(delta_H0)

        if is_sn_realistic(r):
            sn_only_realistic.append(delta_H0)

        if is_calib_realistic(r):
            calib_only_realistic.append(delta_H0)

    all_delta_H0 = np.array(all_delta_H0)
    realistic_delta_H0 = np.array(realistic_delta_H0)
    moderate_delta_H0 = np.array(moderate_delta_H0)
    sn_only_realistic = np.array(sn_only_realistic)
    calib_only_realistic = np.array(calib_only_realistic)

    # Compute statistics
    def compute_stats(arr, name):
        if len(arr) == 0:
            return {
                "name": name,
                "count": 0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }
        return {
            "name": name,
            "count": len(arr),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    stats = {
        "all": compute_stats(all_delta_H0, "All scenarios"),
        "realistic": compute_stats(realistic_delta_H0, "Realistic (SN + calib)"),
        "moderate": compute_stats(moderate_delta_H0, "Moderate (SN + calib)"),
        "sn_realistic": compute_stats(sn_only_realistic, "SN realistic only"),
        "calib_realistic": compute_stats(calib_only_realistic, "Calib realistic only"),
    }

    # Count scenarios exceeding thresholds
    thresholds = [2.0, 3.0, 4.0, 5.0, 6.0]
    counts = {}
    for thresh in thresholds:
        counts[f"all_ge_{thresh}"] = int(np.sum(all_delta_H0 >= thresh))
        counts[f"realistic_ge_{thresh}"] = int(np.sum(realistic_delta_H0 >= thresh))
        counts[f"moderate_ge_{thresh}"] = int(np.sum(moderate_delta_H0 >= thresh))

    # Find top contributors (positive and negative bias)
    sorted_by_bias = sorted(scenarios, key=lambda x: x["delta_H0"], reverse=True)

    top_5_positive = []
    for s in sorted_by_bias[:5]:
        top_5_positive.append({
            "alpha_pop": s.get("alpha_pop", 0.0),
            "gamma_Z": s.get("gamma_Z", 0.0),
            "delta_step": s.get("delta_step_true", 0.0),
            "delta_beta": s.get("delta_beta", 0.0),
            "delta_mu_global": s.get("delta_mu_global", 0.0),
            "k_mu_Z": s.get("k_mu_Z", 0.0),
            "delta_mu_crowd": s.get("delta_mu_crowd", 0.0),
            "delta_H0": s["delta_H0"],
            "H0_fit": s["H0_fit"],
        })

    top_5_negative = []
    for s in sorted_by_bias[-5:]:
        top_5_negative.append({
            "alpha_pop": s.get("alpha_pop", 0.0),
            "gamma_Z": s.get("gamma_Z", 0.0),
            "delta_step": s.get("delta_step_true", 0.0),
            "delta_beta": s.get("delta_beta", 0.0),
            "delta_mu_global": s.get("delta_mu_global", 0.0),
            "k_mu_Z": s.get("k_mu_Z", 0.0),
            "delta_mu_crowd": s.get("delta_mu_crowd", 0.0),
            "delta_H0": s["delta_H0"],
            "H0_fit": s["H0_fit"],
        })

    return {
        "stats": stats,
        "counts": counts,
        "top_5_positive": top_5_positive,
        "top_5_negative": top_5_negative,
        "total_scenarios": len(scenarios),
    }


def print_summary(analysis: dict, results: dict):
    """Print formatted SIMULATION 11C SUMMARY block."""
    stats = analysis["stats"]
    counts = analysis["counts"]
    top_pos = analysis["top_5_positive"]
    top_neg = analysis["top_5_negative"]

    true_H0 = results.get("true_cosmology", {}).get("H0", 67.5)

    print("\n" + "=" * 75)
    print("SIMULATION 11C SUMMARY: Combined Calibrator + SN Ia Systematics")
    print("=" * 75)

    print(f"\nTrue cosmology: H0 = {true_H0:.1f} km/s/Mpc")
    print(f"Total scenarios scanned: {analysis['total_scenarios']}")

    print("\n" + "-" * 75)
    print("PARAMETER SPACE COVERAGE")
    print("-" * 75)
    print("SN Ia Systematics:")
    print(f"  α_pop (population drift):    [0.0, 0.05, 0.10]")
    print(f"  γ_Z (metallicity effect):    [0.0, 0.05, 0.10]")
    print(f"  ΔM_step (host mass step):    [0.0, 0.05, 0.10]")
    print(f"  Δβ (color law mismatch):     [0.0, 0.3, 0.5]")
    print("\nCalibrator Biases:")
    print(f"  δμ_global (zero-point):      [0.0, 0.02, 0.04] mag")
    print(f"  k_μZ (metallicity dep):      [0.0, 0.03, 0.06] mag/dex")
    print(f"  δμ_crowd (crowding/blend):   [0.0, 0.03, 0.06] mag")

    print("\n" + "-" * 75)
    print("REGION DEFINITIONS")
    print("-" * 75)
    print("Realistic: SN(α_pop, γ_Z, ΔM_step ≤ 0.05, Δβ ≤ 0.3) AND")
    print("           Calib(δμ_glob ≤ 0.02, k_μZ ≤ 0.03, δμ_crowd ≤ 0.03)")
    print("Moderate:  SN(α_pop, γ_Z, ΔM_step ≤ 0.10, Δβ ≤ 0.5) AND")
    print("           Calib(δμ_glob ≤ 0.04, k_μZ ≤ 0.06, δμ_crowd ≤ 0.06)")

    print("\n" + "-" * 75)
    print("MAX |ΔH0| BY REGION")
    print("-" * 75)
    print(f"  Overall (all {stats['all']['count']} scenarios):".ljust(45) +
          f"{stats['all']['max']:.2f} km/s/Mpc")
    print(f"  Realistic region ({stats['realistic']['count']} scenarios):".ljust(45) +
          f"{stats['realistic']['max']:.2f} km/s/Mpc")
    print(f"  Moderate region ({stats['moderate']['count']} scenarios):".ljust(45) +
          f"{stats['moderate']['max']:.2f} km/s/Mpc")

    print("\n" + "-" * 75)
    print("DISTRIBUTION STATISTICS")
    print("-" * 75)
    print(f"{'Region':<25} {'Count':>8} {'Max':>8} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("-" * 70)
    for key in ["all", "realistic", "moderate"]:
        s = stats[key]
        print(f"{s['name']:<25} {s['count']:>8} {s['max']:>8.2f} "
              f"{s['mean']:>8.2f} {s['median']:>8.2f} {s['std']:>8.2f}")

    print("\n" + "-" * 75)
    print("SCENARIO COUNTS BY |ΔH0| THRESHOLD")
    print("-" * 75)
    print(f"{'Threshold':<15} {'All':>12} {'Realistic':>12} {'Moderate':>12}")
    print("-" * 55)
    for thresh in [2.0, 3.0, 4.0, 5.0, 6.0]:
        all_c = counts.get(f"all_ge_{thresh}", 0)
        real_c = counts.get(f"realistic_ge_{thresh}", 0)
        mod_c = counts.get(f"moderate_ge_{thresh}", 0)
        print(f"|ΔH0| >= {thresh:<5.1f}  {all_c:>12} {real_c:>12} {mod_c:>12}")

    print("\n" + "-" * 75)
    print("TOP 5 POSITIVE BIAS CONTRIBUTORS (H0 TOO HIGH)")
    print("-" * 75)
    print(f"{'α_pop':>6} {'γ_Z':>6} {'ΔMstep':>6} {'Δβ':>5} | "
          f"{'δμ_gl':>6} {'k_μZ':>6} {'δμ_cr':>6} | {'ΔH0':>8} {'H0_fit':>8}")
    print("-" * 75)
    for t in top_pos:
        print(f"{t['alpha_pop']:>6.2f} {t['gamma_Z']:>6.2f} {t['delta_step']:>6.2f} "
              f"{t['delta_beta']:>5.1f} | "
              f"{t['delta_mu_global']:>6.2f} {t['k_mu_Z']:>6.2f} {t['delta_mu_crowd']:>6.2f} | "
              f"{t['delta_H0']:>+8.2f} {t['H0_fit']:>8.2f}")

    print("\n" + "-" * 75)
    print("TOP 5 NEGATIVE BIAS CONTRIBUTORS (H0 TOO LOW)")
    print("-" * 75)
    print(f"{'α_pop':>6} {'γ_Z':>6} {'ΔMstep':>6} {'Δβ':>5} | "
          f"{'δμ_gl':>6} {'k_μZ':>6} {'δμ_cr':>6} | {'ΔH0':>8} {'H0_fit':>8}")
    print("-" * 75)
    for t in reversed(top_neg):
        print(f"{t['alpha_pop']:>6.2f} {t['gamma_Z']:>6.2f} {t['delta_step']:>6.2f} "
              f"{t['delta_beta']:>5.1f} | "
              f"{t['delta_mu_global']:>6.2f} {t['k_mu_Z']:>6.2f} {t['delta_mu_crowd']:>6.2f} | "
              f"{t['delta_H0']:>+8.2f} {t['H0_fit']:>8.2f}")

    print("\n" + "=" * 75)
    print("KEY FINDINGS")
    print("=" * 75)

    max_realistic = stats['realistic']['max']
    max_moderate = stats['moderate']['max']
    max_overall = stats['all']['max']

    # Check if we can reach Hubble tension levels
    tension_level = 5.0

    if max_realistic >= tension_level:
        print(f"  ✓ REALISTIC systematics CAN produce |ΔH0| >= {tension_level:.1f} km/s/Mpc!")
        print(f"    Max in realistic region: {max_realistic:.2f} km/s/Mpc")
    else:
        print(f"  ✗ Realistic systematics alone produce max |ΔH0| = {max_realistic:.2f} km/s/Mpc")
        print(f"    (Below the ~{tension_level:.1f} km/s/Mpc Hubble tension)")

    if max_moderate >= tension_level:
        print(f"  ✓ MODERATE systematics CAN produce |ΔH0| >= {tension_level:.1f} km/s/Mpc!")
        print(f"    Max in moderate region: {max_moderate:.2f} km/s/Mpc")
    elif max_overall >= tension_level:
        print(f"  ⚠ Moderate region max: {max_moderate:.2f} km/s/Mpc")
        print(f"    Full scan max: {max_overall:.2f} km/s/Mpc (requires extreme params)")

    # Determine dominant bias direction
    if len(top_pos) > 0 and len(top_neg) > 0:
        max_pos = max(t['delta_H0'] for t in top_pos)
        min_neg = min(t['delta_H0'] for t in top_neg)
        if abs(max_pos) > abs(min_neg):
            print(f"\n  → Largest biases are POSITIVE (H0 inferred too HIGH)")
            print(f"    This is consistent with the Hubble tension direction")
        else:
            print(f"\n  → Largest biases are NEGATIVE (H0 inferred too LOW)")
            print(f"    This is OPPOSITE to the Hubble tension direction")

    # Note about calibrator vs SN contribution
    cnt_5_all = counts.get("all_ge_5.0", 0)
    if cnt_5_all > 0:
        print(f"\n  ⚠ {cnt_5_all} scenarios produce |ΔH0| >= 5 km/s/Mpc")

    print("=" * 75 + "\n")


def main():
    """Main analysis routine."""
    results_dir = Path("results/simulation_11c_calibrator_plus_sn")

    print("Loading SIMULATION 11C results...")
    results = load_results(results_dir)

    print("Analyzing results...")
    analysis = analyze_results(results)

    # Save analysis
    output_file = results_dir / "summary.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {output_file}")

    # Print summary
    print_summary(analysis, results)


if __name__ == "__main__":
    main()
