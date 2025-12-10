#!/usr/bin/env python3
"""
Analyze SIMULATION 11B results: SH0ES-like two-step ladder systematics.

Computes:
- Max |ΔH0| overall and in realistic/moderate parameter regions
- Distribution statistics (mean, median, std)
- Counts of scenarios exceeding various |ΔH0| thresholds
"""

import json
import numpy as np
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load scan results from JSON file."""
    results_file = results_dir / "scan_results.json"
    with open(results_file, "r") as f:
        return json.load(f)


def classify_scenario(params: dict) -> dict:
    """
    Classify scenario into realistic/moderate regions.

    Realistic region: all drift/metallicity/step params <= 0.05, |Δβ| <= 0.5
    Moderate region: all drift/metallicity/step params <= 0.10, |Δβ| <= 0.5
    """
    alpha_pop = params.get("alpha_pop", 0.0)
    gamma_Z = params.get("gamma_Z", 0.0)
    delta_step = params.get("delta_step", params.get("delta_M_step_true", 0.0))
    delta_beta = params.get("delta_beta", 0.0)

    # Check realistic region
    realistic = (
        alpha_pop <= 0.05 and
        gamma_Z <= 0.05 and
        delta_step <= 0.05 and
        abs(delta_beta) <= 0.5
    )

    # Check moderate region
    moderate = (
        alpha_pop <= 0.10 and
        gamma_Z <= 0.10 and
        delta_step <= 0.10 and
        abs(delta_beta) <= 0.5
    )

    return {
        "realistic": realistic,
        "moderate": moderate,
    }


def analyze_results(results: dict) -> dict:
    """
    Analyze simulation results and compute summary statistics.
    """
    # Handle both "scenarios" and "results" keys
    scenarios = results.get("scenarios", results.get("results", []))

    # Extract |ΔH0| values
    all_delta_H0 = []
    realistic_delta_H0 = []
    moderate_delta_H0 = []

    for scenario in scenarios:
        # Handle both nested "params" and flat structure
        if "params" in scenario:
            params = scenario["params"]
        else:
            params = scenario  # flat structure
        delta_H0 = abs(scenario["delta_H0"])
        all_delta_H0.append(delta_H0)

        classification = classify_scenario(params)

        if classification["realistic"]:
            realistic_delta_H0.append(delta_H0)

        if classification["moderate"]:
            moderate_delta_H0.append(delta_H0)

    all_delta_H0 = np.array(all_delta_H0)
    realistic_delta_H0 = np.array(realistic_delta_H0)
    moderate_delta_H0 = np.array(moderate_delta_H0)

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
        "realistic": compute_stats(realistic_delta_H0, "Realistic region"),
        "moderate": compute_stats(moderate_delta_H0, "Moderate region"),
    }

    # Count scenarios exceeding thresholds
    thresholds = [3.0, 4.0, 5.0, 6.0]
    counts = {}
    for thresh in thresholds:
        counts[f"all_ge_{thresh}"] = int(np.sum(all_delta_H0 >= thresh))
        counts[f"realistic_ge_{thresh}"] = int(np.sum(realistic_delta_H0 >= thresh))
        counts[f"moderate_ge_{thresh}"] = int(np.sum(moderate_delta_H0 >= thresh))

    # Find top contributors
    sorted_scenarios = sorted(scenarios, key=lambda x: abs(x["delta_H0"]), reverse=True)
    top_5 = []
    for s in sorted_scenarios[:5]:
        p = s.get("params", s)  # Handle flat or nested structure
        top_5.append({
            "alpha_pop": p.get("alpha_pop", 0.0),
            "gamma_Z": p.get("gamma_Z", 0.0),
            "delta_M_step": p.get("delta_step", p.get("delta_M_step_true", 0.0)),
            "delta_beta": p.get("delta_beta", 0.0),
            "delta_H0": s["delta_H0"],
            "H0_fit": s["H0_fit"],
        })

    return {
        "stats": stats,
        "counts": counts,
        "top_5_contributors": top_5,
        "total_scenarios": len(scenarios),
    }


def print_summary(analysis: dict, results: dict):
    """Print formatted SIMULATION 11B SUMMARY block."""
    stats = analysis["stats"]
    counts = analysis["counts"]
    top_5 = analysis["top_5_contributors"]

    print("\n" + "=" * 70)
    print("SIMULATION 11B SUMMARY: SH0ES-like Two-Step Ladder Systematics")
    print("=" * 70)

    print(f"\nTrue cosmology: H0 = {results['true_cosmology']['H0']:.1f} km/s/Mpc")
    print(f"Total scenarios scanned: {analysis['total_scenarios']}")

    print("\n" + "-" * 70)
    print("PARAMETER SPACE COVERAGE")
    print("-" * 70)
    print(f"  α_pop (population drift):    [0.0, 0.05, 0.10]")
    print(f"  γ_Z (metallicity effect):    [0.0, 0.05, 0.10]")
    print(f"  ΔM_step (host mass step):    [0.0, 0.05, 0.10]")
    print(f"  Δβ (color law mismatch):     [0.0, 0.3, 0.5]")

    print("\n" + "-" * 70)
    print("MAX |ΔH0| BY REGION")
    print("-" * 70)
    print(f"  Overall (all {stats['all']['count']} scenarios):".ljust(40) +
          f"{stats['all']['max']:.2f} km/s/Mpc")
    print(f"  Realistic region ({stats['realistic']['count']} scenarios):".ljust(40) +
          f"{stats['realistic']['max']:.2f} km/s/Mpc")
    print(f"  Moderate region ({stats['moderate']['count']} scenarios):".ljust(40) +
          f"{stats['moderate']['max']:.2f} km/s/Mpc")

    print("\n" + "-" * 70)
    print("DISTRIBUTION STATISTICS")
    print("-" * 70)
    print(f"{'Region':<20} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("-" * 50)
    for key in ["all", "realistic", "moderate"]:
        s = stats[key]
        print(f"{s['name']:<20} {s['mean']:>10.2f} {s['median']:>10.2f} {s['std']:>10.2f}")

    print("\n" + "-" * 70)
    print("SCENARIO COUNTS BY |ΔH0| THRESHOLD")
    print("-" * 70)
    print(f"{'Threshold':<15} {'All':>12} {'Realistic':>12} {'Moderate':>12}")
    print("-" * 55)
    for thresh in [3.0, 4.0, 5.0, 6.0]:
        all_c = counts[f"all_ge_{thresh}"]
        real_c = counts[f"realistic_ge_{thresh}"]
        mod_c = counts[f"moderate_ge_{thresh}"]
        print(f"|ΔH0| >= {thresh:<5.1f}  {all_c:>12} {real_c:>12} {mod_c:>12}")

    print("\n" + "-" * 70)
    print("TOP 5 BIAS CONTRIBUTORS")
    print("-" * 70)
    print(f"{'α_pop':>8} {'γ_Z':>8} {'ΔM_step':>8} {'Δβ':>8} {'ΔH0':>12} {'H0_fit':>12}")
    print("-" * 60)
    for t in top_5:
        print(f"{t['alpha_pop']:>8.2f} {t['gamma_Z']:>8.2f} {t['delta_M_step']:>8.2f} "
              f"{t['delta_beta']:>8.2f} {t['delta_H0']:>+12.2f} {t['H0_fit']:>12.2f}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Determine key finding
    max_realistic = stats['realistic']['max']
    max_moderate = stats['moderate']['max']
    max_overall = stats['all']['max']

    if max_realistic >= 5.0:
        print(f"  ⚠ Even 'realistic' systematics (all params ≤ 0.05) can produce")
        print(f"    |ΔH0| up to {max_realistic:.2f} km/s/Mpc")

    if max_moderate >= 5.0:
        print(f"  ⚠ 'Moderate' systematics (all params ≤ 0.10) can produce")
        print(f"    |ΔH0| up to {max_moderate:.2f} km/s/Mpc")

    if counts.get("all_ge_5.0", counts.get("all_ge_5", 0)) > 0:
        cnt = counts.get("all_ge_5.0", counts.get("all_ge_5", 0))
        print(f"  ⚠ {cnt} scenarios produce |ΔH0| >= 5 km/s/Mpc")
        print(f"    (comparable to the ~5 km/s/Mpc Hubble tension)")

    # Note about positive vs negative bias
    positive_bias = sum(1 for s in top_5 if s["delta_H0"] > 0)
    if positive_bias == len(top_5):
        print(f"  → All top bias contributors produce POSITIVE ΔH0 (higher H0)")
        print(f"    This is consistent with the direction of the Hubble tension")

    print("=" * 70 + "\n")


def main():
    """Main analysis routine."""
    results_dir = Path("results/simulation_11b_ladder_systematics")

    print("Loading SIMULATION 11B results...")
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
