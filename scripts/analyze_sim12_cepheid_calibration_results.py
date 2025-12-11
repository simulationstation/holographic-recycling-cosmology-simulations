#!/usr/bin/env python3
"""
Analyze SIMULATION 12 results: Full Cepheid/TRGB Calibration Chain.

Defines "realistic" and "moderate" priors for both SN systematics and
Cepheid/TRGB calibration biases, then computes summary statistics.
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


def is_cepheid_realistic(r: dict) -> bool:
    """
    Check if Cepheid/TRGB systematics are in 'realistic' range.

    Realistic = very conservative, well-constrained values.
    """
    return (
        abs(r.get("delta_mu_anchor_global", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_N4258", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_LMC", 0.0)) <= 0.02 and
        abs(r.get("delta_mu_MW", 0.0)) <= 0.02 and
        abs(r.get("delta_M_W0", 0.0)) <= 0.03 and
        abs(r.get("delta_b_W", 0.0)) <= 0.03 and
        abs(r.get("delta_gamma_W", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_crowd_anchor", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_crowd_hosts", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_TRGB_global", 0.0)) <= 0.02
    )


def is_cepheid_moderate(r: dict) -> bool:
    """
    Check if Cepheid/TRGB systematics are in 'moderate' range.

    Moderate = upper bound but still plausible.
    """
    return (
        abs(r.get("delta_mu_anchor_global", 0.0)) <= 0.04 and
        abs(r.get("delta_mu_N4258", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_LMC", 0.0)) <= 0.03 and
        abs(r.get("delta_mu_MW", 0.0)) <= 0.03 and
        abs(r.get("delta_M_W0", 0.0)) <= 0.05 and
        abs(r.get("delta_b_W", 0.0)) <= 0.05 and
        abs(r.get("delta_gamma_W", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_crowd_anchor", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_crowd_hosts", 0.0)) <= 0.05 and
        abs(r.get("delta_mu_TRGB_global", 0.0)) <= 0.03
    )


def is_realistic(r: dict) -> bool:
    """Combined realistic region: both SN and Cepheid/TRGB systematics modest."""
    return is_sn_realistic(r) and is_cepheid_realistic(r)


def is_moderate(r: dict) -> bool:
    """Combined moderate region."""
    return is_sn_moderate(r) and is_cepheid_moderate(r)


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
    cepheid_only_realistic = []
    cepheid_only_mode = []
    trgb_mode = []

    for r in scenarios:
        delta_H0 = r["delta_H0"]
        abs_delta_H0 = abs(delta_H0)
        all_delta_H0.append(abs_delta_H0)

        if is_realistic(r):
            realistic_delta_H0.append(abs_delta_H0)

        if is_moderate(r):
            moderate_delta_H0.append(abs_delta_H0)

        if is_sn_realistic(r):
            sn_only_realistic.append(abs_delta_H0)

        if is_cepheid_realistic(r):
            cepheid_only_realistic.append(abs_delta_H0)

        # Separate by TRGB usage
        if not r.get("use_trgb", False):
            cepheid_only_mode.append(abs_delta_H0)
        else:
            trgb_mode.append(abs_delta_H0)

    all_delta_H0 = np.array(all_delta_H0)
    realistic_delta_H0 = np.array(realistic_delta_H0)
    moderate_delta_H0 = np.array(moderate_delta_H0)
    sn_only_realistic = np.array(sn_only_realistic)
    cepheid_only_realistic = np.array(cepheid_only_realistic)
    cepheid_only_mode = np.array(cepheid_only_mode)
    trgb_mode = np.array(trgb_mode)

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
        "realistic": compute_stats(realistic_delta_H0, "Realistic (SN + Cepheid)"),
        "moderate": compute_stats(moderate_delta_H0, "Moderate (SN + Cepheid)"),
        "sn_realistic": compute_stats(sn_only_realistic, "SN realistic only"),
        "cepheid_realistic": compute_stats(cepheid_only_realistic, "Cepheid realistic only"),
        "cepheid_only_mode": compute_stats(cepheid_only_mode, "Cepheid-only calibration"),
        "trgb_mode": compute_stats(trgb_mode, "Cepheid+TRGB calibration"),
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
            "delta_mu_anchor_global": s.get("delta_mu_anchor_global", 0.0),
            "delta_M_W0": s.get("delta_M_W0", 0.0),
            "delta_mu_crowd_hosts": s.get("delta_mu_crowd_hosts", 0.0),
            "use_trgb": s.get("use_trgb", False),
            "delta_H0": s["delta_H0"],
            "H0_fit": s["H0_fit"],
            "mean_mu_bias": s.get("mean_mu_bias", 0.0),
        })

    top_5_negative = []
    for s in sorted_by_bias[-5:]:
        top_5_negative.append({
            "alpha_pop": s.get("alpha_pop", 0.0),
            "gamma_Z": s.get("gamma_Z", 0.0),
            "delta_step": s.get("delta_step_true", 0.0),
            "delta_beta": s.get("delta_beta", 0.0),
            "delta_mu_anchor_global": s.get("delta_mu_anchor_global", 0.0),
            "delta_M_W0": s.get("delta_M_W0", 0.0),
            "delta_mu_crowd_hosts": s.get("delta_mu_crowd_hosts", 0.0),
            "use_trgb": s.get("use_trgb", False),
            "delta_H0": s["delta_H0"],
            "H0_fit": s["H0_fit"],
            "mean_mu_bias": s.get("mean_mu_bias", 0.0),
        })

    return {
        "stats": stats,
        "counts": counts,
        "top_5_positive": top_5_positive,
        "top_5_negative": top_5_negative,
        "total_scenarios": len(scenarios),
    }


def print_summary(analysis: dict, results: dict):
    """Print formatted SIMULATION 12 SUMMARY block."""
    stats = analysis["stats"]
    counts = analysis["counts"]
    top_pos = analysis["top_5_positive"]
    top_neg = analysis["top_5_negative"]

    true_H0 = results.get("true_cosmology", {}).get("H0", 67.5)

    print("\n" + "=" * 80)
    print("SIMULATION 12 SUMMARY: Full Cepheid/TRGB Calibration Chain")
    print("=" * 80)

    print(f"\nTrue cosmology: H0 = {true_H0:.1f} km/s/Mpc")
    print(f"Total scenarios scanned: {analysis['total_scenarios']}")

    # Print anchors and hosts
    if "anchors" in results:
        print(f"\nAnchors: {', '.join([a['name'] for a in results['anchors']])}")
    if "cepheid_hosts" in results:
        print(f"Cepheid SN hosts: {', '.join([h['name'] for h in results['cepheid_hosts']])}")

    print("\n" + "-" * 80)
    print("PARAMETER SPACE COVERAGE")
    print("-" * 80)
    print("SN Ia Systematics:")
    print(f"  α_pop (population drift):    {results['parameter_grids']['sn_systematics']['alpha_pop']}")
    print(f"  γ_Z (metallicity effect):    {results['parameter_grids']['sn_systematics']['gamma_Z']}")
    print(f"  ΔM_step (host mass step):    {results['parameter_grids']['sn_systematics']['delta_step']}")
    print(f"  Δβ (color law mismatch):     {results['parameter_grids']['sn_systematics']['delta_beta']}")
    print("\nAnchor Systematics:")
    print(f"  δμ_global (all anchors):     {results['parameter_grids']['anchor_biases']['delta_mu_anchor_global']}")
    print(f"  δμ_N4258:                    {results['parameter_grids']['anchor_biases']['delta_mu_N4258']}")
    print(f"  δμ_LMC:                      {results['parameter_grids']['anchor_biases']['delta_mu_LMC']}")
    print(f"  δμ_MW:                       {results['parameter_grids']['anchor_biases']['delta_mu_MW']}")
    print("\nCepheid PL Relation Biases:")
    print(f"  δM_W0 (zero-point):          {results['parameter_grids']['pl_biases']['delta_M_W0']}")
    print(f"  δb_W (slope):                {results['parameter_grids']['pl_biases']['delta_b_W']}")
    print(f"  δγ_W (metallicity):          {results['parameter_grids']['pl_biases']['delta_gamma_W']}")
    print("\nCrowding/Blending:")
    print(f"  δμ_crowd_anchor:             {results['parameter_grids']['crowding']['delta_mu_crowd_anchor']}")
    print(f"  δμ_crowd_hosts:              {results['parameter_grids']['crowding']['delta_mu_crowd_hosts']}")
    print("\nTRGB:")
    print(f"  δμ_TRGB_global:              {results['parameter_grids']['trgb']['delta_mu_TRGB_global']}")
    print(f"  use_trgb:                    {results['parameter_grids']['trgb']['use_trgb']}")

    print("\n" + "-" * 80)
    print("REGION DEFINITIONS")
    print("-" * 80)
    print("Realistic: SN(α_pop, γ_Z, ΔM_step ≤ 0.05, Δβ ≤ 0.3) AND")
    print("           Cepheid(δμ_glob ≤ 0.02, δμ_indiv ≤ 0.02, δM_W0,δb_W,δγ_W ≤ 0.03,")
    print("                   δμ_crowd ≤ 0.03, δμ_TRGB ≤ 0.02)")
    print("Moderate:  SN(α_pop, γ_Z, ΔM_step ≤ 0.10, Δβ ≤ 0.5) AND")
    print("           Cepheid(δμ_glob ≤ 0.04, δμ_indiv ≤ 0.03, δM_W0,δb_W,δγ_W ≤ 0.05,")
    print("                   δμ_crowd ≤ 0.05, δμ_TRGB ≤ 0.03)")

    print("\n" + "-" * 80)
    print("MAX |ΔH0| BY REGION")
    print("-" * 80)
    print(f"  Overall (all {stats['all']['count']} scenarios):".ljust(50) +
          f"{stats['all']['max']:.2f} km/s/Mpc")
    print(f"  Realistic region ({stats['realistic']['count']} scenarios):".ljust(50) +
          f"{stats['realistic']['max']:.2f} km/s/Mpc")
    print(f"  Moderate region ({stats['moderate']['count']} scenarios):".ljust(50) +
          f"{stats['moderate']['max']:.2f} km/s/Mpc")

    print("\n" + "-" * 80)
    print("DISTRIBUTION STATISTICS")
    print("-" * 80)
    print(f"{'Region':<30} {'Count':>8} {'Max':>8} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("-" * 76)
    for key in ["all", "realistic", "moderate", "cepheid_only_mode", "trgb_mode"]:
        s = stats[key]
        print(f"{s['name']:<30} {s['count']:>8} {s['max']:>8.2f} "
              f"{s['mean']:>8.2f} {s['median']:>8.2f} {s['std']:>8.2f}")

    print("\n" + "-" * 80)
    print("SCENARIO COUNTS BY |ΔH0| THRESHOLD")
    print("-" * 80)
    print(f"{'Threshold':<18} {'All':>12} {'Realistic':>12} {'Moderate':>12}")
    print("-" * 58)
    for thresh in [2.0, 3.0, 4.0, 5.0, 6.0]:
        all_c = counts.get(f"all_ge_{thresh}", 0)
        real_c = counts.get(f"realistic_ge_{thresh}", 0)
        mod_c = counts.get(f"moderate_ge_{thresh}", 0)
        print(f"|ΔH0| >= {thresh:<8.1f} {all_c:>12} {real_c:>12} {mod_c:>12}")

    print("\n" + "-" * 80)
    print("TOP 5 POSITIVE BIAS CONTRIBUTORS (H0 TOO HIGH)")
    print("-" * 80)
    print(f"{'α_pop':>5} {'δμ_anch':>7} {'δM_W0':>6} {'δμ_cr':>6} {'TRGB':>5} | "
          f"{'ΔH0':>8} {'H0_fit':>8} {'μ_bias':>8}")
    print("-" * 72)
    for t in top_pos:
        trgb_str = "Y" if t["use_trgb"] else "N"
        print(f"{t['alpha_pop']:>5.2f} {t['delta_mu_anchor_global']:>7.2f} "
              f"{t['delta_M_W0']:>6.2f} {t['delta_mu_crowd_hosts']:>6.2f} "
              f"{trgb_str:>5} | "
              f"{t['delta_H0']:>+8.2f} {t['H0_fit']:>8.2f} {t['mean_mu_bias']:>+8.3f}")

    print("\n" + "-" * 80)
    print("TOP 5 NEGATIVE BIAS CONTRIBUTORS (H0 TOO LOW)")
    print("-" * 80)
    print(f"{'α_pop':>5} {'δμ_anch':>7} {'δM_W0':>6} {'δμ_cr':>6} {'TRGB':>5} | "
          f"{'ΔH0':>8} {'H0_fit':>8} {'μ_bias':>8}")
    print("-" * 72)
    for t in reversed(top_neg):
        trgb_str = "Y" if t["use_trgb"] else "N"
        print(f"{t['alpha_pop']:>5.2f} {t['delta_mu_anchor_global']:>7.2f} "
              f"{t['delta_M_W0']:>6.2f} {t['delta_mu_crowd_hosts']:>6.2f} "
              f"{trgb_str:>5} | "
              f"{t['delta_H0']:>+8.2f} {t['H0_fit']:>8.2f} {t['mean_mu_bias']:>+8.3f}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    max_realistic = stats['realistic']['max']
    max_moderate = stats['moderate']['max']
    max_overall = stats['all']['max']

    # Check if we can reach Hubble tension levels
    tension_level = 5.0

    if max_realistic >= tension_level:
        print(f"  REALISTIC calibration chain CAN produce |ΔH0| >= {tension_level:.1f} km/s/Mpc!")
        print(f"    Max in realistic region: {max_realistic:.2f} km/s/Mpc")
    else:
        print(f"  Realistic calibration chain produces max |ΔH0| = {max_realistic:.2f} km/s/Mpc")
        print(f"    (Below the ~{tension_level:.1f} km/s/Mpc Hubble tension)")

    if max_moderate >= tension_level:
        print(f"  MODERATE calibration chain CAN produce |ΔH0| >= {tension_level:.1f} km/s/Mpc!")
        print(f"    Max in moderate region: {max_moderate:.2f} km/s/Mpc")
    elif max_overall >= tension_level:
        print(f"  Moderate region max: {max_moderate:.2f} km/s/Mpc")
        print(f"    Full scan max: {max_overall:.2f} km/s/Mpc (requires extreme params)")

    # Determine dominant bias direction
    if len(top_pos) > 0 and len(top_neg) > 0:
        max_pos = max(t['delta_H0'] for t in top_pos)
        min_neg = min(t['delta_H0'] for t in top_neg)
        if abs(max_pos) > abs(min_neg):
            print(f"\n  -> Largest biases are POSITIVE (H0 inferred too HIGH)")
            print(f"     This is consistent with the Hubble tension direction")
        else:
            print(f"\n  -> Largest biases are NEGATIVE (H0 inferred too LOW)")
            print(f"     This is OPPOSITE to the Hubble tension direction")

    # Compare Cepheid-only vs TRGB modes
    ceph_max = stats['cepheid_only_mode']['max']
    trgb_max = stats['trgb_mode']['max']
    print(f"\n  Calibration mode comparison:")
    print(f"    Cepheid-only max |ΔH0|: {ceph_max:.2f} km/s/Mpc")
    print(f"    Cepheid+TRGB max |ΔH0|: {trgb_max:.2f} km/s/Mpc")
    if trgb_max < ceph_max:
        print(f"    -> TRGB averaging reduces bias by {ceph_max - trgb_max:.2f} km/s/Mpc")

    # Count scenarios reaching thresholds
    cnt_5_all = counts.get("all_ge_5.0", 0)
    cnt_5_real = counts.get("realistic_ge_5.0", 0)
    if cnt_5_all > 0:
        print(f"\n  {cnt_5_all} scenarios produce |ΔH0| >= 5 km/s/Mpc")
        if cnt_5_real > 0:
            print(f"    Of which {cnt_5_real} are in the realistic region")
    else:
        print(f"\n  No scenarios produce |ΔH0| >= 5 km/s/Mpc")

    print("=" * 80 + "\n")


def main():
    """Main analysis routine."""
    results_dir = Path("results/simulation_12_cepheid_calibration")

    print("Loading SIMULATION 12 results...")
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
