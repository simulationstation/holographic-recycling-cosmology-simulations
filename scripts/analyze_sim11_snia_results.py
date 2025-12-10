#!/usr/bin/env python3
"""
SIMULATION 11: Analysis of SN Ia Systematics Results

Loads the scan results and produces:
1. Summary statistics
2. Identification of scenarios reaching >= 3 and >= 5 km/s/Mpc
3. Assessment of whether realistic systematics can explain the Hubble tension
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Constants
RESULTS_DIR = 'results/simulation_11_snia_systematics'
H0_TENSION = 5.0  # km/s/Mpc - approximate magnitude of Hubble tension


def load_results() -> Dict[str, Any]:
    """Load scan results from JSON."""
    json_path = os.path.join(RESULTS_DIR, 'scan_results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results not found at {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze scan results and compute summary statistics."""
    results = data['results']
    H0_true = data['true_cosmology']['H0']

    delta_H0 = np.array([r['delta_H0'] for r in results])
    abs_delta_H0 = np.abs(delta_H0)

    # Basic statistics
    stats = {
        'n_scenarios': len(results),
        'H0_true': H0_true,
        'max_delta_H0': float(np.max(delta_H0)),
        'min_delta_H0': float(np.min(delta_H0)),
        'max_abs_delta_H0': float(np.max(abs_delta_H0)),
        'mean_abs_delta_H0': float(np.mean(abs_delta_H0)),
        'median_abs_delta_H0': float(np.median(abs_delta_H0)),
        'std_delta_H0': float(np.std(delta_H0)),
    }

    # Count scenarios at different thresholds
    stats['n_ge_1'] = int(np.sum(abs_delta_H0 >= 1.0))
    stats['n_ge_2'] = int(np.sum(abs_delta_H0 >= 2.0))
    stats['n_ge_3'] = int(np.sum(abs_delta_H0 >= 3.0))
    stats['n_ge_4'] = int(np.sum(abs_delta_H0 >= 4.0))
    stats['n_ge_5'] = int(np.sum(abs_delta_H0 >= 5.0))

    # Identify scenarios at >= 3 and >= 5
    scenarios_ge3 = [r for r in results if abs(r['delta_H0']) >= 3.0]
    scenarios_ge5 = [r for r in results if abs(r['delta_H0']) >= 5.0]

    # "Realistic" region analysis
    realistic = [r for r in results
                 if r['alpha_pop'] <= 0.05 and r['gamma_Z'] <= 0.05
                 and r['delta_m_malm'] <= 0.05]
    if realistic:
        real_dH0 = np.array([r['delta_H0'] for r in realistic])
        stats['realistic_n'] = len(realistic)
        stats['realistic_max_abs'] = float(np.max(np.abs(real_dH0)))
        stats['realistic_mean_abs'] = float(np.mean(np.abs(real_dH0)))
    else:
        stats['realistic_n'] = 0
        stats['realistic_max_abs'] = 0.0
        stats['realistic_mean_abs'] = 0.0

    # "Moderate" region analysis (params <= 0.10)
    moderate = [r for r in results
                if r['alpha_pop'] <= 0.10 and r['gamma_Z'] <= 0.10
                and r['delta_m_malm'] <= 0.10]
    if moderate:
        mod_dH0 = np.array([r['delta_H0'] for r in moderate])
        stats['moderate_n'] = len(moderate)
        stats['moderate_max_abs'] = float(np.max(np.abs(mod_dH0)))
        stats['moderate_mean_abs'] = float(np.mean(np.abs(mod_dH0)))
    else:
        stats['moderate_n'] = 0
        stats['moderate_max_abs'] = 0.0
        stats['moderate_mean_abs'] = 0.0

    # Find best/worst models
    max_idx = np.argmax(delta_H0)
    min_idx = np.argmin(delta_H0)
    max_abs_idx = np.argmax(abs_delta_H0)

    return {
        'statistics': stats,
        'scenarios_ge3': scenarios_ge3,
        'scenarios_ge5': scenarios_ge5,
        'max_positive': results[max_idx],
        'max_negative': results[min_idx],
        'max_absolute': results[max_abs_idx],
    }


def determine_interpretation(analysis: Dict[str, Any]) -> str:
    """Generate interpretation based on analysis."""
    stats = analysis['statistics']
    max_abs = stats['max_abs_delta_H0']
    realistic_max = stats['realistic_max_abs']
    moderate_max = stats['moderate_max_abs']
    n_ge5 = stats['n_ge_5']

    lines = []

    if max_abs >= H0_TENSION:
        lines.append(
            f"SN Ia systematics CAN produce H0 biases >= {H0_TENSION} km/s/Mpc "
            f"(max: {max_abs:.2f} km/s/Mpc)."
        )
        if n_ge5 > 0:
            lines.append(
                f"{n_ge5} scenario(s) in the grid reach |delta_H0| >= 5 km/s/Mpc."
            )
    else:
        lines.append(
            f"In the explored parameter space, max |delta_H0| = {max_abs:.2f} km/s/Mpc, "
            f"which is below the ~{H0_TENSION} km/s/Mpc tension."
        )

    # Realistic region
    if realistic_max >= 2.0:
        lines.append(
            f"Even 'realistic' systematics (all params <= 0.05) produce "
            f"|delta_H0| up to {realistic_max:.2f} km/s/Mpc."
        )
    elif realistic_max > 0:
        lines.append(
            f"'Realistic' systematics (all params <= 0.05) produce only "
            f"up to {realistic_max:.2f} km/s/Mpc bias."
        )

    # Moderate region
    if moderate_max >= 3.0:
        lines.append(
            f"'Moderate' systematics (all params <= 0.10) can produce "
            f"|delta_H0| up to {moderate_max:.2f} km/s/Mpc."
        )

    # Key insight
    if max_abs >= H0_TENSION:
        lines.append(
            "\nKey insight: Combinations of population drift, metallicity effects, "
            "and Malmquist bias CAN plausibly produce H0 biases comparable to the "
            "full Hubble tension, but only with aggressive systematic values."
        )
    else:
        lines.append(
            "\nKey insight: Individual systematics at plausible levels contribute "
            "~1-3 km/s/Mpc each. Getting to ~5 km/s/Mpc requires either extreme "
            "values or additional unmodeled systematics."
        )

    return "\n".join(lines)


def print_summary(data: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Print the SIMULATION 11 SUMMARY block."""
    stats = analysis['statistics']

    print()
    print("=" * 65)
    print("               SIMULATION 11 SUMMARY")
    print("         SN Ia Distance-Ladder Systematics")
    print("=" * 65)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"True H0: {stats['H0_true']} km/s/Mpc")
    print(f"Number of scenarios: {stats['n_scenarios']}")
    print()

    print("Parameter ranges explored:")
    grids = data['parameter_grids']
    print(f"  alpha_pop (pop. drift): {grids['alpha_pop']}")
    print(f"  gamma_Z (metallicity):  {grids['gamma_Z']}")
    print(f"  delta_m_malm (Malm.):   {grids['delta_m_malm']}")
    print()

    print("-" * 65)
    print("H0 BIAS STATISTICS")
    print("-" * 65)
    print(f"Max |delta_H0|: {stats['max_abs_delta_H0']:.2f} km/s/Mpc")
    print(f"Mean |delta_H0|: {stats['mean_abs_delta_H0']:.2f} km/s/Mpc")
    print(f"Median |delta_H0|: {stats['median_abs_delta_H0']:.2f} km/s/Mpc")
    print()

    print(f"Scenarios with |delta_H0| >= 1 km/s/Mpc: {stats['n_ge_1']}")
    print(f"Scenarios with |delta_H0| >= 2 km/s/Mpc: {stats['n_ge_2']}")
    print(f"Scenarios with |delta_H0| >= 3 km/s/Mpc: {stats['n_ge_3']}")
    print(f"Scenarios with |delta_H0| >= 4 km/s/Mpc: {stats['n_ge_4']}")
    print(f"Scenarios with |delta_H0| >= 5 km/s/Mpc: {stats['n_ge_5']}")
    print()

    print("-" * 65)
    print("EXTREME SCENARIOS")
    print("-" * 65)
    r = analysis['max_positive']
    print(f"Largest positive bias (H0 too HIGH):")
    print(f"  alpha_pop={r['alpha_pop']:.2f}, gamma_Z={r['gamma_Z']:.2f}, "
          f"malm={r['delta_m_malm']:.2f}")
    print(f"  H0_fit={r['H0_fit']:.2f} km/s/Mpc, delta_H0={r['delta_H0']:+.2f}")
    print()

    r = analysis['max_negative']
    print(f"Largest negative bias (H0 too LOW):")
    print(f"  alpha_pop={r['alpha_pop']:.2f}, gamma_Z={r['gamma_Z']:.2f}, "
          f"malm={r['delta_m_malm']:.2f}")
    print(f"  H0_fit={r['H0_fit']:.2f} km/s/Mpc, delta_H0={r['delta_H0']:+.2f}")
    print()

    # Scenarios with |delta_H0| >= 5
    if stats['n_ge_5'] > 0:
        print("-" * 65)
        print("SCENARIOS WITH |delta_H0| >= 5 km/s/Mpc:")
        print("-" * 65)
        for r in analysis['scenarios_ge5']:
            print(f"  alpha={r['alpha_pop']:.2f}, gamma={r['gamma_Z']:.2f}, "
                  f"malm={r['delta_m_malm']:.2f} => "
                  f"delta_H0={r['delta_H0']:+.2f} km/s/Mpc")
        print()

    # "Realistic" and "Moderate" regions
    print("-" * 65)
    print("PARAMETER REGION ANALYSIS")
    print("-" * 65)
    print(f"'Realistic' region (all params <= 0.05):")
    print(f"  N scenarios: {stats['realistic_n']}")
    print(f"  Max |delta_H0|: {stats['realistic_max_abs']:.2f} km/s/Mpc")
    print()
    print(f"'Moderate' region (all params <= 0.10):")
    print(f"  N scenarios: {stats['moderate_n']}")
    print(f"  Max |delta_H0|: {stats['moderate_max_abs']:.2f} km/s/Mpc")
    print()

    # Interpretation
    print("-" * 65)
    print("INTERPRETATION")
    print("-" * 65)
    interpretation = determine_interpretation(analysis)
    # Word wrap
    for line in interpretation.split('\n'):
        words = line.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= 63:
                current += word + " "
            else:
                print("  " + current.strip())
                current = word + " "
        if current.strip():
            print("  " + current.strip())
    print()

    print("=" * 65)


def save_summary(data: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Save summary JSON."""
    stats = analysis['statistics']
    interpretation = determine_interpretation(analysis)

    summary = {
        'simulation': 'SIM11_SNIA_SYSTEMATICS',
        'date': datetime.now().isoformat(),
        'true_H0': stats['H0_true'],
        'n_scenarios': stats['n_scenarios'],
        'statistics': stats,
        'interpretation': interpretation,
        'max_positive_scenario': analysis['max_positive'],
        'max_negative_scenario': analysis['max_negative'],
        'n_scenarios_ge5': stats['n_ge_5'],
        'scenarios_ge5': analysis['scenarios_ge5'],
        'verdict': 'CAN_EXPLAIN_TENSION' if stats['n_ge_5'] > 0 else 'INSUFFICIENT',
    }

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {RESULTS_DIR}/summary.json")


def main():
    """Main entry point."""
    print("Loading SIMULATION 11 results...")

    try:
        data = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run scripts/run_sim11_snia_systematics.py first.")
        sys.exit(1)

    print(f"Loaded {len(data['results'])} scenarios")

    # Analyze
    analysis = analyze_results(data)

    # Print summary
    print_summary(data, analysis)

    # Save summary
    save_summary(data, analysis)


if __name__ == '__main__':
    main()
