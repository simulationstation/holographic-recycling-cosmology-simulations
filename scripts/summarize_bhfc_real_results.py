#!/usr/bin/env python3
"""
Summarize BHFC Real Scenario Results

Loads scan results and generates final summary with verdict on whether
early BH-dominated cosmology can generate meaningful early-late H0 split.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> dict:
    """Load scan results from JSON.

    Args:
        results_dir: Directory containing scan_results.json

    Returns:
        Results dictionary
    """
    json_path = os.path.join(results_dir, 'scan_results.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_results(data: dict) -> dict:
    """Analyze scan results and compute summary statistics.

    Args:
        data: Loaded JSON data

    Returns:
        Analysis dictionary
    """
    results = data['results']

    # Filter
    successful = [r for r in results if r.get('success', False)]
    passing = [r for r in successful if r.get('passes_all', False)]

    analysis = {
        'N_total': len(results),
        'N_successful': len(successful),
        'N_passing': len(passing),
        'pass_rate': len(passing) / max(len(successful), 1),
    }

    if len(passing) == 0:
        analysis['max_Delta_H0'] = 0.0
        analysis['best_model'] = None
        analysis['verdict'] = 'NO_VIABLE_MODELS'
        analysis['interpretation'] = (
            "No BHFC models passed all observational constraints. "
            "Early BH formation at the levels tested is inconsistent with CMB/BAO data."
        )
        return analysis

    # Analyze passing models
    Delta_H0_vals = [abs(r.get('Delta_H0', 0)) for r in passing]
    analysis['max_Delta_H0'] = max(Delta_H0_vals)
    analysis['mean_Delta_H0'] = np.mean(Delta_H0_vals)
    analysis['std_Delta_H0'] = np.std(Delta_H0_vals)

    # Best model (max |Delta_H0|)
    best = max(passing, key=lambda x: abs(x.get('Delta_H0', 0)))
    analysis['best_model'] = {
        'params': best['params'],
        'H0_Early': best.get('H0_Early'),
        'H0_Late': best.get('H0_Late'),
        'Delta_H0': best.get('Delta_H0'),
        'delta_theta_s_percent': best.get('delta_theta_s_percent'),
    }

    # Determine verdict
    max_dH0 = analysis['max_Delta_H0']
    if max_dH0 < 1.0:
        analysis['verdict'] = 'NO_SPLIT'
        analysis['interpretation'] = (
            "No PASS models produce |Delta_H0| >= 1 km/s/Mpc. "
            "Early BH-dominated phase cannot generate meaningful H0 split "
            "while satisfying observational constraints."
        )
    elif max_dH0 < 3.0:
        analysis['verdict'] = 'SMALL_SPLIT'
        analysis['interpretation'] = (
            f"Some PASS models produce 1 <= |Delta_H0| <= 3 km/s/Mpc (max: {max_dH0:.2f}). "
            "Early BH-dominated phase can generate a small early-late H0 split, "
            "but not at the level of the full Hubble tension (~5 km/s/Mpc)."
        )
    else:
        analysis['verdict'] = 'TENSION_SCALE_SPLIT'
        analysis['interpretation'] = (
            f"Some PASS models produce |Delta_H0| >= 3 km/s/Mpc (max: {max_dH0:.2f}). "
            "Early BH-dominated phase CAN potentially generate H0 splits "
            "approaching the level of the Hubble tension."
        )

    # Parameter trends
    if len(passing) > 1:
        # Group by f_BH_init
        by_fBH = {}
        for r in passing:
            f = r['params']['f_BH_init']
            if f not in by_fBH:
                by_fBH[f] = []
            by_fBH[f].append(abs(r.get('Delta_H0', 0)))

        analysis['Delta_H0_by_f_BH'] = {
            str(k): {'mean': np.mean(v), 'max': max(v), 'count': len(v)}
            for k, v in by_fBH.items()
        }

        # Group by z_form
        by_zform = {}
        for r in passing:
            z = r['params']['z_form']
            if z not in by_zform:
                by_zform[z] = []
            by_zform[z].append(abs(r.get('Delta_H0', 0)))

        analysis['Delta_H0_by_z_form'] = {
            str(k): {'mean': np.mean(v), 'max': max(v), 'count': len(v)}
            for k, v in by_zform.items()
        }

    return analysis


def write_summary(analysis: dict, output_dir: str):
    """Write summary JSON file.

    Args:
        analysis: Analysis dictionary
        output_dir: Output directory
    """
    summary = {
        'simulation': 'T09_BHFC_REAL - Early Black-Hole Dominated Fertility Cosmology',
        'date': datetime.now().isoformat(),
        'models_scanned': analysis['N_total'],
        'models_successful': analysis['N_successful'],
        'models_passing': analysis['N_passing'],
        'pass_rate_percent': analysis['pass_rate'] * 100,
        'max_Delta_H0_kmsMpc': analysis['max_Delta_H0'],
        'verdict': analysis['verdict'],
        'interpretation': analysis['interpretation'],
        'best_model': analysis.get('best_model'),
        'parameter_trends': {
            'by_f_BH_init': analysis.get('Delta_H0_by_f_BH', {}),
            'by_z_form': analysis.get('Delta_H0_by_z_form', {}),
        },
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Summary saved to {output_dir}/summary.json")


def print_summary(analysis: dict):
    """Print formatted summary to console."""
    print()
    print("=" * 60)
    print("            REAL BHFC SUMMARY")
    print("=" * 60)
    print(f"Models scanned:              {analysis['N_total']}")
    print(f"Successful evaluations:      {analysis['N_successful']}")
    print(f"Models passing constraints:  {analysis['N_passing']}")
    print(f"Pass rate:                   {analysis['pass_rate']*100:.1f}%")
    print()

    if analysis['N_passing'] > 0:
        print(f"Max |Delta_H0| among PASS models: {analysis['max_Delta_H0']:.2f} km/s/Mpc")

        if analysis['best_model']:
            bm = analysis['best_model']
            print()
            print("Best model parameters:")
            print(f"  f_BH_init  = {bm['params']['f_BH_init']}")
            print(f"  z_form     = {bm['params']['z_form']:.0e}")
            print(f"  z_evap     = {bm['params']['z_evap']}")
            print(f"  A_eff      = {bm['params']['A_eff']}")
            print(f"  f_evap     = {bm['params']['f_evap']}")
            print()
            print(f"  H0_Early   = {bm['H0_Early']:.2f} km/s/Mpc")
            print(f"  H0_Late    = {bm['H0_Late']:.2f} km/s/Mpc")
            print(f"  Delta_H0   = {bm['Delta_H0']:.2f} km/s/Mpc")
    else:
        print("No models passed all constraints!")

    print()
    print(f"Verdict: {analysis['verdict']}")
    print()
    print("Interpretation:")
    print(f"  {analysis['interpretation']}")
    print("=" * 60)


def main():
    """Main entry point."""
    results_dir = 'results/simulation_9_bhfc_real'

    # Check if results exist
    if not os.path.exists(os.path.join(results_dir, 'scan_results.json')):
        print(f"Error: No results found in {results_dir}/")
        print("Run the scan first: python scripts/run_bhfc_real_scenario.py")
        sys.exit(1)

    # Load and analyze
    print("Loading BHFC scan results...")
    data = load_results(results_dir)

    print("Analyzing results...")
    analysis = analyze_results(data)

    # Write summary
    write_summary(analysis, results_dir)

    # Print summary
    print_summary(analysis)


if __name__ == '__main__':
    main()
