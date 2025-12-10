#!/usr/bin/env python3
"""
SIMULATION 10: Final Summary of WHBC Primordial P(k) Results

Combines scan results and effective H0 estimates to produce final summary
and interpretation of how much primordial physics can contribute to H0 tension.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scan_results(results_dir: str) -> Dict[str, Any]:
    """Load scan results."""
    json_path = os.path.join(results_dir, 'scan_results.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def load_h0_results(results_dir: str) -> Dict[str, Any]:
    """Load H0 effective results."""
    json_path = os.path.join(results_dir, 'h0_effective.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def compute_summary_statistics(
    scan_data: Dict[str, Any],
    h0_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute comprehensive summary statistics.

    Args:
        scan_data: Scan results dictionary
        h0_data: H0 effective results dictionary

    Returns:
        Summary statistics dictionary
    """
    results = scan_data.get('results', [])
    successful = [r for r in results if r.get('success', False)]
    passing = [r for r in successful if r.get('passes_all', False)]

    stats = {
        'n_total': len(results),
        'n_successful': len(successful),
        'n_passing': len(passing),
        'pass_rate': 100 * len(passing) / max(len(successful), 1),
    }

    # TT/EE RMS statistics
    if len(passing) > 0:
        tt_rms = [r['TT_RMS_30_2000'] for r in passing]
        ee_rms = [r['EE_RMS_30_2000'] for r in passing]
        sigma8_shifts = [abs(r['sigma8_ratio'] - 1.0) for r in passing]
        S8_shifts = [abs(r['S8_ratio'] - 1.0) for r in passing]
        delta_h0_approx = [abs(r['delta_H0_approx']) for r in passing]

        stats['TT_RMS_max'] = float(max(tt_rms))
        stats['TT_RMS_mean'] = float(np.mean(tt_rms))
        stats['EE_RMS_max'] = float(max(ee_rms))
        stats['EE_RMS_mean'] = float(np.mean(ee_rms))
        stats['sigma8_shift_max'] = float(max(sigma8_shifts))
        stats['S8_shift_max'] = float(max(S8_shifts))
        stats['delta_H0_approx_max'] = float(max(delta_h0_approx))
        stats['delta_H0_approx_mean'] = float(np.mean(delta_h0_approx))

    # H0 effective results
    if h0_data is not None and 'summary' in h0_data:
        h0_summary = h0_data['summary']
        stats['H0_eff_max_abs'] = h0_summary.get('max_abs_delta_H0', 0.0)
        stats['H0_eff_mean_abs'] = h0_summary.get('mean_abs_delta_H0', 0.0)
        stats['H0_eff_median_abs'] = h0_summary.get('median_abs_delta_H0', 0.0)
    elif h0_data is not None and 'results' in h0_data:
        h0_results = h0_data['results']
        if len(h0_results) > 0:
            abs_delta = [abs(r['delta_H0_eff']) for r in h0_results]
            stats['H0_eff_max_abs'] = float(max(abs_delta))
            stats['H0_eff_mean_abs'] = float(np.mean(abs_delta))
            stats['H0_eff_median_abs'] = float(np.median(abs_delta))

    # Tight filter: models with TT_RMS < 0.03 and EE_RMS < 0.06
    tight_pass = [r for r in passing
                  if r['TT_RMS_30_2000'] < 0.03 and r['EE_RMS_30_2000'] < 0.06]
    stats['n_tight_pass'] = len(tight_pass)

    if len(tight_pass) > 0:
        delta_h0_tight = [abs(r['delta_H0_approx']) for r in tight_pass]
        stats['delta_H0_approx_max_tight'] = float(max(delta_h0_tight))

    return stats


def determine_interpretation(stats: Dict[str, Any]) -> str:
    """Determine interpretation based on statistics."""
    max_h0 = stats.get('H0_eff_max_abs', stats.get('delta_H0_approx_max', 0.0))

    if max_h0 < 0.5:
        return (
            "WHBC-like primordial features consistent with CMB TT/EE spectra "
            "and sigma8/S8 produce only sub-km/s/Mpc shifts in the effective LCDM H0. "
            "Primordial P(k) modifications alone cannot meaningfully address "
            "the Hubble tension."
        )
    elif max_h0 < 2.0:
        return (
            "WHBC-like primordial features can induce O(0.5-2 km/s/Mpc) shifts in the "
            "effective LCDM H0 while remaining CMB-compatible, but this is insufficient "
            "to fully account for the ~5 km/s/Mpc Hubble tension. Pure P(k) modifications "
            "on LCDM background have limited H0-shifting power."
        )
    elif max_h0 < 3.5:
        return (
            "There exists a region in parameter space where WHBC-like primordial "
            f"features induce >2 km/s/Mpc (max: {max_h0:.1f}) apparent H0 shifts while "
            "remaining within the CMB residual bounds. This warrants further investigation "
            "with full likelihood analysis, though still short of the full tension."
        )
    else:
        return (
            f"WHBC primordial P(k) modifications can produce up to {max_h0:.1f} km/s/Mpc "
            "H0 shifts while marginally satisfying CMB constraints. This approaches "
            "the scale of the Hubble tension, though full likelihood analysis with "
            "Planck covariance is needed to confirm viability."
        )


def print_summary(stats: Dict[str, Any], interpretation: str):
    """Print formatted summary."""
    print()
    print("=" * 60)
    print("        WHBC PRIMORDIAL SUMMARY")
    print("=" * 60)
    print(f"Models scanned: {stats['n_total']}")
    print(f"Models passing CMB/sigma8/S8 filters: {stats['n_passing']}")
    print(f"Pass rate: {stats['pass_rate']:.1f}%")
    print()

    max_h0 = stats.get('H0_eff_max_abs', stats.get('delta_H0_approx_max', 0.0))
    median_h0 = stats.get('H0_eff_median_abs', stats.get('delta_H0_approx_mean', 0.0))

    print(f"Max |delta_H0_eff| among PASS models: {max_h0:.2f} km/s/Mpc")
    print(f"Typical |delta_H0_eff| (median/mean): {median_h0:.2f} km/s/Mpc")
    print()

    if 'TT_RMS_max' in stats:
        print(f"Max TT_RMS (ell=30-2000) among PASS: {stats['TT_RMS_max']:.4f}")
        print(f"Max EE_RMS (ell=30-2000) among PASS: {stats['EE_RMS_max']:.4f}")
        print(f"Max |sigma8 shift| among PASS: {stats['sigma8_shift_max']:.4f}")
        print()

    if 'n_tight_pass' in stats:
        print(f"Models with TT<3% & EE<6% (tight): {stats['n_tight_pass']}")
        if 'delta_H0_approx_max_tight' in stats:
            print(f"Max |delta_H0| among tight PASS: {stats['delta_H0_approx_max_tight']:.2f} km/s/Mpc")
        print()

    print("Interpretation:")
    # Word wrap the interpretation
    words = interpretation.split()
    lines = []
    current_line = "  "
    for word in words:
        if len(current_line) + len(word) + 1 <= 58:
            current_line += word + " "
        else:
            lines.append(current_line.rstrip())
            current_line = "  " + word + " "
    if current_line.strip():
        lines.append(current_line.rstrip())
    for line in lines:
        print(line)

    print()
    print("=" * 60)


def write_summary(
    stats: Dict[str, Any],
    interpretation: str,
    output_dir: str,
    scan_data: Dict[str, Any],
    h0_data: Dict[str, Any],
):
    """Write summary JSON file."""
    # Find best model
    results = scan_data.get('results', [])
    passing = [r for r in results if r.get('success', False) and r.get('passes_all', False)]

    best_model = None
    if len(passing) > 0:
        best = max(passing, key=lambda x: abs(x.get('delta_H0_approx', 0)))
        best_model = {
            'params': best['params'],
            'TT_RMS': best['TT_RMS_30_2000'],
            'EE_RMS': best['EE_RMS_30_2000'],
            'sigma8_ratio': best['sigma8_ratio'],
            'delta_H0_approx': best['delta_H0_approx'],
        }

    summary = {
        'simulation': 'T10_WHBC_PRIMORDIAL - WHBC-like Primordial P(k) on LCDM Background',
        'date': datetime.now().isoformat(),
        'statistics': stats,
        'interpretation': interpretation,
        'best_model': best_model,
        'verdict': 'SMALL_EFFECT' if stats.get('H0_eff_max_abs', 0) < 2.0 else 'MODERATE_EFFECT',
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults written to: {output_dir}/summary.json")


def main():
    """Main entry point."""
    print("="*60)
    print("SIMULATION 10: WHBC Primordial Summary")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print()

    results_dir = 'results/simulation_10_whbc_pk'

    # Load results
    try:
        scan_data = load_scan_results(results_dir)
        print(f"Loaded scan results: {scan_data['total_models']} models")
    except FileNotFoundError:
        print(f"Error: Scan results not found in {results_dir}/")
        print("Run scripts/run_whbc_primordial_scan.py first.")
        sys.exit(1)

    h0_data = load_h0_results(results_dir)
    if h0_data:
        print(f"Loaded H0 effective results: {h0_data.get('n_analyzed', 0)} models")
    else:
        print("H0 effective results not found (optional)")

    # Compute statistics
    stats = compute_summary_statistics(scan_data, h0_data)

    # Determine interpretation
    interpretation = determine_interpretation(stats)

    # Print summary
    print_summary(stats, interpretation)

    # Write summary
    write_summary(stats, interpretation, results_dir, scan_data, h0_data)


if __name__ == '__main__':
    main()
