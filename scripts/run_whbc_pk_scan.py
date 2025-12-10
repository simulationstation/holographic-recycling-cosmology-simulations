#!/usr/bin/env python3
"""
T08_WHBC_PK: WHBC Primordial Power Spectrum Parameter Scan

This script scans the WHBC primordial P(k) parameter space to test
whether primordial spectrum modifications can help resolve the Hubble
tension while satisfying CMB constraints.

The WHBC P(k) modification is:
    P_WHBC(k) = P_LCDM(k) * F_WHBC(k)

where:
    F_WHBC(k) = 1
                + A_cut * exp[-(k / k_cut)^p_cut]     # IR cutoff
                + A_osc * sin(omega*ln(k/k_pivot) + phi) * exp[-(k/k_damp)^2]

This tests whether boundary-condition-motivated primordial features
can shift CMB-inferred H0 while maintaining consistency with theta_s.

Author: HRC Collaboration
Date: December 2025
"""

import os
import sys
import json
import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.primordial import (
    WHBCPrimordialParameters,
    primordial_ratio,
    primordial_PK_whbc,
    analyze_whbc_primordial,
    approximate_cmb_effects,
    HAS_CAMB,
    HAS_CLASS,
)

# Try to import CAMB for more accurate calculations
if HAS_CAMB:
    from hrc2.primordial.class_interface import run_camb_with_whbc, compute_cmb_chi2


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = "results/simulation_8_whbc_pk"

# Parameter grids to scan (reduced for faster execution)
PARAM_GRID = {
    # IR cutoff parameters
    'A_cut': [-0.03, 0.0, 0.03],                    # Amplitude of IR modification
    'k_cut': [0.001],                               # Cutoff scale [Mpc^-1]
    'p_cut': [2.0],                                  # Power of cutoff (fixed)

    # Oscillation parameters
    'A_osc': [0.0, 0.03],                           # Amplitude of oscillations
    'omega_WH': [0.0, 5.0],                         # Frequency in ln(k)
    'phi_WH': [0.0],                                # Phase (fixed)
    'k_damp': [0.5],                                # UV damping scale [Mpc^-1]
}

# Planck constraints
THETA_S_PLANCK = 1.04109e-2  # 100*theta_s
THETA_S_ERR = 0.00030e-2
SIGMA8_PLANCK = 0.811
SIGMA8_ERR = 0.006


# ============================================================================
# Analysis Functions
# ============================================================================

def evaluate_whbc_pk_point(params_dict):
    """
    Evaluate a single WHBC P(k) parameter point.

    Args:
        params_dict: Dictionary of WHBC P(k) parameters

    Returns:
        Dictionary with results
    """
    try:
        # Create parameters
        whbc_params = WHBCPrimordialParameters(**params_dict)

        # Get approximate CMB effects
        effects = approximate_cmb_effects(whbc_params)

        # Analyze primordial spectrum
        analysis = analyze_whbc_primordial(whbc_params)

        # Compute sigma8 ratio from primordial analysis
        sigma8_ratio = analysis.sigma8_ratio

        # Estimate chi^2 based on sigma8 modification
        # P(k) modifications primarily affect sigma8, not theta_s
        sigma8_predicted = SIGMA8_PLANCK * sigma8_ratio
        chi2_sigma8 = ((sigma8_predicted - SIGMA8_PLANCK) / SIGMA8_ERR) ** 2

        # theta_s is essentially unchanged by P(k) modifications
        # (theta_s depends on r_s and D_A, not primordial spectrum)
        chi2_theta_s = 0.0

        chi2_total = chi2_sigma8 + chi2_theta_s

        # Check if passes constraints
        passes = (
            abs(sigma8_ratio - 1.0) < 0.05  # sigma8 within 5%
        )

        result = {
            'params': params_dict,
            'sigma8_ratio': float(sigma8_ratio),
            'sigma8_predicted': float(sigma8_predicted),
            'F_at_pivot': float(effects['F_at_pivot']),
            'F_at_peak1': float(effects['F_at_peak1']),
            'F_at_damping': float(effects['F_at_damping']),
            'peak_ratio': float(effects['peak_ratio_approx']),
            'n_s_eff': float(analysis.n_s_eff_pivot),
            'running': float(analysis.alpha_pivot),
            'chi2_sigma8': float(chi2_sigma8),
            'chi2_theta_s': float(chi2_theta_s),
            'chi2_total': float(chi2_total),
            'passes_constraints': bool(passes),
            'success': True,
            'message': 'OK',
        }

        return result

    except Exception as e:
        return {
            'params': params_dict,
            'success': False,
            'message': str(e),
            'chi2_total': float('inf'),
            'passes_constraints': False,
        }


def evaluate_with_camb(params_dict):
    """
    Evaluate using CAMB for more accurate results.

    Args:
        params_dict: Dictionary of WHBC P(k) parameters

    Returns:
        Dictionary with results including CAMB CMB spectra
    """
    if not HAS_CAMB:
        return evaluate_whbc_pk_point(params_dict)

    try:
        whbc_params = WHBCPrimordialParameters(**params_dict)

        # Run CAMB
        result = run_camb_with_whbc(whbc_params, lmax=2000)

        if not result.success:
            return {
                'params': params_dict,
                'success': False,
                'message': result.message,
                'chi2_total': float('inf'),
                'passes_constraints': False,
            }

        # Compute chi^2
        chi2 = compute_cmb_chi2(result)

        # Check constraints
        passes = (
            abs(result.theta_s - THETA_S_PLANCK * 100) < 3 * THETA_S_ERR * 100 and
            abs(result.sigma8 - SIGMA8_PLANCK) < 3 * SIGMA8_ERR
        )

        return {
            'params': params_dict,
            'theta_s': float(result.theta_s),
            'sigma8': float(result.sigma8),
            'r_s': float(result.r_s),
            'chi2_total': float(chi2) if np.isfinite(chi2) else float('inf'),
            'passes_constraints': bool(passes),
            'success': True,
            'message': 'OK (CAMB)',
        }

    except Exception as e:
        return {
            'params': params_dict,
            'success': False,
            'message': str(e),
            'chi2_total': float('inf'),
            'passes_constraints': False,
        }


def run_scan(use_camb=False, parallel=True, n_jobs=-1):
    """
    Run full parameter space scan.

    Args:
        use_camb: Whether to use CAMB (slower but more accurate)
        parallel: Whether to parallelize
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        List of result dictionaries
    """
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(product(*param_values))

    print(f"Total parameter combinations: {len(combinations)}")

    # Create parameter dictionaries
    param_dicts = [dict(zip(param_names, combo)) for combo in combinations]

    # Choose evaluation function
    eval_func = evaluate_with_camb if use_camb else evaluate_whbc_pk_point

    # Run scan
    if parallel and len(param_dicts) > 1:
        if n_jobs == -1:
            n_jobs = min(mp.cpu_count(), 8)  # Cap at 8 for memory

        print(f"Running parallel scan with {n_jobs} workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(eval_func, param_dicts))
    else:
        print("Running sequential scan...")
        results = []
        for i, params in enumerate(param_dicts):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(param_dicts)}")
            results.append(eval_func(params))

    return results


def analyze_results(results):
    """
    Analyze scan results.

    Args:
        results: List of result dictionaries

    Returns:
        Summary dictionary
    """
    successful = [r for r in results if r.get('success', False)]
    passing = [r for r in successful if r.get('passes_constraints', False)]

    # Find best model
    if successful:
        best = min(successful, key=lambda x: x.get('chi2_total', float('inf')))
    else:
        best = None

    # Compute statistics
    summary = {
        'total_points': len(results),
        'successful_points': len(successful),
        'passing_points': len(passing),
        'pass_fraction': len(passing) / len(results) if results else 0,
    }

    if best:
        summary['best_model'] = best

    # Analyze sigma8 range in passing models
    if passing:
        sigma8_ratios = [r.get('sigma8_ratio', 1.0) for r in passing]
        summary['sigma8_ratio_range'] = [min(sigma8_ratios), max(sigma8_ratios)]
    else:
        summary['sigma8_ratio_range'] = [1.0, 1.0]

    return summary


def create_plots(results, output_dir):
    """
    Create analysis plots.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    successful = [r for r in results if r.get('success', False)]

    if not successful:
        print("No successful results to plot")
        return

    # Extract data
    A_cut_vals = [r['params']['A_cut'] for r in successful]
    A_osc_vals = [r['params']['A_osc'] for r in successful]
    sigma8_ratios = [r.get('sigma8_ratio', 1.0) for r in successful]
    chi2_vals = [r.get('chi2_total', np.inf) for r in successful]
    passes = [r.get('passes_constraints', False) for r in successful]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: A_cut vs sigma8 ratio
    ax = axes[0, 0]
    colors = ['green' if p else 'red' for p in passes]
    ax.scatter(A_cut_vals, sigma8_ratios, c=colors, alpha=0.6, s=30)
    ax.axhline(1.0, color='k', linestyle='--', label='LCDM')
    ax.axhline(1.05, color='gray', linestyle=':', label='5% bounds')
    ax.axhline(0.95, color='gray', linestyle=':')
    ax.set_xlabel('$A_{cut}$')
    ax.set_ylabel('$\\sigma_8^{WHBC} / \\sigma_8^{LCDM}$')
    ax.set_title('IR Cutoff Effect on $\\sigma_8$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: A_osc vs sigma8 ratio
    ax = axes[0, 1]
    ax.scatter(A_osc_vals, sigma8_ratios, c=colors, alpha=0.6, s=30)
    ax.axhline(1.0, color='k', linestyle='--')
    ax.axhline(1.05, color='gray', linestyle=':')
    ax.axhline(0.95, color='gray', linestyle=':')
    ax.set_xlabel('$A_{osc}$')
    ax.set_ylabel('$\\sigma_8^{WHBC} / \\sigma_8^{LCDM}$')
    ax.set_title('Oscillation Amplitude Effect on $\\sigma_8$')
    ax.grid(True, alpha=0.3)

    # Plot 3: chi^2 histogram
    ax = axes[1, 0]
    chi2_finite = [c for c in chi2_vals if np.isfinite(c) and c < 100]
    if chi2_finite:
        ax.hist(chi2_finite, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('$\\chi^2$')
    ax.set_ylabel('Count')
    ax.set_title('$\\chi^2$ Distribution')
    ax.grid(True, alpha=0.3)

    # Plot 4: Example P(k) modifications
    ax = axes[1, 1]
    k_test = np.logspace(-4, 1, 200)

    # LCDM
    ax.semilogx(k_test, np.ones_like(k_test), 'k-', lw=2, label='LCDM')

    # Best model if available
    best_models = sorted(successful, key=lambda x: x.get('chi2_total', np.inf))[:3]
    for i, model in enumerate(best_models):
        params = WHBCPrimordialParameters(**model['params'])
        F = primordial_ratio(k_test, params)
        ax.semilogx(k_test, F, '--', alpha=0.8,
                   label=f"Best {i+1}: $\\chi^2$={model.get('chi2_total', np.inf):.1f}")

    ax.set_xlabel('k [Mpc$^{-1}$]')
    ax.set_ylabel('$F_{WHBC}(k) = P_{WHBC}/P_{LCDM}$')
    ax.set_title('Best-fit P(k) Modifications')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'whbc_pk_scan_results.png'), dpi=150)
    plt.close()

    print(f"Saved plots to {output_dir}/whbc_pk_scan_results.png")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("T08_WHBC_PK: White-Hole Boundary Primordial P(k) Scan")
    print("=" * 70)
    print(f"\nDate: {datetime.datetime.now().isoformat()}")
    print(f"CAMB available: {HAS_CAMB}")
    print(f"CLASS available: {HAS_CLASS}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run scan (use CAMB if available, otherwise use approximations)
    use_camb = HAS_CAMB
    print(f"\nUsing {'CAMB' if use_camb else 'approximations'} for CMB calculations")

    results = run_scan(use_camb=use_camb, parallel=True)

    # Analyze results
    summary = analyze_results(results)

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total points scanned: {summary['total_points']}")
    print(f"Successful evaluations: {summary['successful_points']}")
    print(f"Passing constraints: {summary['passing_points']}")
    print(f"Pass fraction: {summary['pass_fraction']*100:.1f}%")

    if summary.get('best_model'):
        best = summary['best_model']
        print(f"\nBest model:")
        print(f"  A_cut = {best['params']['A_cut']}")
        print(f"  k_cut = {best['params']['k_cut']} Mpc^-1")
        print(f"  A_osc = {best['params']['A_osc']}")
        print(f"  omega_WH = {best['params']['omega_WH']}")
        print(f"  sigma8_ratio = {best.get('sigma8_ratio', 'N/A')}")
        print(f"  chi^2 = {best.get('chi2_total', 'N/A')}")

    # Key findings for Hubble tension
    print(f"\n{'=' * 70}")
    print("HUBBLE TENSION ASSESSMENT")
    print(f"{'=' * 70}")
    print("""
WHBC P(k) modifications affect sigma8 but NOT theta_s or r_s directly.
The Hubble tension involves the CMB sound horizon r_s and angular scale theta_s,
which are determined by background cosmology, not the primordial spectrum.

Therefore, P(k) modifications ALONE cannot resolve the Hubble tension.

Key insight: To shift H0 inference from CMB, one needs to modify:
1. The sound horizon r_s (requires early-universe physics changes)
2. The angular diameter distance D_A (requires late-time cosmology changes)

Primordial P(k) features can help with S8 tension (sigma8 x Omega_m^0.5)
but not H0 tension.
""")

    # Determine if model helps at all
    can_help = (
        summary['passing_points'] > 0 and
        summary['sigma8_ratio_range'][0] < 0.98  # Can reduce sigma8
    )

    summary['can_help_s8_tension'] = can_help
    summary['can_help_h0_tension'] = False  # P(k) cannot help H0
    summary['assessment'] = (
        "WHBC P(k) modifications CAN potentially help S8 tension by reducing sigma8, "
        "but CANNOT help H0 tension as they don't affect r_s or D_A."
    )

    # Save results
    output = {
        'simulation': 'T08_WHBC_PK - WHBC Primordial Power Spectrum',
        'date': datetime.datetime.now().isoformat(),
        'parameters_scanned': PARAM_GRID,
        **summary,
        'all_results': results,
    }

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/summary.json")

    # Create plots
    create_plots(results, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print("SIMULATION COMPLETE")
    print(f"{'=' * 70}")

    return summary


if __name__ == "__main__":
    main()
