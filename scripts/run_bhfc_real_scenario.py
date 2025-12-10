#!/usr/bin/env python3
"""
SIMULATION 9: T09_BHFC_REAL - Early Black-Hole Dominated Fertility Cosmology

This script runs a parameter scan over BHFC models to test whether:
1. An early BH-dominated phase can be consistent with CMB/BAO/SN constraints
2. Such models can generate a meaningful early-late H0 split

GR-consistent: All energy gravitates through T_mu_nu.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.components.black_hole_fertility import (
    BHFCRealParameters,
    BHFCBackgroundCosmology,
    compute_H0_early_late,
    check_constraints,
)


# =============================================================================
# T09_BHFC_REAL Parameter Grid
# =============================================================================

PARAM_GRID = {
    'f_BH_init': [0.0, 0.1, 0.3, 0.5],      # Fraction of matter into BHs
    'z_form': [1e4, 1e5, 1e6],               # Formation redshift
    'z_evap': [None, 1e4],                    # Evaporation redshift (None = no evap)
    'A_eff': [0.5, 1.0, 2.0],                # Formation sharpness
    'f_evap': [0.0, 0.5],                     # Evaporation fraction
}


def evaluate_single_model(params_dict: Dict) -> Dict[str, Any]:
    """Evaluate a single BHFC model.

    Args:
        params_dict: Dictionary of BHFC parameters

    Returns:
        Results dictionary with distances, H0 split, and constraint checks
    """
    try:
        # Create parameters
        params = BHFCRealParameters(
            f_BH_init=params_dict['f_BH_init'],
            z_form=params_dict['z_form'],
            z_evap=params_dict['z_evap'],
            A_eff=params_dict['A_eff'],
            f_evap=params_dict['f_evap'],
        )

        # Create cosmology
        cosmo = BHFCBackgroundCosmology(params)

        # Compute distances
        distances = cosmo.compute_distances()

        # Compute H0 early vs late
        h0_split = compute_H0_early_late(cosmo)

        # Check constraints
        constraints = check_constraints(cosmo)

        # Compute H ratio at key redshifts
        z_array = np.array([0, 10, 100, 1000, 1e4, 1e5])
        z_array = z_array[z_array <= max(params.z_form * 10, 3000)]
        H_ratios = cosmo.H_ratio_vs_LCDM(z_array)

        # BH fraction evolution
        z_bh = np.logspace(0, np.log10(max(params.z_form * 2, 1e4)), 50)
        bh_fractions = cosmo.BH_fraction(z_bh)
        max_bh_frac = float(np.max(bh_fractions))

        return {
            'params': params_dict,
            'success': True,
            'distances': distances,
            'h0_split': h0_split,
            'constraints': constraints,
            'passes_all': constraints['passes_all'],
            'H0_Early': h0_split['H0_Early'],
            'H0_Late': h0_split['H0_Late'],
            'Delta_H0': h0_split['Delta_H0'],
            'theta_s': distances['theta_s'],
            'r_s': distances['r_s'],
            'max_H_ratio': float(np.max(np.abs(H_ratios - 1.0))),
            'max_BH_fraction': max_bh_frac,
            'delta_theta_s_percent': constraints['delta_theta_s_percent'],
            'delta_D_M_percent': constraints['delta_D_M_percent'],
        }

    except Exception as e:
        return {
            'params': params_dict,
            'success': False,
            'error': str(e),
            'passes_all': False,
            'Delta_H0': 0.0,
        }


def run_parameter_scan(n_workers: int = 4) -> List[Dict]:
    """Run full parameter scan over BHFC models.

    Args:
        n_workers: Number of parallel workers

    Returns:
        List of result dictionaries
    """
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    all_combos = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        # Skip invalid combinations
        if combo['z_evap'] is not None and combo['z_evap'] >= combo['z_form']:
            continue  # Evaporation must be after formation (lower z)
        if combo['f_evap'] > 0 and combo['z_evap'] is None:
            continue  # Can't have evaporation without evaporation redshift
        if combo['f_BH_init'] == 0 and combo['f_evap'] > 0:
            continue  # Can't evaporate if no BHs
        all_combos.append(combo)

    print(f"Total parameter combinations: {len(all_combos)}")

    # Run scan
    results = []
    if n_workers == 1:
        for i, combo in enumerate(all_combos):
            print(f"  [{i+1}/{len(all_combos)}] f_BH={combo['f_BH_init']}, "
                  f"z_form={combo['z_form']:.0e}, A_eff={combo['A_eff']}")
            result = evaluate_single_model(combo)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_single_model, c): c for c in all_combos}
            for i, future in enumerate(as_completed(futures)):
                combo = futures[future]
                result = future.result()
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(all_combos)}")

    return results


def generate_plots(results: List[Dict], output_dir: str):
    """Generate diagnostic plots.

    Args:
        results: List of result dictionaries
        output_dir: Output directory for figures
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    passing = [r for r in successful if r.get('passes_all', False)]

    print(f"\nGenerating plots...")
    print(f"  Successful evaluations: {len(successful)}")
    print(f"  Passing constraints: {len(passing)}")

    # 1. H ratio examples
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select example models
    if len(passing) > 0:
        # Best passing model (max |Delta_H0|)
        best_pass = max(passing, key=lambda x: abs(x.get('Delta_H0', 0)))
        params = BHFCRealParameters(**{k: v for k, v in best_pass['params'].items()})
        cosmo = BHFCBackgroundCosmology(params)
        z_arr = np.logspace(-1, 5, 200)
        z_arr = z_arr[z_arr <= params.z_form * 10]
        H_ratio = cosmo.H_ratio_vs_LCDM(z_arr)
        ax.semilogx(z_arr, H_ratio, 'b-', linewidth=2, label=f'Best PASS: f_BH={params.f_BH_init}')

    # Also plot a typical failing model
    failing = [r for r in successful if not r.get('passes_all', False) and r['params']['f_BH_init'] > 0]
    if len(failing) > 0:
        worst = max(failing, key=lambda x: x.get('delta_theta_s_percent', 0))
        params = BHFCRealParameters(**{k: v for k, v in worst['params'].items()})
        cosmo = BHFCBackgroundCosmology(params)
        z_arr = np.logspace(-1, 5, 200)
        z_arr = z_arr[z_arr <= params.z_form * 10]
        H_ratio = cosmo.H_ratio_vs_LCDM(z_arr)
        ax.semilogx(z_arr, H_ratio, 'r--', linewidth=2, label=f'FAIL: f_BH={params.f_BH_init}')

    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel(r'$H_{\rm BHFC}(z) / H_{\rm \Lambda CDM}(z)$')
    ax.set_title('BHFC Hubble Parameter Ratio vs LCDM')
    ax.legend()
    ax.set_ylim(0.9, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'H_ratio_examples.png'), dpi=150)
    plt.close()

    # 2. BH fraction evolution
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in successful[:5]:  # Plot first few
        if r['params']['f_BH_init'] > 0:
            params = BHFCRealParameters(**{k: v for k, v in r['params'].items()})
            cosmo = BHFCBackgroundCosmology(params)
            z_arr = np.logspace(0, np.log10(params.z_form * 2), 100)
            bh_frac = cosmo.BH_fraction(z_arr)
            label = f"f_BH={params.f_BH_init}, z_form={params.z_form:.0e}"
            ax.semilogx(z_arr, bh_frac * 100, label=label)

    ax.set_xlabel('Redshift z')
    ax.set_ylabel(r'$\rho_{\rm BH} / \rho_{\rm total}$ (%)')
    ax.set_title('Black Hole Fraction Evolution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'BH_fraction_evolution.png'), dpi=150)
    plt.close()

    # 3. H0_Early vs H0_Late scatter
    fig, ax = plt.subplots(figsize=(8, 8))

    H0_early = [r['H0_Early'] for r in successful if 'H0_Early' in r]
    H0_late = [r['H0_Late'] for r in successful if 'H0_Late' in r]
    passes = [r['passes_all'] for r in successful if 'H0_Early' in r]

    colors = ['green' if p else 'red' for p in passes]
    ax.scatter(H0_early, H0_late, c=colors, alpha=0.6, s=50)

    # Reference lines
    ax.axhline(67.4, color='blue', linestyle='--', alpha=0.5, label='Planck H0')
    ax.axhline(73.0, color='orange', linestyle='--', alpha=0.5, label='SH0ES H0')
    ax.axvline(67.4, color='blue', linestyle='--', alpha=0.5)
    ax.plot([60, 80], [60, 80], 'k:', alpha=0.5)

    ax.set_xlabel('H0_Early (km/s/Mpc)')
    ax.set_ylabel('H0_Late (km/s/Mpc)')
    ax.set_title('Early vs Late H0 Inference')
    ax.legend()
    ax.set_xlim(60, 80)
    ax.set_ylim(60, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'H0_Early_vs_H0_Late.png'), dpi=150)
    plt.close()

    # 4. Viability map over (f_BH_init, z_form)
    fig, ax = plt.subplots(figsize=(10, 6))

    f_BH_vals = sorted(set(r['params']['f_BH_init'] for r in successful))
    z_form_vals = sorted(set(r['params']['z_form'] for r in successful))

    pass_matrix = np.zeros((len(z_form_vals), len(f_BH_vals)))
    count_matrix = np.zeros((len(z_form_vals), len(f_BH_vals)))

    for r in successful:
        f_idx = f_BH_vals.index(r['params']['f_BH_init'])
        z_idx = z_form_vals.index(r['params']['z_form'])
        count_matrix[z_idx, f_idx] += 1
        if r['passes_all']:
            pass_matrix[z_idx, f_idx] += 1

    # Compute pass fraction
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_matrix = np.where(count_matrix > 0, pass_matrix / count_matrix, 0)

    im = ax.imshow(frac_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   origin='lower')
    ax.set_xticks(range(len(f_BH_vals)))
    ax.set_xticklabels([f'{v:.1f}' for v in f_BH_vals])
    ax.set_yticks(range(len(z_form_vals)))
    ax.set_yticklabels([f'{v:.0e}' for v in z_form_vals])
    ax.set_xlabel('f_BH_init')
    ax.set_ylabel('z_form')
    ax.set_title('Viability Map: Fraction Passing Constraints')
    plt.colorbar(im, ax=ax, label='Pass Fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'viability_map.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


def save_results(results: List[Dict], output_dir: str):
    """Save scan results.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        jr = {
            'params': r['params'],
            'success': r.get('success', False),
            'passes_all': r.get('passes_all', False),
        }
        if r.get('success', False):
            jr.update({
                'H0_Early': r.get('H0_Early', np.nan),
                'H0_Late': r.get('H0_Late', np.nan),
                'Delta_H0': r.get('Delta_H0', 0),
                'theta_s': r.get('theta_s', np.nan),
                'r_s': r.get('r_s', np.nan),
                'delta_theta_s_percent': r.get('delta_theta_s_percent', np.nan),
                'delta_D_A_percent': r.get('delta_D_A_percent', np.nan),
                'max_H_ratio': r.get('max_H_ratio', np.nan),
                'max_BH_fraction': r.get('max_BH_fraction', 0),
            })
        else:
            jr['error'] = r.get('error', 'Unknown error')
        json_results.append(jr)

    # Save JSON
    with open(os.path.join(output_dir, 'scan_results.json'), 'w') as f:
        json.dump({
            'simulation': 'T09_BHFC_REAL - Early Black-Hole Dominated Fertility Cosmology',
            'date': datetime.now().isoformat(),
            'param_grid': PARAM_GRID,
            'total_models': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'passing': sum(1 for r in results if r.get('passes_all', False)),
            'results': json_results,
        }, f, indent=2, default=str)

    # Save NPZ for numerical analysis
    successful = [r for r in results if r.get('success', False)]
    if len(successful) > 0:
        np.savez(
            os.path.join(output_dir, 'scan_results.npz'),
            f_BH_init=np.array([r['params']['f_BH_init'] for r in successful]),
            z_form=np.array([r['params']['z_form'] for r in successful]),
            A_eff=np.array([r['params']['A_eff'] for r in successful]),
            f_evap=np.array([r['params']['f_evap'] for r in successful]),
            passes_all=np.array([r['passes_all'] for r in successful]),
            Delta_H0=np.array([r.get('Delta_H0', 0) for r in successful]),
            H0_Early=np.array([r.get('H0_Early', np.nan) for r in successful]),
            H0_Late=np.array([r.get('H0_Late', np.nan) for r in successful]),
            delta_theta_s=np.array([r.get('delta_theta_s_percent', np.nan) for r in successful]),
        )

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: List[Dict]):
    """Print summary of scan results."""
    successful = [r for r in results if r.get('success', False)]
    passing = [r for r in successful if r.get('passes_all', False)]

    print("\n" + "="*60)
    print("T09_BHFC_REAL SCAN SUMMARY")
    print("="*60)
    print(f"Total models evaluated: {len(results)}")
    print(f"Successful evaluations: {len(successful)}")
    print(f"Passing all constraints: {len(passing)}")
    print(f"Pass rate: {100*len(passing)/max(len(successful),1):.1f}%")

    if len(passing) > 0:
        Delta_H0_vals = [abs(r['Delta_H0']) for r in passing]
        best = max(passing, key=lambda x: abs(x['Delta_H0']))

        print(f"\nMax |Delta_H0| among PASS models: {max(Delta_H0_vals):.2f} km/s/Mpc")
        print(f"Best model parameters:")
        print(f"  f_BH_init = {best['params']['f_BH_init']}")
        print(f"  z_form    = {best['params']['z_form']:.0e}")
        print(f"  z_evap    = {best['params']['z_evap']}")
        print(f"  A_eff     = {best['params']['A_eff']}")
        print(f"  f_evap    = {best['params']['f_evap']}")
        print(f"  H0_Early  = {best['H0_Early']:.2f} km/s/Mpc")
        print(f"  H0_Late   = {best['H0_Late']:.2f} km/s/Mpc")
    else:
        print("\nNo models passed all constraints!")

    print("="*60)


def main():
    """Main entry point."""
    print("="*60)
    print("SIMULATION 9: T09_BHFC_REAL")
    print("Early Black-Hole Dominated Fertility Cosmology")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print()

    # Run scan
    print("Running parameter scan...")
    results = run_parameter_scan(n_workers=4)

    # Output directories
    results_dir = 'results/simulation_9_bhfc_real'
    figures_dir = 'figures/simulation_9_bhfc_real'

    # Save results
    save_results(results, results_dir)

    # Generate plots
    generate_plots(results, figures_dir)

    # Print summary
    print_summary(results)

    print(f"\nResults: {results_dir}/scan_results.json")
    print(f"Figures: {figures_dir}/")


if __name__ == '__main__':
    main()
