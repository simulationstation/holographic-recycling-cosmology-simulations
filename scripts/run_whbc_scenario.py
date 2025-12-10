#!/usr/bin/env python3
"""
SIMULATION 7: White-Hole Boundary Cosmology (WHBC) - T07_WHBC

This script runs the complete WHBC parameter scan and generates:
1. Modified H(z), r_s, theta_s calculations
2. Distance measures: D_L, D_A, chi(z), D_V(z)
3. Perturbation spectra P(k) with damping
4. Viability tests against CMB, BAO, SNe, growth
5. Plots and CLASS export files

Usage:
    python scripts/run_whbc_scenario.py [--parallel] [--n-jobs N]
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.horizon_models.refinement_whbc import (
    WHBCParameters, WHBCModel, WHBCResult,
    create_whbc_model, K_PIVOT, Z_REC, Z_DRAG
)


# Output directories
RESULTS_DIR = "results/simulation_7_whbc"
FIGURES_DIR = "figures/simulation_7_whbc"
CLASS_EXPORT_DIR = f"{RESULTS_DIR}/class_export"


def setup_directories():
    """Create output directories."""
    for d in [RESULTS_DIR, FIGURES_DIR, CLASS_EXPORT_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"Output directories created: {RESULTS_DIR}, {FIGURES_DIR}")


def evaluate_whbc_point(params_tuple):
    """
    Evaluate a single WHBC parameter point.

    Args:
        params_tuple: (epsilon_WH, beta_WH, gamma_WH, xi_WH)

    Returns:
        WHBCResult or None if error
    """
    eps, beta, gamma, xi = params_tuple
    try:
        model = create_whbc_model(
            epsilon_WH=eps,
            beta_WH=beta,
            gamma_WH=gamma,
            xi_WH=xi
        )
        result = model.solve()
        return result
    except Exception as e:
        print(f"Error at eps={eps}, beta={beta}, gamma={gamma}, xi={xi}: {e}")
        return None


def run_parameter_scan(parallel=True, n_jobs=-1):
    """
    Run full WHBC parameter scan.

    Args:
        parallel: Whether to parallelize
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        List of WHBCResult objects
    """
    print("\n" + "="*60)
    print("SIMULATION 7: WHITE-HOLE BOUNDARY COSMOLOGY (WHBC)")
    print("="*60)

    # Parameter grids (as specified)
    epsilon_values = [0.00, 0.05, 0.10, 0.15]
    beta_values = [0.00, 0.02, 0.04, 0.06]
    gamma_values = [0.0, 0.5, 1.0]
    xi_values = [0.0, 0.05]

    # Generate all parameter combinations
    param_combos = []
    for eps in epsilon_values:
        for beta in beta_values:
            for gamma in gamma_values:
                for xi in xi_values:
                    param_combos.append((eps, beta, gamma, xi))

    n_total = len(param_combos)
    print(f"\nParameter grid:")
    print(f"  epsilon_WH: {epsilon_values}")
    print(f"  beta_WH: {beta_values}")
    print(f"  gamma_WH: {gamma_values}")
    print(f"  xi_WH: {xi_values}")
    print(f"\nTotal parameter points: {n_total}")

    start_time = time.time()

    if parallel:
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        print(f"\nRunning parallel scan with {n_jobs} workers...")

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(evaluate_whbc_point, p): p for p in param_combos}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    results.append(result)

                if (i + 1) % 10 == 0 or (i + 1) == n_total:
                    elapsed = time.time() - start_time
                    pct = 100 * (i + 1) / n_total
                    print(f"  Progress: {i+1}/{n_total} ({pct:.1f}%), elapsed: {elapsed:.1f}s")
    else:
        print("\nRunning sequential scan...")
        results = []
        for i, params in enumerate(param_combos):
            result = evaluate_whbc_point(params)
            if result is not None:
                results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == n_total:
                elapsed = time.time() - start_time
                pct = 100 * (i + 1) / n_total
                print(f"  Progress: {i+1}/{n_total} ({pct:.1f}%), elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"\nScan completed in {elapsed_total:.1f} seconds")
    print(f"Successfully evaluated: {len(results)}/{n_total}")

    return results


def analyze_results(results):
    """
    Analyze WHBC results and compute statistics.

    Args:
        results: List of WHBCResult objects

    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*60)
    print("WHBC RESULTS ANALYSIS")
    print("="*60)

    # Count passing points
    passing = [r for r in results if r.passes_constraints]
    n_passing = len(passing)
    n_total = len(results)

    print(f"\nConstraint Summary:")
    print(f"  Total evaluated: {n_total}")
    print(f"  Passed all constraints: {n_passing} ({100*n_passing/n_total:.1f}%)")

    # Find best models (lowest chi2)
    sorted_results = sorted(results, key=lambda r: r.chi2_total)
    best_results = sorted_results[:5]

    print(f"\nTop 5 models by chi^2:")
    print("-" * 80)
    print(f"{'eps_WH':>8} {'beta_WH':>8} {'gamma_WH':>8} {'xi_WH':>8} {'chi2':>12} {'theta_s_dev':>12} {'Pass':>6}")
    print("-" * 80)

    for r in best_results:
        p = r.params
        theta_dev = 100 * abs(r.theta_s_WHBC - r.theta_s_LCDM) / r.theta_s_LCDM
        print(f"{p.epsilon_WH:8.3f} {p.beta_WH:8.3f} {p.gamma_WH:8.3f} {p.xi_WH:8.3f} "
              f"{r.chi2_total:12.2f} {theta_dev:11.4f}% {'Yes' if r.passes_constraints else 'No':>6}")

    # Find models that might help Hubble tension
    # Look for models where H ratio at z~1000 differs from 1
    print("\n" + "-"*60)
    print("HUBBLE TENSION ANALYSIS")
    print("-"*60)

    # Get H ratios at recombination
    h_ratios_rec = []
    for r in results:
        idx = np.argmin(np.abs(r.z_array - 1000))
        h_ratio = r.H_ratio[idx]
        h_ratios_rec.append((r, h_ratio))

    # Sort by H ratio deviation
    h_ratios_rec.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)

    print(f"\nModels with largest H(z=1000) deviation:")
    print(f"{'eps':>6} {'beta':>6} {'gamma':>6} {'xi':>6} {'H_ratio':>10} {'theta_dev':>12} {'Pass':>6}")
    for r, h_ratio in h_ratios_rec[:5]:
        p = r.params
        theta_dev = 100 * abs(r.theta_s_WHBC - r.theta_s_LCDM) / r.theta_s_LCDM
        print(f"{p.epsilon_WH:6.2f} {p.beta_WH:6.2f} {p.gamma_WH:6.2f} {p.xi_WH:6.2f} "
              f"{h_ratio:10.4f} {theta_dev:11.4f}% {'Yes' if r.passes_constraints else 'No':>6}")

    # Build analysis dictionary
    analysis = {
        'n_total': n_total,
        'n_passing': n_passing,
        'pass_fraction': n_passing / n_total if n_total > 0 else 0,
        'best_model': best_results[0].to_dict() if best_results else None,
        'best_5_models': [r.to_dict() for r in best_results],
        'largest_H_deviation': [
            {
                'params': r.params.to_dict(),
                'H_ratio_z1000': float(h_ratio),
                'theta_s_dev_pct': float(100 * abs(r.theta_s_WHBC - r.theta_s_LCDM) / r.theta_s_LCDM),
                'passes': r.passes_constraints
            }
            for r, h_ratio in h_ratios_rec[:5]
        ]
    }

    return analysis


def generate_plots(results):
    """
    Generate all WHBC visualization plots.

    Args:
        results: List of WHBCResult objects
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Get LCDM reference
    lcdm_model = create_whbc_model()  # Default params = LCDM
    lcdm_result = lcdm_model.solve()

    # Select representative models for plotting
    # Pick a few with varying parameters
    plot_results = []
    eps_vals = [0.0, 0.05, 0.10]
    beta_vals = [0.0, 0.02, 0.04]

    for r in results:
        p = r.params
        if p.gamma_WH == 0.0 and p.xi_WH == 0.0:
            if (p.epsilon_WH in eps_vals and p.beta_WH == 0.0) or \
               (p.beta_WH in beta_vals and p.epsilon_WH == 0.0):
                plot_results.append(r)

    # Also add some with gamma and xi
    for r in results:
        p = r.params
        if p.epsilon_WH == 0.05 and p.beta_WH == 0.02:
            if p.gamma_WH in [0.5, 1.0] or p.xi_WH == 0.05:
                plot_results.append(r)

    # Remove duplicates
    seen = set()
    unique_results = []
    for r in plot_results:
        p = r.params
        key = (p.epsilon_WH, p.beta_WH, p.gamma_WH, p.xi_WH)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    plot_results = unique_results[:8]  # Limit to 8 for clarity

    # 1. H_ratio plot
    print("  Generating H_ratio.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    z_plot = np.logspace(-2, 3.2, 200)
    ax.axhline(y=1.0, color='k', linestyle='--', label='LCDM', linewidth=2)

    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_results)))
    for r, color in zip(plot_results, colors):
        p = r.params
        model = create_whbc_model(
            epsilon_WH=p.epsilon_WH,
            beta_WH=p.beta_WH,
            gamma_WH=p.gamma_WH,
            xi_WH=p.xi_WH
        )
        H_ratio = [model.H_ratio(z) for z in z_plot]
        label = f"eps={p.epsilon_WH:.2f}, beta={p.beta_WH:.2f}"
        ax.plot(z_plot, H_ratio, color=color, label=label, linewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel(r'$H_{WHBC}/H_{LCDM}$', fontsize=12)
    ax.set_title('WHBC Hubble Parameter Ratio', fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim([0.01, 2000])
    ax.axvline(x=Z_REC, color='red', linestyle=':', alpha=0.5, label='z_rec')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/H_ratio.png", dpi=150)
    plt.close()

    # 2. r_s shift plot
    print("  Generating r_s_shift.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    eps_arr = np.linspace(0, 0.15, 20)
    beta_arr = np.linspace(0, 0.06, 20)

    # Plot r_s vs epsilon for beta=0
    r_s_vs_eps = []
    for eps in eps_arr:
        m = create_whbc_model(epsilon_WH=eps, beta_WH=0.0)
        res = m.solve()
        r_s_vs_eps.append(res.r_s_WHBC)
    ax.plot(eps_arr, r_s_vs_eps, 'b-', linewidth=2, label=r'$\beta_{WH}=0$')

    # Plot r_s vs beta for epsilon=0
    r_s_vs_beta = []
    for beta in beta_arr:
        m = create_whbc_model(epsilon_WH=0.0, beta_WH=beta)
        res = m.solve()
        r_s_vs_beta.append(res.r_s_WHBC)

    ax2 = ax.twiny()
    ax2.plot(beta_arr, r_s_vs_beta, 'r--', linewidth=2, label=r'$\epsilon_{WH}=0$')
    ax2.set_xlabel(r'$\beta_{WH}$', fontsize=12, color='red')
    ax2.tick_params(axis='x', labelcolor='red')

    ax.axhline(y=lcdm_result.r_s_LCDM, color='k', linestyle=':', label=f'LCDM: {lcdm_result.r_s_LCDM:.2f} Mpc')
    ax.set_xlabel(r'$\epsilon_{WH}$', fontsize=12, color='blue')
    ax.set_ylabel(r'$r_s$ [Mpc]', fontsize=12)
    ax.set_title('WHBC Sound Horizon Modification', fontsize=14)
    ax.tick_params(axis='x', labelcolor='blue')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/r_s_shift.png", dpi=150)
    plt.close()

    # 3. theta_s shift plot
    print("  Generating theta_s_shift.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute theta_s deviation across parameter space
    theta_dev_grid = np.zeros((len(eps_arr), len(beta_arr)))
    for i, eps in enumerate(eps_arr):
        for j, beta in enumerate(beta_arr):
            m = create_whbc_model(epsilon_WH=eps, beta_WH=beta)
            res = m.solve()
            theta_dev_grid[i, j] = 100 * abs(res.theta_s_WHBC - res.theta_s_LCDM) / res.theta_s_LCDM

    E, B = np.meshgrid(eps_arr, beta_arr)
    contour = ax.contourf(E, B, theta_dev_grid.T, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label=r'$|\Delta\theta_s/\theta_s|$ [%]')

    # Mark 0.05% contour (constraint threshold)
    ax.contour(E, B, theta_dev_grid.T, levels=[0.05], colors='red', linewidths=2)

    ax.set_xlabel(r'$\epsilon_{WH}$', fontsize=12)
    ax.set_ylabel(r'$\beta_{WH}$', fontsize=12)
    ax.set_title(r'WHBC $\theta_s$ Deviation from LCDM', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/theta_s_shift.png", dpi=150)
    plt.close()

    # 4. BAO distance ratios
    print("  Generating BAO_distance_ratios.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    z_bao = np.array([0.38, 0.51, 0.61, 1.0, 1.5, 2.0])

    for r, color in zip(plot_results[:5], colors[:5]):
        p = r.params
        model = create_whbc_model(
            epsilon_WH=p.epsilon_WH,
            beta_WH=p.beta_WH,
            gamma_WH=p.gamma_WH,
            xi_WH=p.xi_WH
        )

        # Compute D_V ratios
        D_V_whbc = np.array([model.D_V(z) for z in z_bao])
        D_V_lcdm = np.array([model._compute_D_V_lcdm(z) for z in z_bao])
        ratio = D_V_whbc / D_V_lcdm

        label = f"eps={p.epsilon_WH:.2f}, beta={p.beta_WH:.2f}"
        ax.plot(z_bao, ratio, 'o-', color=color, label=label, markersize=8)

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2)
    ax.axhspan(0.99, 1.01, alpha=0.2, color='green', label='1% constraint')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel(r'$D_V^{WHBC} / D_V^{LCDM}$', fontsize=12)
    ax.set_title('BAO Volume Distance Ratio', fontsize=14)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/BAO_distance_ratios.png", dpi=150)
    plt.close()

    # 5. Primordial damping window
    print("  Generating primordial_damping_window.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    k_arr = np.logspace(-4, 1, 200)
    gamma_vals = [0.0, 0.25, 0.5, 1.0, 2.0]

    for gamma in gamma_vals:
        model = create_whbc_model(gamma_WH=gamma)
        damping = np.array([model.damping_window(k) for k in k_arr])
        ax.plot(k_arr, damping, linewidth=2, label=f'$\\gamma_{{WH}}={gamma}$')

    ax.axvline(x=K_PIVOT, color='gray', linestyle='--', alpha=0.7, label=f'$k_{{pivot}}={K_PIVOT}$ Mpc$^{{-1}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=12)
    ax.set_ylabel(r'$D_{WH}(k)$', fontsize=12)
    ax.set_title('WHBC Perturbation Damping Window', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/primordial_damping_window.png", dpi=150)
    plt.close()

    # 6. Viability map
    print("  Generating viability_map.png...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Map results to grid
    eps_unique = sorted(set(r.params.epsilon_WH for r in results))
    beta_unique = sorted(set(r.params.beta_WH for r in results))
    gamma_unique = sorted(set(r.params.gamma_WH for r in results))
    xi_unique = sorted(set(r.params.xi_WH for r in results))

    # Panel 1: eps vs beta (gamma=0, xi=0)
    ax = axes[0, 0]
    viab = np.zeros((len(eps_unique), len(beta_unique)))
    for r in results:
        if r.params.gamma_WH == 0.0 and r.params.xi_WH == 0.0:
            i = eps_unique.index(r.params.epsilon_WH)
            j = beta_unique.index(r.params.beta_WH)
            viab[i, j] = 1 if r.passes_constraints else 0

    E, B = np.meshgrid(eps_unique, beta_unique)
    c = ax.pcolormesh(E, B, viab.T, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel(r'$\epsilon_{WH}$')
    ax.set_ylabel(r'$\beta_{WH}$')
    ax.set_title(r'Viability: $\gamma_{WH}=0, \xi_{WH}=0$')

    # Panel 2: eps vs gamma (beta=0, xi=0)
    ax = axes[0, 1]
    viab = np.zeros((len(eps_unique), len(gamma_unique)))
    for r in results:
        if r.params.beta_WH == 0.0 and r.params.xi_WH == 0.0:
            i = eps_unique.index(r.params.epsilon_WH)
            j = gamma_unique.index(r.params.gamma_WH)
            viab[i, j] = 1 if r.passes_constraints else 0

    E, G = np.meshgrid(eps_unique, gamma_unique)
    ax.pcolormesh(E, G, viab.T, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel(r'$\epsilon_{WH}$')
    ax.set_ylabel(r'$\gamma_{WH}$')
    ax.set_title(r'Viability: $\beta_{WH}=0, \xi_{WH}=0$')

    # Panel 3: beta vs gamma (eps=0, xi=0)
    ax = axes[1, 0]
    viab = np.zeros((len(beta_unique), len(gamma_unique)))
    for r in results:
        if r.params.epsilon_WH == 0.0 and r.params.xi_WH == 0.0:
            i = beta_unique.index(r.params.beta_WH)
            j = gamma_unique.index(r.params.gamma_WH)
            viab[i, j] = 1 if r.passes_constraints else 0

    B, G = np.meshgrid(beta_unique, gamma_unique)
    ax.pcolormesh(B, G, viab.T, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel(r'$\beta_{WH}$')
    ax.set_ylabel(r'$\gamma_{WH}$')
    ax.set_title(r'Viability: $\epsilon_{WH}=0, \xi_{WH}=0$')

    # Panel 4: chi2 distribution
    ax = axes[1, 1]
    chi2_vals = [r.chi2_total for r in results if r.chi2_total < 1e10]
    ax.hist(chi2_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(chi2_vals), color='red', linestyle='--',
               label=f'Median: {np.median(chi2_vals):.1f}')
    ax.set_xlabel(r'$\chi^2_{total}$')
    ax.set_ylabel('Count')
    ax.set_title(r'$\chi^2$ Distribution')
    ax.legend()
    ax.set_xlim([0, max(chi2_vals)])

    plt.suptitle('WHBC Viability Map', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/viability_map.png", dpi=150)
    plt.close()

    print(f"  All plots saved to {FIGURES_DIR}/")


def export_class_configs(results):
    """
    Export CLASS configuration files for best WHBC models.

    Args:
        results: List of WHBCResult objects
    """
    print("\n" + "="*60)
    print("EXPORTING CLASS CONFIGURATIONS")
    print("="*60)

    # Get best passing models
    passing = [r for r in results if r.passes_constraints]
    if not passing:
        # If none pass, get best by chi2
        passing = sorted(results, key=lambda r: r.chi2_total)[:3]
    else:
        passing = sorted(passing, key=lambda r: r.chi2_total)[:3]

    for i, r in enumerate(passing):
        p = r.params

        # CLASS .ini file
        ini_content = f"""# CLASS configuration for WHBC model
# Generated by run_whbc_scenario.py
# Date: {datetime.now().isoformat()}

# WHBC Parameters:
# epsilon_WH = {p.epsilon_WH}
# beta_WH = {p.beta_WH}
# gamma_WH = {p.gamma_WH}
# xi_WH = {p.xi_WH}

# Standard cosmological parameters
h = {p.H0/100:.4f}
Omega_b = {p.Omega_b:.5f}
Omega_cdm = {p.Omega_m - p.Omega_b:.5f}
Omega_Lambda = {p.Omega_Lambda:.5f}
n_s = {p.ns:.4f}
A_s = {p.As:.4e}
tau_reio = 0.054

# WHBC modifications (as comments - need custom CLASS)
# Modified sound horizon: r_s = {r.r_s_WHBC:.4f} Mpc
# Modified theta_s: theta_s = {r.theta_s_WHBC:.6e} rad
# theta_s deviation: {100*abs(r.theta_s_WHBC - r.theta_s_LCDM)/r.theta_s_LCDM:.4f}%

# Output settings
output = tCl, pCl, lCl, mPk
lensing = yes
l_max_scalars = 2500
P_k_max_h/Mpc = 1.0

# Precision settings
perturbations_verbose = 1
"""

        filename = f"{CLASS_EXPORT_DIR}/whbc_model_{i+1}.ini"
        with open(filename, 'w') as f:
            f.write(ini_content)

        print(f"  Saved: {filename}")
        print(f"    eps={p.epsilon_WH:.3f}, beta={p.beta_WH:.3f}, "
              f"gamma={p.gamma_WH:.3f}, xi={p.xi_WH:.3f}")

    # Also export transfer function modifications
    # Save as JSON for potential CAMB/CLASS modification
    transfer_mods = {
        'models': [
            {
                'index': i+1,
                'params': r.params.to_dict(),
                'r_s_ratio': r.r_s_WHBC / r.r_s_LCDM,
                'theta_s_ratio': r.theta_s_WHBC / r.theta_s_LCDM,
                'damping_k_pivot': K_PIVOT,
                'damping_gamma': r.params.gamma_WH
            }
            for i, r in enumerate(passing)
        ]
    }

    with open(f"{CLASS_EXPORT_DIR}/transfer_modifications.json", 'w') as f:
        json.dump(transfer_mods, f, indent=2)

    print(f"  Saved: {CLASS_EXPORT_DIR}/transfer_modifications.json")


def write_summary(results, analysis):
    """
    Write final summary file.

    Args:
        results: List of WHBCResult objects
        analysis: Analysis dictionary
    """
    print("\n" + "="*60)
    print("WRITING SUMMARY")
    print("="*60)

    # Compute additional statistics
    passing = [r for r in results if r.passes_constraints]

    # Check if WHBC can alleviate Hubble tension
    # Look at H ratio variations
    max_H_ratio = 1.0
    best_H_tension_model = None
    for r in results:
        if r.passes_constraints:
            idx = np.argmin(np.abs(r.z_array - 1000))
            h_ratio = r.H_ratio[idx]
            if abs(h_ratio - 1.0) > abs(max_H_ratio - 1.0):
                max_H_ratio = h_ratio
                best_H_tension_model = r

    # Hubble tension assessment
    # If WHBC can shift early H by a few percent while passing constraints,
    # it might help with tension
    can_help_tension = abs(max_H_ratio - 1.0) > 0.01  # More than 1% H shift

    summary = {
        'simulation': 'T07_WHBC - White-Hole Boundary Cosmology',
        'date': datetime.now().isoformat(),
        'parameters_scanned': {
            'epsilon_WH': [0.00, 0.05, 0.10, 0.15],
            'beta_WH': [0.00, 0.02, 0.04, 0.06],
            'gamma_WH': [0.0, 0.5, 1.0],
            'xi_WH': [0.0, 0.05]
        },
        'total_points': len(results),
        'passing_points': len(passing),
        'pass_fraction': len(passing) / len(results) if results else 0,
        'best_model': analysis['best_model'],
        'min_chi2': analysis['best_model']['chi2_total'] if analysis['best_model'] else None,
        'delta_theta_s_best': analysis['best_model']['theta_s_deviation_percent'] if analysis['best_model'] else None,
        'delta_r_s_best': (1 - analysis['best_model']['r_s_ratio']) * 100 if analysis['best_model'] else None,
        'max_H_ratio_z1000_passing': float(max_H_ratio),
        'can_alleviate_hubble_tension': can_help_tension,
        'hubble_tension_assessment': (
            "WHBC can modify early-universe H(z) while satisfying constraints"
            if can_help_tension else
            "WHBC modifications are too constrained to significantly help Hubble tension"
        ),
        'recommended_next_steps': [
            "Run full CMB power spectrum with CLASS/CAMB",
            "Perform MCMC parameter estimation",
            "Test perturbation growth predictions",
            "Explore higher epsilon_WH with adjusted beta_WH compensation"
        ],
        'key_findings': [
            f"Pass rate: {100*len(passing)/len(results):.1f}% of parameter points",
            f"Best chi^2: {analysis['best_model']['chi2_total']:.2f}" if analysis['best_model'] else "N/A",
            f"Maximum H(z=1000) shift in passing models: {100*(max_H_ratio-1):.2f}%",
            "WHBC primarily affects early-universe via epsilon_WH and r_s via beta_WH",
            "Perturbation damping (gamma_WH) has minimal effect on background"
        ]
    }

    # Save summary
    summary_file = f"{RESULTS_DIR}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved to: {summary_file}")

    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    for finding in summary['key_findings']:
        print(f"  - {finding}")

    print(f"\nHubble Tension Assessment:")
    print(f"  {summary['hubble_tension_assessment']}")

    return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run WHBC Simulation 7')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel execution (default: True)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameter grid')
    args = parser.parse_args()

    # Setup
    setup_directories()

    # Run parameter scan
    results = run_parameter_scan(parallel=args.parallel, n_jobs=args.n_jobs)

    # Analyze results
    analysis = analyze_results(results)

    # Generate plots
    generate_plots(results)

    # Export CLASS configs
    export_class_configs(results)

    # Write summary
    summary = write_summary(results, analysis)

    # Save full results
    full_results = {
        'all_results': [r.to_dict() for r in results]
    }
    with open(f"{RESULTS_DIR}/all_results.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "="*60)
    print("SIMULATION 7 COMPLETE")
    print("="*60)
    print(f"Results: {RESULTS_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"CLASS exports: {CLASS_EXPORT_DIR}/")

    return summary


if __name__ == "__main__":
    main()
