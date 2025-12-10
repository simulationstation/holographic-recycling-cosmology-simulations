#!/usr/bin/env python3
"""
SIMULATION 11B: SH0ES-like Distance Ladder Systematics

Implements a realistic two-step distance ladder:
1. Calibrator SNe with known distances (Cepheid/TRGB hosts)
2. Hubble-flow SNe to measure H0

Systematics explored:
- Population drift (alpha_pop): M_B evolution with redshift
- Metallicity dependence (gamma_Z): mag per dex
- Host mass step (delta_M_step_true): true step vs ignored by fitter
- Color/beta mismatch (delta_beta): beta_fit != beta_true

Goal: Quantify H0 bias from realistic vs moderate systematics.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.ladder.cosmology_baseline import TrueCosmology, mu_of_z
from hrc2.ladder.host_population import HostPopulationParams, sample_hosts
from hrc2.ladder.snia_salt2 import (
    SNSystematicParameters11B,
    simulate_snia_with_hosts,
    apply_magnitude_limit,
)
from hrc2.ladder.ladder_pipeline import run_ladder


# Configuration
TRUE_COSMOLOGY = TrueCosmology(H0=67.5, Omega_m=0.315, Omega_L=0.685)
RANDOM_SEED = 12345

# Sample sizes (SH0ES-like)
N_CALIB = 40        # Cepheid host SNe
N_FLOW = 200        # Hubble flow SNe

# Redshift ranges
Z_CALIB_MIN, Z_CALIB_MAX = 0.005, 0.03
Z_FLOW_MIN, Z_FLOW_MAX = 0.01, 0.15

# Systematic parameter grids
ALPHA_POP_VALUES = [0.0, 0.05, 0.10]        # Population drift (mag at z=0.5)
GAMMA_Z_VALUES = [0.0, 0.05, 0.10]          # Metallicity effect (mag/dex)
DELTA_STEP_VALUES = [0.0, 0.05, 0.10]       # Host mass step (true step)
DELTA_BETA_VALUES = [0.0, 0.3, 0.5]         # Color law mismatch

# Output directories
RESULTS_DIR = 'results/simulation_11b_ladder_systematics'
FIGURES_DIR = 'figures/simulation_11b_ladder_systematics'


def run_single_scenario(
    z_calib: np.ndarray,
    z_flow: np.ndarray,
    hosts_calib,
    hosts_flow,
    alpha_pop: float,
    gamma_Z: float,
    delta_step: float,
    delta_beta: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single systematics scenario.

    Args:
        z_calib, z_flow: Redshift arrays
        hosts_calib, hosts_flow: Host galaxy lists
        alpha_pop, gamma_Z, delta_step, delta_beta: Systematic parameters
        rng: Random number generator

    Returns:
        Dictionary with scenario results
    """
    # Create parameters
    params = SNSystematicParameters11B(
        M_B_0=-19.3,
        alpha_true=0.14,
        beta_true=3.1,
        alpha_fit=0.14,
        beta_fit=3.1 - delta_beta,  # Fitter uses wrong beta
        alpha_pop=alpha_pop,
        z_ref_pop=0.5,
        gamma_Z=gamma_Z,
        M_step_threshold=10.5,
        delta_M_step_true=delta_step,   # Nature has this step
        delta_M_step_fit=0.0,           # Fitter ignores it
        R_V_true=3.1,
        R_V_fit=3.1,
        m_lim_calib=18.5,
        m_lim_flow=19.5,
        sigma_int=0.10,
        sigma_meas=0.08,
    )

    # Simulate calibrator sample
    calib_raw = simulate_snia_with_hosts(
        z_calib, hosts_calib, params, TRUE_COSMOLOGY, rng
    )
    calib = apply_magnitude_limit(calib_raw, params.m_lim_calib)

    # Simulate Hubble flow sample
    flow_raw = simulate_snia_with_hosts(
        z_flow, hosts_flow, params, TRUE_COSMOLOGY, rng
    )
    flow = apply_magnitude_limit(flow_raw, params.m_lim_flow)

    # Run two-step ladder
    result = run_ladder(calib, flow, params, TRUE_COSMOLOGY)

    return {
        'alpha_pop': alpha_pop,
        'gamma_Z': gamma_Z,
        'delta_step': delta_step,
        'delta_beta': delta_beta,
        'H0_true': TRUE_COSMOLOGY.H0,
        'H0_fit': result.H0_fit,
        'M_B_fit': result.M_B_fit,
        'delta_H0': result.delta_H0,
        'chi2_flow': result.chi2_flow,
        'dof_flow': result.dof_flow,
        'N_calib': result.N_calib,
        'N_flow': result.N_flow,
    }


def run_grid_scan() -> List[Dict[str, Any]]:
    """Run the full systematics grid scan."""
    print("=" * 65)
    print("SIMULATION 11B: SH0ES-like Ladder Systematics Scan")
    print("=" * 65)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"True H0: {TRUE_COSMOLOGY.H0} km/s/Mpc")
    print(f"N_calib: {N_CALIB}, N_flow: {N_FLOW}")
    print()

    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate fixed redshift distributions
    z_calib = rng.uniform(Z_CALIB_MIN, Z_CALIB_MAX, size=N_CALIB)
    z_flow = rng.uniform(Z_FLOW_MIN, Z_FLOW_MAX, size=N_FLOW)

    print(f"Calibrators: z in [{Z_CALIB_MIN}, {Z_CALIB_MAX}], "
          f"mean z = {z_calib.mean():.3f}")
    print(f"Hubble flow: z in [{Z_FLOW_MIN}, {Z_FLOW_MAX}], "
          f"mean z = {z_flow.mean():.3f}")
    print()

    # Sample host galaxies (fixed for all scenarios)
    host_params = HostPopulationParams()
    hosts_calib = sample_hosts(N_CALIB, "calib", host_params, rng)
    hosts_flow = sample_hosts(N_FLOW, "flow", host_params, rng)

    # Generate grid
    grid = list(product(
        ALPHA_POP_VALUES,
        GAMMA_Z_VALUES,
        DELTA_STEP_VALUES,
        DELTA_BETA_VALUES,
    ))
    n_scenarios = len(grid)
    print(f"Total scenarios: {n_scenarios}")
    print()

    results = []
    for i, (alpha_pop, gamma_Z, delta_step, delta_beta) in enumerate(grid):
        # Different seed for each scenario (reproducible)
        scenario_rng = np.random.default_rng(RANDOM_SEED + i)

        result = run_single_scenario(
            z_calib, z_flow,
            hosts_calib, hosts_flow,
            alpha_pop, gamma_Z, delta_step, delta_beta,
            scenario_rng,
        )
        results.append(result)

        # Progress
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n_scenarios}] α_pop={alpha_pop:.2f}, "
                  f"γ_Z={gamma_Z:.2f}, ΔM_step={delta_step:.2f}, "
                  f"Δβ={delta_beta:.1f} => ΔH0={result['delta_H0']:+.2f}")

    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save scan results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        'simulation': 'SIM11B_LADDER_SYSTEMATICS',
        'date': datetime.now().isoformat(),
        'true_cosmology': {
            'H0': TRUE_COSMOLOGY.H0,
            'Omega_m': TRUE_COSMOLOGY.Omega_m,
        },
        'sample_sizes': {
            'N_calib': N_CALIB,
            'N_flow': N_FLOW,
            'z_calib_range': [Z_CALIB_MIN, Z_CALIB_MAX],
            'z_flow_range': [Z_FLOW_MIN, Z_FLOW_MAX],
        },
        'parameter_grids': {
            'alpha_pop': ALPHA_POP_VALUES,
            'gamma_Z': GAMMA_Z_VALUES,
            'delta_step': DELTA_STEP_VALUES,
            'delta_beta': DELTA_BETA_VALUES,
        },
        'n_scenarios': len(results),
        'results': results,
    }

    with open(os.path.join(RESULTS_DIR, 'scan_results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/scan_results.json")


def create_plots(results: List[Dict[str, Any]]) -> None:
    """Create diagnostic plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Extract data
    delta_H0 = np.array([r['delta_H0'] for r in results])
    alpha_pop = np.array([r['alpha_pop'] for r in results])
    gamma_Z = np.array([r['gamma_Z'] for r in results])
    delta_step = np.array([r['delta_step'] for r in results])
    delta_beta = np.array([r['delta_beta'] for r in results])

    # Figure 1: H0 bias distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(delta_H0, bins=25, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', label='No bias')
    ax.axvline(5.0, color='r', linestyle='--', label='H0 tension (~5 km/s/Mpc)')
    ax.axvline(-5.0, color='r', linestyle='--')
    ax.set_xlabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('SIM 11B: H0 Bias Distribution (SH0ES-like Ladder)', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_distribution.png'), dpi=150)
    plt.close()

    # Figure 2: H0 bias vs parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: vs alpha_pop
    for db in DELTA_BETA_VALUES:
        mask = delta_beta == db
        means = [np.mean(delta_H0[(mask) & (alpha_pop == a)])
                 for a in ALPHA_POP_VALUES]
        axes[0, 0].plot(ALPHA_POP_VALUES, means, 'o-', label=f'Δβ={db:.1f}')
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel(r'$\alpha_{pop}$ [mag @ z=0.5]', fontsize=11)
    axes[0, 0].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=11)
    axes[0, 0].set_title('vs Population Drift', fontsize=11)
    axes[0, 0].legend(fontsize=8)

    # Panel 2: vs gamma_Z
    for db in DELTA_BETA_VALUES:
        mask = delta_beta == db
        means = [np.mean(delta_H0[(mask) & (gamma_Z == g)])
                 for g in GAMMA_Z_VALUES]
        axes[0, 1].plot(GAMMA_Z_VALUES, means, 'o-', label=f'Δβ={db:.1f}')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel(r'$\gamma_Z$ [mag/dex]', fontsize=11)
    axes[0, 1].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=11)
    axes[0, 1].set_title('vs Metallicity Effect', fontsize=11)
    axes[0, 1].legend(fontsize=8)

    # Panel 3: vs delta_step
    for db in DELTA_BETA_VALUES:
        mask = delta_beta == db
        means = [np.mean(delta_H0[(mask) & (delta_step == ds)])
                 for ds in DELTA_STEP_VALUES]
        axes[1, 0].plot(DELTA_STEP_VALUES, means, 'o-', label=f'Δβ={db:.1f}')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel(r'$\Delta M_{step}$ [mag]', fontsize=11)
    axes[1, 0].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=11)
    axes[1, 0].set_title('vs Host Mass Step', fontsize=11)
    axes[1, 0].legend(fontsize=8)

    # Panel 4: vs delta_beta
    for ap in ALPHA_POP_VALUES:
        mask = alpha_pop == ap
        means = [np.mean(delta_H0[(mask) & (delta_beta == db)])
                 for db in DELTA_BETA_VALUES]
        axes[1, 1].plot(DELTA_BETA_VALUES, means, 'o-', label=f'α_pop={ap:.2f}')
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel(r'$\Delta\beta$ (color law mismatch)', fontsize=11)
    axes[1, 1].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=11)
    axes[1, 1].set_title('vs Color Law Mismatch', fontsize=11)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_vs_params.png'), dpi=150)
    plt.close()

    # Figure 3: Heatmap (alpha_pop vs gamma_Z) for Δβ=0.3
    fig, ax = plt.subplots(figsize=(8, 6))
    db_fixed = 0.3
    ds_fixed = 0.05
    mask = (delta_beta == db_fixed) & (delta_step == ds_fixed)

    n_alpha = len(ALPHA_POP_VALUES)
    n_gamma = len(GAMMA_Z_VALUES)
    grid = np.zeros((n_gamma, n_alpha))

    for i, a in enumerate(ALPHA_POP_VALUES):
        for j, g in enumerate(GAMMA_Z_VALUES):
            idx = np.where((mask) & (alpha_pop == a) & (gamma_Z == g))[0]
            if len(idx) > 0:
                grid[j, i] = delta_H0[idx[0]]

    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[min(ALPHA_POP_VALUES) - 0.025, max(ALPHA_POP_VALUES) + 0.025,
                           min(GAMMA_Z_VALUES) - 0.025, max(GAMMA_Z_VALUES) + 0.025],
                   cmap='RdBu_r', vmin=-8, vmax=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)

    ax.set_xlabel(r'$\alpha_{pop}$ [mag @ z=0.5]', fontsize=12)
    ax.set_ylabel(r'$\gamma_Z$ [mag/dex]', fontsize=12)
    ax.set_title(f'H0 Bias Map (Δβ={db_fixed}, ΔM_step={ds_fixed})', fontsize=14)

    # Add contour at 5 km/s/Mpc
    X, Y = np.meshgrid(ALPHA_POP_VALUES, GAMMA_Z_VALUES)
    ax.contour(X, Y, grid, levels=[5.0], colors='yellow', linewidths=2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_heatmap.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {FIGURES_DIR}/")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    delta_H0 = np.array([r['delta_H0'] for r in results])
    abs_delta_H0 = np.abs(delta_H0)

    n_ge3 = int(np.sum(abs_delta_H0 >= 3.0))
    n_ge4 = int(np.sum(abs_delta_H0 >= 4.0))
    n_ge5 = int(np.sum(abs_delta_H0 >= 5.0))
    n_ge6 = int(np.sum(abs_delta_H0 >= 6.0))

    print()
    print("=" * 65)
    print("SIMULATION 11B SCAN SUMMARY")
    print("=" * 65)
    print(f"True H0: {TRUE_COSMOLOGY.H0} km/s/Mpc")
    print(f"N scenarios: {len(results)}")
    print()
    print(f"Max |ΔH0|: {np.max(abs_delta_H0):.2f} km/s/Mpc")
    print(f"Mean |ΔH0|: {np.mean(abs_delta_H0):.2f} km/s/Mpc")
    print(f"Median |ΔH0|: {np.median(abs_delta_H0):.2f} km/s/Mpc")
    print()
    print(f"Scenarios with |ΔH0| >= 3: {n_ge3}")
    print(f"Scenarios with |ΔH0| >= 4: {n_ge4}")
    print(f"Scenarios with |ΔH0| >= 5: {n_ge5}")
    print(f"Scenarios with |ΔH0| >= 6: {n_ge6}")
    print()

    # Find extreme cases
    max_idx = np.argmax(delta_H0)
    min_idx = np.argmin(delta_H0)

    print("Largest positive bias (H0 too HIGH):")
    r = results[max_idx]
    print(f"  α_pop={r['alpha_pop']:.2f}, γ_Z={r['gamma_Z']:.2f}, "
          f"ΔM_step={r['delta_step']:.2f}, Δβ={r['delta_beta']:.1f}")
    print(f"  H0_fit={r['H0_fit']:.2f} km/s/Mpc, ΔH0={r['delta_H0']:+.2f}")
    print()

    print("Largest negative bias (H0 too LOW):")
    r = results[min_idx]
    print(f"  α_pop={r['alpha_pop']:.2f}, γ_Z={r['gamma_Z']:.2f}, "
          f"ΔM_step={r['delta_step']:.2f}, Δβ={r['delta_beta']:.1f}")
    print(f"  H0_fit={r['H0_fit']:.2f} km/s/Mpc, ΔH0={r['delta_H0']:+.2f}")
    print()

    # List scenarios with |ΔH0| >= 5
    if n_ge5 > 0:
        print("Scenarios with |ΔH0| >= 5 km/s/Mpc:")
        for r in results:
            if abs(r['delta_H0']) >= 5.0:
                print(f"  α_pop={r['alpha_pop']:.2f}, γ_Z={r['gamma_Z']:.2f}, "
                      f"ΔM_step={r['delta_step']:.2f}, Δβ={r['delta_beta']:.1f} "
                      f"=> ΔH0={r['delta_H0']:+.2f}")
        print()

    # Realistic region analysis
    realistic = [r for r in results
                 if r['alpha_pop'] <= 0.05 and r['gamma_Z'] <= 0.05
                 and r['delta_step'] <= 0.05 and r['delta_beta'] <= 0.5]
    if realistic:
        real_dH0 = [r['delta_H0'] for r in realistic]
        print(f"'Realistic' region (all params modest):")
        print(f"  N scenarios: {len(realistic)}")
        print(f"  Max |ΔH0|: {np.max(np.abs(real_dH0)):.2f} km/s/Mpc")
        print()

    # Moderate region
    moderate = [r for r in results
                if r['alpha_pop'] <= 0.10 and r['gamma_Z'] <= 0.10
                and r['delta_step'] <= 0.10 and r['delta_beta'] <= 0.5]
    if moderate:
        mod_dH0 = [r['delta_H0'] for r in moderate]
        print(f"'Moderate' region (params <= 0.10):")
        print(f"  N scenarios: {len(moderate)}")
        print(f"  Max |ΔH0|: {np.max(np.abs(mod_dH0)):.2f} km/s/Mpc")

    print("=" * 65)


def main():
    """Main entry point."""
    # Run grid scan
    results = run_grid_scan()

    # Save results
    save_results(results)

    # Create plots
    create_plots(results)

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
