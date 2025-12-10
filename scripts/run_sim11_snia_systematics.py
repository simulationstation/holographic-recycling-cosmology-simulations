#!/usr/bin/env python3
"""
SIMULATION 11: SN Ia Distance-Ladder Systematics and H0 Bias

This script simulates SN Ia data with various systematics and fits
with a naive ladder model to measure the induced H0 bias.

Systematics explored:
1. Population drift in M_B with redshift (alpha_pop)
2. Metallicity-dependent luminosity (gamma_Z)
3. Malmquist/selection bias (delta_m_malm)

The goal is to quantify whether realistic combinations of these
systematics can produce a ~5 km/s/Mpc H0 shift.
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
from hrc2.ladder.snia_population import (
    SNSystematicParameters,
    simulate_snia_sample,
    generate_realistic_z_distribution,
)
from hrc2.ladder.naive_fitter import fit_naive_H0_M_B


# Configuration
TRUE_COSMOLOGY = TrueCosmology(H0=67.5, Omega_m=0.315, Omega_L=0.685)
N_SNE = 300
RANDOM_SEED = 42

# Systematic parameter grids
ALPHA_POP_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]     # mag drift by z=0.5
GAMMA_Z_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]       # mag per dex
DELTA_M_MALM_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]  # brightening bias

# Output directories
RESULTS_DIR = 'results/simulation_11_snia_systematics'
FIGURES_DIR = 'figures/simulation_11_snia_systematics'


def run_single_scenario(
    alpha_pop: float,
    gamma_Z: float,
    delta_m_malm: float,
    z_array: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single systematics scenario.

    Args:
        alpha_pop: Population drift parameter
        gamma_Z: Metallicity dependence parameter
        delta_m_malm: Malmquist bias magnitude
        z_array: Redshift array
        rng: Random number generator

    Returns:
        Dictionary with scenario results
    """
    # Create systematic parameters
    params = SNSystematicParameters(
        M_B_0=-19.3,
        alpha_pop=alpha_pop,
        z_ref_pop=0.5,
        gamma_Z=gamma_Z,
        R_V_true=3.1,
        R_V_fit=3.1,
        delta_m_malm=delta_m_malm,
        z_malm=0.1,
        sigma_int=0.10,
        sigma_meas=0.08,
    )

    # Simulate SN sample
    data = simulate_snia_sample(z_array, params, TRUE_COSMOLOGY, rng)

    # Fit with naive model
    fit_result = fit_naive_H0_M_B(data, H0_init=70.0, M_B_init=-19.3)

    # Compute bias
    delta_H0 = fit_result.H0_fit - TRUE_COSMOLOGY.H0

    return {
        'alpha_pop': alpha_pop,
        'gamma_Z': gamma_Z,
        'delta_m_malm': delta_m_malm,
        'H0_true': TRUE_COSMOLOGY.H0,
        'H0_fit': fit_result.H0_fit,
        'M_B_fit': fit_result.M_B_fit,
        'delta_H0': delta_H0,
        'chi2': fit_result.chi2,
        'chi2_per_dof': fit_result.chi2_per_dof,
        'success': fit_result.success,
    }


def run_grid_scan() -> List[Dict[str, Any]]:
    """
    Run the full grid scan over systematic parameters.

    Returns:
        List of scenario results
    """
    print("=" * 60)
    print("SIMULATION 11: SN Ia Systematics Grid Scan")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"True H0: {TRUE_COSMOLOGY.H0} km/s/Mpc")
    print(f"N_SNe: {N_SNE}")
    print()

    # Generate fixed redshift distribution
    rng_z = np.random.default_rng(RANDOM_SEED)
    z_array = generate_realistic_z_distribution(N_SNE, rng_z)

    print(f"Redshift distribution: z_min={z_array.min():.3f}, "
          f"z_max={z_array.max():.3f}, z_mean={z_array.mean():.3f}")
    print()

    # Generate grid
    grid = list(product(ALPHA_POP_VALUES, GAMMA_Z_VALUES, DELTA_M_MALM_VALUES))
    n_scenarios = len(grid)
    print(f"Total scenarios: {n_scenarios}")
    print()

    results = []
    for i, (alpha_pop, gamma_Z, delta_m_malm) in enumerate(grid):
        # Use different seed for each scenario but reproducible
        rng = np.random.default_rng(RANDOM_SEED + i)

        result = run_single_scenario(
            alpha_pop, gamma_Z, delta_m_malm, z_array, rng
        )
        results.append(result)

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n_scenarios}] alpha={alpha_pop:.2f}, "
                  f"gamma={gamma_Z:.2f}, malm={delta_m_malm:.2f} => "
                  f"delta_H0={result['delta_H0']:+.2f} km/s/Mpc")

    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save scan results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        'simulation': 'SIM11_SNIA_SYSTEMATICS',
        'date': datetime.now().isoformat(),
        'true_cosmology': {
            'H0': TRUE_COSMOLOGY.H0,
            'Omega_m': TRUE_COSMOLOGY.Omega_m,
            'Omega_L': TRUE_COSMOLOGY.Omega_L,
        },
        'n_sne': N_SNE,
        'random_seed': RANDOM_SEED,
        'parameter_grids': {
            'alpha_pop': ALPHA_POP_VALUES,
            'gamma_Z': GAMMA_Z_VALUES,
            'delta_m_malm': DELTA_M_MALM_VALUES,
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
    delta_m_malm = np.array([r['delta_m_malm'] for r in results])

    # Figure 1: H0 bias distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(delta_H0, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', label='No bias')
    ax.axvline(5.0, color='r', linestyle='--', label='H0 tension (~5 km/s/Mpc)')
    ax.axvline(-5.0, color='r', linestyle='--')
    ax.set_xlabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of H0 Bias from SN Systematics', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_distribution.png'), dpi=150)
    plt.close()

    # Figure 2: Delta_H0 vs alpha_pop (averaged over other params)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: vs alpha_pop
    for malm in DELTA_M_MALM_VALUES:
        mask = delta_m_malm == malm
        means = [np.mean(delta_H0[(mask) & (alpha_pop == a)])
                 for a in ALPHA_POP_VALUES]
        axes[0].plot(ALPHA_POP_VALUES, means, 'o-', label=f'malm={malm:.2f}')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel(r'$\alpha_{pop}$ [mag @ z=0.5]', fontsize=12)
    axes[0].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)
    axes[0].set_title('H0 Bias vs Population Drift', fontsize=12)
    axes[0].legend(fontsize=8)

    # Panel 2: vs gamma_Z
    for malm in DELTA_M_MALM_VALUES:
        mask = delta_m_malm == malm
        means = [np.mean(delta_H0[(mask) & (gamma_Z == g)])
                 for g in GAMMA_Z_VALUES]
        axes[1].plot(GAMMA_Z_VALUES, means, 'o-', label=f'malm={malm:.2f}')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel(r'$\gamma_Z$ [mag/dex]', fontsize=12)
    axes[1].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)
    axes[1].set_title('H0 Bias vs Metallicity Effect', fontsize=12)
    axes[1].legend(fontsize=8)

    # Panel 3: vs delta_m_malm
    for alpha in ALPHA_POP_VALUES:
        mask = alpha_pop == alpha
        means = [np.mean(delta_H0[(mask) & (delta_m_malm == m)])
                 for m in DELTA_M_MALM_VALUES]
        axes[2].plot(DELTA_M_MALM_VALUES, means, 'o-', label=f'alpha={alpha:.2f}')
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2].axhline(5, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel(r'$\delta m_{malm}$ [mag]', fontsize=12)
    axes[2].set_ylabel(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)
    axes[2].set_title('H0 Bias vs Malmquist Bias', fontsize=12)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_vs_params.png'), dpi=150)
    plt.close()

    # Figure 3: 2D heatmap (alpha_pop vs gamma_Z) for fixed malm=0.10
    fig, ax = plt.subplots(figsize=(8, 6))
    malm_fixed = 0.10
    mask = delta_m_malm == malm_fixed

    # Create 2D grid
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
                   cmap='RdBu_r', vmin=-10, vmax=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta H_0$ [km/s/Mpc]', fontsize=12)

    ax.set_xlabel(r'$\alpha_{pop}$ [mag @ z=0.5]', fontsize=12)
    ax.set_ylabel(r'$\gamma_Z$ [mag/dex]', fontsize=12)
    ax.set_title(f'H0 Bias Map (Malmquist bias = {malm_fixed} mag)', fontsize=14)

    # Add contour at 5 km/s/Mpc
    X, Y = np.meshgrid(ALPHA_POP_VALUES, GAMMA_Z_VALUES)
    ax.contour(X, Y, grid, levels=[5.0], colors='yellow', linewidths=2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'H0_bias_heatmap.png'), dpi=150)
    plt.close()

    # Figure 4: Example Hubble diagram for high-bias case
    # Find scenario with highest |delta_H0|
    max_idx = np.argmax(np.abs(delta_H0))
    max_result = results[max_idx]

    # Recreate the data for this scenario
    rng_z = np.random.default_rng(RANDOM_SEED)
    z_array = generate_realistic_z_distribution(N_SNE, rng_z)
    rng = np.random.default_rng(RANDOM_SEED + max_idx)

    params = SNSystematicParameters(
        M_B_0=-19.3,
        alpha_pop=max_result['alpha_pop'],
        z_ref_pop=0.5,
        gamma_Z=max_result['gamma_Z'],
        delta_m_malm=max_result['delta_m_malm'],
        z_malm=0.1,
        sigma_int=0.10,
        sigma_meas=0.08,
    )
    data = simulate_snia_sample(z_array, params, TRUE_COSMOLOGY, rng)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Top: Hubble diagram
    z_plot = data['z']
    mu_true = data['mu_true']
    m_obs = data['m_obs']
    M_B_fit = max_result['M_B_fit']
    mu_obs = m_obs - M_B_fit  # "Observed" distance modulus

    # Model curves
    z_model = np.linspace(0.01, 0.25, 100)
    mu_model_true = np.array([mu_of_z(z, TRUE_COSMOLOGY) for z in z_model])

    cosmo_fit = TrueCosmology(H0=max_result['H0_fit'])
    mu_model_fit = np.array([mu_of_z(z, cosmo_fit) for z in z_model])

    axes[0].scatter(z_plot, mu_obs, s=10, alpha=0.5, label='Data')
    axes[0].plot(z_model, mu_model_true, 'k-', lw=2,
                 label=f"True: H0={TRUE_COSMOLOGY.H0:.1f}")
    axes[0].plot(z_model, mu_model_fit, 'r--', lw=2,
                 label=f"Fit: H0={max_result['H0_fit']:.1f}")
    axes[0].set_xlabel('Redshift z', fontsize=12)
    axes[0].set_ylabel(r'$\mu$ [mag]', fontsize=12)
    axes[0].set_title(f"Hubble Diagram - Max Bias Scenario "
                      f"(alpha={max_result['alpha_pop']:.2f}, "
                      f"gamma={max_result['gamma_Z']:.2f}, "
                      f"malm={max_result['delta_m_malm']:.2f})",
                      fontsize=12)
    axes[0].legend()

    # Bottom: Residuals
    mu_model_at_z = np.array([mu_of_z(z, TRUE_COSMOLOGY) for z in z_plot])
    residuals = mu_obs - mu_model_at_z

    axes[1].scatter(z_plot, residuals, s=10, alpha=0.5)
    axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_xlabel('Redshift z', fontsize=12)
    axes[1].set_ylabel(r'$\mu_{obs} - \mu_{true}$ [mag]', fontsize=12)
    axes[1].set_title(f"Residuals (delta_H0 = {max_result['delta_H0']:+.2f} km/s/Mpc)",
                      fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hubble_diagram_max_bias.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {FIGURES_DIR}/")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    delta_H0 = np.array([r['delta_H0'] for r in results])
    abs_delta_H0 = np.abs(delta_H0)

    n_ge3 = np.sum(abs_delta_H0 >= 3.0)
    n_ge5 = np.sum(abs_delta_H0 >= 5.0)

    print()
    print("=" * 60)
    print("SIMULATION 11 SCAN SUMMARY")
    print("=" * 60)
    print(f"True H0: {TRUE_COSMOLOGY.H0} km/s/Mpc")
    print(f"N scenarios: {len(results)}")
    print()
    print(f"Max |delta_H0|: {np.max(abs_delta_H0):.2f} km/s/Mpc")
    print(f"Mean |delta_H0|: {np.mean(abs_delta_H0):.2f} km/s/Mpc")
    print(f"Median |delta_H0|: {np.median(abs_delta_H0):.2f} km/s/Mpc")
    print()
    print(f"Scenarios with |delta_H0| >= 3: {n_ge3}")
    print(f"Scenarios with |delta_H0| >= 5: {n_ge5}")
    print()

    # Find best/worst cases
    max_idx = np.argmax(delta_H0)
    min_idx = np.argmin(delta_H0)

    print("Largest positive bias:")
    r = results[max_idx]
    print(f"  alpha_pop={r['alpha_pop']:.2f}, gamma_Z={r['gamma_Z']:.2f}, "
          f"malm={r['delta_m_malm']:.2f}")
    print(f"  H0_fit={r['H0_fit']:.2f}, delta_H0={r['delta_H0']:+.2f}")

    print()
    print("Largest negative bias:")
    r = results[min_idx]
    print(f"  alpha_pop={r['alpha_pop']:.2f}, gamma_Z={r['gamma_Z']:.2f}, "
          f"malm={r['delta_m_malm']:.2f}")
    print(f"  H0_fit={r['H0_fit']:.2f}, delta_H0={r['delta_H0']:+.2f}")

    # List scenarios with |delta_H0| >= 5
    if n_ge5 > 0:
        print()
        print("Scenarios with |delta_H0| >= 5 km/s/Mpc:")
        for r in results:
            if abs(r['delta_H0']) >= 5.0:
                print(f"  alpha={r['alpha_pop']:.2f}, gamma={r['gamma_Z']:.2f}, "
                      f"malm={r['delta_m_malm']:.2f} => "
                      f"delta_H0={r['delta_H0']:+.2f}")

    # "Realistic" region: all params <= 0.05
    realistic = [r for r in results
                 if r['alpha_pop'] <= 0.05 and r['gamma_Z'] <= 0.05
                 and r['delta_m_malm'] <= 0.05]
    if realistic:
        real_dH0 = [r['delta_H0'] for r in realistic]
        print()
        print("In 'realistic' region (all params <= 0.05):")
        print(f"  N scenarios: {len(realistic)}")
        print(f"  Max |delta_H0|: {np.max(np.abs(real_dH0)):.2f} km/s/Mpc")

    print("=" * 60)


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
