#!/usr/bin/env python3
"""
MCMC Analysis for Holographic Recycling Cosmology

This script performs Bayesian parameter estimation for the HRC model
using observational data from Planck, DESI, SH0ES, and Pantheon+.

Key questions addressed:
1. Can HRC resolve the Hubble tension?
2. What are the constrained values of new physics parameters?
3. How does HRC compare to standard ΛCDM?

Usage:
    python run_mcmc.py [--n_walkers N] [--n_steps N] [--output DIR]

Requirements:
    - emcee (for MCMC)
    - corner (for visualization)
    - numpy, scipy, matplotlib

Author: HRC Analysis Pipeline
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will be skipped.")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    warnings.warn("corner not available. Corner plots will be skipped.")

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    sys.exit("Error: emcee is required. Install with: pip install emcee")

# Import HRC modules
from hrc_observations import (
    ObservationalData,
    HRCPredictions,
    HRCLikelihood,
    LCDMCosmology,
    FisherMatrix,
    compute_bayes_factor
)


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory for results."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(results: dict, output_dir: Path, prefix: str = "hrc"):
    """Save analysis results to JSON file."""
    filename = output_dir / f"{prefix}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to: {filename}")
    return filename


def plot_chains(sampler, param_names: list, output_dir: Path, burn_in: int = 0):
    """Plot MCMC chains for convergence diagnostics."""
    if not HAS_MATPLOTLIB:
        return

    samples = sampler.get_chain()
    n_steps, n_walkers, n_dim = samples.shape

    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        for j in range(n_walkers):
            ax.plot(samples[:, j, i], alpha=0.3, lw=0.5)
        ax.axvline(burn_in, color='red', linestyle='--', label='Burn-in')
        ax.set_ylabel(name)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Step')
    fig.suptitle('MCMC Chain Traces')
    plt.tight_layout()

    filename = output_dir / 'mcmc_chains.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Chain plot saved to: {filename}")


def plot_corner(sampler, param_names: list, output_dir: Path,
                burn_in: int = 0, truths: list = None):
    """Create corner plot of posterior distributions."""
    if not HAS_CORNER or not HAS_MATPLOTLIB:
        return

    samples = sampler.get_chain(discard=burn_in, flat=True)

    fig = corner.corner(
        samples,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        truths=truths,
    )

    filename = output_dir / 'corner_plot.png'
    fig.savefig(filename, dpi=150)
    plt.close()
    print(f"Corner plot saved to: {filename}")


def plot_hubble_comparison(results: dict, output_dir: Path):
    """Plot H0 predictions vs observations."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Observations
    obs_data = [
        ('Planck CMB', 67.4, 0.5, 'blue'),
        ('SH0ES Local', 73.04, 1.04, 'red'),
        ('TRGB', 69.8, 1.7, 'orange'),
    ]

    y_positions = list(range(len(obs_data)))

    for i, (name, H0, err, color) in enumerate(obs_data):
        ax.errorbar(H0, i, xerr=err, fmt='o', color=color,
                   capsize=5, markersize=10, label=f'{name}: {H0:.1f}±{err:.1f}')

    # HRC predictions
    if 'H0_local_derived' in results and 'H0_cmb_derived' in results:
        H0_local = results['H0_local_derived']['mean']
        H0_local_err = results['H0_local_derived']['std']
        H0_cmb = results['H0_cmb_derived']['mean']
        H0_cmb_err = results['H0_cmb_derived']['std']

        ax.errorbar(H0_local, len(obs_data), xerr=H0_local_err, fmt='s',
                   color='green', capsize=5, markersize=10,
                   label=f'HRC Local: {H0_local:.1f}±{H0_local_err:.1f}')
        ax.errorbar(H0_cmb, len(obs_data) + 1, xerr=H0_cmb_err, fmt='s',
                   color='purple', capsize=5, markersize=10,
                   label=f'HRC CMB: {H0_cmb:.1f}±{H0_cmb_err:.1f}')

    ax.set_xlabel('$H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_yticks(list(range(len(obs_data) + 2)))
    ax.set_yticklabels([d[0] for d in obs_data] + ['HRC Local', 'HRC CMB'])
    ax.axvline(70, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Hubble Constant Comparison: Observations vs HRC Predictions')
    ax.set_xlim(64, 78)

    plt.tight_layout()
    filename = output_dir / 'h0_comparison.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"H0 comparison plot saved to: {filename}")


def plot_tension_resolution(output_dir: Path):
    """Plot how HRC parameters affect the Hubble tension."""
    if not HAS_MATPLOTLIB:
        return

    # Scan parameter space
    xi_values = np.linspace(-0.03, 0.03, 25)
    phi_values = np.linspace(0, 0.3, 25)

    delta_H0_grid = np.zeros((len(phi_values), len(xi_values)))

    for i, phi_0 in enumerate(phi_values):
        for j, xi in enumerate(xi_values):
            params = {
                'H0_true': 70.0,
                'Omega_m': 0.315,
                'xi': xi,
                'alpha': 0.01,
                'phi_0': phi_0,
            }
            try:
                pred = HRCPredictions(params)
                delta_H0_grid[i, j] = pred.H0_local() - pred.H0_cmb()
            except:
                delta_H0_grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot
    X, Y = np.meshgrid(xi_values, phi_values)
    levels = np.linspace(-10, 10, 21)

    cf = ax.contourf(X, Y, delta_H0_grid, levels=levels, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='$\\Delta H_0$ [km/s/Mpc]')

    # Mark target tension
    target = 5.6  # Observed tension
    cs = ax.contour(X, Y, delta_H0_grid, levels=[target], colors='lime', linewidths=2)
    ax.clabel(cs, fmt=f'Target: {target:.1f}')

    ax.set_xlabel('$\\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel('$\\phi_0$ (scalar field today)', fontsize=12)
    ax.set_title('HRC Parameter Space: Hubble Tension Resolution')

    plt.tight_layout()
    filename = output_dir / 'tension_parameter_space.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Tension resolution plot saved to: {filename}")


def run_analysis(n_walkers: int = 32, n_steps: int = 2000,
                burn_in: int = 500, output_dir: str = './results'):
    """
    Run full MCMC analysis.

    Parameters
    ----------
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    burn_in : int
        Number of burn-in steps to discard
    output_dir : str
        Directory for output files
    """
    print("=" * 70)
    print(" HRC MCMC ANALYSIS")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    output_path = setup_output_dir(output_dir)
    print(f"Output directory: {output_path}")

    # Load data
    print("\n1. LOADING OBSERVATIONAL DATA")
    print("-" * 50)
    data = ObservationalData()
    print(data.summary())

    # Setup likelihood
    print("\n2. SETTING UP LIKELIHOOD")
    print("-" * 50)
    likelihood = HRCLikelihood(data)
    print(f"Parameters: {likelihood.param_names}")
    print(f"Bounds: {likelihood.param_bounds}")

    # Fisher matrix estimate
    print("\n3. FISHER MATRIX QUICK ESTIMATE")
    print("-" * 50)

    fiducial = {
        'H0': 70.0,
        'Omega_m': 0.315,
        'xi': 0.01,
        'alpha': 0.01,
        'phi_0': 0.1,
    }

    fisher = FisherMatrix(likelihood, fiducial)
    fisher_errors = fisher.parameter_errors()

    print("Fisher matrix parameter uncertainties:")
    for name, err in fisher_errors.items():
        print(f"  σ({name}) = {err:.6f}")

    # Model comparison
    print("\n4. MODEL COMPARISON (HRC vs ΛCDM)")
    print("-" * 50)

    bf = compute_bayes_factor(data, fiducial)
    print(f"Log Bayes factor: ln(B) = {bf['ln_B']:.2f}")
    print(f"ΔBIC = {bf['delta_bic']:.2f}")
    print(f"Interpretation: {bf['interpretation']}")

    # MCMC
    print("\n5. MCMC SAMPLING")
    print("-" * 50)
    print(f"Walkers: {n_walkers}")
    print(f"Steps: {n_steps}")
    print(f"Burn-in: {burn_in}")

    sampler = likelihood.run_mcmc(
        n_walkers=n_walkers,
        n_steps=n_steps,
        initial_params=fiducial,
        progress=True
    )

    # Analyze chains
    print("\n6. ANALYZING CHAINS")
    print("-" * 50)

    mcmc_results = likelihood.analyze_chains(sampler, burn_in=burn_in)

    print("\nMCMC Parameter Constraints:")
    for name in likelihood.param_names:
        r = mcmc_results[name]
        print(f"  {name:12s} = {r['median']:8.4f} +{r['q84']-r['median']:8.4f} -{r['median']-r['q16']:8.4f}")

    print("\nDerived H0 Values:")
    print(f"  H0_local = {mcmc_results['H0_local_derived']['mean']:.2f} ± {mcmc_results['H0_local_derived']['std']:.2f} km/s/Mpc")
    print(f"  H0_cmb   = {mcmc_results['H0_cmb_derived']['mean']:.2f} ± {mcmc_results['H0_cmb_derived']['std']:.2f} km/s/Mpc")
    print(f"  ΔH0      = {mcmc_results['tension_resolved']['delta_H0']:.2f} km/s/Mpc")

    # Compile results
    results = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'burn_in': burn_in,
        },
        'fiducial_params': fiducial,
        'fisher_errors': fisher_errors,
        'bayes_factor': bf,
        'mcmc_results': mcmc_results,
        'acceptance_fraction': float(np.mean(sampler.acceptance_fraction)),
    }

    # Save results
    print("\n7. SAVING RESULTS")
    print("-" * 50)
    save_results(results, output_path)

    # Generate plots
    print("\n8. GENERATING PLOTS")
    print("-" * 50)
    plot_chains(sampler, likelihood.param_names, output_path, burn_in)
    plot_corner(sampler, likelihood.param_names, output_path, burn_in)
    plot_hubble_comparison(mcmc_results, output_path)
    plot_tension_resolution(output_path)

    # Summary
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 50)

    delta_H0 = mcmc_results['tension_resolved']['delta_H0']
    observed_tension = 5.6  # km/s/Mpc

    if abs(delta_H0 - observed_tension) < 2.0:
        print(f"✓ HRC CAN resolve Hubble tension (ΔH0 = {delta_H0:.1f} vs observed {observed_tension:.1f})")
    else:
        print(f"✗ HRC CANNOT fully resolve tension (ΔH0 = {delta_H0:.1f} vs observed {observed_tension:.1f})")

    xi_mean = mcmc_results['xi']['mean']
    print(f"\nPreferred non-minimal coupling: ξ = {xi_mean:.4f}")
    if abs(xi_mean) < 0.01:
        print("  → Close to minimal coupling (small new physics contribution)")
    else:
        print("  → Significant deviation from minimal coupling")

    print(f"\nModel comparison: {bf['interpretation']}")

    print(f"\nResults saved in: {output_path}")
    print("=" * 70)

    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='MCMC Analysis for Holographic Recycling Cosmology'
    )
    parser.add_argument('--n_walkers', type=int, default=32,
                       help='Number of MCMC walkers (default: 32)')
    parser.add_argument('--n_steps', type=int, default=2000,
                       help='Number of MCMC steps (default: 2000)')
    parser.add_argument('--burn_in', type=int, default=500,
                       help='Burn-in steps to discard (default: 500)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer samples')

    args = parser.parse_args()

    if args.quick:
        args.n_walkers = 16
        args.n_steps = 500
        args.burn_in = 100

    run_analysis(
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
