#!/usr/bin/env python3
"""Run HRC 2.0 coupling test suite.

This script:
1. Loops over registered test scenarios
2. For each scenario: runs 5x5 parallel scan, saves results and figures
3. Writes status.json with summary metrics for each test
4. Skips tests already marked completed (safe restart)

Usage:
    python scripts/run_coupling_test_suite.py                 # Run all scenarios
    python scripts/run_coupling_test_suite.py T01             # Run specific scenario
    python scripts/run_coupling_test_suite.py --list          # List all scenarios
"""

import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hrc2.theory import CouplingFamily, PotentialType, HRC2Parameters
from hrc2.utils.config import PerformanceConfig
from hrc2.analysis.xi_tradeoff import run_xi_tradeoff_parallel, XiTradeoffResultHRC2
from hrc2.analysis.test_scenarios import TEST_SCENARIOS, TestScenario, get_scenario_by_id, list_scenario_ids
from hrc2.plots.xi_tradeoff import plot_xi_tradeoff_hrc2
from hrc2.constraints.observational import estimate_delta_H0


def build_grid(xi_range, phi0_range, nx, nphi):
    """Build parameter grids from ranges."""
    xi_values = np.logspace(np.log10(xi_range[0]), np.log10(xi_range[1]), nx)
    phi0_values = np.linspace(phi0_range[0], phi0_range[1], nphi)
    return xi_values, phi0_values


def save_xi_tradeoff_result(result: XiTradeoffResultHRC2, path: str) -> None:
    """Save XiTradeoffResultHRC2 to npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        coupling_family=result.coupling_family.value,
        potential_type=result.potential_type.value,
        xi_values=result.xi_values,
        phi0_values=result.phi0_values,
        stable_mask=result.stable_mask,
        obs_allowed_mask=result.obs_allowed_mask,
        delta_G_over_G=result.delta_G_over_G,
        stable_fraction=result.stable_fraction,
        obs_allowed_fraction=result.obs_allowed_fraction,
        max_delta_G_stable=result.max_delta_G_stable,
        max_delta_G_allowed=result.max_delta_G_allowed,
        z_max=result.z_max,
        constraint_level=result.constraint_level,
    )
    print(f"Saved results to {path}")


def compute_summary(result: XiTradeoffResultHRC2) -> dict:
    """Compute summary statistics from scan result."""
    n_points = result.stable_mask.size
    n_stable = int(result.stable_mask.sum())
    n_allowed = int(result.obs_allowed_mask.sum())

    # Flatten delta_G arrays
    delta_G = np.abs(result.delta_G_over_G).flatten()
    valid = ~np.isnan(delta_G)

    max_abs_deltaG = float(np.nanmax(delta_G[valid])) if valid.any() else float("nan")

    # Compute max delta_H0 from allowed points
    if n_allowed > 0:
        allowed_dG = result.delta_G_over_G[result.obs_allowed_mask]
        max_allowed_dG = float(np.nanmax(np.abs(allowed_dG)))
        max_delta_H0 = estimate_delta_H0(max_allowed_dG)
    else:
        max_allowed_dG = float("nan")
        max_delta_H0 = float("nan")

    return {
        "n_points": n_points,
        "n_stable": n_stable,
        "n_allowed": n_allowed,
        "stable_fraction": float(n_stable / n_points) if n_points > 0 else 0.0,
        "allowed_fraction": float(n_allowed / n_points) if n_points > 0 else 0.0,
        "max_abs_deltaG": max_abs_deltaG,
        "max_abs_deltaG_allowed": max_allowed_dG,
        "max_abs_deltaH0_km_s_Mpc": max_delta_H0,
    }


def plot_scenario_results(result: XiTradeoffResultHRC2, scenario: TestScenario, output_dir: str) -> None:
    """Generate plots for a scenario."""
    os.makedirs(output_dir, exist_ok=True)

    # Main xi-tradeoff plot
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_xi_tradeoff_hrc2(result, ax=ax, save_path=f'{output_dir}/xi_tradeoff.png')
    plt.close(fig)

    # 2D heatmap of delta_G_over_G
    fig, ax = plt.subplots(figsize=(10, 8))
    data = result.delta_G_over_G.T  # Transpose for imshow
    im = ax.imshow(
        data,
        origin='lower',
        aspect='auto',
        extent=[
            np.log10(result.xi_values[0]),
            np.log10(result.xi_values[-1]),
            result.phi0_values[0],
            result.phi0_values[-1]
        ],
        cmap='viridis'
    )
    plt.colorbar(im, ax=ax, label=r'$|\Delta G/G|$')
    ax.set_xlabel(r'$\log_{10}(\xi)$')
    ax.set_ylabel(r'$\phi_0$')
    ax.set_title(f'{scenario.id}\n{scenario.description}')
    plt.savefig(f'{output_dir}/heatmap_deltaG.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Stability heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    stable_int = result.stable_mask.astype(int) + result.obs_allowed_mask.astype(int)
    im = ax.imshow(
        stable_int.T,
        origin='lower',
        aspect='auto',
        extent=[
            np.log10(result.xi_values[0]),
            np.log10(result.xi_values[-1]),
            result.phi0_values[0],
            result.phi0_values[-1]
        ],
        cmap='RdYlGn',
        vmin=0, vmax=2
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Unstable', 'Stable only', 'Allowed'])
    ax.set_xlabel(r'$\log_{10}(\xi)$')
    ax.set_ylabel(r'$\phi_0$')
    ax.set_title(f'{scenario.id}: Stability & Constraints')
    plt.savefig(f'{output_dir}/heatmap_stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_test_scenario(scenario: TestScenario, perf: PerformanceConfig) -> bool:
    """Run a single test scenario.

    Returns:
        True if completed successfully, False if skipped or failed
    """
    base_results_dir = os.path.join("results", "tests", scenario.id)
    base_fig_dir = os.path.join("figures", "tests", scenario.id)
    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(base_fig_dir, exist_ok=True)

    status_path = os.path.join(base_results_dir, "status.json")

    # Check if already completed (skip)
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            status = json.load(f)
        if status.get("completed"):
            print(f"[SKIP] {scenario.id} already completed")
            return False

    print()
    print("=" * 70)
    print(f"Running scenario: {scenario.id}")
    print(f"Description: {scenario.description}")
    print(f"Coupling: {scenario.coupling_family.value}")
    print(f"Grid: {scenario.nx} xi x {scenario.nphi} phi0 = {scenario.nx * scenario.nphi} points")
    print("=" * 70)

    xi_values, phi0_values = build_grid(
        scenario.xi_range,
        scenario.phi0_range,
        scenario.nx,
        scenario.nphi,
    )

    start_time = time.time()

    # Run parallel scan
    result = run_xi_tradeoff_parallel(
        xi_values=xi_values,
        phi0_values=phi0_values,
        coupling_family=scenario.coupling_family,
        potential_type=scenario.potential_type,
        perf=perf,
        z_max=scenario.z_max,
        z_points=300,
        constraint_level="conservative",
        verbose=True,
    )

    elapsed = time.time() - start_time

    # Save results
    result_path = os.path.join(base_results_dir, "scan.npz")
    save_xi_tradeoff_result(result, result_path)

    # Generate plots
    plot_scenario_results(result, scenario, base_fig_dir)

    # Compute summary
    summary = compute_summary(result)

    # Write status.json
    status = {
        "id": scenario.id,
        "description": scenario.description,
        "coupling_family": scenario.coupling_family.value,
        "potential_type": scenario.potential_type.value,
        "coupling_params": scenario.coupling_params,
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        **summary,
    }

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    print()
    print(f"Scenario {scenario.id} COMPLETED in {elapsed:.1f}s")
    print(f"  Stable: {summary['n_stable']}/{summary['n_points']} ({100*summary['stable_fraction']:.0f}%)")
    print(f"  Allowed: {summary['n_allowed']}/{summary['n_points']} ({100*summary['allowed_fraction']:.0f}%)")
    print(f"  Max |dG/G| (allowed): {summary['max_abs_deltaG_allowed']:.4f}")
    print(f"  Max dH0: {summary['max_abs_deltaH0_km_s_Mpc']:.1f} km/s/Mpc")
    print()

    return True


def print_final_summary():
    """Print summary of all completed tests."""
    print()
    print("=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print()

    results = []
    for scenario in TEST_SCENARIOS:
        status_path = os.path.join("results", "tests", scenario.id, "status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                status = json.load(f)
                if status.get("completed"):
                    results.append(status)

    if not results:
        print("No completed tests found.")
        return

    # Table header
    print(f"{'ID':<30} | {'dH0 (km/s/Mpc)':>15} | {'Allowed%':>10} | {'Max dG/G':>10}")
    print("-" * 70)

    best = None
    best_dH0 = 0.0

    for r in results:
        dH0 = r.get("max_abs_deltaH0_km_s_Mpc", float("nan"))
        allowed_pct = 100 * r.get("allowed_fraction", 0)
        max_dG = r.get("max_abs_deltaG_allowed", float("nan"))

        dH0_str = f"{dH0:.1f}" if not np.isnan(dH0) else "N/A"
        max_dG_str = f"{max_dG:.4f}" if not np.isnan(max_dG) else "N/A"

        print(f"{r['id']:<30} | {dH0_str:>15} | {allowed_pct:>9.0f}% | {max_dG_str:>10}")

        if not np.isnan(dH0) and dH0 > best_dH0:
            best_dH0 = dH0
            best = r

    print("-" * 70)

    if best:
        print()
        print(f"BEST RESULT: {best['id']}")
        print(f"  Max dH0: {best_dH0:.1f} km/s/Mpc")
        if best_dH0 >= 5.0:
            print("  VERDICT: Can potentially resolve Hubble tension!")
        elif best_dH0 >= 3.5:
            print("  VERDICT: Marginal - matches HRC 1.x limit")
        else:
            print("  VERDICT: Insufficient effect")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run HRC 2.0 coupling test suite")
    parser.add_argument("scenario_id", nargs="?", default=None,
                        help="Run specific scenario by ID (e.g., T01_evap_boundary_plateau)")
    parser.add_argument("--list", action="store_true", help="List all available scenarios")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    if args.list:
        print("Available test scenarios:")
        print("-" * 70)
        for s in TEST_SCENARIOS:
            print(f"  {s.id}: {s.description}")
        return

    perf = PerformanceConfig(n_workers=args.workers)

    if args.scenario_id:
        # Run single scenario
        try:
            scenario = get_scenario_by_id(args.scenario_id)
            run_test_scenario(scenario, perf)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available: {list_scenario_ids()}")
            return
    else:
        # Run all scenarios
        print()
        print("=" * 70)
        print("HRC 2.0 COUPLING TEST SUITE")
        print(f"Running {len(TEST_SCENARIOS)} scenarios")
        print("=" * 70)

        for scenario in TEST_SCENARIOS:
            run_test_scenario(scenario, perf)

    # Print final summary
    print_final_summary()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
