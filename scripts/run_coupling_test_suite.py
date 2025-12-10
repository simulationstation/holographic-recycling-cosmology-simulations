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
from hrc2.background import BackgroundCosmology


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


def run_ede_fluid_scenario(scenario: TestScenario) -> bool:
    """Run the EDE fluid scenario (T05) with a custom f_EDE scan.

    This function bypasses the standard xi/phi0 scan and instead scans
    over f_EDE values at fixed z_c, using pure GR (xi=0, phi0=0).

    Returns:
        True if completed successfully, False if skipped or failed
    """
    base_results_dir = os.path.join("results", "tests", scenario.id)
    os.makedirs(base_results_dir, exist_ok=True)

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
    print(f"Running EDE scenario: {scenario.id}")
    print(f"Description: {scenario.description}")
    print("=" * 70)

    start_time = time.time()

    # Scan over f_EDE values (5 points from 0 to 0.05)
    f_EDE_values = np.linspace(0.0, 0.05, 5)
    z_c = 3000.0  # Fixed characteristic EDE peak redshift
    sigma_ln_a = 0.5  # Fixed width

    results = []
    a_BBN = 1e-9  # Scale factor at BBN

    for f_EDE in f_EDE_values:
        # Create HRC2Parameters with pure GR (xi=0) and EDE params
        params = HRC2Parameters(
            xi=0.0,
            phi_0=0.0,
            coupling_family=CouplingFamily.QUADRATIC,
            potential_type=PotentialType.QUADRATIC,
            f_EDE=f_EDE,
            z_c=z_c,
            sigma_ln_a=sigma_ln_a,
        )

        # Create BackgroundCosmology instance
        cosmo = BackgroundCosmology(params)

        # Compute scale factor at z_c
        a_c = 1.0 / (1.0 + z_c)

        # Evaluate densities at a_c
        rho_ede_ac = cosmo.rho_EDE_component(a_c)
        rho_tot_ac = cosmo.total_density(a_c)

        # Effective f_EDE at z_c
        f_EDE_eff = rho_ede_ac / rho_tot_ac if rho_tot_ac > 0 else 0.0

        # BBN check: require rho_EDE << rho_rad at a_BBN
        rho_rad_bbn = cosmo.radiation_density(a_BBN)
        rho_ede_bbn = cosmo.rho_EDE_component(a_BBN)
        bbn_ok = rho_ede_bbn < 0.1 * rho_rad_bbn

        result_entry = {
            "f_EDE_input": float(f_EDE),
            "f_EDE_eff": float(f_EDE_eff),
            "rho_EDE_at_zc": float(rho_ede_ac),
            "rho_tot_at_zc": float(rho_tot_ac),
            "rho_EDE_at_BBN": float(rho_ede_bbn),
            "rho_rad_at_BBN": float(rho_rad_bbn),
            "BBN_ok": bool(bbn_ok),
        }
        results.append(result_entry)

        print(f"  f_EDE={f_EDE:.3f}: f_EDE_eff={f_EDE_eff:.4f}, BBN_ok={bbn_ok}")

    elapsed = time.time() - start_time

    # Find best result (highest f_EDE_eff that passes BBN)
    best = None
    best_f_eff = 0.0
    for r in results:
        if r["BBN_ok"] and r["f_EDE_eff"] > best_f_eff:
            best_f_eff = r["f_EDE_eff"]
            best = r

    # Write status.json
    status = {
        "id": scenario.id,
        "description": scenario.description,
        "coupling_family": scenario.coupling_family.value,
        "potential_type": scenario.potential_type.value,
        "z_c": z_c,
        "sigma_ln_a": sigma_ln_a,
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_points": len(f_EDE_values),
        "n_bbn_ok": sum(1 for r in results if r["BBN_ok"]),
        "max_f_EDE_eff": best_f_eff,
        "scan_results": results,
    }

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    print()
    print(f"Scenario {scenario.id} COMPLETED in {elapsed:.1f}s")
    print(f"  Points scanned: {len(f_EDE_values)}")
    print(f"  BBN-allowed: {sum(1 for r in results if r['BBN_ok'])}/{len(f_EDE_values)}")
    print(f"  Max f_EDE_eff (BBN-allowed): {best_f_eff:.4f}")
    print()

    return True


def run_horizon_memory_scenario(scenario: TestScenario) -> bool:
    """Run the horizon-memory scenario (T06) with a 2D (lambda_hor, tau_hor) scan.

    This function bypasses the standard xi/phi0 scan and instead scans
    over a 2D grid of lambda_hor and tau_hor values, using pure GR (xi=0, phi0=0).

    The memory field M(a) is integrated from a_start with initial condition M=0,
    evolving according to:
        dM/d(ln a) = (S_norm(a) - M) / tau_hor
    where S_norm(a) = (H0/H(a))^2.

    Returns:
        True if completed successfully, False if skipped or failed
    """
    from scipy.integrate import solve_ivp

    base_results_dir = os.path.join("results", "tests", scenario.id)
    os.makedirs(base_results_dir, exist_ok=True)

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
    print(f"Running horizon-memory scenario: {scenario.id}")
    print(f"Description: {scenario.description}")
    print("=" * 70)

    start_time = time.time()

    # 2D grid: 10x10 over lambda_hor and tau_hor
    n_lambda = 10
    n_tau = 10
    lambda_values = np.linspace(0.0, 0.2, n_lambda)
    tau_values = np.linspace(0.1, 3.0, n_tau)

    # Integration range
    a_start = 1e-6  # Very early (z ~ 10^6)
    a_end = 1.0     # Today (z = 0)
    ln_a_start = np.log(a_start)
    ln_a_end = np.log(a_end)

    results = []
    best_delta_H0 = 0.0
    best_params = None

    print(f"  Scanning {n_lambda} x {n_tau} = {n_lambda * n_tau} grid points")
    print(f"  lambda_hor: {lambda_values[0]:.3f} to {lambda_values[-1]:.3f}")
    print(f"  tau_hor: {tau_values[0]:.3f} to {tau_values[-1]:.3f}")
    print()

    for lam in lambda_values:
        for tau in tau_values:
            # Create HRC2Parameters with pure GR (xi=0) and horizon-memory params
            params = HRC2Parameters(
                xi=0.0,
                phi_0=0.0,
                coupling_family=CouplingFamily.QUADRATIC,
                potential_type=PotentialType.QUADRATIC,
                lambda_hor=lam,
                tau_hor=tau,
            )

            # Create BackgroundCosmology instance
            cosmo = BackgroundCosmology(params)

            # Define ODE for M(ln_a)
            # dM/d(ln a) = (S_norm - M) / tau_hor
            def memory_ode(ln_a, y):
                M = y[0]
                a = np.exp(ln_a)
                H = cosmo.H_of_a_gr(a)
                S_n = cosmo.S_norm(H)
                dM_dlna = (S_n - M) / tau
                return [dM_dlna]

            # Initial condition: M(a_start) = 0
            M0 = [0.0]

            # Integrate from ln(a_start) to ln(a_end)
            sol = solve_ivp(
                memory_ode,
                (ln_a_start, ln_a_end),
                M0,
                method='RK45',
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                print(f"  Warning: Integration failed for lambda={lam:.3f}, tau={tau:.3f}")
                continue

            # Evaluate at z=0 (ln_a = 0)
            M_today = sol.sol(0.0)[0]

            # Evaluate at z=1100 (a ~ 1/1101, ln_a ~ -7.0)
            ln_a_rec = np.log(1.0 / 1101.0)
            M_rec = sol.sol(ln_a_rec)[0]

            # Compute rho_hor at both epochs
            rho_hor_0 = cosmo.rho_horizon_memory(M_today)
            rho_hor_rec = cosmo.rho_horizon_memory(M_rec)

            # Total densities
            rho_tot_0 = cosmo.total_density(1.0)
            rho_tot_rec = cosmo.total_density(1.0 / 1101.0)

            frac_0 = rho_hor_0 / rho_tot_0 if rho_tot_0 > 0 else 0.0
            frac_rec = rho_hor_rec / rho_tot_rec if rho_tot_rec > 0 else 0.0

            # Compute delta_H0 proxy
            delta_H0_frac = cosmo.delta_H0_proxy(M_today)

            result_entry = {
                "lambda_hor": float(lam),
                "tau_hor": float(tau),
                "M_today": float(M_today),
                "M_rec": float(M_rec),
                "rho_hor_z0": float(rho_hor_0),
                "rho_tot_z0": float(rho_tot_0),
                "rho_hor_frac_z0": float(frac_0),
                "rho_hor_z1100": float(rho_hor_rec),
                "rho_tot_z1100": float(rho_tot_rec),
                "rho_hor_frac_z1100": float(frac_rec),
                "delta_H0_frac": float(delta_H0_frac),
            }
            results.append(result_entry)

            # Track best result
            if delta_H0_frac > best_delta_H0:
                best_delta_H0 = delta_H0_frac
                best_params = (lam, tau)

    elapsed = time.time() - start_time

    # Print summary
    print()
    print(f"  Sample results:")
    for r in results[:5]:
        print(f"    lambda={r['lambda_hor']:.3f}, tau={r['tau_hor']:.3f}: "
              f"M_today={r['M_today']:.4f}, delta_H0={r['delta_H0_frac']:.4f}")
    if len(results) > 5:
        print(f"    ... and {len(results) - 5} more")

    # Write status.json
    status = {
        "id": scenario.id,
        "description": scenario.description,
        "coupling_family": scenario.coupling_family.value,
        "potential_type": scenario.potential_type.value,
        "n_lambda": n_lambda,
        "n_tau": n_tau,
        "lambda_range": [float(lambda_values[0]), float(lambda_values[-1])],
        "tau_range": [float(tau_values[0]), float(tau_values[-1])],
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_points": len(results),
        "best_delta_H0_frac": float(best_delta_H0),
        "best_lambda_hor": float(best_params[0]) if best_params else None,
        "best_tau_hor": float(best_params[1]) if best_params else None,
        "scan_results": results,
    }

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    print()
    print(f"Scenario {scenario.id} COMPLETED in {elapsed:.1f}s")
    print(f"  Points scanned: {len(results)}")
    print(f"  Best delta_H0 (fractional): {best_delta_H0:.4f}")
    if best_params:
        print(f"  Best params: lambda_hor={best_params[0]:.3f}, tau_hor={best_params[1]:.3f}")
    print()

    return True


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
    if scenario.alpha_rec > 0:
        print(f"Recycling: alpha_rec = {scenario.alpha_rec}")
    if scenario.gamma_rec > 0:
        print(f"Horizon potential: gamma_rec = {scenario.gamma_rec}")
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
        alpha_rec=scenario.alpha_rec,
        gamma_rec=scenario.gamma_rec,
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
        "alpha_rec": scenario.alpha_rec,
        "gamma_rec": scenario.gamma_rec,
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
            # Handle special scenarios
            if scenario.id == "T05_EDE_fluid":
                run_ede_fluid_scenario(scenario)
            elif scenario.id == "T06_horizon_memory_nonlocal":
                run_horizon_memory_scenario(scenario)
            else:
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
            # Handle special scenarios
            if scenario.id == "T05_EDE_fluid":
                run_ede_fluid_scenario(scenario)
            elif scenario.id == "T06_horizon_memory_nonlocal":
                run_horizon_memory_scenario(scenario)
            else:
                run_test_scenario(scenario, perf)

    # Print final summary
    print_final_summary()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
