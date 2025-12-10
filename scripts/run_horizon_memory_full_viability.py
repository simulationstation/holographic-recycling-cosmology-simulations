#!/usr/bin/env python3
"""Unified full viability report for horizon-memory cosmology.

This script combines all cosmological tests for horizon-memory models:
1. Background evolution (SN/BAO constraints)
2. Growth-of-structure (f*sigma_8)
3. CMB distance constraints
4. Perturbation stability

It produces a comprehensive viability report for each model with a final
assessment of whether the model is viable, marginal, or ruled out.

Usage:
    python scripts/run_horizon_memory_full_viability.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp, quad

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.theory import HRC2Parameters, CouplingFamily, PotentialType
from hrc2.background import BackgroundCosmology
from hrc2.perturbations.horizon_memory import (
    HorizonMemoryPerturbations,
    create_perturbations_from_background,
)
from hrc2.analysis.interface_class import (
    HorizonMemoryClassInterface,
    export_to_class_format,
)


# Physical constants
Z_STAR = 1089.0
H0_PLANCK = 67.4
SIGMA8_PLANCK = 0.811


@dataclass
class FullViabilityResult:
    """Container for full viability assessment."""
    model_id: str
    lambda_hor: float
    tau_hor: float

    # Background tests
    passes_SN_BAO: bool
    max_H_deviation: float
    max_D_L_deviation: float

    # Growth tests
    passes_growth: str  # "true", "marginal", "false"
    max_f_sigma8_deviation: float

    # CMB tests
    passes_CMB_distance: str  # "true", "marginal", "false"
    cmb_distance_deviation: float

    # Perturbation stability
    perturbation_stability: str  # "ok", "warning", "fail"
    perturbation_notes: str

    # Overall
    overall_viability: str  # "viable", "marginal", "ruled out"


def integrate_memory_field(cosmo: BackgroundCosmology, z_max: float = 1200.0) -> callable:
    """Integrate the memory field ODE and return interpolator."""
    a_start = 1.0 / (1.0 + z_max)
    a_end = 1.0
    ln_a_start, ln_a_end = np.log(a_start), np.log(a_end)

    def memory_ode(ln_a, y):
        M = y[0]
        a = np.exp(ln_a)
        H = cosmo.H_of_a_gr(a)
        S_n = cosmo.S_norm(H)
        dM_dlna = (S_n - M) / cosmo.tau_hor
        return [dM_dlna]

    sol = solve_ivp(
        memory_ode,
        (ln_a_start, ln_a_end),
        [0.0],
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Memory field integration failed: {sol.message}")

    return sol.sol


def run_background_test(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    z_max: float = 3.0,
    n_points: int = 200,
) -> Dict:
    """Run background evolution test (SN/BAO)."""
    z_array = np.linspace(0, z_max, n_points)

    H_hm = np.zeros_like(z_array)
    H_lcdm = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]

        H_hm[i] = cosmo.H_of_a_selfconsistent(a, M)
        H_lcdm[i] = cosmo.H_of_a_gr_baseline(a)

    H_ratio = H_hm / H_lcdm

    # Compute distances
    def compute_chi(H_func, z_max):
        def integrand(z):
            a = 1.0 / (1.0 + z)
            return 1.0 / H_func(a)
        chi, _ = quad(integrand, 0, z_max, limit=100)
        return chi

    def H_hm_func(a):
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M)

    def H_lcdm_func(a):
        return cosmo.H_of_a_gr_baseline(a)

    # Compute D_L ratio at z=1
    chi_hm_z1 = compute_chi(H_hm_func, 1.0)
    chi_lcdm_z1 = compute_chi(H_lcdm_func, 1.0)
    D_L_ratio_z1 = (2.0 * chi_hm_z1) / (2.0 * chi_lcdm_z1) if chi_lcdm_z1 > 0 else 1.0

    # Maximum deviations
    mask_low_z = (z_array > 0) & (z_array < 1)
    max_H_deviation = np.max(np.abs(H_ratio[mask_low_z] - 1.0)) if np.any(mask_low_z) else 0.0
    max_D_L_deviation = abs(D_L_ratio_z1 - 1.0)

    passes = max_H_deviation < 0.10 and max_D_L_deviation < 0.10

    return {
        "passes": passes,
        "max_H_deviation": float(max_H_deviation),
        "max_D_L_deviation": float(max_D_L_deviation),
        "z_array": z_array,
        "H_ratio": H_ratio,
    }


def run_growth_test(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    n_points: int = 200,
) -> Dict:
    """Run growth-of-structure test (f*sigma_8)."""
    from scripts.run_horizon_memory_growth import solve_growth_equation, compute_f_sigma8

    # Solve for both models
    a_hm, D_hm, f_hm = solve_growth_equation(cosmo, M_interp, use_hm=True, n_points=n_points)
    a_lcdm, D_lcdm, f_lcdm = solve_growth_equation(cosmo, M_interp, use_hm=False, n_points=n_points)

    # Compute f*sigma_8
    f_sigma8_hm = compute_f_sigma8(a_hm, D_hm, f_hm)
    f_sigma8_lcdm = compute_f_sigma8(a_lcdm, D_lcdm, f_lcdm)

    # Restrict to z < 2
    z_hm = 1.0 / a_hm - 1.0
    mask = z_hm <= 2.0

    # Relative deviation
    valid = f_sigma8_lcdm[mask] > 0
    rel_dev = np.zeros_like(f_sigma8_hm[mask])
    rel_dev[valid] = (f_sigma8_hm[mask][valid] - f_sigma8_lcdm[mask][valid]) / f_sigma8_lcdm[mask][valid]

    max_deviation = np.max(np.abs(rel_dev))

    if max_deviation < 0.05:
        status = "true"
    elif max_deviation < 0.10:
        status = "marginal"
    else:
        status = "false"

    return {
        "passes": status,
        "max_deviation": float(max_deviation),
    }


def run_cmb_distance_test(
    cosmo: BackgroundCosmology,
    M_interp: callable,
) -> Dict:
    """Run CMB distance constraint test."""
    def H_ratio_hm(z):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        ln_a = max(ln_a, np.log(1.0 / (Z_STAR + 100)))
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0

    def H_ratio_lcdm(z):
        a = 1.0 / (1.0 + z)
        return cosmo.H_of_a_gr_baseline(a) / cosmo.H0

    # Compute D_*
    D_star_hm, _ = quad(lambda z: 1.0 / H_ratio_hm(z), 0, Z_STAR, limit=500)
    D_star_lcdm, _ = quad(lambda z: 1.0 / H_ratio_lcdm(z), 0, Z_STAR, limit=500)

    ratio = D_star_hm / D_star_lcdm if D_star_lcdm > 0 else 1.0
    deviation_percent = abs(ratio - 1.0) * 100

    if deviation_percent < 0.5:
        status = "true"
    elif deviation_percent < 1.0:
        status = "marginal"
    else:
        status = "false"

    return {
        "passes": status,
        "deviation_percent": float(deviation_percent),
        "ratio": float(ratio),
    }


def run_perturbation_stability_test(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    n_points: int = 100,
) -> Dict:
    """Run perturbation stability test."""
    # Compute w_hor(z) for perturbation solver
    z_array = np.linspace(0, 3, n_points)

    def compute_w_hor(z, eps=1e-4):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)

        M = M_interp(ln_a)[0]
        if M <= 0:
            return -1.0

        ln_a_plus = min(ln_a + eps, 0.0)
        ln_a_minus = ln_a - eps

        M_plus = M_interp(ln_a_plus)[0]
        M_minus = M_interp(ln_a_minus)[0]

        if M_plus <= 0 or M_minus <= 0:
            return -1.0

        d_ln_M = (np.log(M_plus) - np.log(M_minus)) / (2 * eps)
        return np.clip(-1.0 - d_ln_M / 3.0, -2.5, 0.5)

    w_array = np.array([compute_w_hor(z) for z in z_array])

    # H(z) array
    def H_func(z):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0

    H_array = np.array([H_func(z) for z in z_array])

    # Create perturbation solver
    pert = create_perturbations_from_background(
        z_array, w_array, H_array, c_s_squared=1.0
    )

    # Test several k modes
    k_modes = [0.01, 0.1, 0.2]  # h/Mpc
    stability_ok = True
    notes = []

    for k in k_modes:
        result = pert.solve(k=k, a_init=1e-3, a_final=1.0, n_points=100)

        if not result.success:
            stability_ok = False
            notes.append(f"k={k}: integration failed")
            continue

        # Check for blow-up
        max_delta = np.nanmax(np.abs(result.delta))
        if max_delta > 10.0:
            stability_ok = False
            notes.append(f"k={k}: delta blew up (max={max_delta:.2f})")
        elif max_delta > 1.0:
            notes.append(f"k={k}: delta grew large (max={max_delta:.2f})")

    if not stability_ok:
        status = "fail"
    elif len(notes) > 0:
        status = "warning"
    else:
        status = "ok"
        notes.append("All k modes stable")

    return {
        "status": status,
        "notes": "; ".join(notes),
        "k_modes_tested": k_modes,
    }


def make_perturbation_plots(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    model_id: str,
    output_dir: str,
):
    """Generate perturbation diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Compute background quantities
    z_array = np.linspace(0, 3, 100)

    def compute_w_hor(z, eps=1e-4):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        if M <= 0:
            return -1.0
        ln_a_plus = min(ln_a + eps, 0.0)
        ln_a_minus = ln_a - eps
        M_plus = M_interp(ln_a_plus)[0]
        M_minus = M_interp(ln_a_minus)[0]
        if M_plus <= 0 or M_minus <= 0:
            return -1.0
        d_ln_M = (np.log(M_plus) - np.log(M_minus)) / (2 * eps)
        return np.clip(-1.0 - d_ln_M / 3.0, -2.5, 0.5)

    w_array = np.array([compute_w_hor(z) for z in z_array])

    def H_func(z):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0

    H_array = np.array([H_func(z) for z in z_array])

    # Create perturbation solver
    pert = create_perturbations_from_background(z_array, w_array, H_array)

    # Solve for multiple k modes
    k_modes = [0.01, 0.1, 0.2]
    results = {}

    for k in k_modes:
        results[k] = pert.solve(k=k, a_init=1e-3, a_final=1.0, n_points=100)

    # Plot delta evolution
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 7)})

    fig, ax = plt.subplots()

    colors = ['blue', 'green', 'red']
    for k, color in zip(k_modes, colors):
        result = results[k]
        if result.success:
            ax.plot(result.a, result.delta, color=color, linewidth=2,
                   label=f'k = {k} h/Mpc')

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Scale factor $a$')
    ax.set_ylabel('$\\delta_{hor}(k, a)$')
    ax.set_title(f'Horizon-Memory Density Perturbation: {model_id}')
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_hor_evolution.png"), dpi=150)
    plt.close()

    # Plot theta evolution
    fig, ax = plt.subplots()

    for k, color in zip(k_modes, colors):
        result = results[k]
        if result.success:
            ax.plot(result.a, result.theta, color=color, linewidth=2,
                   label=f'k = {k} h/Mpc')

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Scale factor $a$')
    ax.set_ylabel('$\\theta_{hor}(k, a)$')
    ax.set_title(f'Horizon-Memory Velocity Perturbation: {model_id}')
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "theta_hor_evolution.png"), dpi=150)
    plt.close()

    print(f"    Perturbation plots saved to {output_dir}/")


def analyze_full_viability(model_data: dict) -> FullViabilityResult:
    """Run complete viability analysis for a single model."""
    lambda_hor = model_data["lambda_hor"]
    tau_hor = model_data["tau_hor"]
    model_id = f"{lambda_hor:.3f}_{tau_hor:.2f}"

    print(f"\n  Analyzing model {model_id}...")

    # Create cosmology
    params = HRC2Parameters(
        xi=0.0,
        phi_0=0.0,
        coupling_family=CouplingFamily.QUADRATIC,
        potential_type=PotentialType.QUADRATIC,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
    )
    cosmo = BackgroundCosmology(params)

    # Integrate memory field
    M_interp = integrate_memory_field(cosmo)
    M_today = M_interp(0.0)[0]
    cosmo.set_M_today(M_today)

    # Run all tests
    print("    Running background test...")
    bg_result = run_background_test(cosmo, M_interp)

    print("    Running growth test...")
    growth_result = run_growth_test(cosmo, M_interp)

    print("    Running CMB distance test...")
    cmb_result = run_cmb_distance_test(cosmo, M_interp)

    print("    Running perturbation stability test...")
    pert_result = run_perturbation_stability_test(cosmo, M_interp)

    # Determine overall viability
    bg_ok = bg_result["passes"]
    growth_ok = growth_result["passes"] in ["true", "marginal"]
    cmb_ok = cmb_result["passes"] in ["true", "marginal"]
    pert_ok = pert_result["status"] in ["ok", "warning"]

    if bg_ok and growth_ok and cmb_ok and pert_ok:
        if (growth_result["passes"] == "true" and
            cmb_result["passes"] == "true" and
            pert_result["status"] == "ok"):
            overall = "viable"
        else:
            overall = "marginal"
    else:
        overall = "ruled out"

    # Generate perturbation plots
    fig_dir = f"figures/tests/T06_full_viability/{model_id}"
    make_perturbation_plots(cosmo, M_interp, model_id, fig_dir)

    # Export CLASS files
    print("    Exporting CLASS files...")
    class_dir = f"results/class_export/{model_id}"
    export_to_class_format(model_id, class_dir)

    return FullViabilityResult(
        model_id=model_id,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        passes_SN_BAO=bg_result["passes"],
        max_H_deviation=bg_result["max_H_deviation"],
        max_D_L_deviation=bg_result["max_D_L_deviation"],
        passes_growth=growth_result["passes"],
        max_f_sigma8_deviation=growth_result["max_deviation"],
        passes_CMB_distance=cmb_result["passes"],
        cmb_distance_deviation=cmb_result["deviation_percent"],
        perturbation_stability=pert_result["status"],
        perturbation_notes=pert_result["notes"],
        overall_viability=overall,
    )


def save_viability_report(result: FullViabilityResult, output_dir: str):
    """Save full viability report to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "model_id": result.model_id,
        "lambda_hor": float(result.lambda_hor),
        "tau_hor": float(result.tau_hor),
        "tests": {
            "SN_BAO": {
                "passes": result.passes_SN_BAO,
                "max_H_deviation": float(result.max_H_deviation),
                "max_D_L_deviation": float(result.max_D_L_deviation),
            },
            "growth": {
                "passes": result.passes_growth,
                "max_f_sigma8_deviation": float(result.max_f_sigma8_deviation),
            },
            "CMB_distance": {
                "passes": result.passes_CMB_distance,
                "deviation_percent": float(result.cmb_distance_deviation),
            },
            "perturbation_stability": {
                "status": result.perturbation_stability,
                "notes": result.perturbation_notes,
            },
        },
        "overall_viability": result.overall_viability,
    }

    with open(os.path.join(output_dir, "viability.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"    Report saved to {output_dir}/viability.json")


def print_final_summary(results: List[FullViabilityResult]):
    """Print final summary to console."""
    print("\n" + "="*80)
    print("HORIZON-MEMORY FULL VIABILITY ANALYSIS - FINAL SUMMARY")
    print("="*80)

    print("\n" + "-"*80)
    print("INDIVIDUAL MODEL RESULTS:")
    print("-"*80)

    for result in results:
        print(f"\nModel: lambda_hor={result.lambda_hor:.3f}, tau_hor={result.tau_hor:.2f}")
        print(f"  SN/BAO:           {'PASS' if result.passes_SN_BAO else 'FAIL'} "
              f"(H dev: {result.max_H_deviation*100:.2f}%, D_L dev: {result.max_D_L_deviation*100:.2f}%)")
        print(f"  Growth (f*s8):    {result.passes_growth.upper()} "
              f"(max dev: {result.max_f_sigma8_deviation*100:.2f}%)")
        print(f"  CMB Distance:     {result.passes_CMB_distance.upper()} "
              f"(dev: {result.cmb_distance_deviation:.4f}%)")
        print(f"  Perturbations:    {result.perturbation_stability.upper()} "
              f"({result.perturbation_notes})")
        print(f"  ----------------------------------------")
        print(f"  OVERALL:          {result.overall_viability.upper()}")

    # Count results
    viable = [r for r in results if r.overall_viability == "viable"]
    marginal = [r for r in results if r.overall_viability == "marginal"]
    ruled_out = [r for r in results if r.overall_viability == "ruled out"]

    print("\n" + "-"*80)
    print("SUMMARY STATISTICS:")
    print("-"*80)
    print(f"\n  Total models analyzed: {len(results)}")
    print(f"  VIABLE:     {len(viable)}")
    print(f"  MARGINAL:   {len(marginal)}")
    print(f"  RULED OUT:  {len(ruled_out)}")

    if len(viable) > 0:
        print("\n  VIABLE MODELS:")
        for r in viable:
            print(f"    - lambda_hor={r.lambda_hor:.3f}, tau_hor={r.tau_hor:.2f}")
    elif len(marginal) > 0:
        print("\n  Best MARGINAL models:")
        for r in marginal:
            print(f"    - lambda_hor={r.lambda_hor:.3f}, tau_hor={r.tau_hor:.2f}")

    print("\n" + "="*80)


def main():
    print("="*80)
    print("HORIZON-MEMORY FULL VIABILITY ANALYSIS")
    print("="*80)

    # Load T06 models
    summary_path = "results/tests/T06_background_analysis/summary.json"

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {summary_path} not found.")
        print("Run the T06 viability analysis first:")
        print("  python scripts/run_T06_viability_analysis.py")
        return 1

    models = summary["models"]
    print(f"\nLoaded {len(models)} models from T06 analysis.")

    # Analyze each model
    results = []

    for model_data in models:
        result = analyze_full_viability(model_data)
        results.append(result)

        # Save individual report
        output_dir = f"results/tests/T06_full_viability/{result.model_id}"
        save_viability_report(result, output_dir)

    # Print final summary
    print_final_summary(results)

    # Save combined summary
    combined_output = {
        "analysis_type": "T06_full_viability",
        "n_models": len(results),
        "n_viable": len([r for r in results if r.overall_viability == "viable"]),
        "n_marginal": len([r for r in results if r.overall_viability == "marginal"]),
        "n_ruled_out": len([r for r in results if r.overall_viability == "ruled out"]),
        "models": [
            {
                "model_id": r.model_id,
                "lambda_hor": r.lambda_hor,
                "tau_hor": r.tau_hor,
                "overall_viability": r.overall_viability,
            }
            for r in results
        ],
    }

    os.makedirs("results/tests/T06_full_viability", exist_ok=True)
    with open("results/tests/T06_full_viability/summary.json", "w") as f:
        json.dump(combined_output, f, indent=2)

    print("\nAnalysis complete!")
    print(f"Results saved to results/tests/T06_full_viability/")
    print(f"Plots saved to figures/tests/T06_full_viability/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
