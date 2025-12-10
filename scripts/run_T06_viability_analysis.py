#!/usr/bin/env python3
"""T06 Horizon-Memory Model Viability Analysis.

This script performs a comprehensive background-level viability check for the
horizon-memory Friedmann correction model (T06).

Tasks:
1. Load T06 scan results and identify promising parameter points
2. Compute full background evolution for top 3 models
3. Generate publication-quality diagnostic plots
4. Produce structured evaluation report
5. Print final conclusion

Usage:
    python scripts/run_T06_viability_analysis.py
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


@dataclass
class ModelCandidate:
    """Container for a promising horizon-memory model candidate."""
    lambda_hor: float
    tau_hor: float
    M_today: float
    rho_hor_frac_z0: float
    rho_hor_frac_z1100: float
    delta_H0_proxy: float
    Omega_hor0: float
    Omega_L0_eff: float
    Omega_L0_base: float


@dataclass
class BackgroundEvolution:
    """Container for full background evolution data."""
    z_array: np.ndarray
    H_hm: np.ndarray  # Horizon-memory H(z)
    H_lcdm: np.ndarray  # Baseline LCDM H(z)
    chi_hm: np.ndarray  # Comoving distance
    chi_lcdm: np.ndarray
    D_L_hm: np.ndarray  # Luminosity distance
    D_L_lcdm: np.ndarray
    D_A_hm: np.ndarray  # Angular diameter distance
    D_A_lcdm: np.ndarray
    Omega_hor: np.ndarray  # Horizon energy fraction
    w_hor: np.ndarray  # Effective equation of state
    M_z: np.ndarray  # Memory field evolution


# ===========================================================================
# TASK 1: Load and filter T06 scan results
# ===========================================================================

def load_T06_scan_results(path: str = "results/tests/T06_horizon_memory_nonlocal/status.json") -> dict:
    """Load T06 scan results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def filter_promising_points(scan_data: dict) -> List[ModelCandidate]:
    """Filter points satisfying viability criteria.

    Criteria:
    - rho_hor_frac_z1100 < 1e-3 (safe early-time behavior)
    - 0.03 <= rho_hor_frac_z0 <= 0.30 (non-negligible late-time component)
    - 0.03 <= delta_H0_proxy <= 0.15 (proxy suggests meaningful effect)
    """
    candidates = []

    for r in scan_data["scan_results"]:
        # Apply filters
        early_safe = r["rho_hor_frac_z1100"] < 1e-3
        late_significant = 0.03 <= r["rho_hor_frac_z0"] <= 0.30
        proxy_meaningful = 0.03 <= r["delta_H0_proxy"] <= 0.15

        if early_safe and late_significant and proxy_meaningful:
            candidates.append(ModelCandidate(
                lambda_hor=r["lambda_hor"],
                tau_hor=r["tau_hor"],
                M_today=r["M_today"],
                rho_hor_frac_z0=r["rho_hor_frac_z0"],
                rho_hor_frac_z1100=r["rho_hor_frac_z1100"],
                delta_H0_proxy=r["delta_H0_proxy"],
                Omega_hor0=r["Omega_hor0"],
                Omega_L0_eff=r["Omega_L0_eff"],
                Omega_L0_base=r["Omega_L0_base"],
            ))

    # Sort by descending delta_H0_proxy
    candidates.sort(key=lambda x: x.delta_H0_proxy, reverse=True)

    return candidates


# ===========================================================================
# TASK 2: Compute full background evolution
# ===========================================================================

def integrate_memory_field(cosmo: BackgroundCosmology) -> callable:
    """Integrate the memory field ODE and return interpolator."""
    a_start, a_end = 1e-6, 1.0
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


def compute_comoving_distance(z_array: np.ndarray, H_func: callable) -> np.ndarray:
    """Compute comoving distance chi(z) = integral_0^z dz'/H(z')."""
    chi = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        if z == 0:
            chi[i] = 0.0
            continue

        def integrand(zp):
            H = H_func(zp)
            return 1.0 / H if H > 0 else 0.0

        chi[i], _ = quad(integrand, 0, z, limit=100)

    return chi


def compute_background_evolution(candidate: ModelCandidate, n_points: int = 200, z_max: float = 3.0) -> BackgroundEvolution:
    """Compute full background evolution for a model candidate."""

    # Create cosmology instance
    params = HRC2Parameters(
        xi=0.0,
        phi_0=0.0,
        coupling_family=CouplingFamily.QUADRATIC,
        potential_type=PotentialType.QUADRATIC,
        lambda_hor=candidate.lambda_hor,
        tau_hor=candidate.tau_hor,
    )
    cosmo = BackgroundCosmology(params)

    # Integrate memory field
    M_interp = integrate_memory_field(cosmo)
    M_today = M_interp(0.0)[0]

    # Set up self-consistent Lambda
    cosmo.set_M_today(M_today)

    # Create redshift array
    z_array = np.linspace(0, z_max, n_points)

    # Arrays for results
    H_hm = np.zeros_like(z_array)
    H_lcdm = np.zeros_like(z_array)
    M_z = np.zeros_like(z_array)
    Omega_hor = np.zeros_like(z_array)

    # Compute H(z) for both models
    for i, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)

        # Memory field at this z
        M = M_interp(ln_a)[0]
        M_z[i] = M

        # Horizon-memory H(z)
        H_hm[i] = cosmo.H_of_a_selfconsistent(a, M)

        # Baseline LCDM H(z)
        H_lcdm[i] = cosmo.H_of_a_gr_baseline(a)

        # Horizon energy fraction: Omega_hor(z) = rho_hor / rho_crit = rho_hor / (H^2 / H0^2)
        rho_hor = cosmo.lambda_hor * M
        H_ratio_sq = (H_hm[i] / cosmo.H0)**2 if cosmo.H0 > 0 else 1.0
        Omega_hor[i] = rho_hor / H_ratio_sq if H_ratio_sq > 0 else 0.0

    # Compute comoving distances
    def H_hm_func(z):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M)

    def H_lcdm_func(z):
        a = 1.0 / (1.0 + z)
        return cosmo.H_of_a_gr_baseline(a)

    chi_hm = compute_comoving_distance(z_array, H_hm_func)
    chi_lcdm = compute_comoving_distance(z_array, H_lcdm_func)

    # Luminosity and angular diameter distances
    D_L_hm = (1.0 + z_array) * chi_hm
    D_L_lcdm = (1.0 + z_array) * chi_lcdm
    D_A_hm = chi_hm / (1.0 + z_array)
    D_A_lcdm = chi_lcdm / (1.0 + z_array)
    # Fix D_A at z=0 (0/0)
    D_A_hm[0] = 0.0
    D_A_lcdm[0] = 0.0

    # Compute effective equation of state w_hor(z)
    # w_hor = -1 - (1/3) * d ln(rho_hor) / d ln(a)
    w_hor = np.zeros_like(z_array)
    eps = 1e-4  # Small step for numerical derivative

    for i, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)

        M = M_interp(ln_a)[0]
        rho = cosmo.lambda_hor * M

        if rho > 0 and i > 0 and i < len(z_array) - 1:
            # Numerical derivative using centered difference
            ln_a_plus = min(ln_a + eps, 0.0)
            ln_a_minus = ln_a - eps

            M_plus = M_interp(ln_a_plus)[0]
            M_minus = M_interp(ln_a_minus)[0]

            rho_plus = cosmo.lambda_hor * M_plus
            rho_minus = cosmo.lambda_hor * M_minus

            if rho_plus > 0 and rho_minus > 0:
                d_ln_rho = (np.log(rho_plus) - np.log(rho_minus)) / (2 * eps)
                w_hor[i] = -1.0 - d_ln_rho / 3.0
            else:
                w_hor[i] = np.nan
        else:
            w_hor[i] = np.nan

    return BackgroundEvolution(
        z_array=z_array,
        H_hm=H_hm,
        H_lcdm=H_lcdm,
        chi_hm=chi_hm,
        chi_lcdm=chi_lcdm,
        D_L_hm=D_L_hm,
        D_L_lcdm=D_L_lcdm,
        D_A_hm=D_A_hm,
        D_A_lcdm=D_A_lcdm,
        Omega_hor=Omega_hor,
        w_hor=w_hor,
        M_z=M_z,
    )


# ===========================================================================
# TASK 3: Generate diagnostic plots
# ===========================================================================

def make_diagnostic_plots(
    candidate: ModelCandidate,
    evolution: BackgroundEvolution,
    output_dir: str,
):
    """Generate publication-quality diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    z = evolution.z_array

    # Plot style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (10, 7),
    })

    # =======================================================================
    # Plot 1: H(z)/H_LCDM(z) ratio
    # =======================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    H_ratio = evolution.H_hm / evolution.H_lcdm

    # Tolerance bands
    ax.fill_between(z, 0.95, 1.05, alpha=0.2, color='green', label='5% tolerance')
    ax.fill_between(z, 0.90, 1.10, alpha=0.1, color='orange', label='10% tolerance')

    ax.plot(z, H_ratio, 'b-', linewidth=2, label='Horizon-memory / LCDM')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('H(z) / H_LCDM(z)')
    ax.set_title(f'Hubble Parameter Ratio: $\\lambda_{{hor}}={candidate.lambda_hor:.3f}$, $\\tau_{{hor}}={candidate.tau_hor:.2f}$')
    ax.legend(loc='best')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.85, 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "H_ratio.png"), dpi=150)
    plt.close()

    # =======================================================================
    # Plot 2: Distance ratios
    # =======================================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # D_L ratio (skip z=0 where both are 0)
    valid = z > 0.01
    D_L_ratio = np.ones_like(z)
    D_L_ratio[valid] = evolution.D_L_hm[valid] / evolution.D_L_lcdm[valid]

    D_A_ratio = np.ones_like(z)
    D_A_ratio[valid] = evolution.D_A_hm[valid] / evolution.D_A_lcdm[valid]

    # Luminosity distance
    ax1 = axes[0]
    ax1.fill_between(z, 0.95, 1.05, alpha=0.2, color='green', label='5% tolerance')
    ax1.fill_between(z, 0.90, 1.10, alpha=0.1, color='orange', label='10% tolerance')
    ax1.plot(z, D_L_ratio, 'r-', linewidth=2, label='$D_L$ ratio')
    ax1.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('$D_L(z) / D_{L,LCDM}(z)$')
    ax1.set_title(f'Distance Ratios: $\\lambda_{{hor}}={candidate.lambda_hor:.3f}$, $\\tau_{{hor}}={candidate.tau_hor:.2f}$')
    ax1.legend(loc='best')
    ax1.set_ylim(0.85, 1.15)
    ax1.grid(True, alpha=0.3)

    # Angular diameter distance
    ax2 = axes[1]
    ax2.fill_between(z, 0.95, 1.05, alpha=0.2, color='green', label='5% tolerance')
    ax2.fill_between(z, 0.90, 1.10, alpha=0.1, color='orange', label='10% tolerance')
    ax2.plot(z, D_A_ratio, 'purple', linewidth=2, label='$D_A$ ratio')
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('$D_A(z) / D_{A,LCDM}(z)$')
    ax2.legend(loc='best')
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0.85, 1.15)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_ratios.png"), dpi=150)
    plt.close()

    # =======================================================================
    # Plot 3: Horizon energy density evolution
    # =======================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(z, evolution.Omega_hor, 'g-', linewidth=2)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('$\\Omega_{hor}(z)$')
    ax.set_title(f'Horizon Energy Fraction: $\\lambda_{{hor}}={candidate.lambda_hor:.3f}$, $\\tau_{{hor}}={candidate.tau_hor:.2f}$')
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.3)

    # Annotate z=0 value
    ax.annotate(f'$\\Omega_{{hor,0}} = {candidate.Omega_hor0:.4f}$',
                xy=(0, evolution.Omega_hor[0]),
                xytext=(0.5, evolution.Omega_hor[0] * 1.2),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Omega_hor.png"), dpi=150)
    plt.close()

    # =======================================================================
    # Plot 4: Effective equation of state
    # =======================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    valid_w = ~np.isnan(evolution.w_hor)
    ax.plot(z[valid_w], evolution.w_hor[valid_w], 'b-', linewidth=2, label='$w_{hor}(z)$')

    # Reference lines
    ax.axhline(-1, color='purple', linestyle='--', alpha=0.7, label='$w = -1$ (cosmological constant)')
    ax.axhline(0, color='green', linestyle='--', alpha=0.7, label='$w = 0$ (matter)')
    ax.axhline(1, color='red', linestyle='--', alpha=0.7, label='$w = +1$ (stiff matter)')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('$w_{hor}(z)$')
    ax.set_title(f'Effective Equation of State: $\\lambda_{{hor}}={candidate.lambda_hor:.3f}$, $\\tau_{{hor}}={candidate.tau_hor:.2f}$')
    ax.legend(loc='best')
    ax.set_xlim(0, 3)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "w_hor.png"), dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


# ===========================================================================
# TASK 4: Produce evaluation report
# ===========================================================================

def evaluate_model(candidate: ModelCandidate, evolution: BackgroundEvolution) -> dict:
    """Evaluate a single model and return metrics dictionary."""

    z = evolution.z_array

    # H(z) ratio deviation in 0 < z < 1
    H_ratio = evolution.H_hm / evolution.H_lcdm
    mask_low_z = (z > 0) & (z < 1)
    max_H_deviation = np.max(np.abs(H_ratio[mask_low_z] - 1.0)) if np.any(mask_low_z) else 0.0

    # Distance ratio deviations
    valid = z > 0.01
    D_L_ratio = np.ones_like(z)
    D_L_ratio[valid] = evolution.D_L_hm[valid] / evolution.D_L_lcdm[valid]
    D_A_ratio = np.ones_like(z)
    D_A_ratio[valid] = evolution.D_A_hm[valid] / evolution.D_A_lcdm[valid]

    max_D_L_deviation = np.max(np.abs(D_L_ratio - 1.0))
    max_D_A_deviation = np.max(np.abs(D_A_ratio - 1.0))

    # Assess w_hor stability
    valid_w = ~np.isnan(evolution.w_hor)
    if np.sum(valid_w) > 10:
        w_range = np.max(evolution.w_hor[valid_w]) - np.min(evolution.w_hor[valid_w])
        if w_range < 0.5:
            w_stability = "smooth"
        elif w_range < 2.0:
            w_stability = "mild variation"
        else:
            w_stability = "pathological"
    else:
        w_stability = "insufficient data"

    # Check if within tolerance
    within_5pct = max_H_deviation < 0.05 and max_D_L_deviation < 0.05
    within_10pct = max_H_deviation < 0.10 and max_D_L_deviation < 0.10

    # Assess Hubble tension relief potential
    # In the self-consistent model, we look at evolution differences
    # A model "relieves" tension if it modifies expansion history while staying within bounds
    relieves_tension = within_10pct and candidate.Omega_hor0 > 0.03

    # Worth pursuing further?
    worth_pursuing = within_10pct and w_stability != "pathological"

    return {
        "lambda_hor": float(candidate.lambda_hor),
        "tau_hor": float(candidate.tau_hor),
        "Omega_hor0": float(candidate.Omega_hor0),
        "Omega_L0_eff": float(candidate.Omega_L0_eff),
        "delta_H0_proxy": float(candidate.delta_H0_proxy),
        "max_H_deviation_z01": float(max_H_deviation),
        "max_D_L_deviation": float(max_D_L_deviation),
        "max_D_A_deviation": float(max_D_A_deviation),
        "w_hor_stability": w_stability,
        "within_5pct_tolerance": bool(within_5pct),
        "within_10pct_tolerance": bool(within_10pct),
        "plausibly_relieves_tension": bool(relieves_tension),
        "worth_perturbation_analysis": bool(worth_pursuing),
    }


def generate_summary_report(evaluations: List[dict], output_path: str):
    """Generate summary JSON report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = {
        "analysis_type": "T06_horizon_memory_viability",
        "n_models_analyzed": len(evaluations),
        "models": evaluations,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSummary report saved to {output_path}")


# ===========================================================================
# TASK 5: Print final conclusion
# ===========================================================================

def print_final_conclusion(evaluations: List[dict]):
    """Print human-readable summary to console."""

    print("\n" + "="*80)
    print("T06 HORIZON-MEMORY MODEL VIABILITY ANALYSIS - FINAL CONCLUSION")
    print("="*80)

    viable_models = [e for e in evaluations if e["within_10pct_tolerance"]]
    pursuit_worthy = [e for e in evaluations if e["worth_perturbation_analysis"]]

    print(f"\nModels analyzed: {len(evaluations)}")
    print(f"Models within 10% SN/BAO tolerance: {len(viable_models)}")
    print(f"Models worth perturbation analysis: {len(pursuit_worthy)}")

    print("\n" + "-"*80)
    print("INDIVIDUAL MODEL ASSESSMENTS:")
    print("-"*80)

    for i, e in enumerate(evaluations):
        print(f"\nModel {i+1}: lambda_hor={e['lambda_hor']:.3f}, tau_hor={e['tau_hor']:.2f}")
        print(f"  Omega_hor0 = {e['Omega_hor0']:.4f}")
        print(f"  Omega_L0_eff = {e['Omega_L0_eff']:.4f}")
        print(f"  delta_H0_proxy = {e['delta_H0_proxy']*100:.2f}%")
        print(f"  max|H/H_LCDM - 1| (z<1) = {e['max_H_deviation_z01']*100:.2f}%")
        print(f"  max|D_L ratio - 1| = {e['max_D_L_deviation']*100:.2f}%")
        print(f"  w_hor stability: {e['w_hor_stability']}")
        print(f"  Within 10% tolerance: {'YES' if e['within_10pct_tolerance'] else 'NO'}")
        print(f"  Worth perturbation analysis: {'YES' if e['worth_perturbation_analysis'] else 'NO'}")

    print("\n" + "-"*80)
    print("OVERALL ASSESSMENT:")
    print("-"*80)

    if len(viable_models) == 0:
        print("\n[RESULT] No models pass the 10% SN/BAO tolerance threshold.")
        print("         The horizon-memory mechanism in its current form produces")
        print("         deviations too large for background-level viability.")
        mechanism_promising = False
    elif len(pursuit_worthy) > 0:
        print(f"\n[RESULT] {len(pursuit_worthy)} model(s) look viable at background level!")
        print("         The horizon-memory mechanism shows promise for Hubble tension relief.")
        mechanism_promising = True

        best = max(pursuit_worthy, key=lambda x: x["delta_H0_proxy"])
        print(f"\n         BEST CANDIDATE: lambda_hor={best['lambda_hor']:.3f}, tau_hor={best['tau_hor']:.2f}")
    else:
        print("\n[RESULT] Models are within tolerance but show other issues (w_hor pathological).")
        mechanism_promising = False

    print("\n" + "-"*80)
    print("RECOMMENDED NEXT STEPS:")
    print("-"*80)

    if mechanism_promising:
        print("""
1. PERTURBATION ANALYSIS
   - Implement scalar perturbation equations for horizon-memory field
   - Check growth factor f*sigma8 against BOSS/eBOSS data
   - Verify ISW effect consistency

2. CMB CONSTRAINTS
   - Interface with CLASS-like Boltzmann solver
   - Check angular diameter distance to last scattering
   - Verify CMB peak positions

3. MEMORY KERNEL REFINEMENT
   - Test alternative memory evolution laws (exponential, power-law)
   - Explore temperature-dependent tau_hor

4. COMBINED ANALYSIS
   - Run MCMC with SN Ia + BAO + CMB data
   - Map out allowed (lambda_hor, tau_hor) region
""")
    else:
        print("""
1. MODIFY THE MODEL
   - Consider non-linear memory evolution M(a)
   - Add coupling to matter sector
   - Try scale-dependent lambda_hor

2. ALTERNATIVE MECHANISMS
   - Return to scalar-tensor couplings (T01-T04)
   - Explore EDE variants (T05)
   - Consider interacting dark energy
""")

    print("="*80)

    return mechanism_promising


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    print("="*80)
    print("T06 HORIZON-MEMORY MODEL VIABILITY ANALYSIS")
    print("="*80)

    # TASK 1: Load and filter
    print("\n[TASK 1] Loading T06 scan results and filtering promising points...")

    try:
        scan_data = load_T06_scan_results()
    except FileNotFoundError:
        print("ERROR: T06 scan results not found. Run T06 scan first:")
        print("  python scripts/run_coupling_test_suite.py T06_horizon_memory_nonlocal")
        return 1

    candidates = filter_promising_points(scan_data)

    print(f"  Found {len(candidates)} points satisfying criteria:")
    print("    - rho_hor_frac_z1100 < 1e-3")
    print("    - 0.03 <= rho_hor_frac_z0 <= 0.30")
    print("    - 0.03 <= delta_H0_proxy <= 0.15")

    if len(candidates) == 0:
        print("\n  WARNING: No candidates found! Relaxing criteria...")
        # Relax criteria
        for r in scan_data["scan_results"]:
            early_safe = r["rho_hor_frac_z1100"] < 0.01
            late_significant = 0.01 <= r["rho_hor_frac_z0"] <= 0.50
            proxy_meaningful = 0.01 <= r["delta_H0_proxy"] <= 0.20

            if early_safe and late_significant and proxy_meaningful:
                candidates.append(ModelCandidate(
                    lambda_hor=r["lambda_hor"],
                    tau_hor=r["tau_hor"],
                    M_today=r["M_today"],
                    rho_hor_frac_z0=r["rho_hor_frac_z0"],
                    rho_hor_frac_z1100=r["rho_hor_frac_z1100"],
                    delta_H0_proxy=r["delta_H0_proxy"],
                    Omega_hor0=r["Omega_hor0"],
                    Omega_L0_eff=r["Omega_L0_eff"],
                    Omega_L0_base=r["Omega_L0_base"],
                ))
        candidates.sort(key=lambda x: x.delta_H0_proxy, reverse=True)
        print(f"  After relaxing: {len(candidates)} candidates")

    # Keep top 3
    top_candidates = candidates[:3]

    print(f"\n  Top {len(top_candidates)} candidates for analysis:")
    for i, c in enumerate(top_candidates):
        print(f"    {i+1}. lambda_hor={c.lambda_hor:.3f}, tau_hor={c.tau_hor:.2f}, "
              f"delta_H0_proxy={c.delta_H0_proxy*100:.2f}%")

    if len(top_candidates) == 0:
        print("\nERROR: No candidates found even with relaxed criteria.")
        return 1

    # TASK 2: Compute background evolution
    print("\n[TASK 2] Computing full background evolution for each model...")

    evolutions = []
    for i, candidate in enumerate(top_candidates):
        print(f"  Model {i+1}/{len(top_candidates)}: lambda_hor={candidate.lambda_hor:.3f}, tau_hor={candidate.tau_hor:.2f}")
        evolution = compute_background_evolution(candidate)
        evolutions.append(evolution)
        print(f"    H(z=0) ratio: {evolution.H_hm[0]/evolution.H_lcdm[0]:.6f}")
        print(f"    Omega_hor(z=0): {evolution.Omega_hor[0]:.6f}")

    # TASK 3: Generate plots
    print("\n[TASK 3] Generating diagnostic plots...")

    for i, (candidate, evolution) in enumerate(zip(top_candidates, evolutions)):
        output_dir = f"figures/tests/T06_background_analysis/{candidate.lambda_hor:.3f}_{candidate.tau_hor:.2f}"
        print(f"  Model {i+1}:")
        make_diagnostic_plots(candidate, evolution, output_dir)

    # TASK 4: Generate report
    print("\n[TASK 4] Generating evaluation report...")

    evaluations = []
    for candidate, evolution in zip(top_candidates, evolutions):
        eval_result = evaluate_model(candidate, evolution)
        evaluations.append(eval_result)

    generate_summary_report(evaluations, "results/tests/T06_background_analysis/summary.json")

    # TASK 5: Print conclusion
    print("\n[TASK 5] Final conclusion...")
    mechanism_promising = print_final_conclusion(evaluations)

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
