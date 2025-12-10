#!/usr/bin/env python3
"""Growth-of-structure predictions for horizon-memory cosmology.

This script computes the linear growth factor D(a) and growth rate f*sigma_8
for horizon-memory models, comparing them to LCDM predictions.

The linear growth equation is:
    D''(a) + (3/a + H'/H)*D'(a) - (3*Omega_m(a))/(2*a^2*(H/H0)^2) * D(a) = 0

where:
    - D(a) is the linear growth factor
    - H(a) is the Hubble parameter
    - Omega_m(a) = Omega_m0 * (1+z)^3 / (H/H0)^2
    - Primes denote d/d(ln a)

The growth rate is:
    f(a) = d ln D / d ln a

And the observable:
    f*sigma_8(z) = f(z) * sigma_8 * D(z)/D(z=0)

Usage:
    python scripts/run_horizon_memory_growth.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.theory import HRC2Parameters, CouplingFamily, PotentialType
from hrc2.background import BackgroundCosmology


# Planck 2018 values
OMEGA_M0 = 0.315
OMEGA_R0 = 9.0e-5
H0_PLANCK = 67.4  # km/s/Mpc
SIGMA8_PLANCK = 0.811


@dataclass
class GrowthResult:
    """Container for growth factor computation results."""
    model_id: str
    lambda_hor: float
    tau_hor: float

    # Arrays
    z_array: np.ndarray
    a_array: np.ndarray
    D_hm: np.ndarray  # Growth factor for horizon-memory
    D_lcdm: np.ndarray  # Growth factor for LCDM
    f_hm: np.ndarray  # Growth rate for horizon-memory
    f_lcdm: np.ndarray  # Growth rate for LCDM
    f_sigma8_hm: np.ndarray  # f*sigma_8 for horizon-memory
    f_sigma8_lcdm: np.ndarray  # f*sigma_8 for LCDM

    # Metrics
    max_f_sigma8_deviation: float
    rms_f_sigma8_deviation: float
    growth_status: str  # "good", "marginal", "bad"


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


def compute_H_ratio(
    a: float,
    cosmo: BackgroundCosmology,
    M_interp: callable,
    use_hm: bool = True,
) -> float:
    """Compute H(a)/H0 for the given model.

    Args:
        a: Scale factor
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator
        use_hm: If True, use horizon-memory H; if False, use baseline LCDM

    Returns:
        H/H0
    """
    if use_hm:
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        return cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0
    else:
        return cosmo.H_of_a_gr_baseline(a) / cosmo.H0


def compute_Omega_m(
    a: float,
    cosmo: BackgroundCosmology,
    M_interp: callable,
    use_hm: bool = True,
) -> float:
    """Compute Omega_m(a) = rho_m / rho_crit.

    Args:
        a: Scale factor
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator
        use_hm: If True, use horizon-memory H; if False, use baseline LCDM

    Returns:
        Omega_m(a)
    """
    z = 1.0 / a - 1.0
    rho_m = cosmo.Omega_m0 * (1 + z)**3

    H_ratio = compute_H_ratio(a, cosmo, M_interp, use_hm)

    # Omega_m = rho_m / rho_crit = rho_m / (H^2/H0^2)
    return rho_m / (H_ratio**2) if H_ratio > 0 else 0.0


def solve_growth_equation(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    use_hm: bool = True,
    a_init: float = 1e-4,
    a_final: float = 1.0,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the linear growth equation.

    The growth equation in ln(a) is:
        D'' + (2 + H'/H) * D' - (3/2) * Omega_m(a) * D = 0

    where primes are d/d(ln a) and H'/H = d ln H / d ln a.

    We convert to first-order system:
        y[0] = D
        y[1] = D' = dD/d(ln a)

    Args:
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator
        use_hm: If True, use horizon-memory; if False, use LCDM
        a_init: Initial scale factor (deep in matter domination)
        a_final: Final scale factor
        n_points: Number of output points

    Returns:
        Tuple (a_array, D_array, f_array) where f = d ln D / d ln a
    """
    ln_a_init = np.log(a_init)
    ln_a_final = np.log(a_final)
    ln_a_eval = np.linspace(ln_a_init, ln_a_final, n_points)

    # Precompute H'/H using finite differences
    eps = 1e-4

    def H_prime_over_H(a: float) -> float:
        """Compute d ln H / d ln a."""
        ln_a = np.log(a)
        H_plus = compute_H_ratio(np.exp(ln_a + eps), cosmo, M_interp, use_hm)
        H_minus = compute_H_ratio(np.exp(ln_a - eps), cosmo, M_interp, use_hm)

        if H_plus > 0 and H_minus > 0:
            return (np.log(H_plus) - np.log(H_minus)) / (2 * eps)
        return 0.0

    def growth_ode(ln_a: float, y: np.ndarray) -> np.ndarray:
        """ODE system for growth equation."""
        D, D_prime = y
        a = np.exp(ln_a)

        # Background quantities
        Omega_m_a = compute_Omega_m(a, cosmo, M_interp, use_hm)
        d_ln_H_d_ln_a = H_prime_over_H(a)

        # Growth equation: D'' + (2 + H'/H)*D' - (3/2)*Omega_m*D = 0
        # D'' = (3/2)*Omega_m*D - (2 + H'/H)*D'
        D_double_prime = 1.5 * Omega_m_a * D - (2.0 + d_ln_H_d_ln_a) * D_prime

        return np.array([D_prime, D_double_prime])

    # Initial conditions: deep in matter domination, D ~ a
    # So D(a_init) ~ a_init and D'(a_init) = d(ln D)/d(ln a) * D ~ 1 * D
    D_init = a_init
    D_prime_init = D_init  # Since D ~ a implies d ln D / d ln a = 1

    y0 = np.array([D_init, D_prime_init])

    sol = solve_ivp(
        growth_ode,
        (ln_a_init, ln_a_final),
        y0,
        method='RK45',
        t_eval=ln_a_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Growth equation integration failed: {sol.message}")

    a_array = np.exp(sol.t)
    D_array = sol.y[0]
    D_prime_array = sol.y[1]

    # Normalize D so that D(a=1) = 1
    D_today = np.interp(1.0, a_array, D_array)
    D_array = D_array / D_today

    # Growth rate f = d ln D / d ln a = D' / D
    f_array = D_prime_array / D_array

    return a_array, D_array, f_array


def compute_f_sigma8(
    a_array: np.ndarray,
    D_array: np.ndarray,
    f_array: np.ndarray,
    sigma8_0: float = SIGMA8_PLANCK,
) -> np.ndarray:
    """Compute f*sigma_8(z).

    f*sigma_8(z) = f(z) * sigma_8(z) = f(z) * sigma8_0 * D(z)/D(z=0)

    Since we normalized D(z=0) = 1:
        f*sigma_8(z) = f(z) * sigma8_0 * D(z)

    Args:
        a_array: Scale factor array
        D_array: Growth factor array (normalized to D(a=1)=1)
        f_array: Growth rate array
        sigma8_0: sigma_8 at z=0

    Returns:
        f*sigma_8 array
    """
    return f_array * sigma8_0 * D_array


def analyze_model(
    model_data: dict,
    n_points: int = 200,
    z_max: float = 2.0,
) -> GrowthResult:
    """Analyze growth-of-structure for a single model.

    Args:
        model_data: Dictionary with model parameters
        n_points: Number of output points
        z_max: Maximum redshift

    Returns:
        GrowthResult with full analysis
    """
    lambda_hor = model_data["lambda_hor"]
    tau_hor = model_data["tau_hor"]

    model_id = f"{lambda_hor:.3f}_{tau_hor:.2f}"
    print(f"  Analyzing model {model_id}...")

    # Create cosmology instance
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

    # Set self-consistent Lambda
    cosmo.set_M_today(M_today)

    # Solve growth equation for horizon-memory model
    a_hm, D_hm, f_hm = solve_growth_equation(
        cosmo, M_interp, use_hm=True, n_points=n_points
    )

    # Solve growth equation for baseline LCDM
    a_lcdm, D_lcdm, f_lcdm = solve_growth_equation(
        cosmo, M_interp, use_hm=False, n_points=n_points
    )

    # Convert to redshift arrays (both should be same since same a_eval)
    z_hm = 1.0 / a_hm - 1.0
    z_lcdm = 1.0 / a_lcdm - 1.0

    # Compute f*sigma_8
    f_sigma8_hm = compute_f_sigma8(a_hm, D_hm, f_hm)
    f_sigma8_lcdm = compute_f_sigma8(a_lcdm, D_lcdm, f_lcdm)

    # Restrict to z <= z_max
    mask_hm = z_hm <= z_max
    mask_lcdm = z_lcdm <= z_max

    z_array = z_hm[mask_hm]
    a_array = a_hm[mask_hm]
    D_hm_out = D_hm[mask_hm]
    D_lcdm_out = D_lcdm[mask_lcdm]
    f_hm_out = f_hm[mask_hm]
    f_lcdm_out = f_lcdm[mask_lcdm]
    f_sigma8_hm_out = f_sigma8_hm[mask_hm]
    f_sigma8_lcdm_out = f_sigma8_lcdm[mask_lcdm]

    # Compute deviation metrics
    # Relative deviation in f*sigma_8
    valid = f_sigma8_lcdm_out > 0
    relative_deviation = np.zeros_like(f_sigma8_hm_out)
    relative_deviation[valid] = (f_sigma8_hm_out[valid] - f_sigma8_lcdm_out[valid]) / f_sigma8_lcdm_out[valid]

    max_deviation = np.max(np.abs(relative_deviation))
    rms_deviation = np.sqrt(np.mean(relative_deviation**2))

    # Determine status
    if max_deviation < 0.05:
        status = "good"
    elif max_deviation < 0.10:
        status = "marginal"
    else:
        status = "bad"

    return GrowthResult(
        model_id=model_id,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        z_array=z_array,
        a_array=a_array,
        D_hm=D_hm_out,
        D_lcdm=D_lcdm_out,
        f_hm=f_hm_out,
        f_lcdm=f_lcdm_out,
        f_sigma8_hm=f_sigma8_hm_out,
        f_sigma8_lcdm=f_sigma8_lcdm_out,
        max_f_sigma8_deviation=max_deviation,
        rms_f_sigma8_deviation=rms_deviation,
        growth_status=status,
    )


def make_growth_plots(result: GrowthResult, output_dir: str):
    """Generate diagnostic plots for growth analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    z = result.z_array

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (10, 7),
    })

    # Plot 1: f*sigma_8 comparison
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(z, result.f_sigma8_hm, 'b-', linewidth=2, label='Horizon-memory')
    ax.plot(z, result.f_sigma8_lcdm, 'k--', linewidth=2, label='$\\Lambda$CDM')

    # Tolerance bands
    ax.fill_between(z, result.f_sigma8_lcdm * 0.95, result.f_sigma8_lcdm * 1.05,
                    alpha=0.2, color='green', label='5% tolerance')
    ax.fill_between(z, result.f_sigma8_lcdm * 0.90, result.f_sigma8_lcdm * 1.10,
                    alpha=0.1, color='orange', label='10% tolerance')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('$f\\sigma_8(z)$')
    ax.set_title(f'Growth Rate: $\\lambda_{{hor}}={result.lambda_hor:.3f}$, $\\tau_{{hor}}={result.tau_hor:.2f}$')
    ax.legend(loc='best')
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f_sigma8_comparison.png"), dpi=150)
    plt.close()

    # Plot 2: Growth factor comparison
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(z, result.D_hm, 'b-', linewidth=2, label='Horizon-memory $D(z)$')
    ax.plot(z, result.D_lcdm, 'k--', linewidth=2, label='$\\Lambda$CDM $D(z)$')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Growth factor $D(z)$')
    ax.set_title(f'Linear Growth Factor: $\\lambda_{{hor}}={result.lambda_hor:.3f}$, $\\tau_{{hor}}={result.tau_hor:.2f}$')
    ax.legend(loc='best')
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "growth_factor.png"), dpi=150)
    plt.close()

    # Plot 3: Growth rate f comparison
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(z, result.f_hm, 'b-', linewidth=2, label='Horizon-memory $f(z)$')
    ax.plot(z, result.f_lcdm, 'k--', linewidth=2, label='$\\Lambda$CDM $f(z)$')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Growth rate $f(z) = d\\ln D / d\\ln a$')
    ax.set_title(f'Linear Growth Rate: $\\lambda_{{hor}}={result.lambda_hor:.3f}$, $\\tau_{{hor}}={result.tau_hor:.2f}$')
    ax.legend(loc='best')
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "growth_rate.png"), dpi=150)
    plt.close()

    print(f"    Plots saved to {output_dir}/")


def save_growth_results(result: GrowthResult, output_dir: str):
    """Save growth analysis results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "model_id": result.model_id,
        "lambda_hor": float(result.lambda_hor),
        "tau_hor": float(result.tau_hor),
        "max_f_sigma8_deviation": float(result.max_f_sigma8_deviation),
        "rms_f_sigma8_deviation": float(result.rms_f_sigma8_deviation),
        "growth_status": result.growth_status,
        "z_values": result.z_array.tolist(),
        "D_horizon_memory": result.D_hm.tolist(),
        "D_lcdm": result.D_lcdm.tolist(),
        "f_horizon_memory": result.f_hm.tolist(),
        "f_lcdm": result.f_lcdm.tolist(),
        "f_sigma8_horizon_memory": result.f_sigma8_hm.tolist(),
        "f_sigma8_lcdm": result.f_sigma8_lcdm.tolist(),
    }

    with open(os.path.join(output_dir, "growth.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Results saved to {output_dir}/growth.json")


def main():
    print("="*80)
    print("HORIZON-MEMORY GROWTH-OF-STRUCTURE ANALYSIS")
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

    print("\n[TASK] Computing growth factors and f*sigma_8...")

    for i, model_data in enumerate(models):
        result = analyze_model(model_data)
        results.append(result)

        # Save results
        output_dir = f"results/tests/T06_growth/{result.model_id}"
        save_growth_results(result, output_dir)

        # Generate plots
        fig_dir = f"figures/tests/T06_growth/{result.model_id}"
        make_growth_plots(result, fig_dir)

    # Print summary
    print("\n" + "="*80)
    print("GROWTH ANALYSIS SUMMARY")
    print("="*80)

    for result in results:
        print(f"\nModel: lambda_hor={result.lambda_hor:.3f}, tau_hor={result.tau_hor:.2f}")
        print(f"  Max |f*sigma8 deviation|: {result.max_f_sigma8_deviation*100:.2f}%")
        print(f"  RMS f*sigma8 deviation: {result.rms_f_sigma8_deviation*100:.2f}%")
        print(f"  Growth status: {result.growth_status.upper()}")

    # Overall assessment
    good_models = [r for r in results if r.growth_status == "good"]
    marginal_models = [r for r in results if r.growth_status == "marginal"]

    print("\n" + "-"*80)
    if len(good_models) > 0:
        print(f"[RESULT] {len(good_models)} model(s) pass growth constraints (<5% deviation)")
    elif len(marginal_models) > 0:
        print(f"[RESULT] {len(marginal_models)} model(s) are marginal (5-10% deviation)")
    else:
        print("[RESULT] All models show significant growth deviations (>10%)")

    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
