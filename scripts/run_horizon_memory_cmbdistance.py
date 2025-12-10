#!/usr/bin/env python3
"""CMB distance constraints for horizon-memory cosmology.

This script computes the comoving angular diameter distance to last scattering
for horizon-memory models and compares to LCDM predictions.

The key observable is:
    D_* = integral_0^z_* dz / H(z)

where z_* = 1089 is the redshift of last scattering.

The ratio R = D_*(horizon-memory) / D_*(LCDM) measures how much the
model deviates from standard cosmology at the CMB epoch.

Constraints:
    - |R - 1| < 0.5%  -> passes CMB distance test
    - 0.5% <= |R - 1| < 1.0% -> marginal
    - |R - 1| >= 1.0% -> fails

Usage:
    python scripts/run_horizon_memory_cmbdistance.py
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


# CMB parameters
Z_STAR = 1089.0  # Redshift of last scattering
H0_PLANCK = 67.4  # km/s/Mpc


@dataclass
class CMBDistanceResult:
    """Container for CMB distance computation results."""
    model_id: str
    lambda_hor: float
    tau_hor: float

    # Distance quantities
    D_star_hm: float  # Comoving distance to last scattering (horizon-memory)
    D_star_lcdm: float  # Comoving distance to last scattering (LCDM)
    ratio: float  # D_star_hm / D_star_lcdm
    deviation_percent: float  # |ratio - 1| * 100

    # Status
    cmb_status: str  # "passes", "marginal", "fails"

    # Optional: integrand for plotting
    z_plot: Optional[np.ndarray] = None
    integrand_hm: Optional[np.ndarray] = None
    integrand_lcdm: Optional[np.ndarray] = None


def integrate_memory_field(cosmo: BackgroundCosmology, z_max: float = 1200.0) -> callable:
    """Integrate the memory field ODE and return interpolator.

    Extended to high z for CMB analysis.
    """
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


def compute_D_star(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    z_star: float = Z_STAR,
    use_hm: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute comoving distance to last scattering.

    D_* = c/H0 * integral_0^z_* dz / E(z)

    where E(z) = H(z)/H0.

    We return D_* in units where c/H0 = 1 (i.e., just the integral).

    Args:
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator
        z_star: Redshift of last scattering
        use_hm: If True, use horizon-memory H; if False, use baseline LCDM

    Returns:
        Tuple (D_star, z_plot, integrand)
    """
    def H_ratio(z: float) -> float:
        """Compute H(z)/H0."""
        a = 1.0 / (1.0 + z)

        if use_hm:
            ln_a = np.log(a)
            # Make sure ln_a is in valid range
            ln_a = max(ln_a, np.log(1.0 / (1.0 + z_star + 100)))
            M = M_interp(ln_a)[0]
            return cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0
        else:
            return cosmo.H_of_a_gr_baseline(a) / cosmo.H0

    def integrand(z: float) -> float:
        """Integrand 1/E(z) = H0/H(z)."""
        E = H_ratio(z)
        return 1.0 / E if E > 0 else 0.0

    # Compute integral
    D_star, _ = quad(integrand, 0, z_star, limit=500)

    # Generate plot data
    z_plot = np.logspace(-2, np.log10(z_star), 200)
    integrand_plot = np.array([integrand(z) for z in z_plot])

    return D_star, z_plot, integrand_plot


def analyze_cmb_distance(model_data: dict) -> CMBDistanceResult:
    """Analyze CMB distance constraints for a single model.

    Args:
        model_data: Dictionary with model parameters

    Returns:
        CMBDistanceResult with analysis
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

    # Integrate memory field to high z
    M_interp = integrate_memory_field(cosmo, z_max=1200.0)
    M_today = M_interp(0.0)[0]

    # Set self-consistent Lambda
    cosmo.set_M_today(M_today)

    # Compute D_* for horizon-memory
    D_star_hm, z_plot, integrand_hm = compute_D_star(
        cosmo, M_interp, z_star=Z_STAR, use_hm=True
    )

    # Compute D_* for baseline LCDM
    D_star_lcdm, _, integrand_lcdm = compute_D_star(
        cosmo, M_interp, z_star=Z_STAR, use_hm=False
    )

    # Compute ratio and deviation
    ratio = D_star_hm / D_star_lcdm if D_star_lcdm > 0 else 1.0
    deviation_percent = abs(ratio - 1.0) * 100

    # Determine status
    if deviation_percent < 0.5:
        status = "passes"
    elif deviation_percent < 1.0:
        status = "marginal"
    else:
        status = "fails"

    return CMBDistanceResult(
        model_id=model_id,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        D_star_hm=D_star_hm,
        D_star_lcdm=D_star_lcdm,
        ratio=ratio,
        deviation_percent=deviation_percent,
        cmb_status=status,
        z_plot=z_plot,
        integrand_hm=integrand_hm,
        integrand_lcdm=integrand_lcdm,
    )


def make_cmb_distance_plot(result: CMBDistanceResult, output_dir: str):
    """Generate diagnostic plot for CMB distance analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if result.z_plot is None:
        return

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (10, 7),
    })

    # Plot: 1/H(z) vs z (integrand for D_*)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(result.z_plot, result.integrand_hm, 'b-', linewidth=2, label='Horizon-memory')
    ax.loglog(result.z_plot, result.integrand_lcdm, 'k--', linewidth=2, label='$\\Lambda$CDM')

    # Mark z_* and z=0
    ax.axvline(Z_STAR, color='red', linestyle=':', alpha=0.7, label=f'$z_* = {Z_STAR:.0f}$')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('$H_0 / H(z)$')
    ax.set_title(f'CMB Distance Integrand: $\\lambda_{{hor}}={result.lambda_hor:.3f}$, $\\tau_{{hor}}={result.tau_hor:.2f}$\n'
                 f'$D_*/D_*^{{\\Lambda CDM}} = {result.ratio:.6f}$ (deviation = {result.deviation_percent:.3f}%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cmb_distance_integrand.png"), dpi=150)
    plt.close()

    print(f"    Plot saved to {output_dir}/")


def save_cmb_distance_results(result: CMBDistanceResult, output_dir: str):
    """Save CMB distance analysis results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "model_id": result.model_id,
        "lambda_hor": float(result.lambda_hor),
        "tau_hor": float(result.tau_hor),
        "z_star": Z_STAR,
        "D_star_horizon_memory": float(result.D_star_hm),
        "D_star_lcdm": float(result.D_star_lcdm),
        "ratio": float(result.ratio),
        "deviation_percent": float(result.deviation_percent),
        "cmb_status": result.cmb_status,
    }

    with open(os.path.join(output_dir, "cmb_distance.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Results saved to {output_dir}/cmb_distance.json")


def main():
    print("="*80)
    print("HORIZON-MEMORY CMB DISTANCE CONSTRAINTS")
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

    print("\n[TASK] Computing CMB distance constraints...")

    for i, model_data in enumerate(models):
        result = analyze_cmb_distance(model_data)
        results.append(result)

        # Save results
        output_dir = f"results/tests/T06_cmbdistance/{result.model_id}"
        save_cmb_distance_results(result, output_dir)

        # Generate plot
        fig_dir = f"figures/tests/T06_cmbdistance/{result.model_id}"
        make_cmb_distance_plot(result, fig_dir)

    # Print summary
    print("\n" + "="*80)
    print("CMB DISTANCE ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nReference: z_* = {Z_STAR}")

    for result in results:
        print(f"\nModel: lambda_hor={result.lambda_hor:.3f}, tau_hor={result.tau_hor:.2f}")
        print(f"  D_*(HM) / D_*(LCDM) = {result.ratio:.6f}")
        print(f"  Deviation: {result.deviation_percent:.4f}%")
        print(f"  CMB status: {result.cmb_status.upper()}")

    # Overall assessment
    passing = [r for r in results if r.cmb_status == "passes"]
    marginal = [r for r in results if r.cmb_status == "marginal"]

    print("\n" + "-"*80)
    if len(passing) > 0:
        print(f"[RESULT] {len(passing)} model(s) PASS CMB distance test (<0.5% deviation)")
    elif len(marginal) > 0:
        print(f"[RESULT] {len(marginal)} model(s) are MARGINAL (0.5-1.0% deviation)")
    else:
        print("[RESULT] All models FAIL CMB distance test (>1% deviation)")

    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
