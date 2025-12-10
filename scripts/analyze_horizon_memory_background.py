#!/usr/bin/env python3
"""Horizon-memory background cosmology analysis.

This script provides detailed analysis of promising horizon-memory parameter points,
computing H(z), distances, and effective equation of state.

Usage:
    python scripts/analyze_horizon_memory_background.py [--lambda_hor LAMBDA] [--tau_hor TAU]
    python scripts/analyze_horizon_memory_background.py --from-scan  # Use best from T06 scan

Outputs:
    - results/horizon_memory_analysis/background_Hz.json: H(z) data
    - results/horizon_memory_analysis/distances.json: D_L(z), D_A(z), chi(z)
    - figures/horizon_memory_analysis/Hz_comparison.png: H(z) plot
    - figures/horizon_memory_analysis/distances.png: Distance measures plot
    - figures/horizon_memory_analysis/w_hor.png: Effective equation of state
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, quad

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.theory import HRC2Parameters, CouplingFamily, PotentialType
from hrc2.background import BackgroundCosmology


def integrate_memory_field(
    cosmo: BackgroundCosmology,
    a_start: float = 1e-6,
    a_end: float = 1.0,
) -> callable:
    """Integrate the memory field ODE and return interpolator.

    Args:
        cosmo: BackgroundCosmology instance with lambda_hor, tau_hor set
        a_start: Initial scale factor
        a_end: Final scale factor

    Returns:
        Callable M(ln_a) returning memory field at given ln(a)
    """
    ln_a_start = np.log(a_start)
    ln_a_end = np.log(a_end)

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
        [0.0],  # M(a_start) = 0
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Memory field integration failed: {sol.message}")

    return sol.sol


def compute_Hz_data(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    z_array: np.ndarray,
) -> dict:
    """Compute H(z) for both horizon-memory and baseline GR.

    Args:
        cosmo: BackgroundCosmology with M_today set
        M_interp: Memory field interpolator
        z_array: Redshift array

    Returns:
        Dictionary with H(z) data
    """
    H_hm = np.zeros_like(z_array)
    H_gr = np.zeros_like(z_array)
    M_z = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)
        M = M_interp(ln_a)[0]
        M_z[i] = M
        H_hm[i] = cosmo.H_of_a_selfconsistent(a, M)
        H_gr[i] = cosmo.H_of_a_gr_baseline(a)

    return {
        "z": z_array.tolist(),
        "H_horizon_memory": H_hm.tolist(),
        "H_baseline_GR": H_gr.tolist(),
        "M_z": M_z.tolist(),
        "H_ratio": (H_hm / H_gr).tolist(),
    }


def compute_distances(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    z_array: np.ndarray,
) -> dict:
    """Compute distance measures: comoving chi(z), D_L(z), D_A(z).

    Args:
        cosmo: BackgroundCosmology with M_today set
        M_interp: Memory field interpolator
        z_array: Redshift array

    Returns:
        Dictionary with distance data
    """
    c = 299792.458  # km/s
    H0 = cosmo.H0  # in units of H0=1, so H0 = 1 here, we'll scale later

    # Comoving distance chi(z) = c * integral_0^z dz' / H(z')
    chi_hm = np.zeros_like(z_array)
    chi_gr = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        if z == 0:
            chi_hm[i] = 0.0
            chi_gr[i] = 0.0
            continue

        # Integrate 1/H(z') from 0 to z
        def integrand_hm(zp):
            a = 1.0 / (1.0 + zp)
            ln_a = np.log(a)
            M = M_interp(ln_a)[0]
            H = cosmo.H_of_a_selfconsistent(a, M)
            return 1.0 / H if H > 0 else 0.0

        def integrand_gr(zp):
            a = 1.0 / (1.0 + zp)
            H = cosmo.H_of_a_gr_baseline(a)
            return 1.0 / H if H > 0 else 0.0

        chi_hm[i], _ = quad(integrand_hm, 0, z, limit=100)
        chi_gr[i], _ = quad(integrand_gr, 0, z, limit=100)

    # chi is in units of c/H0 [Mpc if we use H0 in km/s/Mpc]
    # D_L = (1+z) * chi, D_A = chi / (1+z)
    D_L_hm = (1.0 + z_array) * chi_hm
    D_L_gr = (1.0 + z_array) * chi_gr
    D_A_hm = chi_hm / (1.0 + z_array)
    D_A_gr = chi_gr / (1.0 + z_array)

    return {
        "z": z_array.tolist(),
        "chi_horizon_memory": chi_hm.tolist(),
        "chi_baseline_GR": chi_gr.tolist(),
        "D_L_horizon_memory": D_L_hm.tolist(),
        "D_L_baseline_GR": D_L_gr.tolist(),
        "D_A_horizon_memory": D_A_hm.tolist(),
        "D_A_baseline_GR": D_A_gr.tolist(),
        "D_L_ratio": (D_L_hm / D_L_gr).tolist() if np.all(D_L_gr > 0) else [],
        "chi_ratio": (chi_hm / chi_gr).tolist() if np.all(chi_gr > 0) else [],
    }


def compute_w_hor(
    cosmo: BackgroundCosmology,
    M_interp: callable,
    z_array: np.ndarray,
) -> dict:
    """Compute effective equation of state w_hor(z) for horizon-memory component.

    w_hor = -1 - (1/3) * d ln(rho_hor) / d ln(a)

    Args:
        cosmo: BackgroundCosmology with M_today set
        M_interp: Memory field interpolator
        z_array: Redshift array

    Returns:
        Dictionary with w_hor data
    """
    w_hor = np.zeros_like(z_array)
    rho_hor = np.zeros_like(z_array)
    Omega_hor = np.zeros_like(z_array)

    eps = 1e-4  # Small step for numerical derivative

    for i, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        ln_a = np.log(a)

        # Get M and rho_hor at this redshift
        M = M_interp(ln_a)[0]
        rho = cosmo.lambda_hor * M
        rho_hor[i] = rho

        # Total density for Omega_hor
        H = cosmo.H_of_a_selfconsistent(a, M)
        rho_tot = (H / cosmo.H0)**2 if cosmo.H0 > 0 else 1.0
        Omega_hor[i] = rho / rho_tot if rho_tot > 0 else 0.0

        # Numerical derivative d ln(rho_hor) / d ln(a)
        if rho > 0:
            ln_a_plus = ln_a + eps
            ln_a_minus = ln_a - eps
            M_plus = M_interp(min(ln_a_plus, 0.0))[0]
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

    return {
        "z": z_array.tolist(),
        "w_hor": w_hor.tolist(),
        "rho_hor": rho_hor.tolist(),
        "Omega_hor": Omega_hor.tolist(),
    }


def make_plots(
    Hz_data: dict,
    dist_data: dict,
    w_data: dict,
    params_info: dict,
    fig_dir: str,
):
    """Create analysis plots.

    Args:
        Hz_data: H(z) data dictionary
        dist_data: Distance data dictionary
        w_data: w_hor data dictionary
        params_info: Parameter information
        fig_dir: Output directory for figures
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return

    os.makedirs(fig_dir, exist_ok=True)

    # 1. H(z) comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    z = np.array(Hz_data["z"])
    H_hm = np.array(Hz_data["H_horizon_memory"])
    H_gr = np.array(Hz_data["H_baseline_GR"])

    ax1 = axes[0]
    ax1.plot(z, H_hm, 'b-', label='Horizon-memory', linewidth=2)
    ax1.plot(z, H_gr, 'k--', label='Baseline GR', linewidth=2)
    ax1.set_ylabel('H(z) / H0')
    ax1.legend()
    ax1.set_title(f'H(z) comparison: lambda_hor={params_info["lambda_hor"]:.3f}, tau_hor={params_info["tau_hor"]:.2f}')
    ax1.set_xlim(0, max(z))
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    H_ratio = np.array(Hz_data["H_ratio"])
    ax2.plot(z, (H_ratio - 1) * 100, 'r-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('(H_hm / H_GR - 1) x 100 [%]')
    ax2.set_xlim(0, max(z))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Hz_comparison.png"), dpi=150)
    plt.close()

    # 2. Distance measures plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    z_dist = np.array(dist_data["z"])
    D_L_hm = np.array(dist_data["D_L_horizon_memory"])
    D_L_gr = np.array(dist_data["D_L_baseline_GR"])

    ax1 = axes[0]
    ax1.plot(z_dist, D_L_hm, 'b-', label='D_L horizon-memory', linewidth=2)
    ax1.plot(z_dist, D_L_gr, 'k--', label='D_L baseline GR', linewidth=2)
    ax1.set_ylabel('D_L(z) [c/H0]')
    ax1.legend()
    ax1.set_title('Luminosity distance')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if len(dist_data["D_L_ratio"]) > 0:
        D_L_ratio = np.array(dist_data["D_L_ratio"])
        ax2.plot(z_dist, (D_L_ratio - 1) * 100, 'r-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('(D_L_hm / D_L_GR - 1) x 100 [%]')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "distances.png"), dpi=150)
    plt.close()

    # 3. Effective equation of state plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    z_w = np.array(w_data["z"])
    w_hor = np.array(w_data["w_hor"])
    Omega_hor = np.array(w_data["Omega_hor"])

    ax1 = axes[0]
    valid = ~np.isnan(w_hor)
    ax1.plot(z_w[valid], w_hor[valid], 'b-', linewidth=2)
    ax1.axhline(-1, color='k', linestyle='--', alpha=0.5, label='w = -1 (Lambda)')
    ax1.set_ylabel('w_hor(z)')
    ax1.set_title('Effective equation of state for horizon-memory component')
    ax1.legend()
    ax1.set_ylim(-2, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(z_w, Omega_hor, 'g-', linewidth=2)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Omega_hor(z)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "w_hor.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {fig_dir}/")


def load_best_from_scan(results_dir: str = "results/tests/T06_horizon_memory_nonlocal") -> tuple:
    """Load best parameters from T06 scan results.

    Returns:
        (lambda_hor, tau_hor) tuple
    """
    status_path = os.path.join(results_dir, "status.json")
    if not os.path.exists(status_path):
        raise FileNotFoundError(f"T06 scan results not found at {status_path}")

    with open(status_path, "r") as f:
        status = json.load(f)

    lambda_hor = status.get("best_lambda_hor")
    tau_hor = status.get("best_tau_hor")

    if lambda_hor is None or tau_hor is None:
        raise ValueError("No best parameters found in T06 scan")

    return lambda_hor, tau_hor


def main():
    parser = argparse.ArgumentParser(description="Horizon-memory background analysis")
    parser.add_argument("--lambda_hor", type=float, default=None, help="Horizon-memory amplitude")
    parser.add_argument("--tau_hor", type=float, default=None, help="Memory timescale in ln(a)")
    parser.add_argument("--from-scan", action="store_true", help="Use best params from T06 scan")
    parser.add_argument("--z-max", type=float, default=3.0, help="Maximum redshift for analysis")
    parser.add_argument("--n-points", type=int, default=200, help="Number of redshift points")
    args = parser.parse_args()

    # Determine parameters
    if args.from_scan:
        lambda_hor, tau_hor = load_best_from_scan()
        print(f"Using best parameters from T06 scan: lambda_hor={lambda_hor:.4f}, tau_hor={tau_hor:.3f}")
    elif args.lambda_hor is not None and args.tau_hor is not None:
        lambda_hor = args.lambda_hor
        tau_hor = args.tau_hor
    else:
        # Default values for demonstration
        lambda_hor = 0.1
        tau_hor = 1.0
        print(f"Using default parameters: lambda_hor={lambda_hor}, tau_hor={tau_hor}")

    # Create output directories
    results_dir = "results/horizon_memory_analysis"
    fig_dir = "figures/horizon_memory_analysis"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

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

    print(f"\nParameters:")
    print(f"  lambda_hor = {lambda_hor:.4f}")
    print(f"  tau_hor = {tau_hor:.3f}")
    print(f"  Omega_m0 = {cosmo.Omega_m0:.4f}")
    print(f"  Omega_r0 = {cosmo.Omega_r0:.6f}")
    print(f"  Omega_L0_base = {cosmo.Omega_L0_base:.4f}")

    # Integrate memory field
    print("\nIntegrating memory field...")
    M_interp = integrate_memory_field(cosmo)
    M_today = M_interp(0.0)[0]
    print(f"  M(z=0) = {M_today:.6f}")

    # Set up self-consistent Lambda
    cosmo.set_M_today(M_today)
    print(f"  Omega_hor0 = {cosmo.Omega_hor0:.6f}")
    print(f"  Omega_L0_eff = {cosmo.Omega_L0_eff:.4f}")

    # Compute delta_H0
    delta_H0_result = cosmo.compute_delta_H0(lambda ln_a: M_interp(ln_a))
    print(f"\nH0 shift:")
    print(f"  delta_H0/H0 = {delta_H0_result['delta_H0_frac']*100:.2f}%")
    print(f"  delta_H0 = {delta_H0_result['delta_H0_kmsMpc']:.2f} km/s/Mpc")

    # Create redshift array
    z_array = np.linspace(0, args.z_max, args.n_points)

    # Compute H(z)
    print("\nComputing H(z)...")
    Hz_data = compute_Hz_data(cosmo, M_interp, z_array)

    # Compute distances
    print("Computing distances...")
    dist_data = compute_distances(cosmo, M_interp, z_array)

    # Compute w_hor
    print("Computing w_hor(z)...")
    w_data = compute_w_hor(cosmo, M_interp, z_array)

    # Parameter info for output
    params_info = {
        "lambda_hor": lambda_hor,
        "tau_hor": tau_hor,
        "Omega_m0": cosmo.Omega_m0,
        "Omega_r0": cosmo.Omega_r0,
        "Omega_L0_base": cosmo.Omega_L0_base,
        "Omega_L0_eff": cosmo.Omega_L0_eff,
        "Omega_hor0": cosmo.Omega_hor0,
        "M_today": M_today,
        "delta_H0_frac": delta_H0_result["delta_H0_frac"],
        "delta_H0_kmsMpc": delta_H0_result["delta_H0_kmsMpc"],
    }

    # Save results
    with open(os.path.join(results_dir, "background_Hz.json"), "w") as f:
        json.dump({"params": params_info, "data": Hz_data}, f, indent=2)

    with open(os.path.join(results_dir, "distances.json"), "w") as f:
        json.dump({"params": params_info, "data": dist_data}, f, indent=2)

    with open(os.path.join(results_dir, "w_hor.json"), "w") as f:
        json.dump({"params": params_info, "data": w_data}, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    # Make plots
    make_plots(Hz_data, dist_data, w_data, params_info, fig_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
