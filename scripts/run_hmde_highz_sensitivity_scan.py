#!/usr/bin/env python3
"""TEST 2: High-redshift sensitivity scan for HMDE.

This script quantifies what percentage change in H(z) at various high-z epochs
(z = 10, 100, 500, 1000, 1100) is possible while maintaining:
- < 0.1% change in theta_s (CMB angular scale)
- < 2% change in D_L at z = 0.5 (SNe)
- < 1% change in D_M/r_d at z = 0.5 (BAO)

The goal is to identify if there's ANY allowed parameter space where HMDE
can produce observable effects at high-z without violating low-z constraints.
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.horizon_models.refinement_d import create_dynamical_eos_model


# Physical constants
C_KM_S = 299792.458  # km/s

# Constraint thresholds
THETA_S_MAX_DEV = 0.001  # 0.1% max theta_s deviation
D_L_MAX_DEV = 0.02       # 2% max D_L deviation at z=0.5
BAO_MAX_DEV = 0.01       # 1% max D_M/r_d deviation

# Fiducial Planck cosmology
H0_FID = 67.4
OMEGA_M_FID = 0.315
OMEGA_R_FID = 9e-5

# High-z evaluation points
Z_EVAL = [0.5, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1089.0]


@dataclass
class HighZScanResult:
    """Result from high-z sensitivity scan."""
    delta_w: float
    a_w: float
    lambda_hor: float

    # Constraint metrics
    theta_s_dev: float      # Fractional deviation in theta_s
    D_L_z05_dev: float      # Fractional deviation in D_L(z=0.5)
    bao_z05_dev: float      # Fractional deviation in D_M/r_d(z=0.5)

    # High-z H(z) deviations (percent)
    H_deviations: Dict[float, float]  # z -> delta_H/H in percent

    # Pass constraints?
    passes_theta_s: bool
    passes_D_L: bool
    passes_bao: bool
    passes_all: bool


def compute_lcdm_reference(H0: float = H0_FID, Omega_m: float = OMEGA_M_FID):
    """Compute LCDM reference quantities."""
    Omega_L = 1 - Omega_m - OMEGA_R_FID

    def H_lcdm(z):
        return H0 * np.sqrt(Omega_m * (1+z)**3 + OMEGA_R_FID * (1+z)**4 + Omega_L)

    # Compute reference distances
    z_star = 1089.0
    z_drag = 1060.0
    c_s = C_KM_S / np.sqrt(3)

    # Sound horizon at drag
    r_drag, _ = quad(lambda z: c_s / H_lcdm(z), z_drag, 3000, limit=500)

    # Sound horizon at recombination
    r_star, _ = quad(lambda z: c_s / H_lcdm(z), z_star, 3000, limit=500)

    # Angular diameter distance to last scattering
    D_M_star, _ = quad(lambda z: C_KM_S / H_lcdm(z), 0, z_star, limit=500)
    D_A_star = D_M_star / (1 + z_star)

    # theta_s
    theta_s = r_star / D_A_star

    # D_L at z=0.5
    D_M_05, _ = quad(lambda z: C_KM_S / H_lcdm(z), 0, 0.5, limit=100)
    D_L_05 = D_M_05 * 1.5

    # D_M/r_d at z=0.5
    bao_05 = D_M_05 / r_drag

    # H at all evaluation z
    H_z = {z: H_lcdm(z) for z in Z_EVAL}

    return {
        'H0': H0,
        'theta_s': theta_s,
        'r_drag': r_drag,
        'r_star': r_star,
        'D_L_05': D_L_05,
        'bao_05': bao_05,
        'H_z': H_z,
    }


def evaluate_hmde_point(
    delta_w: float,
    a_w: float,
    lambda_hor: float,
    lcdm_ref: dict,
    tau_hor: float = 0.1,
    m_eos: float = 2.0,
) -> Optional[HighZScanResult]:
    """Evaluate HMDE at a single parameter point."""
    try:
        # Create HMDE model
        model = create_dynamical_eos_model(
            delta_w=delta_w,
            a_w=a_w,
            m_eos=m_eos,
            lambda_hor=lambda_hor,
            tau_hor=tau_hor,
            Omega_m0=OMEGA_M_FID,
            Omega_r0=OMEGA_R_FID,
            H0=H0_FID,
        )

        # Solve background
        result = model.solve(z_max=1200.0, n_points=2000)

        if not result.success:
            return None

        # Build interpolator
        z_arr = result.z
        H_arr = result.H

        def H_interp(z):
            return np.interp(z, z_arr[::-1], H_arr[::-1])

        # Compute HMDE quantities
        z_star = 1089.0
        z_drag = 1060.0
        c_s = C_KM_S / np.sqrt(3)

        # Sound horizon at drag
        r_drag, _ = quad(lambda z: c_s / H_interp(z), z_drag, 1200, limit=500)

        # Sound horizon at recombination
        r_star, _ = quad(lambda z: c_s / H_interp(z), z_star, 1200, limit=500)

        # Angular diameter distance to last scattering
        D_M_star, _ = quad(lambda z: C_KM_S / H_interp(z), 0, z_star, limit=500)
        D_A_star = D_M_star / (1 + z_star)

        # theta_s
        theta_s = r_star / D_A_star
        theta_s_dev = abs(theta_s / lcdm_ref['theta_s'] - 1)

        # D_L at z=0.5
        D_M_05, _ = quad(lambda z: C_KM_S / H_interp(z), 0, 0.5, limit=100)
        D_L_05 = D_M_05 * 1.5
        D_L_dev = abs(D_L_05 / lcdm_ref['D_L_05'] - 1)

        # D_M/r_d at z=0.5
        bao_05 = D_M_05 / r_drag
        bao_dev = abs(bao_05 / lcdm_ref['bao_05'] - 1)

        # H deviations at all z
        H_deviations = {}
        for z in Z_EVAL:
            H_hmde = H_interp(z)
            H_lcdm = lcdm_ref['H_z'][z]
            H_deviations[z] = (H_hmde / H_lcdm - 1) * 100  # Percent

        return HighZScanResult(
            delta_w=delta_w,
            a_w=a_w,
            lambda_hor=lambda_hor,
            theta_s_dev=theta_s_dev,
            D_L_z05_dev=D_L_dev,
            bao_z05_dev=bao_dev,
            H_deviations=H_deviations,
            passes_theta_s=theta_s_dev < THETA_S_MAX_DEV,
            passes_D_L=D_L_dev < D_L_MAX_DEV,
            passes_bao=bao_dev < BAO_MAX_DEV,
            passes_all=(theta_s_dev < THETA_S_MAX_DEV and
                       D_L_dev < D_L_MAX_DEV and
                       bao_dev < BAO_MAX_DEV),
        )

    except Exception as e:
        return None


def run_sensitivity_scan(
    delta_w_range: Tuple[float, float] = (-0.5, 0.1),
    a_w_range: Tuple[float, float] = (0.1, 0.6),
    lambda_hor_range: Tuple[float, float] = (0.01, 0.4),
    n_delta_w: int = 25,
    n_a_w: int = 25,
    n_lambda_hor: int = 15,
    output_dir: str = "results/test2_highz_sensitivity",
):
    """Run 3D parameter scan for high-z sensitivity.

    Args:
        delta_w_range: (min, max) for delta_w
        a_w_range: (min, max) for a_w
        lambda_hor_range: (min, max) for lambda_hor
        n_delta_w: Grid size for delta_w
        n_a_w: Grid size for a_w
        n_lambda_hor: Grid size for lambda_hor
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEST 2: High-redshift Sensitivity Scan")
    print("=" * 60)
    print(f"\nParameter ranges:")
    print(f"  delta_w: [{delta_w_range[0]}, {delta_w_range[1]}], n={n_delta_w}")
    print(f"  a_w: [{a_w_range[0]}, {a_w_range[1]}], n={n_a_w}")
    print(f"  lambda_hor: [{lambda_hor_range[0]}, {lambda_hor_range[1]}], n={n_lambda_hor}")
    print(f"\nTotal points: {n_delta_w * n_a_w * n_lambda_hor}")
    print()

    # Compute LCDM reference
    print("Computing LCDM reference...")
    lcdm_ref = compute_lcdm_reference()
    print(f"  LCDM theta_s = {lcdm_ref['theta_s']:.6e}")
    print(f"  LCDM r_drag = {lcdm_ref['r_drag']:.2f} Mpc")
    print(f"  LCDM D_L(z=0.5) = {lcdm_ref['D_L_05']:.2f} Mpc")
    print()

    # Create grids
    delta_w_vals = np.linspace(delta_w_range[0], delta_w_range[1], n_delta_w)
    a_w_vals = np.linspace(a_w_range[0], a_w_range[1], n_a_w)
    lambda_hor_vals = np.linspace(lambda_hor_range[0], lambda_hor_range[1], n_lambda_hor)

    # Store all results
    all_results = []
    passing_results = []

    # Track max H deviations at each z for points that pass constraints
    max_H_dev_passing = {z: 0.0 for z in Z_EVAL}

    start_time = time.time()
    n_total = n_delta_w * n_a_w * n_lambda_hor
    n_done = 0
    n_passed = 0

    print("Running scan...")

    for i, delta_w in enumerate(delta_w_vals):
        for j, a_w in enumerate(a_w_vals):
            for k, lambda_hor in enumerate(lambda_hor_vals):
                n_done += 1

                if n_done % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = n_done / elapsed
                    remaining = (n_total - n_done) / rate
                    print(f"  Progress: {n_done}/{n_total} ({100*n_done/n_total:.1f}%), "
                          f"passed: {n_passed}, "
                          f"ETA: {remaining:.0f}s")

                res = evaluate_hmde_point(delta_w, a_w, lambda_hor, lcdm_ref)

                if res is None:
                    continue

                result_dict = {
                    'delta_w': delta_w,
                    'a_w': a_w,
                    'lambda_hor': lambda_hor,
                    'theta_s_dev': res.theta_s_dev,
                    'D_L_z05_dev': res.D_L_z05_dev,
                    'bao_z05_dev': res.bao_z05_dev,
                    'H_deviations': res.H_deviations,
                    'passes_all': res.passes_all,
                }

                all_results.append(result_dict)

                if res.passes_all:
                    n_passed += 1
                    passing_results.append(result_dict)

                    # Track max H deviations
                    for z, dev in res.H_deviations.items():
                        if abs(dev) > abs(max_H_dev_passing[z]):
                            max_H_dev_passing[z] = dev

    elapsed_total = time.time() - start_time
    print(f"\nScan completed in {elapsed_total:.1f} seconds")
    print(f"Total evaluated: {len(all_results)}")
    print(f"Passed all constraints: {n_passed}")

    # Analysis
    print("\n" + "=" * 60)
    print("HIGH-Z SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\nMaximum |delta H(z)/H(z)| (%) for points passing all constraints:")
    print("-" * 50)
    for z in Z_EVAL:
        print(f"  z = {z:7.1f}: {max_H_dev_passing[z]:+.4f}%")

    # Find the parameter point with maximum high-z effect while passing
    if passing_results:
        # Sort by max high-z deviation
        def max_highz_dev(r):
            return max(abs(r['H_deviations'].get(z, 0)) for z in [100, 500, 1000, 1089])

        best_highz = max(passing_results, key=max_highz_dev)

        print(f"\nBest high-z effect while passing constraints:")
        print(f"  delta_w = {best_highz['delta_w']:.4f}")
        print(f"  a_w = {best_highz['a_w']:.4f}")
        print(f"  lambda_hor = {best_highz['lambda_hor']:.4f}")
        print(f"  theta_s deviation = {best_highz['theta_s_dev']*100:.4f}%")
        print(f"  D_L(z=0.5) deviation = {best_highz['D_L_z05_dev']*100:.4f}%")
        print(f"\n  H(z) deviations:")
        for z in Z_EVAL:
            print(f"    z = {z:7.1f}: {best_highz['H_deviations'][z]:+.4f}%")

    # Save results
    output = {
        'config': {
            'delta_w_range': delta_w_range,
            'a_w_range': a_w_range,
            'lambda_hor_range': lambda_hor_range,
            'n_delta_w': n_delta_w,
            'n_a_w': n_a_w,
            'n_lambda_hor': n_lambda_hor,
            'constraint_thresholds': {
                'theta_s_max_dev': THETA_S_MAX_DEV,
                'D_L_max_dev': D_L_MAX_DEV,
                'bao_max_dev': BAO_MAX_DEV,
            }
        },
        'lcdm_reference': {
            'H0': lcdm_ref['H0'],
            'theta_s': lcdm_ref['theta_s'],
            'r_drag': lcdm_ref['r_drag'],
            'D_L_05': lcdm_ref['D_L_05'],
            'bao_05': lcdm_ref['bao_05'],
        },
        'statistics': {
            'n_evaluated': len(all_results),
            'n_passed': n_passed,
            'pass_fraction': n_passed / len(all_results) if all_results else 0,
        },
        'max_H_deviations_passing': {str(z): v for z, v in max_H_dev_passing.items()},
        'best_highz_point': best_highz if passing_results else None,
        'passing_results': passing_results,
    }

    with open(output_path / 'sensitivity_scan_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save all results as numpy arrays for further analysis
    np.savez(
        output_path / 'scan_data.npz',
        delta_w_vals=delta_w_vals,
        a_w_vals=a_w_vals,
        lambda_hor_vals=lambda_hor_vals,
    )

    print(f"\nResults saved to {output_path}")

    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)

    max_recomb_dev = abs(max_H_dev_passing.get(1089.0, 0))
    max_z100_dev = abs(max_H_dev_passing.get(100.0, 0))

    if max_recomb_dev < 0.01:
        print(f"Maximum H(z=1089) deviation: {max_recomb_dev:.4f}%")
        print("=> HMDE has NEGLIGIBLE impact at recombination (< 0.01%)")
        print("=> Cannot detectably alter CMB physics while satisfying constraints")
    elif max_recomb_dev < 0.1:
        print(f"Maximum H(z=1089) deviation: {max_recomb_dev:.4f}%")
        print("=> HMDE has SMALL but non-zero impact at recombination")
    else:
        print(f"Maximum H(z=1089) deviation: {max_recomb_dev:.4f}%")
        print("=> HMDE can have SIGNIFICANT high-z impact!")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TEST 2: High-z Sensitivity Scan")
    parser.add_argument("--n-delta-w", type=int, default=25)
    parser.add_argument("--n-a-w", type=int, default=25)
    parser.add_argument("--n-lambda-hor", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="results/test2_highz_sensitivity")

    args = parser.parse_args()

    run_sensitivity_scan(
        n_delta_w=args.n_delta_w,
        n_a_w=args.n_a_w,
        n_lambda_hor=args.n_lambda_hor,
        output_dir=args.output_dir,
    )
