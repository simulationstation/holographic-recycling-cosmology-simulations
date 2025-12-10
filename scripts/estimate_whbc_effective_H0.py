#!/usr/bin/env python3
"""
SIMULATION 10 - Phase 2: Estimate Effective H0 from WHBC P(k) Modifications

For each PASS model from the WHBC primordial scan, estimate the effective H0
that a standard LCDM analysis would infer when fitting to the WHBC-distorted
CMB spectra.

The key insight: WHBC P(k) modifications change the shape/amplitude of the
CMB power spectrum, but theta_s (the acoustic scale) is unchanged since we
keep the background LCDM. However, parameter degeneracies in the fit can
shift the inferred H0.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.primordial.whbc_primordial import (
    WHBCPrimordialParameters,
    primordial_ratio,
)


# Constants
H0_GRID = np.arange(65.0, 73.2, 0.2)  # H0 grid for chi2 minimization
H0_LCDM = 67.4  # Planck baseline

# LCDM baseline Cl parameters (simplified model)
LCDM_PARAMS = {
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'H0': 67.4,
    'theta_s': 1.04109e-2,  # radians
}


def generate_lcdm_cl(H0: float, ell_max: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate approximate LCDM TT spectrum for a given H0.

    The CMB spectrum depends on H0 primarily through:
    1. The angular diameter distance D_A(z*) which sets theta_s = r_s/D_A
    2. The matter density omega_m = Omega_m * h^2

    For fixed theta_s (which is the CMB observable), higher H0 requires
    different omega_m, omega_b to maintain the same acoustic scale.

    This is a simplified model for rapid scanning.

    Args:
        H0: Hubble constant in km/s/Mpc
        ell_max: Maximum multipole

    Returns:
        (ell, Cl_TT) arrays
    """
    ell = np.arange(2, ell_max + 1)

    # Reference values at H0=67.4
    h = H0 / 100.0
    h_ref = 67.4 / 100.0

    # The acoustic scale theta_s is extremely well measured
    # theta_s ~ r_s / D_A(z*)
    # For fixed theta_s, changing H0 shifts D_A, requiring omega_m changes

    # First peak position scales with theta_s: l_1 ~ pi / theta_s ~ 220
    l_peak1 = 220.0

    # Higher H0 -> smaller D_A -> smaller theta_s -> higher l_peak1
    # But we're keeping theta_s fixed, so l_peak1 stays at 220

    # The main H0 effect comes through:
    # 1. Overall amplitude (through Omega_m h^2 changes)
    # 2. Early ISW effect at low-ell (through Omega_DE)
    # 3. Damping tail shape (through n_s adjustments to fit)

    # Amplitude scaling: sigma8 ~ (omega_m)^0.5 * As^0.5 * (D_+/a)
    # For fixed CMB amplitude at pivot, H0 change requires As change
    amp_ratio = 1.0 + 0.01 * (H0 - 67.4)  # Small amplitude effect

    # Low-ell ISW effect: higher H0 -> earlier DE domination -> more ISW
    isw_boost = 1.0 + 0.005 * (H0 - 67.4)

    # Approximate TT spectrum
    Dl_TT = np.zeros_like(ell, dtype=float)

    # Sachs-Wolfe plateau (low ell)
    low_ell = ell < 30
    Dl_TT[low_ell] = 1000 * isw_boost * (ell[low_ell] / 10.0) ** 0.04

    # Acoustic peaks
    high_ell = ell >= 30
    envelope = 5700 * amp_ratio * np.exp(-(ell[high_ell] - 220)**2 / (150**2))

    for n in range(1, 8):
        peak_pos = l_peak1 * n
        peak_amp = 5000 * amp_ratio / n**1.5
        peak_width = 80 * np.sqrt(n)
        envelope += peak_amp * np.exp(-(ell[high_ell] - peak_pos)**2 / (peak_width**2))

    # Damping tail
    damping = np.exp(-(ell[high_ell] / 1000)**2)
    Dl_TT[high_ell] = envelope * damping

    return ell, Dl_TT


def apply_whbc_to_cl(
    ell: np.ndarray,
    Cl_TT: np.ndarray,
    whbc_params: WHBCPrimordialParameters,
) -> np.ndarray:
    """Apply WHBC P(k) modification to TT spectrum."""
    D_M_star = 14000.0  # Mpc
    k_ell = ell / D_M_star
    F_whbc = primordial_ratio(k_ell, whbc_params)
    return Cl_TT * F_whbc


def compute_chi2(
    Cl_data: np.ndarray,
    Cl_model: np.ndarray,
    ell: np.ndarray,
    ell_min: int = 30,
    ell_max: int = 2000,
) -> float:
    """Compute simplified chi^2 between data and model.

    Uses fractional errors proportional to cosmic variance:
    sigma_ell ~ C_ell / sqrt(2*ell + 1)

    Args:
        Cl_data: "Data" spectrum (WHBC-modified)
        Cl_model: Model spectrum (LCDM at some H0)
        ell: Multipole array
        ell_min: Minimum ell for fit
        ell_max: Maximum ell for fit

    Returns:
        chi^2 value
    """
    mask = (ell >= ell_min) & (ell <= ell_max)

    # Cosmic variance error
    # sigma_l / C_l ~ 1 / sqrt(2*l + 1) for full sky
    # Add 5% systematic floor
    sigma = Cl_data[mask] / np.sqrt(2 * ell[mask] + 1)
    sigma = np.sqrt(sigma**2 + (0.05 * Cl_data[mask])**2)

    diff = Cl_data[mask] - Cl_model[mask]
    chi2 = np.sum((diff / sigma)**2)

    return float(chi2)


def find_best_fit_H0(
    Cl_whbc: np.ndarray,
    ell: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Find best-fit H0 for LCDM fitting to WHBC spectrum.

    Args:
        Cl_whbc: WHBC-modified TT spectrum
        ell: Multipole array

    Returns:
        (H0_best, chi2_min, chi2_array over H0 grid)
    """
    chi2_array = np.zeros_like(H0_GRID)

    for i, H0 in enumerate(H0_GRID):
        _, Cl_model = generate_lcdm_cl(H0, ell_max=len(ell)+1)
        # Match array lengths
        min_len = min(len(Cl_whbc), len(Cl_model))
        chi2_array[i] = compute_chi2(Cl_whbc[:min_len], Cl_model[:min_len], ell[:min_len])

    # Find minimum
    i_min = np.argmin(chi2_array)
    H0_best = H0_GRID[i_min]
    chi2_min = chi2_array[i_min]

    return H0_best, chi2_min, chi2_array


def load_scan_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load scan results from SIMULATION 10 Phase 1."""
    json_path = os.path.join(results_dir, 'scan_results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Scan results not found at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data['results']


def main():
    """Main entry point."""
    print("="*60)
    print("SIMULATION 10 - Phase 2: Effective H0 Estimation")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print()

    # Load scan results
    results_dir = 'results/simulation_10_whbc_pk'
    try:
        scan_results = load_scan_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run scripts/run_whbc_primordial_scan.py first.")
        sys.exit(1)

    # Filter to PASS models
    pass_results = [r for r in scan_results if r.get('success', False) and r.get('passes_all', False)]

    print(f"Total scan results: {len(scan_results)}")
    print(f"PASS models: {len(pass_results)}")
    print()

    if len(pass_results) == 0:
        print("No PASS models to analyze. Exiting.")
        sys.exit(0)

    # Generate baseline LCDM spectrum
    ell_lcdm, Cl_TT_lcdm = generate_lcdm_cl(H0_LCDM)

    # Analyze each PASS model
    h0_results = []

    print("Estimating effective H0 for PASS models...")
    for i, r in enumerate(pass_results):
        params = r['params']
        # Need to use the fixed params from the scan
        full_params = {
            'A_cut': params.get('A_cut', 0.0),
            'k_cut': params.get('k_cut', 0.001),
            'p_cut': 2.0,
            'A_osc': params.get('A_osc', 0.0),
            'omega_WH': params.get('omega_WH', 0.0),
            'phi_WH': 0.0,
            'k_damp': 0.1,
            'k_pivot': 0.05,
            'As': 2.1e-9,
            'ns': 0.9649,
        }

        try:
            whbc_params = WHBCPrimordialParameters(**full_params)
        except Exception as e:
            print(f"  [{i+1}/{len(pass_results)}] Error creating params: {e}")
            continue

        # Generate WHBC-modified spectrum
        Cl_TT_whbc = apply_whbc_to_cl(ell_lcdm, Cl_TT_lcdm, whbc_params)

        # Find best-fit H0
        H0_best, chi2_min, chi2_array = find_best_fit_H0(Cl_TT_whbc, ell_lcdm)
        delta_H0 = H0_best - H0_LCDM

        h0_result = {
            'params': params,
            'H0_eff': float(H0_best),
            'delta_H0_eff': float(delta_H0),
            'chi2_min': float(chi2_min),
            'sigma8_ratio': r.get('sigma8_ratio', 1.0),
            'TT_RMS': r.get('TT_RMS_30_2000', 0.0),
        }
        h0_results.append(h0_result)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(pass_results)}] A_cut={params['A_cut']:.2f}, "
                  f"H0_eff={H0_best:.1f}, delta_H0={delta_H0:+.2f}")

    # Save results
    output = {
        'simulation': 'T10_WHBC_PRIMORDIAL - Effective H0 Estimation',
        'date': datetime.now().isoformat(),
        'H0_LCDM_baseline': H0_LCDM,
        'H0_grid': list(H0_GRID),
        'n_pass_models': len(pass_results),
        'n_analyzed': len(h0_results),
        'results': h0_results,
    }

    # Compute summary statistics
    if len(h0_results) > 0:
        delta_H0_vals = [r['delta_H0_eff'] for r in h0_results]
        abs_delta_H0 = [abs(d) for d in delta_H0_vals]

        output['summary'] = {
            'max_abs_delta_H0': float(max(abs_delta_H0)),
            'min_delta_H0': float(min(delta_H0_vals)),
            'max_delta_H0': float(max(delta_H0_vals)),
            'mean_abs_delta_H0': float(np.mean(abs_delta_H0)),
            'median_abs_delta_H0': float(np.median(abs_delta_H0)),
        }

    with open(os.path.join(results_dir, 'h0_effective.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("EFFECTIVE H0 ESTIMATION SUMMARY")
    print("="*60)
    print(f"PASS models analyzed: {len(h0_results)}")

    if len(h0_results) > 0:
        print(f"\nMax |delta_H0_eff|: {max(abs_delta_H0):.2f} km/s/Mpc")
        print(f"Mean |delta_H0_eff|: {np.mean(abs_delta_H0):.2f} km/s/Mpc")
        print(f"Median |delta_H0_eff|: {np.median(abs_delta_H0):.2f} km/s/Mpc")

        # Best model
        best_idx = np.argmax(abs_delta_H0)
        best = h0_results[best_idx]
        print(f"\nBest model (max |delta_H0|):")
        print(f"  A_cut = {best['params']['A_cut']}")
        print(f"  k_cut = {best['params']['k_cut']:.0e}")
        print(f"  A_osc = {best['params']['A_osc']}")
        print(f"  omega_WH = {best['params']['omega_WH']}")
        print(f"  H0_eff = {best['H0_eff']:.1f} km/s/Mpc")
        print(f"  delta_H0 = {best['delta_H0_eff']:+.2f} km/s/Mpc")

    print("="*60)
    print(f"\nResults saved to {results_dir}/h0_effective.json")


if __name__ == '__main__':
    main()
