#!/usr/bin/env python3
"""
SIMULATION 10: Primordial WHBC-like Power Spectrum Scan

Scans WHBC-inspired primordial P(k) modifications on fixed LCDM background.
All changes come from P(k) only - no background modifications.

Tests how much CMB TT/EE can be modified while remaining compatible with
observations, and estimates the effective H0 drift when fitting LCDM.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrc2.primordial.whbc_primordial import (
    WHBCPrimordialParameters,
    primordial_ratio,
    primordial_PK_whbc,
    primordial_PK_lcdm,
    compute_sigma8_ratio,
)
from hrc2.primordial.class_interface import (
    approximate_cmb_effects,
    run_boltzmann_with_whbc,
    HAS_CLASS,
    HAS_CAMB,
    PLANCK_COSMO,
)


# =============================================================================
# SIMULATION 10 Parameter Grid
# =============================================================================

# WHBC primordial P(k) modifications
PARAM_GRID = {
    'A_cut': [0.0, -0.05, -0.10],           # IR suppression amplitude
    'k_cut': [1e-4, 5e-4, 1e-3],            # IR cutoff scale [Mpc^-1]
    'A_osc': [0.0, 0.02],                   # Oscillation amplitude
    'omega_WH': [0.0, 10.0],                # Oscillation frequency in log(k)
}

# Fixed parameters
FIXED_PARAMS = {
    'p_cut': 2.0,        # Cutoff power
    'phi_WH': 0.0,       # Oscillation phase
    'k_damp': 0.1,       # UV damping scale
    'k_pivot': 0.05,     # Pivot scale
    'As': 2.1e-9,        # LCDM amplitude
    'ns': 0.9649,        # LCDM spectral index
}

# Viability criteria
VIABILITY_CRITERIA = {
    'TT_RMS_max': 0.05,           # 5% max TT RMS residual (ell=30-2000)
    'EE_RMS_max': 0.10,           # 10% max EE RMS residual
    'delta_sigma8_max': 0.05,     # 5% max sigma8 shift
    'delta_S8_max': 0.05,         # 5% max S8 shift
}

# LCDM baseline values (Planck 2018)
LCDM_BASELINE = {
    'sigma8': 0.811,
    'S8': 0.832,
    'theta_s': 1.04109,
    'H0': 67.4,
}


@dataclass
class WHBCScanResult:
    """Result from evaluating a single WHBC P(k) model."""
    params: Dict[str, float]
    success: bool

    # Primordial spectrum properties
    F_ratio_pivot: float = 1.0
    F_ratio_low_k: float = 1.0  # At k=1e-4
    F_ratio_high_k: float = 1.0  # At k=0.1

    # CMB effects
    TT_RMS_30_2000: float = 0.0
    EE_RMS_30_2000: float = 0.0
    low_ell_TT_ratio: float = 1.0  # Average Cl ratio for ell<30

    # Derived cosmology
    sigma8: float = 0.0
    sigma8_ratio: float = 1.0
    S8: float = 0.0
    S8_ratio: float = 1.0

    # Viability
    passes_TT: bool = False
    passes_EE: bool = False
    passes_sigma8: bool = False
    passes_S8: bool = False
    passes_all: bool = False

    # Approximate H0 effect
    delta_H0_approx: float = 0.0

    error: str = ""


def generate_lcdm_cl_baseline() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate LCDM baseline CMB spectra for comparison.

    Returns:
        (ell, Cl_TT, Cl_EE) arrays
    """
    # Use approximate formula for rapid evaluation
    # D_ell = ell(ell+1)/(2pi) * C_ell in microK^2

    ell = np.arange(2, 2501)

    # Approximate LCDM TT spectrum (simplified model)
    # First peak at ell ~ 220, subsequent peaks
    theta_s = 1.04109e-2  # rad
    l_A = np.pi / theta_s  # ~ 302

    # Sachs-Wolfe plateau + acoustic peaks
    Dl_TT = np.zeros_like(ell, dtype=float)

    # Sachs-Wolfe at low ell
    low_ell = ell < 30
    Dl_TT[low_ell] = 1000 * (ell[low_ell] / 10.0) ** 0.04

    # Acoustic peaks approximation
    high_ell = ell >= 30
    # Main envelope
    envelope = 5700 * np.exp(-(ell[high_ell] - 220)**2 / (150**2))
    # Add peaks
    for n in range(1, 8):
        peak_pos = 220 * n
        peak_amp = 5000 / n**1.5
        peak_width = 80 * np.sqrt(n)
        envelope += peak_amp * np.exp(-(ell[high_ell] - peak_pos)**2 / (peak_width**2))

    # Damping tail
    damping = np.exp(-(ell[high_ell] / 1000)**2)
    Dl_TT[high_ell] = envelope * damping

    # EE spectrum (roughly 0.05 of TT with different peak structure)
    Dl_EE = 0.05 * Dl_TT

    return ell, Dl_TT, Dl_EE


def apply_whbc_to_cl(
    ell: np.ndarray,
    Cl_TT: np.ndarray,
    Cl_EE: np.ndarray,
    params: WHBCPrimordialParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply WHBC P(k) modification to CMB Cls.

    Approximation: Cl_ell ~ integral P(k) * Transfer^2
    So P(k) modifications map approximately to Cl modifications
    with k ~ ell / D_A(z*)

    Args:
        ell: Multipole array
        Cl_TT: LCDM TT spectrum
        Cl_EE: LCDM EE spectrum
        params: WHBC primordial parameters

    Returns:
        (Cl_TT_mod, Cl_EE_mod)
    """
    # Approximate k-ell relation: k ~ ell / D_M(z*)
    # D_M(z*) ~ 14000 Mpc for Planck cosmology
    D_M_star = 14000.0  # Mpc
    k_ell = ell / D_M_star  # Mpc^-1

    # Get primordial ratio at each k
    F_whbc = primordial_ratio(k_ell, params)

    # Apply to Cls
    Cl_TT_mod = Cl_TT * F_whbc
    Cl_EE_mod = Cl_EE * F_whbc

    return Cl_TT_mod, Cl_EE_mod


def compute_cl_rms_residual(
    Cl_mod: np.ndarray,
    Cl_base: np.ndarray,
    ell: np.ndarray,
    ell_min: int = 30,
    ell_max: int = 2000,
) -> float:
    """Compute RMS fractional residual in Cl.

    Returns:
        RMS of |Delta Cl / Cl| in the range [ell_min, ell_max]
    """
    mask = (ell >= ell_min) & (ell <= ell_max) & (Cl_base > 0)
    if not np.any(mask):
        return 0.0

    frac_diff = np.abs((Cl_mod[mask] - Cl_base[mask]) / Cl_base[mask])
    return float(np.sqrt(np.mean(frac_diff**2)))


def compute_low_ell_ratio(
    Cl_mod: np.ndarray,
    Cl_base: np.ndarray,
    ell: np.ndarray,
    ell_max: int = 30,
) -> float:
    """Compute average Cl ratio at low ell.

    Returns:
        Mean of Cl_mod / Cl_base for ell < ell_max
    """
    mask = (ell < ell_max) & (Cl_base > 0)
    if not np.any(mask):
        return 1.0
    return float(np.mean(Cl_mod[mask] / Cl_base[mask]))


def estimate_effective_H0_shift(
    sigma8_ratio: float,
    low_ell_ratio: float,
    params: WHBCPrimordialParameters,
) -> float:
    """Estimate effective H0 shift when fitting LCDM to WHBC spectrum.

    The CMB acoustic scale theta_s is extremely well measured and largely
    independent of P(k) modifications. However, changes in P(k) can shift
    the inferred H0 through degeneracies with:
    - As (amplitude changes)
    - ns (tilt changes)
    - tau (reionization optical depth)

    This is a rough estimate based on parameter degeneracies.

    Returns:
        Approximate Delta_H0 in km/s/Mpc
    """
    # The key insight: if WHBC P(k) changes amplitude/shape differently
    # at CMB vs. BAO/local scales, there's potential for H0 shift

    # Effect 1: sigma8 changes require As adjustment
    # dln(sigma8) = 0.5 * dln(As)
    # H0 is correlated with As through late-time amplitude
    delta_As_eff = 2.0 * np.log(sigma8_ratio)  # Fractional change in effective As

    # Effect 2: Low-ell modifications can be absorbed by tau changes
    # tau changes don't directly affect H0 much

    # Effect 3: Shape changes can be absorbed by ns changes
    # ns affects peak heights and spacing slightly

    # Combined estimate: very small effect on H0
    # H0-sigma8 degeneracy direction is roughly d(H0)/d(sigma8) ~ 10-15 (km/s/Mpc)/0.01
    # But this is mostly about late-time structure, not CMB

    # For pure P(k) modifications on fixed background:
    # H0 shift is subdominant because theta_s is unchanged
    # Main effect is through sigma8 -> H0 degeneracy in CMB+LSS

    delta_sigma8 = sigma8_ratio - 1.0  # Fractional sigma8 change

    # Approximate degeneracy coefficient (from CMB+LSS analyses)
    # d(H0)/d(sigma8) ~ -15 km/s/Mpc per 0.01 change in sigma8
    # (Lower sigma8 -> higher H0 through degeneracy)
    degeneracy_coeff = -15.0 / 0.01  # (km/s/Mpc) / (sigma8 unit)

    delta_H0 = degeneracy_coeff * delta_sigma8 * 0.1  # Reduced coefficient for pure P(k)

    return float(delta_H0)


def evaluate_single_model(params_dict: Dict[str, float]) -> WHBCScanResult:
    """Evaluate a single WHBC P(k) model.

    Args:
        params_dict: Dictionary of WHBC parameters

    Returns:
        WHBCScanResult with all computed quantities
    """
    try:
        # Merge with fixed parameters
        full_params = {**FIXED_PARAMS, **params_dict}

        # Create WHBC parameters
        whbc_params = WHBCPrimordialParameters(**full_params)

        # Compute primordial ratio at key scales
        k_pivot = 0.05
        k_low = 1e-4
        k_high = 0.1

        F_pivot = primordial_ratio(k_pivot, whbc_params)
        F_low = primordial_ratio(k_low, whbc_params)
        F_high = primordial_ratio(k_high, whbc_params)

        # Compute sigma8 ratio
        sigma8_ratio = compute_sigma8_ratio(whbc_params)
        sigma8 = LCDM_BASELINE['sigma8'] * sigma8_ratio

        # Compute S8 = sigma8 * sqrt(Omega_m / 0.3)
        # Omega_m is unchanged on LCDM background
        Omega_m = 0.315
        S8_ratio = sigma8_ratio
        S8 = sigma8 * np.sqrt(Omega_m / 0.3)

        # Generate baseline Cl
        ell, Cl_TT_base, Cl_EE_base = generate_lcdm_cl_baseline()

        # Apply WHBC modification
        Cl_TT_mod, Cl_EE_mod = apply_whbc_to_cl(ell, Cl_TT_base, Cl_EE_base, whbc_params)

        # Compute RMS residuals
        TT_RMS = compute_cl_rms_residual(Cl_TT_mod, Cl_TT_base, ell, 30, 2000)
        EE_RMS = compute_cl_rms_residual(Cl_EE_mod, Cl_EE_base, ell, 30, 2000)

        # Compute low-ell ratio
        low_ell_ratio = compute_low_ell_ratio(Cl_TT_mod, Cl_TT_base, ell, 30)

        # Check viability
        passes_TT = TT_RMS < VIABILITY_CRITERIA['TT_RMS_max']
        passes_EE = EE_RMS < VIABILITY_CRITERIA['EE_RMS_max']
        passes_sigma8 = abs(sigma8_ratio - 1.0) < VIABILITY_CRITERIA['delta_sigma8_max']
        passes_S8 = abs(S8_ratio - 1.0) < VIABILITY_CRITERIA['delta_S8_max']
        passes_all = passes_TT and passes_EE and passes_sigma8 and passes_S8

        # Estimate H0 shift
        delta_H0 = estimate_effective_H0_shift(sigma8_ratio, low_ell_ratio, whbc_params)

        return WHBCScanResult(
            params=params_dict,
            success=True,
            F_ratio_pivot=float(F_pivot),
            F_ratio_low_k=float(F_low),
            F_ratio_high_k=float(F_high),
            TT_RMS_30_2000=TT_RMS,
            EE_RMS_30_2000=EE_RMS,
            low_ell_TT_ratio=low_ell_ratio,
            sigma8=sigma8,
            sigma8_ratio=sigma8_ratio,
            S8=S8,
            S8_ratio=S8_ratio,
            passes_TT=passes_TT,
            passes_EE=passes_EE,
            passes_sigma8=passes_sigma8,
            passes_S8=passes_S8,
            passes_all=passes_all,
            delta_H0_approx=delta_H0,
        )

    except Exception as e:
        return WHBCScanResult(
            params=params_dict,
            success=False,
            error=str(e),
        )


def run_parameter_scan() -> List[WHBCScanResult]:
    """Run full parameter scan over WHBC P(k) models.

    Returns:
        List of WHBCScanResult
    """
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    all_combos = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        # Skip invalid combinations
        if combo['A_osc'] > 0 and combo['omega_WH'] == 0:
            continue  # Need frequency for oscillations
        if combo['A_osc'] == 0 and combo['omega_WH'] > 0:
            continue  # No point in frequency without amplitude
        all_combos.append(combo)

    print(f"Total parameter combinations: {len(all_combos)}")

    # Run scan
    results = []
    for i, combo in enumerate(all_combos):
        result = evaluate_single_model(combo)
        results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(all_combos)}] A_cut={combo['A_cut']:.2f}, "
                  f"k_cut={combo['k_cut']:.0e}, A_osc={combo['A_osc']:.2f}")

    return results


def generate_plots(results: List[WHBCScanResult], output_dir: str):
    """Generate diagnostic plots.

    Args:
        results: List of scan results
        output_dir: Output directory for figures
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    successful = [r for r in results if r.success]
    passing = [r for r in successful if r.passes_all]

    print(f"\nGenerating plots...")
    print(f"  Successful evaluations: {len(successful)}")
    print(f"  Passing all criteria: {len(passing)}")

    # 1. P(k) ratio samples
    fig, ax = plt.subplots(figsize=(10, 6))

    k_array = np.logspace(-5, 0, 300)

    # LCDM baseline
    ax.semilogx(k_array, np.ones_like(k_array), 'k-', lw=2, label='LCDM')

    # Plot a few example models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    examples = successful[:5]
    for i, r in enumerate(examples):
        full_params = {**FIXED_PARAMS, **r.params}
        whbc_params = WHBCPrimordialParameters(**full_params)
        F = primordial_ratio(k_array, whbc_params)
        label = f"A_cut={r.params['A_cut']:.2f}, A_osc={r.params['A_osc']:.2f}"
        color = colors[i % len(colors)]
        linestyle = '-' if r.passes_all else '--'
        ax.semilogx(k_array, F, color=color, linestyle=linestyle, lw=1.5, label=label)

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('k [Mpc$^{-1}$]')
    ax.set_ylabel('$P_{WHBC}(k) / P_{LCDM}(k)$')
    ax.set_title('WHBC Primordial P(k) Ratio')
    ax.legend(fontsize=8)
    ax.set_ylim(0.7, 1.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pk_ratio_samples.png'), dpi=150)
    plt.close()

    # 2. CMB TT residuals
    fig, ax = plt.subplots(figsize=(10, 6))

    ell, Cl_TT_base, Cl_EE_base = generate_lcdm_cl_baseline()

    for i, r in enumerate(examples):
        full_params = {**FIXED_PARAMS, **r.params}
        whbc_params = WHBCPrimordialParameters(**full_params)
        Cl_TT_mod, _ = apply_whbc_to_cl(ell, Cl_TT_base, Cl_EE_base, whbc_params)

        frac_diff = (Cl_TT_mod - Cl_TT_base) / Cl_TT_base
        label = f"A_cut={r.params['A_cut']:.2f} (RMS={r.TT_RMS_30_2000:.3f})"
        color = colors[i % len(colors)]
        linestyle = '-' if r.passes_all else '--'
        ax.plot(ell, frac_diff, color=color, linestyle=linestyle, lw=1, label=label, alpha=0.7)

    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.axhline(0.05, color='red', linestyle='--', lw=1, alpha=0.5, label='5% threshold')
    ax.axhline(-0.05, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Delta C_\ell^{TT} / C_\ell^{TT}$')
    ax.set_title('CMB TT Fractional Residuals')
    ax.set_xlim(2, 2500)
    ax.set_ylim(-0.15, 0.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cmb_TT_residuals.png'), dpi=150)
    plt.close()

    # 3. sigma8/S8 shifts
    fig, ax = plt.subplots(figsize=(8, 8))

    sigma8_vals = [r.sigma8 for r in successful]
    S8_vals = [r.S8 for r in successful]
    passes = [r.passes_all for r in successful]

    colors_scatter = ['green' if p else 'red' for p in passes]
    ax.scatter(sigma8_vals, S8_vals, c=colors_scatter, alpha=0.6, s=60)

    # Reference lines
    ax.axhline(LCDM_BASELINE['S8'], color='blue', linestyle='--', alpha=0.5, label='Planck S8')
    ax.axvline(LCDM_BASELINE['sigma8'], color='blue', linestyle='--', alpha=0.5, label='Planck sigma8')

    ax.set_xlabel(r'$\sigma_8$')
    ax.set_ylabel(r'$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$')
    ax.set_title('WHBC Effects on $\\sigma_8$ and $S_8$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma8_S8_shifts.png'), dpi=150)
    plt.close()

    # 4. H0 shift distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    delta_H0_all = [r.delta_H0_approx for r in successful]
    delta_H0_pass = [r.delta_H0_approx for r in passing]

    bins = np.linspace(-2, 2, 21)
    ax.hist(delta_H0_all, bins=bins, alpha=0.5, label='All models', color='gray')
    if len(delta_H0_pass) > 0:
        ax.hist(delta_H0_pass, bins=bins, alpha=0.7, label='PASS models', color='green')

    ax.axvline(0, color='k', linestyle='-', lw=1)
    ax.axvline(5.0, color='orange', linestyle='--', lw=2, alpha=0.7, label='Hubble tension ~5')
    ax.axvline(-5.0, color='orange', linestyle='--', lw=2, alpha=0.7)

    ax.set_xlabel(r'$\Delta H_0$ (km/s/Mpc)')
    ax.set_ylabel('Count')
    ax.set_title('Approximate H0 Shift from WHBC P(k) Modifications')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'H0_shift_distribution.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


def save_results(results: List[WHBCScanResult], output_dir: str):
    """Save scan results.

    Args:
        results: List of scan results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        jr = {
            'params': r.params,
            'success': r.success,
            'passes_all': r.passes_all,
        }
        if r.success:
            jr.update({
                'F_ratio_pivot': r.F_ratio_pivot,
                'F_ratio_low_k': r.F_ratio_low_k,
                'F_ratio_high_k': r.F_ratio_high_k,
                'TT_RMS_30_2000': r.TT_RMS_30_2000,
                'EE_RMS_30_2000': r.EE_RMS_30_2000,
                'low_ell_TT_ratio': r.low_ell_TT_ratio,
                'sigma8': r.sigma8,
                'sigma8_ratio': r.sigma8_ratio,
                'S8': r.S8,
                'S8_ratio': r.S8_ratio,
                'delta_H0_approx': r.delta_H0_approx,
                'passes_TT': r.passes_TT,
                'passes_EE': r.passes_EE,
                'passes_sigma8': r.passes_sigma8,
                'passes_S8': r.passes_S8,
            })
        else:
            jr['error'] = r.error
        json_results.append(jr)

    # Save JSON
    with open(os.path.join(output_dir, 'scan_results.json'), 'w') as f:
        json.dump({
            'simulation': 'T10_WHBC_PRIMORDIAL - WHBC-like Primordial P(k) on LCDM Background',
            'date': datetime.now().isoformat(),
            'param_grid': PARAM_GRID,
            'fixed_params': FIXED_PARAMS,
            'viability_criteria': VIABILITY_CRITERIA,
            'lcdm_baseline': LCDM_BASELINE,
            'total_models': len(results),
            'successful': sum(1 for r in results if r.success),
            'passing': sum(1 for r in results if r.passes_all),
            'has_class': HAS_CLASS,
            'has_camb': HAS_CAMB,
            'results': json_results,
        }, f, indent=2, default=str)

    # Save NPZ for numerical analysis
    successful = [r for r in results if r.success]
    if len(successful) > 0:
        np.savez(
            os.path.join(output_dir, 'scan_results.npz'),
            A_cut=np.array([r.params['A_cut'] for r in successful]),
            k_cut=np.array([r.params['k_cut'] for r in successful]),
            A_osc=np.array([r.params['A_osc'] for r in successful]),
            omega_WH=np.array([r.params['omega_WH'] for r in successful]),
            passes_all=np.array([r.passes_all for r in successful]),
            TT_RMS=np.array([r.TT_RMS_30_2000 for r in successful]),
            EE_RMS=np.array([r.EE_RMS_30_2000 for r in successful]),
            sigma8_ratio=np.array([r.sigma8_ratio for r in successful]),
            delta_H0=np.array([r.delta_H0_approx for r in successful]),
        )

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: List[WHBCScanResult]):
    """Print summary of scan results."""
    successful = [r for r in results if r.success]
    passing = [r for r in successful if r.passes_all]

    print("\n" + "="*60)
    print("T10_WHBC_PRIMORDIAL SCAN SUMMARY")
    print("="*60)
    print(f"Total models evaluated: {len(results)}")
    print(f"Successful evaluations: {len(successful)}")
    print(f"Passing all criteria: {len(passing)}")
    print(f"Pass rate: {100*len(passing)/max(len(successful),1):.1f}%")

    if len(passing) > 0:
        delta_H0_pass = [abs(r.delta_H0_approx) for r in passing]
        TT_RMS_pass = [r.TT_RMS_30_2000 for r in passing]
        sigma8_shift = [abs(r.sigma8_ratio - 1.0) for r in passing]

        print(f"\nAmong PASS models:")
        print(f"  Max |delta_H0|: {max(delta_H0_pass):.2f} km/s/Mpc")
        print(f"  Mean |delta_H0|: {np.mean(delta_H0_pass):.2f} km/s/Mpc")
        print(f"  Max TT_RMS: {max(TT_RMS_pass):.4f}")
        print(f"  Max |sigma8_shift|: {max(sigma8_shift):.4f}")

        # Best model (max H0 effect)
        best = max(passing, key=lambda x: abs(x.delta_H0_approx))
        print(f"\nBest model (max |delta_H0|):")
        print(f"  A_cut = {best.params['A_cut']}")
        print(f"  k_cut = {best.params['k_cut']:.0e}")
        print(f"  A_osc = {best.params['A_osc']}")
        print(f"  omega_WH = {best.params['omega_WH']}")
        print(f"  delta_H0 = {best.delta_H0_approx:.2f} km/s/Mpc")
        print(f"  sigma8_ratio = {best.sigma8_ratio:.4f}")
    else:
        print("\nNo models passed all criteria!")

    print("="*60)


def main():
    """Main entry point."""
    print("="*60)
    print("SIMULATION 10: T10_WHBC_PRIMORDIAL")
    print("WHBC-like Primordial P(k) on Fixed LCDM Background")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"CLASS available: {HAS_CLASS}")
    print(f"CAMB available: {HAS_CAMB}")
    print()

    # Run scan
    print("Running parameter scan...")
    results = run_parameter_scan()

    # Output directories
    results_dir = 'results/simulation_10_whbc_pk'
    figures_dir = 'figures/simulation_10_whbc_pk'

    # Save results
    save_results(results, results_dir)

    # Generate plots
    generate_plots(results, figures_dir)

    # Print summary
    print_summary(results)

    print(f"\nResults: {results_dir}/scan_results.json")
    print(f"Figures: {figures_dir}/")


if __name__ == '__main__':
    main()
