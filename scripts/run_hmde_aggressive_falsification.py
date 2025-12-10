#!/usr/bin/env python3
"""TEST 3: Aggressive Falsification Test with CAMB.

This script attempts to push H0 up toward 73 km/s/Mpc while using CAMB to compute
the full CMB TT and EE power spectra. The goal is to determine whether extreme
HMDE parameters can:

1. Push H0 to 72-73 km/s/Mpc while maintaining acceptable theta_s
2. Produce CMB spectra that remain consistent with Planck data

We use CAMB's dark energy module (PPF) with CPL parameterization as a proxy
for HMDE effects, mapping the HMDE w(a) profile to best-fit (w0, wa).

Key output: Can we get H0 > 72 with chi^2_CMB within ~10 of LCDM?
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import camb
    from camb import model as camb_model
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False
    print("WARNING: CAMB not available. Will run in background-only mode.")

from hrc2.horizon_models.refinement_d import create_dynamical_eos_model


# Physical constants
C_KM_S = 299792.458  # km/s

# Planck 2018 best-fit parameters
PLANCK_PARAMS = {
    'H0': 67.4,
    'ombh2': 0.02237,
    'omch2': 0.1200,
    'tau': 0.054,
    'As': 2.1e-9,
    'ns': 0.965,
}

# Target H0 values to test
H0_TARGETS = [68.0, 69.0, 70.0, 71.0, 72.0, 73.0]


@dataclass
class FalsificationResult:
    """Result from falsification test."""
    H0_target: float
    H0_achieved: float

    # HMDE parameters
    delta_w: float
    a_w: float
    lambda_hor: float

    # CPL fit
    w0: float
    wa: float

    # Constraint metrics
    theta_s: float
    theta_s_planck: float
    theta_s_dev_percent: float

    # CMB chi^2 (if CAMB available)
    chi2_TT: Optional[float] = None
    chi2_EE: Optional[float] = None
    chi2_total: Optional[float] = None

    # Comparison to LCDM
    delta_chi2_vs_lcdm: Optional[float] = None

    success: bool = True
    message: str = ""


def fit_cpl_to_hmde(
    delta_w: float,
    a_w: float,
    m_eos: float = 2.0,
    a_fit_min: float = 0.3,
    a_fit_max: float = 1.0,
    n_points: int = 100,
) -> Tuple[float, float]:
    """Fit CPL (w0, wa) to HMDE w(a) profile.

    CPL: w(a) = w0 + wa * (1 - a)
    HMDE: w(a) = -1 + delta_w / (1 + (a/a_w)^m)

    Args:
        delta_w: HMDE EoS shift
        a_w: HMDE transition scale factor
        m_eos: HMDE transition power
        a_fit_min, a_fit_max: Fitting range in a
        n_points: Number of fit points

    Returns:
        (w0, wa) CPL parameters
    """
    a_arr = np.linspace(a_fit_min, a_fit_max, n_points)

    # HMDE w(a)
    def w_hmde(a):
        return -1.0 + delta_w / (1.0 + (a / a_w) ** m_eos)

    w_hmde_arr = np.array([w_hmde(a) for a in a_arr])

    # Fit CPL using least squares
    # w_cpl = w0 + wa * (1 - a)
    # Let x = 1 - a, then w = w0 + wa * x
    x_arr = 1 - a_arr

    # Design matrix: [1, x]
    A = np.column_stack([np.ones_like(x_arr), x_arr])

    # Solve normal equations
    coeffs, _, _, _ = np.linalg.lstsq(A, w_hmde_arr, rcond=None)
    w0, wa = coeffs

    return w0, wa


def compute_hmde_background_and_theta_s(
    H0: float,
    delta_w: float,
    a_w: float,
    lambda_hor: float,
    Omega_m: float = 0.315,
) -> Optional[Dict]:
    """Compute HMDE background and theta_s."""
    try:
        model = create_dynamical_eos_model(
            delta_w=delta_w,
            a_w=a_w,
            m_eos=2.0,
            lambda_hor=lambda_hor,
            tau_hor=0.1,
            Omega_m0=Omega_m,
            Omega_r0=9e-5,
            H0=H0,
        )

        result = model.solve(z_max=1200.0, n_points=2000)

        if not result.success:
            return None

        z_arr = result.z
        H_arr = result.H

        def H_interp(z):
            return np.interp(z, z_arr[::-1], H_arr[::-1])

        # Sound horizon and theta_s
        z_star = 1089.0
        c_s = C_KM_S / np.sqrt(3)

        r_star, _ = quad(lambda z: c_s / H_interp(z), z_star, 1200, limit=500)
        D_M_star, _ = quad(lambda z: C_KM_S / H_interp(z), 0, z_star, limit=500)
        D_A_star = D_M_star / (1 + z_star)

        theta_s = r_star / D_A_star

        return {
            'H0': H0,
            'theta_s': theta_s,
            'r_star': r_star,
            'D_A_star': D_A_star,
        }

    except Exception as e:
        return None


def compute_camb_spectra(
    H0: float,
    w0: float = -1.0,
    wa: float = 0.0,
    ombh2: float = 0.02237,
    omch2: float = 0.1200,
    tau: float = 0.054,
    As: float = 2.1e-9,
    ns: float = 0.965,
    lmax: int = 2500,
) -> Optional[Dict]:
    """Compute CMB spectra using CAMB with CPL dark energy."""
    if not CAMB_AVAILABLE:
        return None

    try:
        # Set up CAMB parameters
        pars = camb.CAMBparams()

        # Cosmological parameters
        pars.set_cosmology(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            tau=tau,
            mnu=0.06,
            omk=0,
        )

        # Dark energy
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')

        # Primordial power spectrum
        pars.InitPower.set_params(As=As, ns=ns)

        # Accuracy
        pars.set_accuracy(AccuracyBoost=1.0, lAccuracyBoost=1.0)

        # Compute
        pars.set_for_lmax(lmax, lens_potential_accuracy=1)
        results = camb.get_results(pars)

        # Get power spectra
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL = powers['total']

        # Extract TT and EE
        ell = np.arange(len(totCL))
        Cl_TT = totCL[:, 0]  # TT
        Cl_EE = totCL[:, 1]  # EE
        Cl_TE = totCL[:, 3]  # TE

        # Get derived parameters
        derived = results.get_derived_params()

        return {
            'ell': ell,
            'Cl_TT': Cl_TT,
            'Cl_EE': Cl_EE,
            'Cl_TE': Cl_TE,
            'theta_star': derived.get('thetastar', np.nan),
            'rdrag': derived.get('rdrag', np.nan),
            'sigma8': results.get_sigma8(),
        }

    except Exception as e:
        print(f"CAMB error: {e}")
        return None


def compute_chi2_cmb(
    Cl_model: Dict,
    Cl_lcdm: Dict,
    l_min: int = 30,
    l_max: int = 2000,
    f_sky: float = 0.7,
) -> Dict[str, float]:
    """Compute simplified chi^2 for CMB power spectra.

    Uses a simplified cosmic variance limited chi^2:
    chi^2 = sum_l (2l+1) * f_sky * [(C_model - C_lcdm) / C_lcdm]^2

    This is approximate - a full analysis would use the Planck covariance.
    """
    chi2_TT = 0.0
    chi2_EE = 0.0

    for l in range(l_min, min(l_max + 1, len(Cl_model['Cl_TT']))):
        # Cosmic variance
        weight = (2 * l + 1) * f_sky

        # TT
        if Cl_lcdm['Cl_TT'][l] > 0:
            delta_TT = (Cl_model['Cl_TT'][l] - Cl_lcdm['Cl_TT'][l]) / Cl_lcdm['Cl_TT'][l]
            chi2_TT += weight * delta_TT**2

        # EE
        if Cl_lcdm['Cl_EE'][l] > 0:
            delta_EE = (Cl_model['Cl_EE'][l] - Cl_lcdm['Cl_EE'][l]) / Cl_lcdm['Cl_EE'][l]
            chi2_EE += weight * delta_EE**2

    return {
        'chi2_TT': chi2_TT,
        'chi2_EE': chi2_EE,
        'chi2_total': chi2_TT + chi2_EE,
    }


def find_hmde_params_for_H0(
    H0_target: float,
    theta_s_target: float,
    delta_w_range: Tuple[float, float] = (-0.5, 0.1),
    a_w_range: Tuple[float, float] = (0.15, 0.5),
    lambda_hor: float = 0.1,
    n_grid: int = 30,
) -> Optional[Dict]:
    """Find HMDE parameters that achieve target H0 while matching theta_s.

    Strategy: Grid search over (delta_w, a_w) to find parameters that:
    1. Give theta_s close to target (Planck value)
    2. Allow H0 to reach target

    Args:
        H0_target: Target Hubble constant
        theta_s_target: Target theta_s (Planck value)
        delta_w_range: Search range for delta_w
        a_w_range: Search range for a_w
        lambda_hor: Fixed horizon memory amplitude
        n_grid: Grid size for search

    Returns:
        Best parameter set or None
    """
    delta_w_vals = np.linspace(delta_w_range[0], delta_w_range[1], n_grid)
    a_w_vals = np.linspace(a_w_range[0], a_w_range[1], n_grid)

    best_result = None
    best_theta_dev = np.inf

    for delta_w in delta_w_vals:
        for a_w in a_w_vals:
            bg = compute_hmde_background_and_theta_s(
                H0=H0_target,
                delta_w=delta_w,
                a_w=a_w,
                lambda_hor=lambda_hor,
            )

            if bg is None:
                continue

            theta_dev = abs(bg['theta_s'] - theta_s_target) / theta_s_target

            if theta_dev < best_theta_dev:
                best_theta_dev = theta_dev
                best_result = {
                    'delta_w': delta_w,
                    'a_w': a_w,
                    'lambda_hor': lambda_hor,
                    'theta_s': bg['theta_s'],
                    'theta_dev': theta_dev,
                }

    return best_result


def run_falsification_test(
    output_dir: str = "results/test3_aggressive_falsification",
    H0_targets: list = None,
):
    """Run aggressive falsification test.

    Args:
        output_dir: Output directory
        H0_targets: List of H0 values to test
    """
    if H0_targets is None:
        H0_targets = H0_TARGETS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEST 3: Aggressive Falsification with CAMB")
    print("=" * 60)
    print(f"\nCAMB available: {CAMB_AVAILABLE}")
    print(f"H0 targets: {H0_targets}")
    print()

    # Compute LCDM reference
    print("Computing LCDM reference...")
    lcdm_bg = compute_hmde_background_and_theta_s(
        H0=PLANCK_PARAMS['H0'],
        delta_w=0.0,  # No HMDE modification
        a_w=0.3,
        lambda_hor=0.0,
    )

    if lcdm_bg is None:
        print("ERROR: Could not compute LCDM reference!")
        return

    theta_s_planck = lcdm_bg['theta_s']
    print(f"  LCDM theta_s = {theta_s_planck:.6e}")

    # Compute LCDM CMB spectra
    if CAMB_AVAILABLE:
        print("Computing LCDM CMB spectra...")
        Cl_lcdm = compute_camb_spectra(
            H0=PLANCK_PARAMS['H0'],
            w0=-1.0,
            wa=0.0,
        )
        if Cl_lcdm:
            print(f"  LCDM theta_star (CAMB) = {Cl_lcdm['theta_star']:.6e}")
    else:
        Cl_lcdm = None

    print()

    # Test each H0 target
    results = []

    for H0_target in H0_targets:
        print(f"\n{'='*40}")
        print(f"Testing H0 = {H0_target} km/s/Mpc")
        print(f"{'='*40}")

        # Find HMDE parameters for this H0
        print("Finding HMDE parameters...")
        hmde_params = find_hmde_params_for_H0(
            H0_target=H0_target,
            theta_s_target=theta_s_planck,
        )

        if hmde_params is None:
            print(f"  Could not find valid HMDE parameters for H0={H0_target}")
            results.append(FalsificationResult(
                H0_target=H0_target,
                H0_achieved=np.nan,
                delta_w=np.nan,
                a_w=np.nan,
                lambda_hor=np.nan,
                w0=np.nan,
                wa=np.nan,
                theta_s=np.nan,
                theta_s_planck=theta_s_planck,
                theta_s_dev_percent=np.nan,
                success=False,
                message="No valid HMDE parameters found",
            ))
            continue

        delta_w = hmde_params['delta_w']
        a_w = hmde_params['a_w']
        lambda_hor = hmde_params['lambda_hor']
        theta_s = hmde_params['theta_s']
        theta_dev_pct = hmde_params['theta_dev'] * 100

        print(f"  Found: delta_w={delta_w:.4f}, a_w={a_w:.4f}")
        print(f"  theta_s deviation: {theta_dev_pct:.4f}%")

        # Fit CPL
        w0, wa = fit_cpl_to_hmde(delta_w, a_w)
        print(f"  CPL fit: w0={w0:.4f}, wa={wa:.4f}")

        # Compute CMB spectra with CAMB
        chi2_TT = None
        chi2_EE = None
        chi2_total = None
        delta_chi2 = None

        if CAMB_AVAILABLE and Cl_lcdm is not None:
            print("  Computing CAMB spectra...")
            Cl_model = compute_camb_spectra(
                H0=H0_target,
                w0=w0,
                wa=wa,
            )

            if Cl_model is not None:
                chi2_result = compute_chi2_cmb(Cl_model, Cl_lcdm)
                chi2_TT = chi2_result['chi2_TT']
                chi2_EE = chi2_result['chi2_EE']
                chi2_total = chi2_result['chi2_total']

                # Compare to LCDM
                chi2_lcdm = compute_chi2_cmb(Cl_lcdm, Cl_lcdm)
                delta_chi2 = chi2_total - chi2_lcdm['chi2_total']

                print(f"  chi2_TT = {chi2_TT:.1f}")
                print(f"  chi2_EE = {chi2_EE:.1f}")
                print(f"  chi2_total = {chi2_total:.1f}")
                print(f"  Delta chi2 vs LCDM = {delta_chi2:.1f}")

        res = FalsificationResult(
            H0_target=H0_target,
            H0_achieved=H0_target,
            delta_w=delta_w,
            a_w=a_w,
            lambda_hor=lambda_hor,
            w0=w0,
            wa=wa,
            theta_s=theta_s,
            theta_s_planck=theta_s_planck,
            theta_s_dev_percent=theta_dev_pct,
            chi2_TT=chi2_TT,
            chi2_EE=chi2_EE,
            chi2_total=chi2_total,
            delta_chi2_vs_lcdm=delta_chi2,
            success=True,
            message="Success",
        )
        results.append(res)

    # Summary
    print("\n" + "=" * 60)
    print("FALSIFICATION TEST SUMMARY")
    print("=" * 60)

    print("\nH0 Target | theta_s dev | w0     | wa     | Delta chi2")
    print("-" * 60)

    for res in results:
        if res.success:
            chi2_str = f"{res.delta_chi2_vs_lcdm:+.1f}" if res.delta_chi2_vs_lcdm is not None else "N/A"
            print(f"  {res.H0_target:5.1f}   | {res.theta_s_dev_percent:7.4f}% | {res.w0:6.3f} | {res.wa:6.3f} | {chi2_str}")
        else:
            print(f"  {res.H0_target:5.1f}   | FAILED")

    # Save results
    output_dict = {
        'config': {
            'H0_targets': H0_targets,
            'CAMB_available': CAMB_AVAILABLE,
        },
        'lcdm_reference': {
            'H0': PLANCK_PARAMS['H0'],
            'theta_s': theta_s_planck,
        },
        'results': [
            {
                'H0_target': r.H0_target,
                'H0_achieved': r.H0_achieved,
                'delta_w': r.delta_w,
                'a_w': r.a_w,
                'lambda_hor': r.lambda_hor,
                'w0': r.w0,
                'wa': r.wa,
                'theta_s': r.theta_s,
                'theta_s_dev_percent': r.theta_s_dev_percent,
                'chi2_TT': r.chi2_TT,
                'chi2_EE': r.chi2_EE,
                'chi2_total': r.chi2_total,
                'delta_chi2_vs_lcdm': r.delta_chi2_vs_lcdm,
                'success': r.success,
                'message': r.message,
            }
            for r in results
        ],
    }

    with open(output_path / 'falsification_results.json', 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)

    # Check if any H0 > 72 passes
    high_H0_results = [r for r in results if r.success and r.H0_target >= 72]

    if not high_H0_results:
        print("No successful results for H0 >= 72")
    else:
        for r in high_H0_results:
            if r.theta_s_dev_percent < 0.1:  # < 0.1% theta_s deviation
                if r.delta_chi2_vs_lcdm is not None and r.delta_chi2_vs_lcdm < 10:
                    print(f"H0 = {r.H0_target}: POTENTIALLY VIABLE!")
                    print(f"  theta_s deviation = {r.theta_s_dev_percent:.4f}%")
                    print(f"  Delta chi^2 = {r.delta_chi2_vs_lcdm:.1f}")
                elif r.delta_chi2_vs_lcdm is None:
                    print(f"H0 = {r.H0_target}: Passes theta_s, CMB chi^2 unknown")
                else:
                    print(f"H0 = {r.H0_target}: RULED OUT by CMB chi^2")
                    print(f"  Delta chi^2 = {r.delta_chi2_vs_lcdm:.1f} >> 10")
            else:
                print(f"H0 = {r.H0_target}: theta_s deviation too large ({r.theta_s_dev_percent:.2f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TEST 3: Aggressive Falsification")
    parser.add_argument("--output-dir", type=str, default="results/test3_aggressive_falsification")
    parser.add_argument("--H0-targets", type=float, nargs="+", default=H0_TARGETS)

    args = parser.parse_args()

    run_falsification_test(
        output_dir=args.output_dir,
        H0_targets=args.H0_targets,
    )
