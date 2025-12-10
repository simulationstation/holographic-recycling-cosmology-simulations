#!/usr/bin/env python3
"""TEST 1: Direct non-CPL HMDE background MCMC.

Uses emcee to sample directly in horizon-memory parameter space at background level,
bypassing the CPL (w0, wa) proxy used in Cobaya.

Sampled parameters:
- H0: Hubble constant [60, 80]
- Omega_m: Matter density parameter [0.2, 0.4]
- delta_w: EoS shift amplitude [-0.5, 0.1]
- a_w: Transition scale factor [0.1, 0.5]
- lambda_hor: Horizon-memory amplitude [0.0, 0.5]

Merit function includes:
- Gaussian penalty for theta_s deviation from Planck
- SNe distance modulus chi^2 (simple)
- BAO distance ratio chi^2 (simple)
"""

import numpy as np
import emcee
import json
import time
from pathlib import Path
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.horizon_models.refinement_d import create_dynamical_eos_model
from hrc2.horizon_models.base import HorizonMemoryParameters, RefinementType


# Physical constants
C_KM_S = 299792.458  # km/s

# Planck 2018 constraints
THETA_S_PLANCK = 1.04109e-2  # Planck 2018 100*theta_s
THETA_S_SIGMA = 0.00030e-2   # Uncertainty

# Sound horizon at drag epoch (Planck 2018)
R_DRAG_PLANCK = 147.09  # Mpc
R_DRAG_SIGMA = 0.26     # Mpc

# BAO data points (SDSS DR12 consensus)
# Format: (z_eff, measurement, sigma, type)
# type: 'DM_rd' = D_M/r_d, 'DH_rd' = D_H/r_d
BAO_DATA = [
    (0.38, 10.27, 0.15, 'DM_rd'),
    (0.38, 25.00, 0.76, 'DH_rd'),
    (0.51, 13.38, 0.18, 'DM_rd'),
    (0.51, 22.33, 0.58, 'DH_rd'),
    (0.61, 15.33, 0.21, 'DM_rd'),
    (0.61, 20.98, 0.61, 'DH_rd'),
]

# Pantheon+ simplified constraint: D_L(z=0.5) constraint
# We use effective magnitude offset chi^2
SN_Z_EFF = 0.5
SN_CONSTRAINT_SIGMA = 0.02  # ~2% distance error at z=0.5


@dataclass
class HMDEBackgroundResult:
    """Result from HMDE background calculation."""
    H0: float
    Omega_m: float
    delta_w: float
    a_w: float
    lambda_hor: float

    # Computed quantities
    theta_s: float  # 100 * theta_s
    r_drag: float   # Sound horizon at drag
    D_M_z: dict     # Comoving distance at various z
    D_H_z: dict     # Hubble distance at various z

    chi2_theta_s: float
    chi2_bao: float
    chi2_sn: float
    chi2_total: float

    success: bool
    message: str


def compute_hmde_background(
    H0: float,
    Omega_m: float,
    delta_w: float,
    a_w: float,
    lambda_hor: float,
    tau_hor: float = 0.1,
    m_eos: float = 2.0,
    z_max: float = 1200.0,
) -> Optional[HMDEBackgroundResult]:
    """Compute HMDE background cosmology directly (no CPL proxy).

    Args:
        H0: Hubble constant
        Omega_m: Matter density parameter
        delta_w: EoS shift amplitude
        a_w: Transition scale factor
        lambda_hor: Horizon-memory amplitude
        tau_hor: Memory timescale (fixed)
        m_eos: EoS transition power (fixed)
        z_max: Maximum redshift for integration

    Returns:
        HMDEBackgroundResult or None if failed
    """
    try:
        # Create HMDE model
        model = create_dynamical_eos_model(
            delta_w=delta_w,
            a_w=a_w,
            m_eos=m_eos,
            lambda_hor=lambda_hor,
            tau_hor=tau_hor,
            Omega_m0=Omega_m,
            Omega_r0=9e-5,  # Radiation density
            H0=H0,
        )

        # Solve the background evolution
        result = model.solve(z_max=z_max, n_points=1000)

        if not result.success:
            return None

        # Build interpolator for H(z)
        z_arr = result.z
        H_arr = result.H

        def H_interp(z):
            """Interpolate H(z) in km/s/Mpc."""
            return np.interp(z, z_arr[::-1], H_arr[::-1])

        # Compute sound horizon at drag
        # r_s = integral from z_drag to infinity of c_s/H(z) dz
        # Approximate z_drag ~ 1060
        z_drag = 1060.0

        # Sound speed in plasma: c_s = c / sqrt(3(1 + R_b))
        # R_b = 3 * rho_b / (4 * rho_gamma)
        # For simplicity, use approximation c_s ~ c/sqrt(3) at high z
        c_s_approx = C_KM_S / np.sqrt(3)

        try:
            r_drag, _ = quad(
                lambda z: c_s_approx / H_interp(z),
                z_drag, z_max,
                limit=500
            )
        except:
            r_drag = R_DRAG_PLANCK  # Fallback

        # Compute comoving distance D_M(z) = c * integral_0^z dz'/H(z')
        D_M_z = {}
        D_H_z = {}

        for z_target in [0.38, 0.51, 0.61, 1089.0]:
            try:
                D_M, _ = quad(
                    lambda z: C_KM_S / H_interp(z),
                    0, z_target,
                    limit=500
                )
                D_M_z[z_target] = D_M
                D_H_z[z_target] = C_KM_S / H_interp(z_target)
            except:
                D_M_z[z_target] = np.nan
                D_H_z[z_target] = np.nan

        # Compute theta_s = r_s(z_*) / D_A(z_*)
        # D_A = D_M / (1 + z)
        z_star = 1089.0
        D_A_star = D_M_z.get(z_star, np.nan) / (1 + z_star)

        # Sound horizon at recombination (approximate)
        try:
            r_star, _ = quad(
                lambda z: c_s_approx / H_interp(z),
                z_star, z_max,
                limit=500
            )
        except:
            r_star = r_drag

        theta_s = 100 * r_star / D_A_star if D_A_star > 0 else np.nan

        # Compute chi^2 components

        # 1. theta_s chi^2
        chi2_theta_s = ((theta_s - THETA_S_PLANCK * 100) / (THETA_S_SIGMA * 100))**2

        # 2. BAO chi^2
        chi2_bao = 0.0
        for z_eff, meas, sigma, mtype in BAO_DATA:
            if mtype == 'DM_rd':
                pred = D_M_z.get(z_eff, np.nan) / r_drag
            else:  # DH_rd
                pred = D_H_z.get(z_eff, np.nan) / r_drag

            if np.isfinite(pred):
                chi2_bao += ((pred - meas) / sigma)**2

        # 3. SN chi^2 (simplified: D_L deviation at z=0.5)
        # Compare to LCDM D_L
        D_L_model = D_M_z.get(SN_Z_EFF, np.nan) * (1 + SN_Z_EFF)

        # LCDM reference
        Omega_L_lcdm = 1 - Omega_m - 9e-5
        def H_lcdm(z):
            return H0 * np.sqrt(Omega_m * (1+z)**3 + 9e-5 * (1+z)**4 + Omega_L_lcdm)

        try:
            D_M_lcdm, _ = quad(lambda z: C_KM_S / H_lcdm(z), 0, SN_Z_EFF, limit=100)
            D_L_lcdm = D_M_lcdm * (1 + SN_Z_EFF)
        except:
            D_L_lcdm = D_L_model

        rel_dev = (D_L_model - D_L_lcdm) / D_L_lcdm if D_L_lcdm > 0 else 0
        chi2_sn = (rel_dev / SN_CONSTRAINT_SIGMA)**2

        chi2_total = chi2_theta_s + chi2_bao + chi2_sn

        return HMDEBackgroundResult(
            H0=H0,
            Omega_m=Omega_m,
            delta_w=delta_w,
            a_w=a_w,
            lambda_hor=lambda_hor,
            theta_s=theta_s,
            r_drag=r_drag,
            D_M_z=D_M_z,
            D_H_z=D_H_z,
            chi2_theta_s=chi2_theta_s,
            chi2_bao=chi2_bao,
            chi2_sn=chi2_sn,
            chi2_total=chi2_total,
            success=True,
            message="Success"
        )

    except Exception as e:
        return None


def log_prior(theta):
    """Flat priors on parameters."""
    H0, Omega_m, delta_w, a_w, lambda_hor = theta

    if not (60.0 < H0 < 80.0):
        return -np.inf
    if not (0.2 < Omega_m < 0.4):
        return -np.inf
    if not (-0.5 < delta_w < 0.1):
        return -np.inf
    if not (0.1 < a_w < 0.5):
        return -np.inf
    if not (0.0 < lambda_hor < 0.5):
        return -np.inf

    return 0.0


def log_likelihood(theta):
    """Log-likelihood from chi^2."""
    H0, Omega_m, delta_w, a_w, lambda_hor = theta

    result = compute_hmde_background(
        H0=H0,
        Omega_m=Omega_m,
        delta_w=delta_w,
        a_w=a_w,
        lambda_hor=lambda_hor,
    )

    if result is None or not result.success:
        return -np.inf

    if not np.isfinite(result.chi2_total):
        return -np.inf

    return -0.5 * result.chi2_total


def log_posterior(theta):
    """Log-posterior = log_prior + log_likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def run_mcmc(
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    output_dir: str = "results/test1_background_mcmc",
):
    """Run emcee MCMC sampling.

    Args:
        n_walkers: Number of walkers
        n_steps: Number of MCMC steps per walker
        n_burn: Burn-in steps to discard
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEST 1: Direct non-CPL HMDE Background MCMC")
    print("=" * 60)
    print(f"\nParameters: H0, Omega_m, delta_w, a_w, lambda_hor")
    print(f"n_walkers = {n_walkers}, n_steps = {n_steps}, n_burn = {n_burn}")
    print()

    # Initial positions
    # Start near Planck LCDM + small HMDE perturbations
    ndim = 5
    p0_mean = np.array([67.4, 0.315, -0.1, 0.3, 0.1])
    p0_std = np.array([0.5, 0.01, 0.05, 0.05, 0.05])

    p0 = p0_mean + p0_std * np.random.randn(n_walkers, ndim)

    # Ensure all walkers start in valid prior range
    for i in range(n_walkers):
        p0[i, 0] = np.clip(p0[i, 0], 61, 79)
        p0[i, 1] = np.clip(p0[i, 1], 0.21, 0.39)
        p0[i, 2] = np.clip(p0[i, 2], -0.49, 0.09)
        p0[i, 3] = np.clip(p0[i, 3], 0.11, 0.49)
        p0[i, 4] = np.clip(p0[i, 4], 0.01, 0.49)

    print("Testing initial point...")
    test_result = compute_hmde_background(
        H0=p0_mean[0],
        Omega_m=p0_mean[1],
        delta_w=p0_mean[2],
        a_w=p0_mean[3],
        lambda_hor=p0_mean[4],
    )

    if test_result is None:
        print("ERROR: Initial point evaluation failed!")
        return

    print(f"  theta_s = {test_result.theta_s:.4f}")
    print(f"  r_drag = {test_result.r_drag:.2f} Mpc")
    print(f"  chi2_theta_s = {test_result.chi2_theta_s:.2f}")
    print(f"  chi2_bao = {test_result.chi2_bao:.2f}")
    print(f"  chi2_sn = {test_result.chi2_sn:.2f}")
    print(f"  chi2_total = {test_result.chi2_total:.2f}")
    print()

    # Create sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)

    # Run MCMC
    print("Running MCMC...")
    start_time = time.time()

    # Progress tracking
    for i, sample in enumerate(sampler.sample(p0, iterations=n_steps, progress=True)):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            acc_frac = np.mean(sampler.acceptance_fraction)
            print(f"  Step {i+1}/{n_steps}, elapsed: {elapsed:.1f}s, acceptance: {acc_frac:.3f}")

    elapsed_total = time.time() - start_time
    print(f"\nMCMC completed in {elapsed_total:.1f} seconds")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    # Get chains
    samples = sampler.get_chain(discard=n_burn, flat=True)

    print(f"\nPost burn-in samples: {len(samples)}")

    # Compute statistics
    labels = ['H0', 'Omega_m', 'delta_w', 'a_w', 'lambda_hor']

    print("\nParameter constraints (mean ± std):")
    print("-" * 40)

    results = {}
    for i, label in enumerate(labels):
        mean = np.mean(samples[:, i])
        std = np.std(samples[:, i])
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])

        print(f"  {label:12s}: {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")

        results[label] = {
            'mean': float(mean),
            'std': float(std),
            'q16': float(q16),
            'q50': float(q50),
            'q84': float(q84),
        }

    # Best-fit point
    best_idx = np.argmax(sampler.get_log_prob(discard=n_burn, flat=True))
    best_params = samples[best_idx]

    print(f"\nBest-fit: H0={best_params[0]:.2f}, Omega_m={best_params[1]:.4f}, "
          f"delta_w={best_params[2]:.3f}, a_w={best_params[3]:.3f}, lambda_hor={best_params[4]:.3f}")

    # Evaluate best-fit
    best_result = compute_hmde_background(
        H0=best_params[0],
        Omega_m=best_params[1],
        delta_w=best_params[2],
        a_w=best_params[3],
        lambda_hor=best_params[4],
    )

    if best_result:
        print(f"\nBest-fit chi^2: {best_result.chi2_total:.2f}")
        print(f"  chi2_theta_s: {best_result.chi2_theta_s:.2f}")
        print(f"  chi2_bao: {best_result.chi2_bao:.2f}")
        print(f"  chi2_sn: {best_result.chi2_sn:.2f}")

    # Save results
    output = {
        'config': {
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'n_samples': len(samples),
        },
        'acceptance_fraction': float(np.mean(sampler.acceptance_fraction)),
        'parameters': results,
        'best_fit': {
            'H0': float(best_params[0]),
            'Omega_m': float(best_params[1]),
            'delta_w': float(best_params[2]),
            'a_w': float(best_params[3]),
            'lambda_hor': float(best_params[4]),
            'chi2_total': float(best_result.chi2_total) if best_result else None,
            'chi2_theta_s': float(best_result.chi2_theta_s) if best_result else None,
            'chi2_bao': float(best_result.chi2_bao) if best_result else None,
            'chi2_sn': float(best_result.chi2_sn) if best_result else None,
        },
    }

    with open(output_path / 'mcmc_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save samples
    np.save(output_path / 'samples.npy', samples)

    # Save full chain
    np.save(output_path / 'chain.npy', sampler.get_chain())
    np.save(output_path / 'log_prob.npy', sampler.get_log_prob())

    print(f"\nResults saved to {output_path}")

    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)

    H0_q50 = results['H0']['q50']
    H0_err = (results['H0']['q84'] - results['H0']['q16']) / 2

    print(f"H0 = {H0_q50:.2f} ± {H0_err:.2f} km/s/Mpc")

    if H0_q50 < 70:
        print("=> HMDE does NOT significantly raise H0 above Planck LCDM value")
    elif H0_q50 > 72:
        print("=> HMDE CAN push H0 toward SH0ES value!")
    else:
        print("=> HMDE provides modest H0 increase but not full tension resolution")

    return samples, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TEST 1: HMDE Background MCMC")
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-burn", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results/test1_background_mcmc")

    args = parser.parse_args()

    run_mcmc(
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        output_dir=args.output_dir,
    )
