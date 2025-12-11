#!/usr/bin/env python3
"""
SIMULATION 16B: No-Boundary Prior MCMC with Simulated Cosmological Data

This simulation performs MCMC sampling with the Hawking-Hartle no-boundary
prior over primordial parameters, constrained by simulated cosmological data
(Planck-like CMB + BAO + SN data).

Comparison with SIM 15:
- SIM 15: Flat priors on systematics, LCDM background -> P(H0 >= 73) = 0.27%
- SIM 16B: No-boundary prior with epsilon_corr -> P(H0 >= 73) = ?

Key questions:
1. Does the no-boundary prior affect the H0 posterior?
2. Can epsilon_corr (early-time H modification) shift H0 upward?
3. How does P(H0 >= 73 | no-boundary) compare to P(H0 >= 73 | flat priors)?

Usage:
    python scripts/run_sim16b_noboundary_mcmc.py [--n-walkers 32] [--n-steps 3000]
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
import emcee
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.noboundary import (
    NoBoundaryHyperparams,
    NoBoundaryParams,
    log_prior_no_boundary,
    primordial_to_cosmo,
    CosmoParams,
)
from hrc2.background import apply_epsilon_corr, compute_sound_horizon_epsilon_effect


@dataclass
class Sim16BConfig:
    """Configuration for SIM 16B MCMC."""
    # MCMC settings
    n_walkers: int = 32
    n_steps: int = 3000
    n_burn: int = 500
    seed: int = 20241210

    # True cosmology (Planck LCDM)
    H0_true: float = 67.4
    Omega_m_true: float = 0.315
    Omega_b_true: float = 0.0493

    # Simulated data uncertainties
    sigma_H0_cmb: float = 0.5      # CMB-inferred H0 uncertainty
    sigma_theta_s: float = 0.0003  # CMB acoustic scale uncertainty
    sigma_DM_BAO: float = 0.02     # BAO distance modulus uncertainty
    sigma_mu_SN: float = 0.1       # SN distance modulus uncertainty

    # Number of simulated data points
    n_bao_points: int = 3          # BAO redshifts
    n_sn_points: int = 50          # SN redshifts

    # No-boundary prior hyperparameters
    alpha_Ne: float = 0.05
    Ne_min: float = 50.0
    Ne_max: float = 80.0
    mu_logV: float = -10.0
    sigma_logV: float = 1.0
    mu_phi_init: float = 0.1
    sigma_phi_init: float = 0.3
    sigma_epsilon_corr: float = 0.02


# Parameter indices for the sampler
# [Ne, log10_V_scale, phi_init, epsilon_corr, H0]
PARAM_NAMES = ["Ne", "log10_V_scale", "phi_init", "epsilon_corr", "H0"]
NDIM = 5


def generate_simulated_data(config: Sim16BConfig, rng: np.random.Generator) -> Dict[str, Any]:
    """Generate simulated cosmological data based on true cosmology.

    Args:
        config: Simulation configuration
        rng: Random number generator

    Returns:
        Dictionary with simulated data
    """
    # CMB acoustic scale (observed value)
    # theta_s ~ 1.04110 in radians * 100
    theta_s_obs = 1.04110 + rng.normal(0, config.sigma_theta_s)

    # BAO data at z = [0.38, 0.51, 0.70]
    z_bao = np.array([0.38, 0.51, 0.70])

    # Compute true comoving distance at each BAO redshift
    # D_M(z) = (c/H0) * integral_0^z dz'/E(z')
    def E_of_z(z):
        Omega_L = 1 - config.Omega_m_true
        return np.sqrt(config.Omega_m_true * (1+z)**3 + Omega_L)

    from scipy.integrate import quad
    c_over_H0 = 299792.458 / config.H0_true  # Mpc

    DM_true = []
    for z in z_bao:
        integral, _ = quad(lambda zp: 1/E_of_z(zp), 0, z)
        DM_true.append(c_over_H0 * integral)
    DM_true = np.array(DM_true)

    # Add noise
    DM_obs = DM_true * (1 + rng.normal(0, config.sigma_DM_BAO, len(z_bao)))
    DM_err = DM_true * config.sigma_DM_BAO

    # SN data at log-spaced redshifts
    z_sn = np.logspace(np.log10(0.01), np.log10(1.5), config.n_sn_points)

    # Distance modulus mu = 5 * log10(d_L / 10 pc)
    # d_L = (1+z) * D_M
    dL_true = []
    for z in z_sn:
        integral, _ = quad(lambda zp: 1/E_of_z(zp), 0, z)
        dL_true.append((1+z) * c_over_H0 * integral)
    dL_true = np.array(dL_true)

    mu_true = 5 * np.log10(dL_true) + 25  # Mpc to 10 pc
    mu_obs = mu_true + rng.normal(0, config.sigma_mu_SN, len(z_sn))
    mu_err = np.ones(len(z_sn)) * config.sigma_mu_SN

    return {
        "theta_s_obs": theta_s_obs,
        "sigma_theta_s": config.sigma_theta_s,
        "z_bao": z_bao,
        "DM_obs": DM_obs,
        "DM_err": DM_err,
        "z_sn": z_sn,
        "mu_obs": mu_obs,
        "mu_err": mu_err,
    }


def compute_model_predictions(
    theta: np.ndarray,
    z_bao: np.ndarray,
    z_sn: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute model predictions for given parameters.

    Args:
        theta: Parameter vector [Ne, log10_V_scale, phi_init, epsilon_corr, H0]
        z_bao: BAO redshifts
        z_sn: SN redshifts

    Returns:
        (theta_s_model, DM_model, mu_model)
    """
    Ne, log10_V_scale, phi_init, epsilon_corr, H0 = theta

    # Create primordial params
    prim_params = NoBoundaryParams(
        Ne=Ne,
        log10_V_scale=log10_V_scale,
        phi_init=phi_init,
        epsilon_corr=epsilon_corr
    )

    # Map to cosmological parameters
    # Note: For simplicity, we use the H0 parameter directly rather than
    # computing it from the sound horizon. This allows us to sample H0
    # and see how the no-boundary prior affects its posterior.
    Omega_m = 0.315  # Fixed for this analysis
    Omega_L = 1 - Omega_m

    c_over_H0 = 299792.458 / H0  # Mpc

    # Sound horizon is modified by epsilon_corr
    # delta_r_s / r_s ~ -epsilon_corr * (high-z fraction)
    delta_rs_frac = compute_sound_horizon_epsilon_effect(epsilon_corr)

    # Reference sound horizon (Planck LCDM)
    r_s_ref = 147.09  # Mpc
    r_s = r_s_ref * (1 + delta_rs_frac)

    # Angular diameter distance to last scattering
    from scipy.integrate import quad
    def E_of_z(z):
        return np.sqrt(Omega_m * (1+z)**3 + Omega_L)

    z_star = 1089.80
    integral_star, _ = quad(lambda zp: 1/E_of_z(zp), 0, z_star)
    D_A_star = c_over_H0 * integral_star / (1 + z_star)

    # Acoustic scale
    theta_s_model = r_s / D_A_star * 100  # in radians * 100

    # BAO distances
    DM_model = []
    for z in z_bao:
        integral, _ = quad(lambda zp: 1/E_of_z(zp), 0, z)
        DM_model.append(c_over_H0 * integral)
    DM_model = np.array(DM_model)

    # SN distances
    dL_model = []
    for z in z_sn:
        integral, _ = quad(lambda zp: 1/E_of_z(zp), 0, z)
        dL_model.append((1+z) * c_over_H0 * integral)
    dL_model = np.array(dL_model)
    mu_model = 5 * np.log10(dL_model) + 25

    return theta_s_model, DM_model, mu_model


def log_likelihood(
    theta: np.ndarray,
    data: Dict[str, Any],
) -> float:
    """Compute log-likelihood for the data given parameters.

    Args:
        theta: Parameter vector
        data: Simulated data dictionary

    Returns:
        Log-likelihood value
    """
    try:
        theta_s_model, DM_model, mu_model = compute_model_predictions(
            theta, data["z_bao"], data["z_sn"]
        )

        # CMB chi^2
        chi2_cmb = ((theta_s_model - data["theta_s_obs"]) / data["sigma_theta_s"])**2

        # BAO chi^2
        chi2_bao = np.sum(((DM_model - data["DM_obs"]) / data["DM_err"])**2)

        # SN chi^2
        chi2_sn = np.sum(((mu_model - data["mu_obs"]) / data["mu_err"])**2)

        return -0.5 * (chi2_cmb + chi2_bao + chi2_sn)

    except Exception:
        return -np.inf


def log_prior(theta: np.ndarray, hyper: NoBoundaryHyperparams) -> float:
    """Compute log-prior for the parameters.

    Args:
        theta: Parameter vector [Ne, log10_V_scale, phi_init, epsilon_corr, H0]
        hyper: No-boundary hyperparameters

    Returns:
        Log-prior probability
    """
    Ne, log10_V_scale, phi_init, epsilon_corr, H0 = theta

    # H0 prior: Uniform [50, 100]
    if H0 < 50 or H0 > 100:
        return -np.inf

    # No-boundary prior on primordial parameters
    prim_params = NoBoundaryParams(
        Ne=Ne,
        log10_V_scale=log10_V_scale,
        phi_init=phi_init,
        epsilon_corr=epsilon_corr
    )

    log_p_primordial = log_prior_no_boundary(prim_params, hyper)

    if not np.isfinite(log_p_primordial):
        return -np.inf

    # Flat prior on H0 within bounds
    log_p_H0 = -np.log(100 - 50)  # Uniform normalization

    return log_p_primordial + log_p_H0


def log_posterior(
    theta: np.ndarray,
    hyper: NoBoundaryHyperparams,
    data: Dict[str, Any],
) -> float:
    """Compute log-posterior = log-prior + log-likelihood.

    Args:
        theta: Parameter vector
        hyper: No-boundary hyperparameters
        data: Simulated data

    Returns:
        Log-posterior probability
    """
    lp = log_prior(theta, hyper)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, data)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def run_mcmc(config: Sim16BConfig) -> Dict[str, Any]:
    """Run MCMC sampling with no-boundary prior.

    Args:
        config: Simulation configuration

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("    SIMULATION 16B: No-Boundary Prior MCMC")
    print("=" * 70)
    print()

    rng = np.random.default_rng(config.seed)

    # Create hyperparameters
    hyper = NoBoundaryHyperparams(
        alpha_Ne=config.alpha_Ne,
        Ne_min=config.Ne_min,
        Ne_max=config.Ne_max,
        mu_logV=config.mu_logV,
        sigma_logV=config.sigma_logV,
        mu_phi_init=config.mu_phi_init,
        sigma_phi_init=config.sigma_phi_init,
        sigma_epsilon_corr=config.sigma_epsilon_corr,
    )

    print("Configuration:")
    print(f"  True H0: {config.H0_true} km/s/Mpc")
    print(f"  MCMC: {config.n_walkers} walkers, {config.n_steps} steps")
    print(f"  Burn-in: {config.n_burn} steps")
    print()

    # Generate simulated data
    print("Generating simulated cosmological data...")
    data = generate_simulated_data(config, rng)
    print(f"  CMB theta_s: {data['theta_s_obs']:.5f}")
    print(f"  BAO points: {len(data['z_bao'])}")
    print(f"  SN points: {len(data['z_sn'])}")
    print()

    # Initialize walkers
    print("Initializing walkers...")
    initial_pos = []
    for _ in range(config.n_walkers):
        # Random starting position within priors
        Ne = rng.uniform(hyper.Ne_min + 5, hyper.Ne_max - 5)
        log10_V_scale = rng.normal(hyper.mu_logV, hyper.sigma_logV * 0.5)
        phi_init = rng.uniform(0.05, 0.5)
        epsilon_corr = rng.normal(0, hyper.sigma_epsilon_corr * 0.5)
        H0 = rng.uniform(60, 75)
        initial_pos.append([Ne, log10_V_scale, phi_init, epsilon_corr, H0])
    initial_pos = np.array(initial_pos)

    # Set up sampler
    sampler = emcee.EnsembleSampler(
        config.n_walkers,
        NDIM,
        log_posterior,
        args=(hyper, data),
    )

    # Run MCMC
    print(f"Running MCMC ({config.n_steps} steps)...")
    for sample in tqdm(sampler.sample(initial_pos, iterations=config.n_steps),
                       total=config.n_steps, desc="MCMC"):
        pass
    print()

    # Get chain
    chain = sampler.get_chain()
    flat_chain = sampler.get_chain(discard=config.n_burn, flat=True)

    print(f"Chain shape: {chain.shape}")
    print(f"Flat chain shape: {flat_chain.shape}")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print()

    # Extract H0 samples
    H0_samples = flat_chain[:, 4]  # H0 is index 4
    epsilon_samples = flat_chain[:, 3]  # epsilon_corr is index 3
    Ne_samples = flat_chain[:, 0]

    # Compute statistics
    H0_mean = np.mean(H0_samples)
    H0_std = np.std(H0_samples)
    H0_median = np.median(H0_samples)
    H0_16, H0_84 = np.percentile(H0_samples, [16, 84])
    H0_2_5, H0_97_5 = np.percentile(H0_samples, [2.5, 97.5])

    P_H0_ge_73 = np.mean(H0_samples >= 73.0)
    P_H0_ge_70 = np.mean(H0_samples >= 70.0)

    print("-" * 70)
    print("H0 POSTERIOR")
    print("-" * 70)
    print(f"  Mean: {H0_mean:.3f} km/s/Mpc")
    print(f"  Std:  {H0_std:.3f} km/s/Mpc")
    print(f"  Median: {H0_median:.3f} km/s/Mpc")
    print(f"  68% CI: [{H0_16:.3f}, {H0_84:.3f}] km/s/Mpc")
    print(f"  95% CI: [{H0_2_5:.3f}, {H0_97_5:.3f}] km/s/Mpc")
    print()

    print("-" * 70)
    print("KEY PROBABILITIES")
    print("-" * 70)
    print(f"  P(H0 >= 73 | data, no-boundary prior) = {P_H0_ge_73:.6f}")
    print(f"  P(H0 >= 70 | data, no-boundary prior) = {P_H0_ge_70:.6f}")
    print()

    # Compare with SIM 15
    P_H0_ge_73_sim15 = 0.00268  # From SIM 15 results
    print("-" * 70)
    print("COMPARISON WITH SIM 15 (LCDM + Flat Priors)")
    print("-" * 70)
    print(f"  SIM 15 P(H0 >= 73) = {P_H0_ge_73_sim15:.6f}")
    print(f"  SIM 16B P(H0 >= 73) = {P_H0_ge_73:.6f}")
    ratio = P_H0_ge_73 / P_H0_ge_73_sim15 if P_H0_ge_73_sim15 > 0 else np.inf
    print(f"  Ratio: {ratio:.2f}x")
    print()

    # Correlation with epsilon_corr
    corr_H0_eps = np.corrcoef(H0_samples, epsilon_samples)[0, 1]
    print(f"  Correlation(H0, epsilon_corr) = {corr_H0_eps:.4f}")

    # Nuisance parameter posteriors
    param_stats = {}
    for i, name in enumerate(PARAM_NAMES):
        samples = flat_chain[:, i]
        param_stats[name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "16th": float(np.percentile(samples, 16)),
            "84th": float(np.percentile(samples, 84)),
        }

    print("\n" + "-" * 70)
    print("PARAMETER POSTERIORS")
    print("-" * 70)
    for name, stats in param_stats.items():
        print(f"  {name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
1. DATA RECOVERY:
   True H0: {config.H0_true} km/s/Mpc
   Recovered H0: {H0_mean:.2f} +/- {H0_std:.2f} km/s/Mpc
   Bias: {H0_mean - config.H0_true:+.2f} km/s/Mpc

2. HUBBLE TENSION:
   P(H0 >= 73 | data, no-boundary) = {P_H0_ge_73:.6f} ({100*P_H0_ge_73:.3f}%)

   {'The no-boundary prior does NOT resolve the Hubble tension.' if P_H0_ge_73 < 0.05 else 'The no-boundary prior provides some support for higher H0.'}

3. EFFECT OF EPSILON_CORR:
   Correlation with H0: {corr_H0_eps:.3f}
   {'epsilon_corr can shift H0, but not enough to reach 73 km/s/Mpc.' if corr_H0_eps > 0.1 and P_H0_ge_73 < 0.05 else 'Limited effect on H0 posterior.'}

4. COMPARISON WITH SIM 15:
   SIM 16B / SIM 15 ratio: {ratio:.2f}x
   {'No significant difference from LCDM analysis.' if 0.5 < ratio < 2.0 else 'The no-boundary prior has a measurable effect.'}
""")

    # Compile results
    results = {
        "config": {
            "n_walkers": config.n_walkers,
            "n_steps": config.n_steps,
            "n_burn": config.n_burn,
            "seed": config.seed,
            "H0_true": config.H0_true,
            "sigma_epsilon_corr": config.sigma_epsilon_corr,
        },
        "H0": {
            "mean": float(H0_mean),
            "std": float(H0_std),
            "median": float(H0_median),
            "16th": float(H0_16),
            "84th": float(H0_84),
            "2.5th": float(H0_2_5),
            "97.5th": float(H0_97_5),
        },
        "P_H0_ge_73": float(P_H0_ge_73),
        "P_H0_ge_70": float(P_H0_ge_70),
        "P_H0_ge_73_sim15": P_H0_ge_73_sim15,
        "ratio_to_sim15": float(ratio),
        "correlations": {
            "H0_epsilon_corr": float(corr_H0_eps),
        },
        "parameters": param_stats,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }

    return results, chain, flat_chain


def main():
    parser = argparse.ArgumentParser(
        description="SIM 16B: No-Boundary Prior MCMC"
    )
    parser.add_argument(
        "--n-walkers", type=int, default=32,
        help="Number of MCMC walkers (default: 32)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=3000,
        help="Number of MCMC steps (default: 3000)"
    )
    parser.add_argument(
        "--n-burn", type=int, default=500,
        help="Number of burn-in steps (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=20241210,
        help="Random seed (default: 20241210)"
    )
    parser.add_argument(
        "--sigma-epsilon", type=float, default=0.02,
        help="Prior width on epsilon_corr (default: 0.02)"
    )
    args = parser.parse_args()

    config = Sim16BConfig(
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        seed=args.seed,
        sigma_epsilon_corr=args.sigma_epsilon,
    )

    # Create output directory
    output_dir = Path("results/simulation_16b_noboundary_mcmc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run MCMC
    results, chain, flat_chain = run_mcmc(config)

    # Save results
    results_file = output_dir / "mcmc_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Save chains
    np.save(output_dir / "chain.npy", chain)
    np.save(output_dir / "flat_chain.npy", flat_chain)
    print(f"Saved chains to {output_dir}")

    print("\n" + "=" * 70)
    print("SIMULATION 16B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
