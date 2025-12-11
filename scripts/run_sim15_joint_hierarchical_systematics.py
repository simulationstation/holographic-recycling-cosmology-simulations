#!/usr/bin/env python3
"""
SIMULATION 15: Joint Hierarchical Systematics and H₀ Inference (JHS-H0)

This simulation answers: "Given realistic priors on all known systematics,
what is P(H₀ ≥ 73 km/s/Mpc | data, ΛCDM, systematics)?"

Method:
1. Generate synthetic SH0ES-like ladder data with true H0 = 67.5 km/s/Mpc
2. Run MCMC with emcee to sample P(H0, systematics | data, priors)
3. Marginalize over systematics to get P(H0 | data)
4. Compute P(H0 ≥ 73 | data, systematics)

Parameters (10 total):
- H0: Hubble constant [50, 90] km/s/Mpc (uniform prior)
- alpha_pop: Population drift coefficient ~ N(0, 0.05)
- gamma_Z: Metallicity dependence ~ N(0, 0.05)
- delta_M_step: Host mass step bias ~ N(0, 0.03)
- delta_beta: Color-law mismatch ~ N(0, 0.3)
- delta_M_W0: Cepheid PL zero-point bias ~ N(0, 0.03)
- delta_mu_anchor: Anchor distance bias ~ N(0, 0.02)
- delta_mu_crowd: Crowding bias ~ N(0, 0.03)
- delta_ZP_inst: Instrument zero-point offset ~ N(0, 0.02)
- delta_color_inst: Instrument color term ~ N(0, 0.02)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import emcee

from hrc2.ladder import (
    JointSystematicsPriors,
    JointSystematicsParameters,
    theta_to_params,
    params_to_theta,
    joint_log_prior,
    joint_sample_prior,
    JOINT_PARAM_NAMES,
    JOINT_NDIM,
    SyntheticLadderData,
    generate_synthetic_ladder_data,
    joint_log_posterior,
    test_log_posterior,
    find_best_fit_H0_no_systematics,
)
from hrc2.ladder.joint_systematics_model import (
    get_initial_theta,
    get_walker_initialization_scales,
)


# =============================================================================
# Configuration
# =============================================================================

# True cosmology
H0_TRUE = 67.5
OMEGA_M = 0.315

# MCMC parameters
N_WALKERS = 40
N_STEPS = 5000
N_BURN = 1000
SEED = 20241215

# Data generation parameters
N_CALIB = 40
N_FLOW = 200
SIGMA_M_B = 0.13
SIGMA_MU_CALIB = 0.05


def log_posterior_wrapper(theta, data, priors):
    """Wrapper for log_posterior compatible with emcee."""
    return joint_log_posterior(theta, data, priors)


def run_mcmc(
    data: SyntheticLadderData,
    priors: JointSystematicsPriors,
    n_walkers: int = N_WALKERS,
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> emcee.EnsembleSampler:
    """
    Run MCMC sampling with emcee.

    Returns:
        emcee.EnsembleSampler with chain
    """
    rng = np.random.default_rng(seed)

    # Find best-fit H0 to initialize walkers
    print("Finding best-fit H0 for initialization...")
    H0_init, chi2_init = find_best_fit_H0_no_systematics(data)
    print(f"  Best-fit H0 (no systematics): {H0_init:.2f} km/s/Mpc, chi2={chi2_init:.1f}")

    # Initialize walkers
    print(f"\nInitializing {n_walkers} walkers...")
    initial_theta = get_initial_theta(H0_init=H0_init)
    scales = get_walker_initialization_scales()

    # Scatter walkers around initial point
    pos = initial_theta + scales * rng.normal(size=(n_walkers, JOINT_NDIM))

    # Ensure all walkers start in valid prior region
    for i in range(n_walkers):
        # Clip H0 to prior bounds
        pos[i, 0] = np.clip(pos[i, 0], priors.H0_min + 0.1, priors.H0_max - 0.1)

    # Test log_posterior at initial position
    test_lp = joint_log_posterior(initial_theta, data, priors)
    print(f"  Initial log-posterior: {test_lp:.1f}")

    # Create sampler
    print(f"\nStarting MCMC: {n_steps} steps, {n_walkers} walkers...")
    sampler = emcee.EnsembleSampler(
        n_walkers,
        JOINT_NDIM,
        log_posterior_wrapper,
        args=(data, priors),
    )

    # Run MCMC with progress tracking
    start_time = time.time()

    for i, sample in enumerate(sampler.sample(pos, iterations=n_steps, progress=True)):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_steps - i - 1) / rate
            print(f"  Step {i+1}/{n_steps}, "
                  f"acceptance: {np.mean(sampler.acceptance_fraction):.2%}, "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\nMCMC completed in {elapsed:.1f}s")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.2%}")

    return sampler


def analyze_chain(
    sampler: emcee.EnsembleSampler,
    n_burn: int = N_BURN,
) -> Dict[str, Any]:
    """
    Analyze MCMC chain and extract statistics.

    Returns:
        Dict with posterior statistics
    """
    print(f"\nAnalyzing chain (discarding {n_burn} burn-in steps)...")

    # Get flat chain after burn-in
    chain = sampler.get_chain(discard=n_burn, flat=True)
    print(f"  Chain shape: {chain.shape}")

    # Extract H0 posterior
    H0_samples = chain[:, 0]
    n_samples = len(H0_samples)

    # H0 statistics
    H0_mean = np.mean(H0_samples)
    H0_std = np.std(H0_samples)
    H0_median = np.median(H0_samples)
    H0_16 = np.percentile(H0_samples, 16)
    H0_84 = np.percentile(H0_samples, 84)
    H0_2p5 = np.percentile(H0_samples, 2.5)
    H0_97p5 = np.percentile(H0_samples, 97.5)

    # Key probability: P(H0 >= 73)
    P_H0_ge_73 = np.mean(H0_samples >= 73.0)

    # Also compute P(H0 >= 70)
    P_H0_ge_70 = np.mean(H0_samples >= 70.0)

    # Nuisance parameter statistics
    nuisance_stats = {}
    for i, name in enumerate(JOINT_PARAM_NAMES):
        if name == "H0":
            continue
        samples = chain[:, i]
        nuisance_stats[name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "16th": float(np.percentile(samples, 16)),
            "84th": float(np.percentile(samples, 84)),
        }

    results = {
        "n_samples": n_samples,
        "H0": {
            "mean": float(H0_mean),
            "std": float(H0_std),
            "median": float(H0_median),
            "16th": float(H0_16),
            "84th": float(H0_84),
            "2.5th": float(H0_2p5),
            "97.5th": float(H0_97p5),
        },
        "P_H0_ge_73": float(P_H0_ge_73),
        "P_H0_ge_70": float(P_H0_ge_70),
        "nuisance_params": nuisance_stats,
    }

    return results


def main():
    """Run full SIM 15: Joint Hierarchical Systematics MCMC."""
    print("=" * 70)
    print("SIMULATION 15: Joint Hierarchical Systematics and H₀ Inference")
    print("=" * 70)
    print()
    print(f"True H0: {H0_TRUE} km/s/Mpc")
    print(f"MCMC: {N_WALKERS} walkers, {N_STEPS} steps, {N_BURN} burn-in")
    print(f"Parameters: {JOINT_NDIM} ({', '.join(JOINT_PARAM_NAMES)})")
    print()

    # Create output directory
    output_dir = Path("results/simulation_15_joint_systematics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create priors
    priors = JointSystematicsPriors()

    print("-" * 70)
    print("PRIOR WIDTHS")
    print("-" * 70)
    print(f"  H0: Uniform[{priors.H0_min}, {priors.H0_max}]")
    print(f"  alpha_pop: N(0, {priors.sigma_alpha_pop})")
    print(f"  gamma_Z: N(0, {priors.sigma_gamma_Z})")
    print(f"  delta_M_step: N(0, {priors.sigma_delta_M_step})")
    print(f"  delta_beta: N(0, {priors.sigma_delta_beta})")
    print(f"  delta_M_W0: N(0, {priors.sigma_delta_M_W0})")
    print(f"  delta_mu_anchor: N(0, {priors.sigma_delta_mu_anchor})")
    print(f"  delta_mu_crowd: N(0, {priors.sigma_delta_mu_crowd})")
    print(f"  delta_ZP_inst: N(0, {priors.sigma_delta_ZP_inst})")
    print(f"  delta_color_inst: N(0, {priors.sigma_delta_color_inst})")
    print()

    # Generate synthetic data
    print("-" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("-" * 70)
    data = generate_synthetic_ladder_data(
        H0_true=H0_TRUE,
        Omega_m=OMEGA_M,
        N_calib=N_CALIB,
        N_flow=N_FLOW,
        sigma_m_B=SIGMA_M_B,
        sigma_mu_calib=SIGMA_MU_CALIB,
        seed=SEED,
    )
    print(f"  Calibrator SNe: {len(data.calib_z)}")
    print(f"  Hubble flow SNe: {len(data.flow_z)}")
    print(f"  Flow z range: [{data.flow_z.min():.3f}, {data.flow_z.max():.3f}]")
    print()

    # Test log-posterior
    print("Testing log-posterior...")
    lp_test = test_log_posterior(data, priors, H0_test=H0_TRUE)
    print(f"  log-posterior at H0={H0_TRUE}: {lp_test:.1f}")
    print()

    # Run MCMC
    print("-" * 70)
    print("RUNNING MCMC")
    print("-" * 70)
    sampler = run_mcmc(data, priors, n_walkers=N_WALKERS, n_steps=N_STEPS, seed=SEED)

    # Analyze chain
    print("-" * 70)
    print("ANALYZING RESULTS")
    print("-" * 70)
    results = analyze_chain(sampler, n_burn=N_BURN)

    # Print summary
    print()
    print("=" * 70)
    print("                         FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"True H0: {H0_TRUE} km/s/Mpc")
    print()
    print(f"H0 posterior (marginalized over systematics):")
    print(f"  Mean:   {results['H0']['mean']:.2f} km/s/Mpc")
    print(f"  Median: {results['H0']['median']:.2f} km/s/Mpc")
    print(f"  Std:    {results['H0']['std']:.2f} km/s/Mpc")
    print(f"  68% CI: [{results['H0']['16th']:.2f}, {results['H0']['84th']:.2f}] km/s/Mpc")
    print(f"  95% CI: [{results['H0']['2.5th']:.2f}, {results['H0']['97.5th']:.2f}] km/s/Mpc")
    print()
    print("-" * 70)
    print("KEY PROBABILITIES")
    print("-" * 70)
    print()
    print(f"  P(H0 >= 73 km/s/Mpc | data, ΛCDM, systematics) = {results['P_H0_ge_73']:.4f}")
    print(f"  P(H0 >= 70 km/s/Mpc | data, ΛCDM, systematics) = {results['P_H0_ge_70']:.4f}")
    print()
    print("-" * 70)
    print("NUISANCE PARAMETER POSTERIORS")
    print("-" * 70)
    print()
    for name, stats in results["nuisance_params"].items():
        print(f"  {name}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    68% CI: [{stats['16th']:.4f}, {stats['84th']:.4f}]")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Compute delta H0
    delta_H0 = results["H0"]["mean"] - H0_TRUE
    print(f"Bias in H0 recovery: {delta_H0:+.2f} km/s/Mpc")

    # Hubble tension context
    H0_SH0ES = 73.0
    tension_sigma = (H0_SH0ES - results["H0"]["mean"]) / results["H0"]["std"]
    print(f"Distance from SH0ES value (73 km/s/Mpc): {tension_sigma:.1f}σ")
    print()

    if results["P_H0_ge_73"] < 0.01:
        print("CONCLUSION: Even with realistic systematics priors,")
        print(f"P(H0 >= 73) = {results['P_H0_ge_73']:.4f} is EXTREMELY LOW.")
        print("This means the Hubble tension CANNOT be explained by")
        print("known astrophysical and calibration systematics alone.")
    elif results["P_H0_ge_73"] < 0.05:
        print("CONCLUSION: P(H0 >= 73) is low but not negligible.")
        print("Systematics could contribute but are unlikely to fully")
        print("explain the tension.")
    else:
        print("CONCLUSION: P(H0 >= 73) is significant.")
        print("Systematics may play a non-trivial role in the tension.")
    print()

    # Save results
    full_results = {
        "config": {
            "H0_true": H0_TRUE,
            "Omega_m": OMEGA_M,
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burn": N_BURN,
            "n_calib": N_CALIB,
            "n_flow": N_FLOW,
            "seed": SEED,
        },
        "priors": {
            "H0_min": priors.H0_min,
            "H0_max": priors.H0_max,
            "sigma_alpha_pop": priors.sigma_alpha_pop,
            "sigma_gamma_Z": priors.sigma_gamma_Z,
            "sigma_delta_M_step": priors.sigma_delta_M_step,
            "sigma_delta_beta": priors.sigma_delta_beta,
            "sigma_delta_M_W0": priors.sigma_delta_M_W0,
            "sigma_delta_mu_anchor": priors.sigma_delta_mu_anchor,
            "sigma_delta_mu_crowd": priors.sigma_delta_mu_crowd,
            "sigma_delta_ZP_inst": priors.sigma_delta_ZP_inst,
            "sigma_delta_color_inst": priors.sigma_delta_color_inst,
        },
        "results": results,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }

    output_file = output_dir / "mcmc_results.json"
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved results to {output_file}")

    # Save chain for further analysis
    chain = sampler.get_chain(flat=False)
    np.save(output_dir / "chain.npy", chain)
    print(f"Saved chain to {output_dir / 'chain.npy'}")

    # Save flat chain (post burn-in)
    flat_chain = sampler.get_chain(discard=N_BURN, flat=True)
    np.save(output_dir / "flat_chain.npy", flat_chain)
    print(f"Saved flat chain to {output_dir / 'flat_chain.npy'}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
