#!/usr/bin/env python3
"""
Analyze SIMULATION 15: Joint Hierarchical Systematics Results

Reads the MCMC chain and produces:
1. H0 posterior distribution
2. Nuisance parameter posteriors
3. Correlation analysis
4. P(H0 >= 73) with confidence intervals
5. Summary statistics
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.ladder import JOINT_PARAM_NAMES, JOINT_NDIM


def compute_gelman_rubin(chain: np.ndarray) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat statistic for convergence.

    Args:
        chain: Shape (n_steps, n_walkers, n_dim)

    Returns:
        R-hat for each parameter
    """
    n_steps, n_walkers, n_dim = chain.shape

    # Use second half of chain
    samples = chain[n_steps // 2:]
    n = len(samples)
    m = n_walkers

    R_hat = np.zeros(n_dim)

    for i in range(n_dim):
        # Get samples for this parameter from each walker
        walker_means = np.mean(samples[:, :, i], axis=0)  # (n_walkers,)
        walker_vars = np.var(samples[:, :, i], axis=0, ddof=1)  # (n_walkers,)

        # Between-chain variance
        B = n * np.var(walker_means, ddof=1)

        # Within-chain variance
        W = np.mean(walker_vars)

        # Estimate of variance
        var_hat = (1 - 1/n) * W + (1/n) * B

        # R-hat
        R_hat[i] = np.sqrt(var_hat / W) if W > 0 else np.inf

    return R_hat


def compute_effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """
    Compute effective sample size for each parameter.

    Args:
        chain: Shape (n_samples, n_dim)

    Returns:
        ESS for each parameter
    """
    n_samples, n_dim = chain.shape
    ess = np.zeros(n_dim)

    for i in range(n_dim):
        samples = chain[:, i]

        # Simple autocorrelation estimate
        mean = np.mean(samples)
        var = np.var(samples)

        if var == 0:
            ess[i] = 0
            continue

        # Compute autocorrelation up to lag=500
        max_lag = min(500, n_samples // 2)
        rho = np.zeros(max_lag)

        for k in range(max_lag):
            cov = np.mean((samples[:-k-1] - mean) * (samples[k+1:] - mean))
            rho[k] = cov / var

        # Integrated autocorrelation time
        # Sum until first negative
        tau = 1.0
        for k in range(max_lag):
            if rho[k] < 0:
                break
            tau += 2 * rho[k]

        ess[i] = n_samples / tau

    return ess


def main():
    """Analyze SIM 15 MCMC results."""
    input_dir = Path("results/simulation_15_joint_systematics")
    results_file = input_dir / "mcmc_results.json"
    chain_file = input_dir / "chain.npy"
    flat_chain_file = input_dir / "flat_chain.npy"

    if not results_file.exists():
        print(f"ERROR: {results_file} not found. Run SIM 15 first.")
        return

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    print()
    print("=" * 70)
    print("         SIMULATION 15 ANALYSIS: Joint Hierarchical Systematics")
    print("=" * 70)
    print()

    # Configuration summary
    config = results["config"]
    print("-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"True H0: {config['H0_true']} km/s/Mpc")
    print(f"MCMC: {config['n_walkers']} walkers, {config['n_steps']} steps")
    print(f"Burn-in: {config['n_burn']} steps")
    print(f"Calibrator SNe: {config['n_calib']}")
    print(f"Hubble flow SNe: {config['n_flow']}")
    print()

    # Prior summary
    priors = results["priors"]
    print("-" * 70)
    print("PRIORS")
    print("-" * 70)
    print(f"H0: Uniform[{priors['H0_min']}, {priors['H0_max']}]")
    for key, val in priors.items():
        if key.startswith("sigma_"):
            param_name = key.replace("sigma_", "")
            print(f"{param_name}: N(0, {val})")
    print()

    # H0 posterior
    h0_stats = results["results"]["H0"]
    print("-" * 70)
    print("H0 POSTERIOR (marginalized over all systematics)")
    print("-" * 70)
    print(f"Mean:   {h0_stats['mean']:.3f} km/s/Mpc")
    print(f"Median: {h0_stats['median']:.3f} km/s/Mpc")
    print(f"Std:    {h0_stats['std']:.3f} km/s/Mpc")
    print(f"68% CI: [{h0_stats['16th']:.3f}, {h0_stats['84th']:.3f}] km/s/Mpc")
    print(f"95% CI: [{h0_stats['2.5th']:.3f}, {h0_stats['97.5th']:.3f}] km/s/Mpc")
    print()

    # Key probabilities
    P_73 = results["results"]["P_H0_ge_73"]
    P_70 = results["results"]["P_H0_ge_70"]

    print("-" * 70)
    print("KEY PROBABILITIES")
    print("-" * 70)
    print(f"P(H0 >= 73 km/s/Mpc | data, ΛCDM, systematics) = {P_73:.6f}")
    print(f"P(H0 >= 70 km/s/Mpc | data, ΛCDM, systematics) = {P_70:.6f}")
    print()

    # Sigma equivalent
    from scipy.special import erfinv
    if P_73 > 0 and P_73 < 1:
        sigma_equiv = np.sqrt(2) * erfinv(1 - 2 * P_73)
        print(f"H0=73 is at {sigma_equiv:.2f}σ from posterior mean")
    print()

    # Nuisance parameters
    nuisance = results["results"]["nuisance_params"]
    print("-" * 70)
    print("NUISANCE PARAMETER POSTERIORS")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Mean':>10} {'Std':>10} {'68% CI':>25}")
    print("-" * 70)
    for name, stats in nuisance.items():
        ci_str = f"[{stats['16th']:+.4f}, {stats['84th']:+.4f}]"
        print(f"{name:<20} {stats['mean']:>+10.4f} {stats['std']:>10.4f} {ci_str:>25}")
    print()

    # Load chain for additional analysis if available
    if chain_file.exists():
        print("-" * 70)
        print("CONVERGENCE DIAGNOSTICS")
        print("-" * 70)

        chain = np.load(chain_file)
        print(f"Chain shape: {chain.shape}")

        # Gelman-Rubin
        R_hat = compute_gelman_rubin(chain)
        print("\nGelman-Rubin R-hat:")
        for i, name in enumerate(JOINT_PARAM_NAMES):
            status = "OK" if R_hat[i] < 1.1 else "WARNING"
            print(f"  {name}: {R_hat[i]:.3f} [{status}]")

        max_R = np.max(R_hat)
        if max_R < 1.1:
            print(f"\nAll R-hat < 1.1: Chain is well-converged")
        else:
            print(f"\nWARNING: Max R-hat = {max_R:.3f}, chain may not be converged")
        print()

    if flat_chain_file.exists():
        flat_chain = np.load(flat_chain_file)
        print("-" * 70)
        print("EFFECTIVE SAMPLE SIZE")
        print("-" * 70)

        ess = compute_effective_sample_size(flat_chain)
        for i, name in enumerate(JOINT_PARAM_NAMES):
            print(f"  {name}: {ess[i]:.0f}")
        print(f"\nMinimum ESS: {np.min(ess):.0f}")
        print()

        # Correlation analysis
        print("-" * 70)
        print("PARAMETER CORRELATIONS WITH H0")
        print("-" * 70)
        H0_samples = flat_chain[:, 0]
        for i, name in enumerate(JOINT_PARAM_NAMES):
            if i == 0:
                continue
            corr = np.corrcoef(H0_samples, flat_chain[:, i])[0, 1]
            print(f"  Corr(H0, {name}): {corr:+.3f}")
        print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    H0_true = config["H0_true"]
    H0_SH0ES = 73.0
    H0_mean = h0_stats["mean"]
    H0_std = h0_stats["std"]

    print(f"1. DATA RECOVERY:")
    print(f"   True H0: {H0_true} km/s/Mpc")
    print(f"   Recovered H0: {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
    bias = H0_mean - H0_true
    print(f"   Bias: {bias:+.2f} km/s/Mpc ({bias/H0_std:.1f}σ)")
    print()

    print(f"2. HUBBLE TENSION CONTEXT:")
    print(f"   SH0ES H0: {H0_SH0ES} km/s/Mpc")
    tension = (H0_SH0ES - H0_mean) / H0_std
    print(f"   Distance from SH0ES: {tension:.1f}σ")
    print()

    print(f"3. CAN SYSTEMATICS EXPLAIN THE TENSION?")
    print(f"   P(H0 >= 73 | data, ΛCDM, systematics) = {P_73:.6f}")
    print()

    if P_73 < 0.001:
        print("   CONCLUSION: NO")
        print("   Even with generous priors on all known systematics,")
        print("   the probability of reaching H0 = 73 km/s/Mpc is < 0.1%.")
        print("   This means ΛCDM + known systematics CANNOT explain")
        print("   the Hubble tension. New physics or unknown systematics")
        print("   would be required.")
    elif P_73 < 0.01:
        print("   CONCLUSION: VERY UNLIKELY")
        print("   The probability is low but not negligible.")
        print("   Systematics alone are unlikely to explain the full tension.")
    elif P_73 < 0.05:
        print("   CONCLUSION: UNLIKELY")
        print("   There's a small but non-trivial probability.")
        print("   Systematics may contribute but other factors are likely needed.")
    else:
        print("   CONCLUSION: POSSIBLE")
        print("   Systematics could potentially explain a significant portion")
        print("   of the Hubble tension.")
    print()

    # Save extended summary
    summary = {
        "H0_posterior": {
            "mean": H0_mean,
            "std": H0_std,
            "median": h0_stats["median"],
            "68_CI": [h0_stats["16th"], h0_stats["84th"]],
            "95_CI": [h0_stats["2.5th"], h0_stats["97.5th"]],
        },
        "P_H0_ge_73": P_73,
        "P_H0_ge_70": P_70,
        "bias": bias,
        "tension_with_SH0ES_sigma": tension,
        "interpretation": {
            "can_systematics_explain_tension": P_73 >= 0.05,
            "conclusion": "POSSIBLE" if P_73 >= 0.05 else (
                "UNLIKELY" if P_73 >= 0.01 else (
                    "VERY UNLIKELY" if P_73 >= 0.001 else "NO"
                )
            )
        }
    }

    summary_file = input_dir / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
