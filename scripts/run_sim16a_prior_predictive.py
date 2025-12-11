#!/usr/bin/env python3
"""
SIMULATION 16A: Prior Predictive Sampling from Hawking-Hartle No-Boundary Prior

This simulation samples from the no-boundary prior over primordial parameters
and maps each sample to cosmological parameters (H0, Omega_m, etc.), generating
a prior predictive distribution over observables.

Key questions addressed:
1. What is the prior distribution over H0 under the Hawking-Hartle framework?
2. What fraction of prior samples have H0 >= 73 km/s/Mpc?
3. How does epsilon_corr affect the inferred H0?

Usage:
    python scripts/run_sim16a_prior_predictive.py [--n-samples 1000] [--seed 20241210]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.noboundary import (
    NoBoundaryHyperparams,
    NoBoundaryParams,
    sample_no_boundary_prior,
    primordial_to_cosmo,
    CosmoParams,
    compute_early_H0,
    compute_late_H0,
)


@dataclass
class Sim16AConfig:
    """Configuration for SIM 16A."""
    n_samples: int = 1000
    seed: int = 20241210

    # Hyperparameters for no-boundary prior
    alpha_Ne: float = 0.05
    Ne_min: float = 50.0
    Ne_max: float = 80.0
    mu_logV: float = -10.0
    sigma_logV: float = 1.0
    mu_phi_init: float = 0.1
    sigma_phi_init: float = 0.3
    sigma_epsilon_corr: float = 0.02

    # Fiducial cosmology
    Omega_b: float = 0.0493
    Omega_m_base: float = 0.315
    H0_base: float = 67.4


def run_prior_predictive(config: Sim16AConfig) -> Dict[str, Any]:
    """
    Run prior predictive sampling from no-boundary prior.

    Args:
        config: Simulation configuration

    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("    SIMULATION 16A: No-Boundary Prior Predictive Sampling")
    print("=" * 70)
    print()

    # Set up random generator
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

    print("Prior hyperparameters:")
    print(f"  N_e: exp-weighted in [{hyper.Ne_min}, {hyper.Ne_max}], alpha={hyper.alpha_Ne}")
    print(f"  log10(V_scale): N({hyper.mu_logV}, {hyper.sigma_logV})")
    print(f"  phi_init: N({hyper.mu_phi_init}, {hyper.sigma_phi_init})")
    print(f"  epsilon_corr: N(0, {hyper.sigma_epsilon_corr})")
    print()

    # Sample from prior
    print(f"Sampling {config.n_samples} universes from no-boundary prior...")
    primordial_samples = sample_no_boundary_prior(hyper, config.n_samples, rng)

    # Map to cosmological parameters
    print("Mapping primordial parameters to cosmological observables...")
    cosmo_samples: List[CosmoParams] = []
    valid_count = 0

    for i, prim in enumerate(tqdm(primordial_samples, desc="Mapping to cosmo")):
        try:
            cosmo = primordial_to_cosmo(
                prim,
                Omega_b=config.Omega_b,
                Omega_m_base=config.Omega_m_base,
                H0_base=config.H0_base,
            )
            cosmo_samples.append(cosmo)
            valid_count += 1
        except Exception as e:
            # Skip samples that fail (e.g., unphysical parameters)
            continue

    print(f"  Valid samples: {valid_count}/{config.n_samples} ({100*valid_count/config.n_samples:.1f}%)")
    print()

    # Extract distributions
    H0_samples = np.array([c.H0 for c in cosmo_samples])
    Omega_m_samples = np.array([c.Omega_m for c in cosmo_samples])
    Omega_k_samples = np.array([c.Omega_k for c in cosmo_samples])
    n_s_samples = np.array([c.n_s for c in cosmo_samples])
    epsilon_samples = np.array([c.epsilon_corr for c in cosmo_samples])
    r_s_samples = np.array([c.r_s for c in cosmo_samples])
    theta_s_samples = np.array([c.theta_s for c in cosmo_samples])

    # Primordial parameter distributions
    Ne_samples = np.array([p.Ne for p in primordial_samples[:valid_count]])
    logV_samples = np.array([p.log10_V_scale for p in primordial_samples[:valid_count]])
    phi_samples = np.array([p.phi_init for p in primordial_samples[:valid_count]])

    # Compute summary statistics
    print("-" * 70)
    print("PRIOR PREDICTIVE DISTRIBUTIONS")
    print("-" * 70)

    def print_stats(name: str, samples: np.ndarray, unit: str = ""):
        mean = np.mean(samples)
        std = np.std(samples)
        p16, p50, p84 = np.percentile(samples, [16, 50, 84])
        p2_5, p97_5 = np.percentile(samples, [2.5, 97.5])
        print(f"  {name}:")
        print(f"    Mean: {mean:.4f}{unit}")
        print(f"    Median: {p50:.4f}{unit}")
        print(f"    Std: {std:.4f}{unit}")
        print(f"    68% CI: [{p16:.4f}, {p84:.4f}]")
        print(f"    95% CI: [{p2_5:.4f}, {p97_5:.4f}]")
        return {
            "mean": float(mean),
            "std": float(std),
            "median": float(p50),
            "16th": float(p16),
            "84th": float(p84),
            "2.5th": float(p2_5),
            "97.5th": float(p97_5),
        }

    print("\nPrimordial parameters:")
    Ne_stats = print_stats("N_e (e-folds)", Ne_samples)
    logV_stats = print_stats("log10(V_scale)", logV_samples)
    phi_stats = print_stats("phi_init", phi_samples)
    epsilon_stats = print_stats("epsilon_corr", epsilon_samples)

    print("\nCosmological parameters:")
    H0_stats = print_stats("H0", H0_samples, " km/s/Mpc")
    Omega_m_stats = print_stats("Omega_m", Omega_m_samples)
    Omega_k_stats = print_stats("Omega_k", Omega_k_samples)
    n_s_stats = print_stats("n_s", n_s_samples)
    r_s_stats = print_stats("r_s", r_s_samples, " Mpc")
    theta_s_stats = print_stats("100*theta_s", theta_s_samples)

    # Key probabilities
    print("\n" + "-" * 70)
    print("KEY PROBABILITIES")
    print("-" * 70)

    P_H0_ge_73 = np.mean(H0_samples >= 73.0)
    P_H0_ge_70 = np.mean(H0_samples >= 70.0)
    P_H0_le_65 = np.mean(H0_samples <= 65.0)

    print(f"  P(H0 >= 73) = {P_H0_ge_73:.6f} ({100*P_H0_ge_73:.3f}%)")
    print(f"  P(H0 >= 70) = {P_H0_ge_70:.6f} ({100*P_H0_ge_70:.3f}%)")
    print(f"  P(H0 <= 65) = {P_H0_le_65:.6f} ({100*P_H0_le_65:.3f}%)")

    # Correlation with epsilon_corr
    corr_H0_epsilon = np.corrcoef(H0_samples, epsilon_samples)[0, 1]
    print(f"\n  Correlation(H0, epsilon_corr) = {corr_H0_epsilon:.4f}")

    # Effect of epsilon_corr on H0
    epsilon_positive = epsilon_samples > 0
    epsilon_negative = epsilon_samples < 0

    if np.sum(epsilon_positive) > 10 and np.sum(epsilon_negative) > 10:
        H0_pos_mean = np.mean(H0_samples[epsilon_positive])
        H0_neg_mean = np.mean(H0_samples[epsilon_negative])
        print(f"\n  Mean H0 when epsilon_corr > 0: {H0_pos_mean:.2f} km/s/Mpc")
        print(f"  Mean H0 when epsilon_corr < 0: {H0_neg_mean:.2f} km/s/Mpc")
        print(f"  Difference: {H0_pos_mean - H0_neg_mean:.2f} km/s/Mpc")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"""
Under the Hawking-Hartle no-boundary prior with conservative hyperparameters:

1. H0 PRIOR DISTRIBUTION:
   Mean: {H0_stats['mean']:.2f} km/s/Mpc
   Std:  {H0_stats['std']:.2f} km/s/Mpc
   The prior is centered near the Planck value of 67.4 km/s/Mpc.

2. HUBBLE TENSION IMPLICATIONS:
   P(H0 >= 73 | no-boundary prior) = {P_H0_ge_73:.4f} ({100*P_H0_ge_73:.2f}%)

   This is the probability of obtaining H0 >= 73 km/s/Mpc purely from
   the no-boundary prior, BEFORE incorporating any data.

   Compare to SIM 15 result: P(H0 >= 73 | data, LCDM) ~ 0.27%

3. EPSILON_CORR EFFECT:
   Correlation with H0: {corr_H0_epsilon:.3f}

   {'Positive epsilon_corr (larger H at high-z) tends to increase inferred H0.' if corr_H0_epsilon > 0.1 else 'epsilon_corr has limited effect on H0 in this prior.'}

4. CONCLUSION:
   The no-boundary prior {'provides a mechanism for higher H0 values' if P_H0_ge_73 > 0.1 else 'does not naturally favor H0 >= 73'}.
""")

    # Compile results
    results = {
        "config": asdict(config) if hasattr(config, '__dict__') else {
            "n_samples": config.n_samples,
            "seed": config.seed,
            "alpha_Ne": config.alpha_Ne,
            "Ne_min": config.Ne_min,
            "Ne_max": config.Ne_max,
            "mu_logV": config.mu_logV,
            "sigma_logV": config.sigma_logV,
            "mu_phi_init": config.mu_phi_init,
            "sigma_phi_init": config.sigma_phi_init,
            "sigma_epsilon_corr": config.sigma_epsilon_corr,
        },
        "n_valid_samples": valid_count,
        "primordial": {
            "Ne": Ne_stats,
            "log10_V_scale": logV_stats,
            "phi_init": phi_stats,
            "epsilon_corr": epsilon_stats,
        },
        "cosmological": {
            "H0": H0_stats,
            "Omega_m": Omega_m_stats,
            "Omega_k": Omega_k_stats,
            "n_s": n_s_stats,
            "r_s": r_s_stats,
            "theta_s": theta_s_stats,
        },
        "probabilities": {
            "P_H0_ge_73": float(P_H0_ge_73),
            "P_H0_ge_70": float(P_H0_ge_70),
            "P_H0_le_65": float(P_H0_le_65),
        },
        "correlations": {
            "H0_epsilon_corr": float(corr_H0_epsilon),
        },
    }

    # Save samples for later analysis
    samples_data = {
        "primordial": {
            "Ne": Ne_samples.tolist(),
            "log10_V_scale": logV_samples.tolist(),
            "phi_init": phi_samples.tolist(),
            "epsilon_corr": epsilon_samples.tolist(),
        },
        "cosmological": {
            "H0": H0_samples.tolist(),
            "Omega_m": Omega_m_samples.tolist(),
            "Omega_k": Omega_k_samples.tolist(),
            "n_s": n_s_samples.tolist(),
            "r_s": r_s_samples.tolist(),
            "theta_s": theta_s_samples.tolist(),
        },
    }

    return results, samples_data


def main():
    parser = argparse.ArgumentParser(
        description="SIM 16A: No-Boundary Prior Predictive Sampling"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of universes to sample (default: 1000)"
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

    # Create config
    config = Sim16AConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        sigma_epsilon_corr=args.sigma_epsilon,
    )

    # Create output directory
    output_dir = Path("results/simulation_16a_prior_predictive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation
    results, samples = run_prior_predictive(config)

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    samples_file = output_dir / "samples.json"
    with open(samples_file, "w") as f:
        json.dump(samples, f)
    print(f"Saved samples to {samples_file}")

    # Save numpy arrays for efficient loading
    np.savez(
        output_dir / "samples.npz",
        H0=np.array(samples["cosmological"]["H0"]),
        Omega_m=np.array(samples["cosmological"]["Omega_m"]),
        Omega_k=np.array(samples["cosmological"]["Omega_k"]),
        n_s=np.array(samples["cosmological"]["n_s"]),
        r_s=np.array(samples["cosmological"]["r_s"]),
        Ne=np.array(samples["primordial"]["Ne"]),
        epsilon_corr=np.array(samples["primordial"]["epsilon_corr"]),
    )
    print(f"Saved numpy arrays to {output_dir / 'samples.npz'}")

    print("\n" + "=" * 70)
    print("SIMULATION 16A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
