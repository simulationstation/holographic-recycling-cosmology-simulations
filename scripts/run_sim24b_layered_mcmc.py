#!/usr/bin/env python3
"""
SIMULATION 24B: Layered Expansion (Bent-Deck) MCMC Inference

This script performs MCMC sampling to infer the posterior distribution of
the effective H0 under the layered expansion model, given CMB+BAO+SN data.

The parameterization includes:
- Baseline cosmological parameters (Omega_m)
- Layered expansion deviations (delta_nodes array)

The smoothness prior penalizes sharp kinks in the expansion history.

Key outputs:
- Posterior distribution of H0_eff
- P(H0_eff >= 73 | data, layered model)
- Credible intervals and convergence diagnostics
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from scipy.stats import norm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.layered import (
    LayeredExpansionHyperparams,
    LayeredExpansionParams,
    LCDMBackground,
    make_default_nodes,
    make_random_params,
    make_zero_params,
    log_smoothness_prior,
    H_of_z_layered,
    get_H0_effective,
    check_physical_validity,
    compute_chi2_cmb_bao_sn,
    compute_chi2_cmb_layered,
    compute_chi2_bao_layered,
    compute_chi2_sn_layered,
)


@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling."""
    # Model structure
    n_layers: int
    smooth_sigma: float
    mode: str  # "delta_H" or "delta_w"

    # Priors
    Omega_m_prior_mean: float
    Omega_m_prior_std: float
    delta_prior_std: float  # Prior width on each delta_i

    # Sampling
    n_walkers: int
    n_steps: int
    n_burn: int
    thin: int

    # Options
    include_shoes: bool
    seed: int


@dataclass
class MCMCResult:
    """Results from MCMC sampling."""
    # Posterior summaries for H0_eff
    H0_eff_mean: float
    H0_eff_median: float
    H0_eff_std: float
    H0_eff_q16: float  # 16th percentile
    H0_eff_q84: float  # 84th percentile
    H0_eff_q2p5: float  # 2.5th percentile
    H0_eff_q97p5: float  # 97.5th percentile

    # Probability of high H0
    prob_H0_ge_73: float
    prob_H0_ge_71: float
    prob_H0_ge_70: float

    # Convergence
    acceptance_fraction: float
    n_effective_samples: int

    # Best-fit
    best_fit_H0_eff: float
    best_fit_chi2: float
    best_fit_Omega_m: float
    best_fit_delta_nodes: List[float]


def create_log_posterior(
    hyp: LayeredExpansionHyperparams,
    z_nodes: np.ndarray,
    config: MCMCConfig,
) -> Callable[[np.ndarray], float]:
    """
    Create the log-posterior function for MCMC sampling.

    The parameter vector theta = [Omega_m, delta_0, delta_1, ..., delta_{n-1}]

    Parameters
    ----------
    hyp : LayeredExpansionHyperparams
        Fixed hyperparameters
    z_nodes : ndarray
        Redshift nodes for the layered model
    config : MCMCConfig
        MCMC configuration

    Returns
    -------
    Callable
        Log-posterior function
    """
    n_params = 1 + len(z_nodes)  # Omega_m + delta_nodes

    def log_posterior(theta: np.ndarray) -> float:
        # Unpack parameters
        Omega_m = theta[0]
        delta_nodes = theta[1:]

        # --- Prior on Omega_m ---
        if Omega_m < 0.1 or Omega_m > 0.5:
            return -np.inf
        log_prior_Omega_m = norm.logpdf(
            Omega_m, loc=config.Omega_m_prior_mean, scale=config.Omega_m_prior_std
        )

        # --- Prior on delta nodes ---
        # Gaussian prior centered at 0
        log_prior_delta = np.sum(norm.logpdf(
            delta_nodes, loc=0, scale=config.delta_prior_std
        ))

        # --- Create background and params ---
        # Use fixed H0_base = 67.5 for LCDM; the effective H0 comes from the deltas
        lcdm = LCDMBackground(H0=67.5, Omega_m=Omega_m)
        params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

        # --- Smoothness prior ---
        try:
            log_prior_smooth = log_smoothness_prior(params, hyp)
        except ValueError:
            return -np.inf

        # --- Physical validity check ---
        validity = check_physical_validity(lcdm, params, hyp)
        if not validity["valid"]:
            return -np.inf

        # --- Likelihood (chi-squared) ---
        try:
            # CMB constraint
            chi2_cmb, _ = compute_chi2_cmb_layered(lcdm, params, hyp)

            # BAO constraint
            chi2_bao, _ = compute_chi2_bao_layered(lcdm, params, hyp)

            # SN constraint
            chi2_sn, _ = compute_chi2_sn_layered(
                lcdm, params, hyp, use_shoes_prior=config.include_shoes
            )

            chi2_total = chi2_cmb + chi2_bao + chi2_sn

            if not np.isfinite(chi2_total):
                return -np.inf

            log_likelihood = -0.5 * chi2_total

        except Exception:
            return -np.inf

        # --- Total log posterior ---
        log_post = log_prior_Omega_m + log_prior_delta + log_prior_smooth + log_likelihood

        return log_post if np.isfinite(log_post) else -np.inf

    return log_posterior


def compute_H0_eff_from_theta(
    theta: np.ndarray,
    z_nodes: np.ndarray,
    hyp: LayeredExpansionHyperparams,
) -> float:
    """
    Compute H0_eff from parameter vector.

    Parameters
    ----------
    theta : ndarray
        Parameter vector [Omega_m, delta_0, ..., delta_{n-1}]
    z_nodes : ndarray
        Redshift nodes
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float
        Effective H0 in km/s/Mpc
    """
    Omega_m = theta[0]
    delta_nodes = theta[1:]

    lcdm = LCDMBackground(H0=67.5, Omega_m=Omega_m)
    params = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)

    return get_H0_effective(lcdm, params, hyp)


def run_mcmc(
    config: MCMCConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MCMCResult]:
    """
    Run MCMC sampling.

    Parameters
    ----------
    config : MCMCConfig
        MCMC configuration
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (chains, H0_eff_samples, result)
        - chains: shape (n_walkers, n_steps, n_params)
        - H0_eff_samples: shape (n_samples,) after burn-in and thinning
        - result: MCMCResult with summary statistics
    """
    # Setup
    hyp = LayeredExpansionHyperparams(
        n_layers=config.n_layers,
        smooth_sigma=config.smooth_sigma,
        mode=config.mode,
        z_max=6.0,
        spacing="log",
    )
    z_nodes = make_default_nodes(hyp)
    n_params = 1 + len(z_nodes)

    if verbose:
        print(f"Parameter dimensionality: {n_params}")
        print(f"  Omega_m: 1 parameter")
        print(f"  delta_nodes: {len(z_nodes)} parameters")
        print()

    # Create log-posterior
    log_posterior = create_log_posterior(hyp, z_nodes, config)

    # Initialize walkers
    rng = np.random.default_rng(config.seed)

    # Initial positions: near the prior means
    initial = np.zeros((config.n_walkers, n_params))
    initial[:, 0] = rng.normal(
        config.Omega_m_prior_mean,
        config.Omega_m_prior_std * 0.1,
        size=config.n_walkers
    )
    for i in range(len(z_nodes)):
        initial[:, 1 + i] = rng.normal(0, config.delta_prior_std * 0.1, size=config.n_walkers)

    # Try to import emcee
    try:
        import emcee
        use_emcee = True
    except ImportError:
        use_emcee = False
        if verbose:
            print("Warning: emcee not available, using simple MH sampler")

    if use_emcee:
        # Run emcee
        sampler = emcee.EnsembleSampler(config.n_walkers, n_params, log_posterior)

        if verbose:
            print(f"Running emcee with {config.n_walkers} walkers for {config.n_steps} steps...")

        # Run with progress
        sampler.run_mcmc(initial, config.n_steps, progress=verbose)

        chains = sampler.get_chain()  # Shape: (n_steps, n_walkers, n_params)
        chains = np.transpose(chains, (1, 0, 2))  # Shape: (n_walkers, n_steps, n_params)
        log_prob = sampler.get_log_prob().T  # Shape: (n_walkers, n_steps)
        acceptance_fraction = np.mean(sampler.acceptance_fraction)

    else:
        # Simple Metropolis-Hastings
        chains, log_prob, acceptance_fraction = run_simple_mh(
            log_posterior, initial, config.n_steps, n_params, verbose
        )

    # Extract samples after burn-in and thinning
    flat_samples = chains[:, config.n_burn::config.thin, :].reshape(-1, n_params)
    flat_log_prob = log_prob[:, config.n_burn::config.thin].flatten()

    n_effective = len(flat_samples)
    if verbose:
        print(f"\nEffective samples after burn-in and thinning: {n_effective}")
        print(f"Acceptance fraction: {acceptance_fraction:.3f}")

    # Compute H0_eff for all samples
    H0_eff_samples = np.array([
        compute_H0_eff_from_theta(s, z_nodes, hyp)
        for s in flat_samples
    ])

    # Find best-fit (MAP)
    best_idx = np.argmax(flat_log_prob)
    best_theta = flat_samples[best_idx]
    best_H0_eff = H0_eff_samples[best_idx]

    # Compute chi2 for best-fit
    Omega_m_best = best_theta[0]
    delta_best = best_theta[1:]
    lcdm_best = LCDMBackground(H0=67.5, Omega_m=Omega_m_best)
    params_best = LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_best)

    chi2_cmb, _ = compute_chi2_cmb_layered(lcdm_best, params_best, hyp)
    chi2_bao, _ = compute_chi2_bao_layered(lcdm_best, params_best, hyp)
    chi2_sn, _ = compute_chi2_sn_layered(lcdm_best, params_best, hyp, use_shoes_prior=config.include_shoes)
    best_chi2 = chi2_cmb + chi2_bao + chi2_sn

    # Compute summary statistics
    result = MCMCResult(
        H0_eff_mean=float(np.mean(H0_eff_samples)),
        H0_eff_median=float(np.median(H0_eff_samples)),
        H0_eff_std=float(np.std(H0_eff_samples)),
        H0_eff_q16=float(np.percentile(H0_eff_samples, 16)),
        H0_eff_q84=float(np.percentile(H0_eff_samples, 84)),
        H0_eff_q2p5=float(np.percentile(H0_eff_samples, 2.5)),
        H0_eff_q97p5=float(np.percentile(H0_eff_samples, 97.5)),
        prob_H0_ge_73=float(np.mean(H0_eff_samples >= 73)),
        prob_H0_ge_71=float(np.mean(H0_eff_samples >= 71)),
        prob_H0_ge_70=float(np.mean(H0_eff_samples >= 70)),
        acceptance_fraction=float(acceptance_fraction),
        n_effective_samples=n_effective,
        best_fit_H0_eff=float(best_H0_eff),
        best_fit_chi2=float(best_chi2),
        best_fit_Omega_m=float(Omega_m_best),
        best_fit_delta_nodes=delta_best.tolist(),
    )

    return chains, H0_eff_samples, result


def run_simple_mh(
    log_posterior: Callable,
    initial: np.ndarray,
    n_steps: int,
    n_params: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple Metropolis-Hastings sampler (fallback if emcee not available).
    """
    n_walkers = len(initial)
    chains = np.zeros((n_walkers, n_steps, n_params))
    log_prob = np.zeros((n_walkers, n_steps))

    # Proposal scales
    proposal_scale = np.ones(n_params) * 0.01
    proposal_scale[0] = 0.005  # Omega_m

    n_accept = 0
    n_total = 0

    for w in range(n_walkers):
        current = initial[w].copy()
        current_lp = log_posterior(current)

        for step in range(n_steps):
            # Propose
            proposal = current + proposal_scale * np.random.randn(n_params)
            proposal_lp = log_posterior(proposal)

            # Accept/reject
            log_alpha = proposal_lp - current_lp
            if np.log(np.random.rand()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                n_accept += 1

            chains[w, step] = current
            log_prob[w, step] = current_lp
            n_total += 1

        if verbose:
            print(f"Walker {w+1}/{n_walkers} complete")

    acceptance_fraction = n_accept / n_total
    return chains, log_prob, acceptance_fraction


def print_results(result: MCMCResult, config: MCMCConfig) -> None:
    """Print MCMC results."""
    print("\n" + "=" * 70)
    print("SIMULATION 24B: Layered Expansion MCMC - RESULTS")
    print("=" * 70)

    print(f"\nH0_eff posterior summary:")
    print(f"  Mean:   {result.H0_eff_mean:.2f} km/s/Mpc")
    print(f"  Median: {result.H0_eff_median:.2f} km/s/Mpc")
    print(f"  Std:    {result.H0_eff_std:.2f} km/s/Mpc")
    print(f"  68% CI: [{result.H0_eff_q16:.2f}, {result.H0_eff_q84:.2f}] km/s/Mpc")
    print(f"  95% CI: [{result.H0_eff_q2p5:.2f}, {result.H0_eff_q97p5:.2f}] km/s/Mpc")

    print(f"\nProbabilities:")
    print(f"  P(H0_eff >= 70) = {result.prob_H0_ge_70:.4f} ({100*result.prob_H0_ge_70:.2f}%)")
    print(f"  P(H0_eff >= 71) = {result.prob_H0_ge_71:.4f} ({100*result.prob_H0_ge_71:.2f}%)")
    print(f"  P(H0_eff >= 73) = {result.prob_H0_ge_73:.4f} ({100*result.prob_H0_ge_73:.2f}%)")

    print(f"\nBest-fit model:")
    print(f"  H0_eff = {result.best_fit_H0_eff:.2f} km/s/Mpc")
    print(f"  chi2 = {result.best_fit_chi2:.1f}")
    print(f"  Omega_m = {result.best_fit_Omega_m:.4f}")

    print(f"\nConvergence:")
    print(f"  Acceptance fraction: {result.acceptance_fraction:.3f}")
    print(f"  Effective samples: {result.n_effective_samples}")

    print("\n" + "=" * 70)
    print("VERDICT:")
    if result.prob_H0_ge_73 > 0.05:
        print(f"  P(H0 >= 73) = {100*result.prob_H0_ge_73:.1f}% - non-negligible posterior support")
        print("  Layered expansion provides some avenue for reconciliation.")
    elif result.prob_H0_ge_73 > 0.01:
        print(f"  P(H0 >= 73) = {100*result.prob_H0_ge_73:.1f}% - marginal posterior support")
        print("  Layered expansion provides weak avenue for reconciliation.")
    else:
        print(f"  P(H0 >= 73) = {100*result.prob_H0_ge_73:.2f}% - negligible posterior support")
        print("  Layered expansion alone CANNOT rescue H0 = 73.")
    print("=" * 70)


def save_results(
    chains: np.ndarray,
    H0_samples: np.ndarray,
    result: MCMCResult,
    config: MCMCConfig,
    output_dir: Path,
) -> None:
    """Save MCMC results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save chains
    np.savez(
        output_dir / "chains.npz",
        chains=chains,
        H0_eff_samples=H0_samples,
    )
    print(f"Saved chains to {output_dir / 'chains.npz'}")

    # Save summary
    summary = {
        "result": asdict(result),
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_dir / 'summary.json'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SIM 24B: Layered Expansion MCMC"
    )
    parser.add_argument(
        "--n-walkers", type=int, default=32,
        help="Number of walkers (default: 32)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=1000,
        help="Number of steps per walker (default: 1000)"
    )
    parser.add_argument(
        "--n-burn", type=int, default=200,
        help="Burn-in steps (default: 200)"
    )
    parser.add_argument(
        "--thin", type=int, default=1,
        help="Thinning factor (default: 1)"
    )
    parser.add_argument(
        "--n-layers", type=int, default=6,
        help="Number of layers (default: 6)"
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=0.05,
        help="Smoothness prior width (default: 0.05)"
    )
    parser.add_argument(
        "--delta-prior-std", type=float, default=0.1,
        help="Prior std on delta nodes (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-shoes", action="store_true",
        help="Exclude SH0ES H0 prior"
    )
    parser.add_argument(
        "--mode", type=str, default="delta_H", choices=["delta_H", "delta_w"],
        help="Mode (default: delta_H)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run (fewer steps)"
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_walkers = 16
        n_steps = 100
        n_burn = 20
    else:
        n_walkers = args.n_walkers
        n_steps = args.n_steps
        n_burn = args.n_burn

    config = MCMCConfig(
        n_layers=args.n_layers,
        smooth_sigma=args.smooth_sigma,
        mode=args.mode,
        Omega_m_prior_mean=0.315,
        Omega_m_prior_std=0.02,
        delta_prior_std=args.delta_prior_std,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        thin=args.thin,
        include_shoes=not args.no_shoes,
        seed=args.seed,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/simulation_24_layered_mcmc")

    # Print configuration
    print("\n" + "=" * 70)
    print("SIMULATION 24B: Layered Expansion (Bent-Deck) MCMC")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_layers: {config.n_layers}")
    print(f"  smooth_sigma: {config.smooth_sigma}")
    print(f"  mode: {config.mode}")
    print(f"  delta_prior_std: {config.delta_prior_std}")
    print(f"  n_walkers: {config.n_walkers}")
    print(f"  n_steps: {config.n_steps}")
    print(f"  n_burn: {config.n_burn}")
    print(f"  include SH0ES: {config.include_shoes}")
    print(f"  seed: {config.seed}")
    print()

    # Run MCMC
    start_time = time.time()
    chains, H0_samples, result = run_mcmc(config, verbose=True)
    elapsed = time.time() - start_time

    print(f"\nMCMC completed in {elapsed:.1f} seconds")

    # Print results
    print_results(result, config)

    # Save results
    save_results(chains, H0_samples, result, config, output_dir)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
