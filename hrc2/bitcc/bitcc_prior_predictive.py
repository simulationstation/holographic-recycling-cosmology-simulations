"""
Black-Hole Interior Transition Computation Cosmology (BITCC) - Prior Predictive

This module implements prior predictive sampling from the BITCC model,
allowing us to explore what H0 values are typical under this prior
before imposing any observational constraints.

Key functions:
- sample_bitcc_prior(): Draw samples from the BITCC prior
- compute_H0_distribution_from_bitcc(): Compute statistics of H0 distribution
- run_bitcc_prior_predictive(): Full prior predictive analysis
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np

from .bitcc_internal import (
    BITCCInteriorParams,
    BITCCHyperparams,
    BITCCDerivedParams,
    sample_interiors,
    compute_chi_trans,
    map_chi_to_H_init,
    compute_derived_params,
)


@dataclass
class BITCCPriorSample:
    """
    A single sample from the BITCC prior.

    Attributes
    ----------
    interior : BITCCInteriorParams
        The interior configuration of the black hole.

    derived : BITCCDerivedParams
        The derived parameters (χ_trans, H_init).
    """
    interior: BITCCInteriorParams
    derived: BITCCDerivedParams

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "interior": {
                "N_q": self.interior.N_q,
                "N_n": self.interior.N_n,
                "k_trans": self.interior.k_trans,
                "s_q": self.interior.s_q,
                "m_bh": self.interior.m_bh,
            },
            "derived": {
                "chi_trans": self.derived.chi_trans,
                "H_init": self.derived.H_init,
                "weight": self.derived.weight,
            },
        }


def sample_bitcc_prior(
    hyper: BITCCHyperparams,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    H0_ref: float = 67.5,
    gamma_H: float = 0.15,
) -> List[BITCCPriorSample]:
    """
    Sample from the BITCC prior over H_init.

    This function:
    1. Samples interior parameters from the prior distributions.
    2. Computes χ_trans for each sample.
    3. Maps χ_trans to H_init.
    4. Returns the full samples with all parameters.

    Parameters
    ----------
    hyper : BITCCHyperparams
        Hyperparameters defining the prior distributions.

    n_samples : int
        Number of samples to draw.

    rng : np.random.Generator, optional
        Random number generator. If None, creates a new one with default seed.

    H0_ref : float, default=67.5
        Reference H0 for the χ_trans -> H_init mapping.

    gamma_H : float, default=0.15
        Sensitivity parameter for H_init mapping.

    Returns
    -------
    List[BITCCPriorSample]
        List of samples from the BITCC prior.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    # Sample interior parameters
    interiors = sample_interiors(hyper, n_samples, rng)

    # Compute derived parameters for each
    samples = []
    for interior in interiors:
        derived = compute_derived_params(interior, H0_ref=H0_ref, gamma_H=gamma_H)
        samples.append(BITCCPriorSample(interior=interior, derived=derived))

    return samples


def compute_H0_distribution_from_bitcc(
    samples: List[BITCCPriorSample],
) -> Dict[str, Any]:
    """
    Compute summary statistics of the H0 (H_init) distribution from BITCC samples.

    Parameters
    ----------
    samples : List[BITCCPriorSample]
        Samples from the BITCC prior.

    Returns
    -------
    dict
        Summary statistics including:
        - mean, std: Mean and standard deviation of H_init
        - median: Median H_init
        - q16, q84: 16th and 84th percentiles (68% CI)
        - q2p5, q97p5: 2.5th and 97.5th percentiles (95% CI)
        - min, max: Minimum and maximum H_init
        - P_H0_ge_73: Fraction with H_init >= 73
        - P_H0_ge_70: Fraction with H_init >= 70
        - P_H0_le_65: Fraction with H_init <= 65
        - n_samples: Number of samples
    """
    if not samples:
        return {"error": "No samples provided"}

    H_init_values = np.array([s.derived.H_init for s in samples])

    return {
        "mean": float(np.mean(H_init_values)),
        "std": float(np.std(H_init_values)),
        "median": float(np.median(H_init_values)),
        "q16": float(np.percentile(H_init_values, 16)),
        "q84": float(np.percentile(H_init_values, 84)),
        "q2p5": float(np.percentile(H_init_values, 2.5)),
        "q97p5": float(np.percentile(H_init_values, 97.5)),
        "min": float(np.min(H_init_values)),
        "max": float(np.max(H_init_values)),
        "P_H0_ge_73": float(np.mean(H_init_values >= 73)),
        "P_H0_ge_70": float(np.mean(H_init_values >= 70)),
        "P_H0_le_65": float(np.mean(H_init_values <= 65)),
        "n_samples": len(samples),
    }


def compute_chi_trans_statistics(
    samples: List[BITCCPriorSample],
) -> Dict[str, Any]:
    """
    Compute summary statistics of the χ_trans distribution.

    Parameters
    ----------
    samples : List[BITCCPriorSample]
        Samples from the BITCC prior.

    Returns
    -------
    dict
        Summary statistics of χ_trans.
    """
    if not samples:
        return {"error": "No samples provided"}

    chi_values = np.array([s.derived.chi_trans for s in samples])

    return {
        "mean": float(np.mean(chi_values)),
        "std": float(np.std(chi_values)),
        "median": float(np.median(chi_values)),
        "q16": float(np.percentile(chi_values, 16)),
        "q84": float(np.percentile(chi_values, 84)),
        "min": float(np.min(chi_values)),
        "max": float(np.max(chi_values)),
    }


def run_bitcc_prior_predictive(
    hyper: BITCCHyperparams,
    n_samples: int = 10000,
    seed: int = 1234,
    H0_ref: float = 67.5,
    gamma_H: float = 0.15,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run prior predictive BITCC sampling and analysis.

    This function performs a complete prior predictive analysis:
    1. Samples from the BITCC prior.
    2. Computes H_init and χ_trans for each sample.
    3. Calculates summary statistics.
    4. Optionally saves results to JSON.

    Parameters
    ----------
    hyper : BITCCHyperparams
        Hyperparameters defining the prior distributions.

    n_samples : int, default=10000
        Number of samples to draw.

    seed : int, default=1234
        Random seed for reproducibility.

    H0_ref : float, default=67.5
        Reference H0 for mapping.

    gamma_H : float, default=0.15
        Sensitivity parameter for H_init mapping.

    output_path : Path, optional
        If provided, save results to this JSON file.

    verbose : bool, default=True
        Print progress and summary.

    Returns
    -------
    dict
        Full results dictionary including:
        - hyperparams: The hyperparameters used
        - H0_stats: Statistics of H_init distribution
        - chi_trans_stats: Statistics of χ_trans distribution
        - model_params: H0_ref, gamma_H, seed, n_samples
    """
    if verbose:
        print(f"Running BITCC prior predictive with {n_samples} samples...")

    rng = np.random.default_rng(seed)

    # Sample from prior
    samples = sample_bitcc_prior(
        hyper=hyper,
        n_samples=n_samples,
        rng=rng,
        H0_ref=H0_ref,
        gamma_H=gamma_H,
    )

    # Compute statistics
    H0_stats = compute_H0_distribution_from_bitcc(samples)
    chi_trans_stats = compute_chi_trans_statistics(samples)

    # Assemble results
    results = {
        "hyperparams": hyper.to_dict(),
        "model_params": {
            "H0_ref": H0_ref,
            "gamma_H": gamma_H,
            "seed": seed,
            "n_samples": n_samples,
        },
        "H0_stats": H0_stats,
        "chi_trans_stats": chi_trans_stats,
    }

    if verbose:
        print(f"\nH0 prior predictive statistics:")
        print(f"  Mean:   {H0_stats['mean']:.2f} km/s/Mpc")
        print(f"  Std:    {H0_stats['std']:.2f} km/s/Mpc")
        print(f"  Median: {H0_stats['median']:.2f} km/s/Mpc")
        print(f"  68% CI: [{H0_stats['q16']:.2f}, {H0_stats['q84']:.2f}]")
        print(f"  P(H0 >= 73): {100*H0_stats['P_H0_ge_73']:.2f}%")
        print(f"  P(H0 >= 70): {100*H0_stats['P_H0_ge_70']:.2f}%")
        print(f"  P(H0 <= 65): {100*H0_stats['P_H0_le_65']:.2f}%")

    # Save to file if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"\nSaved results to {output_path}")

    return results


def extract_arrays_from_samples(
    samples: List[BITCCPriorSample],
) -> Dict[str, np.ndarray]:
    """
    Extract arrays of parameters from BITCC samples for analysis.

    Parameters
    ----------
    samples : List[BITCCPriorSample]
        Samples from the BITCC prior.

    Returns
    -------
    dict
        Dictionary with arrays:
        - H_init: Array of H_init values
        - chi_trans: Array of χ_trans values
        - N_q, N_n, k_trans, s_q, m_bh: Arrays of interior parameters
    """
    n = len(samples)

    return {
        "H_init": np.array([s.derived.H_init for s in samples]),
        "chi_trans": np.array([s.derived.chi_trans for s in samples]),
        "N_q": np.array([s.interior.N_q for s in samples]),
        "N_n": np.array([s.interior.N_n for s in samples]),
        "k_trans": np.array([s.interior.k_trans for s in samples]),
        "s_q": np.array([s.interior.s_q for s in samples]),
        "m_bh": np.array([s.interior.m_bh for s in samples]),
    }
