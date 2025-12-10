#!/usr/bin/env python3
"""
Host galaxy population model for SN Ia distance ladder simulation.

Implements realistic host galaxy properties including:
- Stellar mass distribution (different for calibrators vs Hubble flow)
- Metallicity-mass relation
- Dust (E(B-V)) distribution
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class HostGalaxy:
    """Properties of a SN Ia host galaxy."""
    logM_star: float    # log10(M*/M_sun)
    Z: float            # metallicity proxy [O/H] - [O/H]_solar or log(Z/Z_sun)
    E_BV: float         # dust reddening E(B-V)


@dataclass
class HostPopulationParams:
    """
    Hyperparameters for host galaxy population.

    Calibrator hosts tend to be lower mass (Cepheid hosts are typically
    late-type spirals), while Hubble-flow hosts span a broader range
    including massive ellipticals.
    """
    # Mass distribution parameters
    logM_star_mean_calib: float = 10.3    # Calibrator host mean
    logM_star_sigma_calib: float = 0.3
    logM_star_mean_flow: float = 10.8     # Hubble flow host mean (more massive)
    logM_star_sigma_flow: float = 0.5

    # Metallicity-mass relation:
    # Z = Z0 + k_ZM * (logM_star - 10.0) + noise
    Z0: float = 0.0                       # Metallicity at logM=10
    k_ZM: float = 0.3                     # Metallicity gradient per dex mass
    sigma_Z: float = 0.1                  # Scatter in Z at fixed mass

    # Dust distribution:
    # E(B-V) ~ max(0, Normal(mu_EBV, sigma_EBV))
    mu_EBV_calib: float = 0.05           # Calibrators typically lower extinction
    sigma_EBV_calib: float = 0.03
    mu_EBV_flow: float = 0.08            # Hubble flow slightly higher extinction
    sigma_EBV_flow: float = 0.05


def sample_hosts(
    N: int,
    sample_type: str,
    pop_params: HostPopulationParams,
    rng: np.random.Generator,
) -> List[HostGalaxy]:
    """
    Sample host galaxies for calibrators or Hubble flow.

    Args:
        N: Number of hosts to sample
        sample_type: "calib" or "flow"
        pop_params: Host population hyperparameters
        rng: Random number generator

    Returns:
        List of HostGalaxy objects
    """
    if sample_type == "calib":
        mu_M = pop_params.logM_star_mean_calib
        sig_M = pop_params.logM_star_sigma_calib
        mu_EBV = pop_params.mu_EBV_calib
        sig_EBV = pop_params.sigma_EBV_calib
    elif sample_type == "flow":
        mu_M = pop_params.logM_star_mean_flow
        sig_M = pop_params.logM_star_sigma_flow
        mu_EBV = pop_params.mu_EBV_flow
        sig_EBV = pop_params.sigma_EBV_flow
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    # Sample stellar masses
    logM = rng.normal(mu_M, sig_M, size=N)

    # Metallicity from mass-metallicity relation + scatter
    Z = (pop_params.Z0 +
         pop_params.k_ZM * (logM - 10.0) +
         rng.normal(0.0, pop_params.sigma_Z, size=N))

    # Dust reddening (truncated at 0)
    E_BV = np.clip(rng.normal(mu_EBV, sig_EBV, size=N), 0.0, None)

    hosts = [
        HostGalaxy(logM_star=lm, Z=z, E_BV=e)
        for lm, z, e in zip(logM, Z, E_BV)
    ]
    return hosts


def get_host_arrays(hosts: List[HostGalaxy]):
    """Convert list of hosts to numpy arrays."""
    logM = np.array([h.logM_star for h in hosts])
    Z = np.array([h.Z for h in hosts])
    E_BV = np.array([h.E_BV for h in hosts])
    return logM, Z, E_BV
