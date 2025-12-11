"""
Black-Hole Interior Transition Computation Cosmology (BITCC) - Internal Module

This module implements the core BITCC model that maps black-hole interior
parameters to an emergent "computational residue" χ_trans, which then determines
the initial expansion scale H_init for a daughter universe.

Conceptual picture:
- Black-hole interiors undergo alternating quiet (computational) and noise
  (mass-inflation/instability) phases.
- During the transitional computational window, a finite amount of "effective
  gravitational computation" occurs.
- The output χ_trans is a scalar encoding the effective computational output.
- χ_trans maps to an initial Hubble scale H_init for the daughter universe.

This is a phenomenological model; we do not claim physical validity but use
it to explore whether such priors naturally prefer certain H0 values.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Internal model constants (phenomenological)
# =============================================================================

# Scale parameters for χ_trans computation
N_Q0 = 5.0       # Reference quiet e-folds (saturation scale for tanh)
N_N0 = 3.0       # Reference noise e-folds (decay scale for exponential)
GAMMA_S = 0.3    # Exponent for stability margin dependence
GAMMA_M = 0.02   # Coefficient for BH mass dependence (weak log scaling)
A_Q = 1.5        # Overall amplitude for χ_trans

# Reference χ_trans value for H_init mapping
CHI_REF = 0.4    # χ_trans value that gives H_init = H0_ref


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class BITCCInteriorParams:
    """
    Interior configuration of a single black hole in the BITCC model.

    These parameters characterize the computational dynamics of the BH interior
    during the transitional window between quiet and noise phases.

    Attributes
    ----------
    N_q : float
        Quiet-phase effective e-folds of interior computational time.
        More e-folds = more time for computation before noise onset.
        Typical range: [1, 20]

    N_n : float
        Noise-phase effective e-folds before full mass inflation.
        More e-folds = more destruction/scrambling of computational output.
        Typical range: [0.5, 10]

    k_trans : float
        Dimensionless parameter controlling transition sharpness.
        Higher values = sharper transition = more intense but brief computation.
        Typical range: [0.1, 5]

    s_q : float
        Dimensionless "stability margin" of quiet phase.
        0 = marginally stable (easily disrupted)
        1 = very stable (robust computation)
        Range: (0, 1)

    m_bh : float
        Black hole mass in units of fiducial mass (M_BH / 10^6 M_sun).
        Typical range: [10^5, 10^9] solar masses -> [0.1, 1000] in these units.
    """
    N_q: float
    N_n: float
    k_trans: float
    s_q: float
    m_bh: float

    def __post_init__(self):
        """Validate parameters."""
        if self.N_q <= 0:
            raise ValueError(f"N_q must be positive, got {self.N_q}")
        if self.N_n <= 0:
            raise ValueError(f"N_n must be positive, got {self.N_n}")
        if self.k_trans <= 0:
            raise ValueError(f"k_trans must be positive, got {self.k_trans}")
        if not (0 < self.s_q < 1):
            raise ValueError(f"s_q must be in (0, 1), got {self.s_q}")
        if self.m_bh <= 0:
            raise ValueError(f"m_bh must be positive, got {self.m_bh}")


@dataclass
class BITCCHyperparams:
    """
    Hyperparameters defining prior distributions over interior parameters.

    These control the population-level distribution of black hole interiors
    that seed daughter universes.

    Attributes
    ----------
    N_q_mean, N_q_sigma : float
        Mean and std for Gaussian prior on N_q (clipped to positive).

    N_n_mean, N_n_sigma : float
        Mean and std for Gaussian prior on N_n (clipped to positive).

    k_trans_mean, k_trans_sigma : float
        Mean and std for Gaussian prior on k_trans (clipped to positive).

    s_q_alpha, s_q_beta : float
        Shape parameters for Beta distribution on s_q in (0, 1).
        alpha=beta=2 gives symmetric distribution peaked at 0.5.
        alpha>beta biases toward higher stability.

    log10_m_bh_min, log10_m_bh_max : float
        Range for log-uniform prior on m_bh.
        e.g., [5, 9] means M_BH from 10^5 to 10^9 times fiducial.
    """
    N_q_mean: float = 8.0
    N_q_sigma: float = 3.0
    N_n_mean: float = 3.0
    N_n_sigma: float = 1.0
    k_trans_mean: float = 1.0
    k_trans_sigma: float = 0.5
    s_q_alpha: float = 2.0
    s_q_beta: float = 2.0
    log10_m_bh_min: float = 5.0
    log10_m_bh_max: float = 9.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "N_q_mean": self.N_q_mean,
            "N_q_sigma": self.N_q_sigma,
            "N_n_mean": self.N_n_mean,
            "N_n_sigma": self.N_n_sigma,
            "k_trans_mean": self.k_trans_mean,
            "k_trans_sigma": self.k_trans_sigma,
            "s_q_alpha": self.s_q_alpha,
            "s_q_beta": self.s_q_beta,
            "log10_m_bh_min": self.log10_m_bh_min,
            "log10_m_bh_max": self.log10_m_bh_max,
        }


@dataclass
class BITCCDerivedParams:
    """
    Derived parameters from BITCC model.

    Attributes
    ----------
    chi_trans : float
        Effective computational residue from the BH interior.
        Dimensionless, typically O(1) for reasonable parameters.

    H_init : float
        Initial expansion scale for daughter universe [km/s/Mpc].
        This becomes the effective H0 in the daughter cosmology.

    weight : float
        Unnormalized prior weight (for importance sampling if needed).
        Default is 1.0 for uniform weighting.
    """
    chi_trans: float
    H_init: float
    weight: float = 1.0


# =============================================================================
# Core functions
# =============================================================================

def compute_chi_trans(params: BITCCInteriorParams) -> float:
    """
    Compute the effective 'computational residue' χ_trans from interior parameters.

    The functional form is phenomenological but captures intuitive dependencies:

    χ_trans = A_q * tanh(N_q / N_q0)           # Saturating benefit of quiet time
            * exp(-N_n / N_n0)                 # Exponential destruction from noise
            * (k_trans / (1 + k_trans))        # Saturating benefit of sharp transition
            * s_q^γ_s                          # Power-law benefit of stability
            * (1 + γ_m * log10(m_bh))          # Weak log-scaling with BH mass

    Parameters
    ----------
    params : BITCCInteriorParams
        The interior configuration of the black hole.

    Returns
    -------
    float
        The computational residue χ_trans (dimensionless, typically O(0.1-1)).

    Notes
    -----
    The model constants are:
    - N_q0 = 10: saturation scale for quiet time
    - N_n0 = 5: decay scale for noise destruction
    - γ_s = 0.5: stability margin exponent
    - γ_m = 0.05: BH mass coefficient (weak dependence)
    - A_q = 1.0: overall amplitude

    Behavior:
    - χ_trans increases with N_q (more quiet time => more computation)
    - χ_trans decreases with N_n (more noise => more destruction)
    - χ_trans increases with k_trans up to saturation
    - χ_trans increases with s_q (more stable => more reliable)
    - χ_trans has weak positive log-dependence on m_bh
    """
    # Saturating quiet time contribution
    quiet_factor = np.tanh(params.N_q / N_Q0)

    # Exponential noise destruction (bounded to avoid numerical issues)
    noise_factor = np.exp(-np.clip(params.N_n / N_N0, 0, 20))

    # Saturating transition sharpness contribution
    trans_factor = params.k_trans / (1.0 + params.k_trans)

    # Power-law stability contribution
    stability_factor = params.s_q ** GAMMA_S

    # Weak log-scaling with BH mass
    # Ensure m_bh > 0 for log10
    log_m = np.log10(max(params.m_bh, 1e-10))
    mass_factor = 1.0 + GAMMA_M * log_m

    # Combine all factors
    chi_trans = A_Q * quiet_factor * noise_factor * trans_factor * stability_factor * mass_factor

    return float(chi_trans)


def map_chi_to_H_init(
    chi_trans: float,
    H0_ref: float = 67.5,
    gamma_H: float = 0.15,
    H0_min: float = 50.0,
    H0_max: float = 80.0,
) -> float:
    """
    Map χ_trans to an initial Hubble scale H_init.

    The mapping uses an exponential form centered at a reference χ_trans:

        H_init = H0_ref * exp(γ_H * (χ_trans - χ_ref))

    where χ_ref is chosen so that χ_trans = χ_ref => H_init = H0_ref.

    Parameters
    ----------
    chi_trans : float
        The computational residue from the BH interior.

    H0_ref : float, default=67.5
        Reference Hubble constant [km/s/Mpc].
        This is the H_init when χ_trans = χ_ref.

    gamma_H : float, default=0.15
        Sensitivity parameter controlling how strongly χ_trans affects H_init.
        Larger values => stronger dependence.
        gamma_H = 0.15 gives ~1-2 km/s/Mpc shift per 0.1 change in χ_trans.

    H0_min : float, default=50.0
        Minimum allowed H_init [km/s/Mpc].

    H0_max : float, default=80.0
        Maximum allowed H_init [km/s/Mpc].

    Returns
    -------
    float
        Initial Hubble scale H_init [km/s/Mpc], clipped to [H0_min, H0_max].

    Notes
    -----
    The reference χ_ref is set by the module constant CHI_REF = 0.5.
    With default parameters:
    - χ_trans = 0.5 => H_init = 67.5 km/s/Mpc
    - χ_trans = 1.0 => H_init ~ 72.9 km/s/Mpc
    - χ_trans = 0.0 => H_init ~ 62.6 km/s/Mpc
    """
    # Compute H_init using exponential mapping
    exponent = gamma_H * (chi_trans - CHI_REF)

    # Clip exponent to avoid numerical overflow
    exponent = np.clip(exponent, -2, 2)

    H_init = H0_ref * np.exp(exponent)

    # Clip to physical range
    H_init = np.clip(H_init, H0_min, H0_max)

    return float(H_init)


def sample_interiors(
    hyper: BITCCHyperparams,
    n_samples: int,
    rng: np.random.Generator,
) -> List[BITCCInteriorParams]:
    """
    Sample interior configurations from the prior distribution.

    Parameters
    ----------
    hyper : BITCCHyperparams
        Hyperparameters defining the prior distributions.

    n_samples : int
        Number of samples to draw.

    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    List[BITCCInteriorParams]
        List of sampled interior configurations.

    Notes
    -----
    Sampling distributions:
    - N_q: Gaussian(mean, sigma), clipped to [0.1, inf)
    - N_n: Gaussian(mean, sigma), clipped to [0.1, inf)
    - k_trans: Gaussian(mean, sigma), clipped to [0.01, inf)
    - s_q: Beta(alpha, beta) in (0, 1)
    - m_bh: log-uniform between 10^log10_min and 10^log10_max
    """
    samples = []

    # Draw all samples at once for efficiency
    N_q_raw = rng.normal(hyper.N_q_mean, hyper.N_q_sigma, size=n_samples)
    N_n_raw = rng.normal(hyper.N_n_mean, hyper.N_n_sigma, size=n_samples)
    k_trans_raw = rng.normal(hyper.k_trans_mean, hyper.k_trans_sigma, size=n_samples)
    s_q_raw = rng.beta(hyper.s_q_alpha, hyper.s_q_beta, size=n_samples)

    # Log-uniform for m_bh
    log10_m_bh = rng.uniform(hyper.log10_m_bh_min, hyper.log10_m_bh_max, size=n_samples)
    m_bh_raw = 10 ** log10_m_bh

    # Clip to valid ranges
    N_q = np.clip(N_q_raw, 0.1, None)
    N_n = np.clip(N_n_raw, 0.1, None)
    k_trans = np.clip(k_trans_raw, 0.01, None)
    # s_q from Beta is already in (0, 1), but clip for safety
    s_q = np.clip(s_q_raw, 0.001, 0.999)

    for i in range(n_samples):
        params = BITCCInteriorParams(
            N_q=float(N_q[i]),
            N_n=float(N_n[i]),
            k_trans=float(k_trans[i]),
            s_q=float(s_q[i]),
            m_bh=float(m_bh_raw[i]),
        )
        samples.append(params)

    return samples


def compute_derived_params(
    params: BITCCInteriorParams,
    H0_ref: float = 67.5,
    gamma_H: float = 0.15,
) -> BITCCDerivedParams:
    """
    Compute derived parameters (χ_trans, H_init) from interior parameters.

    Parameters
    ----------
    params : BITCCInteriorParams
        The interior configuration.

    H0_ref : float
        Reference H0 for mapping.

    gamma_H : float
        Sensitivity parameter for H_init mapping.

    Returns
    -------
    BITCCDerivedParams
        The derived parameters including χ_trans and H_init.
    """
    chi_trans = compute_chi_trans(params)
    H_init = map_chi_to_H_init(chi_trans, H0_ref=H0_ref, gamma_H=gamma_H)

    return BITCCDerivedParams(
        chi_trans=chi_trans,
        H_init=H_init,
        weight=1.0,
    )
