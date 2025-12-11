"""
Hawking-Hartle No-Boundary Prior for Cosmological Parameters

This module implements a structured prior over primordial cosmological
parameters based on the Hawking-Hartle no-boundary wave function.

The no-boundary proposal constrains:
- N_e: Number of inflationary e-folds (controls Omega_k)
- V_scale: Inflationary potential energy scale (controls A_s, n_s)
- phi_init: Initial scalar field value (affects slow-roll)
- epsilon_corr: Early-time H(z) correction at z > z_transition

Theory Reference:
    Hartle, J.B. & Hawking, S.W. (1983). "Wave function of the Universe"
    Physical Review D, 28(12), 2960-2975.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class NoBoundaryHyperparams:
    """
    Hyperparameters for the no-boundary prior distribution.

    These define the shape and scale of priors over primordial parameters.
    The default values are chosen to be conservative (relatively broad)
    while still encoding physical intuition from inflation theory.
    """
    # N_e prior: log-uniform or mildly weighted
    # More e-folds -> more classical, higher weighting
    alpha_Ne: float = 0.05  # Exponential weighting strength (0 = flat)
    Ne_min: float = 50.0    # Minimum e-folds for sufficient inflation
    Ne_max: float = 80.0    # Maximum reasonable e-folds

    # log10(V_scale) prior: Gaussian centered on GUT scale
    mu_logV: float = -10.0     # Mean of log10(V_scale/M_pl^4)
    sigma_logV: float = 1.0    # Width in decades

    # phi_init prior: Gaussian in Planck units
    mu_phi_init: float = 0.1     # Mean initial field value
    sigma_phi_init: float = 0.3  # Width

    # epsilon_corr prior: Gaussian centered on zero (LCDM)
    sigma_epsilon_corr: float = 0.02  # 2% max deviation at high-z

    # Physical bounds
    phi_init_min: float = 0.001  # Avoid phi=0 singularity
    phi_init_max: float = 2.0    # Large field inflation limit

    def __post_init__(self):
        """Validate hyperparameters."""
        assert self.Ne_min > 0, "Ne_min must be positive"
        assert self.Ne_max > self.Ne_min, "Ne_max must exceed Ne_min"
        assert self.sigma_logV > 0, "sigma_logV must be positive"
        assert self.sigma_phi_init > 0, "sigma_phi_init must be positive"
        assert self.sigma_epsilon_corr > 0, "sigma_epsilon_corr must be positive"


@dataclass
class NoBoundaryParams:
    """
    Sampled primordial parameters from the no-boundary prior.

    These are the "fundamental" parameters in the Hawking-Hartle framework
    that get mapped to observable cosmological parameters.
    """
    Ne: float              # Number of inflationary e-folds
    log10_V_scale: float   # log10 of potential scale (in M_pl^4 units)
    phi_init: float        # Initial scalar field value (in M_pl units)
    epsilon_corr: float    # Fractional early-time H(z) correction

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "Ne": self.Ne,
            "log10_V_scale": self.log10_V_scale,
            "phi_init": self.phi_init,
            "epsilon_corr": self.epsilon_corr
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NoBoundaryParams":
        """Construct from dictionary."""
        return cls(
            Ne=d["Ne"],
            log10_V_scale=d["log10_V_scale"],
            phi_init=d["phi_init"],
            epsilon_corr=d["epsilon_corr"]
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [Ne, log10_V_scale, phi_init, epsilon_corr]."""
        return np.array([self.Ne, self.log10_V_scale, self.phi_init, self.epsilon_corr])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NoBoundaryParams":
        """Construct from numpy array."""
        return cls(
            Ne=float(arr[0]),
            log10_V_scale=float(arr[1]),
            phi_init=float(arr[2]),
            epsilon_corr=float(arr[3])
        )


# Parameter names and dimensions
NOBOUNDARY_PARAM_NAMES = ["Ne", "log10_V_scale", "phi_init", "epsilon_corr"]
NOBOUNDARY_NDIM = 4


def log_prior_no_boundary(
    params: NoBoundaryParams,
    hyper: NoBoundaryHyperparams
) -> float:
    """
    Compute log prior probability under the Hawking-Hartle no-boundary framework.

    The prior structure encodes:
    1. N_e: Mildly weighted toward more e-folds (more classical universes)
       p(N_e) ~ exp(alpha_Ne * N_e) for N_e in [Ne_min, Ne_max]

    2. log10_V_scale: Gaussian reflecting uncertainty in inflation scale
       log10(V) ~ N(mu_logV, sigma_logV^2)

    3. phi_init: Gaussian reflecting typical slow-roll initial conditions
       phi_init ~ N(mu_phi_init, sigma_phi_init^2), truncated to [phi_min, phi_max]

    4. epsilon_corr: Gaussian centered on zero (LCDM as reference)
       epsilon_corr ~ N(0, sigma_epsilon_corr^2)

    Args:
        params: NoBoundaryParams to evaluate
        hyper: NoBoundaryHyperparams defining the prior

    Returns:
        Log prior probability (can be -inf for out of bounds)
    """
    log_p = 0.0

    # 1. N_e prior: bounded with optional exponential weighting
    if params.Ne < hyper.Ne_min or params.Ne > hyper.Ne_max:
        return -np.inf

    # Exponential weighting: more e-folds -> more classical -> higher weight
    # Normalize: integral of exp(alpha*N) from N_min to N_max
    if hyper.alpha_Ne > 0:
        norm = (np.exp(hyper.alpha_Ne * hyper.Ne_max) -
                np.exp(hyper.alpha_Ne * hyper.Ne_min)) / hyper.alpha_Ne
        log_p += hyper.alpha_Ne * params.Ne - np.log(norm)
    else:
        # Flat prior
        log_p += -np.log(hyper.Ne_max - hyper.Ne_min)

    # 2. log10_V_scale prior: Gaussian
    log_p += stats.norm.logpdf(params.log10_V_scale,
                                loc=hyper.mu_logV,
                                scale=hyper.sigma_logV)

    # 3. phi_init prior: truncated Gaussian
    if params.phi_init < hyper.phi_init_min or params.phi_init > hyper.phi_init_max:
        return -np.inf

    # Use truncated normal
    a = (hyper.phi_init_min - hyper.mu_phi_init) / hyper.sigma_phi_init
    b = (hyper.phi_init_max - hyper.mu_phi_init) / hyper.sigma_phi_init
    log_p += stats.truncnorm.logpdf(params.phi_init, a, b,
                                     loc=hyper.mu_phi_init,
                                     scale=hyper.sigma_phi_init)

    # 4. epsilon_corr prior: Gaussian centered on zero
    log_p += stats.norm.logpdf(params.epsilon_corr,
                                loc=0.0,
                                scale=hyper.sigma_epsilon_corr)

    return log_p


def sample_no_boundary_prior(
    hyper: NoBoundaryHyperparams,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None
) -> list:
    """
    Sample from the no-boundary prior distribution.

    Args:
        hyper: NoBoundaryHyperparams defining the prior
        n_samples: Number of samples to draw
        rng: Optional numpy random generator for reproducibility

    Returns:
        List of NoBoundaryParams samples
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = []

    for _ in range(n_samples):
        # 1. Sample N_e with exponential weighting
        if hyper.alpha_Ne > 0:
            # Inverse CDF sampling for exp(alpha * N) on [N_min, N_max]
            u = rng.uniform()
            exp_min = np.exp(hyper.alpha_Ne * hyper.Ne_min)
            exp_max = np.exp(hyper.alpha_Ne * hyper.Ne_max)
            Ne = np.log(exp_min + u * (exp_max - exp_min)) / hyper.alpha_Ne
        else:
            # Flat prior
            Ne = rng.uniform(hyper.Ne_min, hyper.Ne_max)

        # 2. Sample log10_V_scale from Gaussian
        log10_V_scale = rng.normal(hyper.mu_logV, hyper.sigma_logV)

        # 3. Sample phi_init from truncated Gaussian
        a = (hyper.phi_init_min - hyper.mu_phi_init) / hyper.sigma_phi_init
        b = (hyper.phi_init_max - hyper.mu_phi_init) / hyper.sigma_phi_init
        phi_init = stats.truncnorm.rvs(a, b,
                                        loc=hyper.mu_phi_init,
                                        scale=hyper.sigma_phi_init,
                                        random_state=rng)

        # 4. Sample epsilon_corr from Gaussian
        epsilon_corr = rng.normal(0.0, hyper.sigma_epsilon_corr)

        samples.append(NoBoundaryParams(
            Ne=Ne,
            log10_V_scale=log10_V_scale,
            phi_init=phi_init,
            epsilon_corr=epsilon_corr
        ))

    return samples


def get_prior_bounds(hyper: NoBoundaryHyperparams) -> Dict[str, Tuple[float, float]]:
    """
    Get effective bounds for each parameter.

    For unbounded parameters (Gaussian), returns 5-sigma bounds.

    Args:
        hyper: NoBoundaryHyperparams

    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    return {
        "Ne": (hyper.Ne_min, hyper.Ne_max),
        "log10_V_scale": (hyper.mu_logV - 5*hyper.sigma_logV,
                          hyper.mu_logV + 5*hyper.sigma_logV),
        "phi_init": (hyper.phi_init_min, hyper.phi_init_max),
        "epsilon_corr": (-5*hyper.sigma_epsilon_corr, 5*hyper.sigma_epsilon_corr)
    }


def compute_no_boundary_weighting(
    Ne: float,
    V_scale: float,
    use_semiclassical: bool = True
) -> float:
    """
    Compute the no-boundary wave function weighting for given parameters.

    In the semiclassical approximation, the no-boundary wave function gives:
    |Psi|^2 ~ exp(24 * pi^2 / V)

    where V is the potential at the start of inflation.

    More e-folds of inflation generally corresponds to a more classical
    (less quantum-dominated) trajectory.

    Args:
        Ne: Number of e-folds
        V_scale: Potential scale (dimensionless, V/M_pl^4)
        use_semiclassical: If True, include semiclassical weighting

    Returns:
        Relative probability weight (not normalized)
    """
    if not use_semiclassical:
        return 1.0

    # Semiclassical weighting from Hartle-Hawking
    # The factor 24*pi^2 comes from the Euclidean action of de Sitter
    if V_scale <= 0:
        return 0.0

    # Avoid numerical overflow - use log scale
    log_weight = 24 * np.pi**2 / V_scale

    # Cap to avoid overflow
    log_weight = min(log_weight, 700)  # exp(700) ~ 10^304

    return np.exp(log_weight)


# Convenience function for MCMC
def log_prior_from_array(
    theta: np.ndarray,
    hyper: NoBoundaryHyperparams
) -> float:
    """
    Compute log prior from parameter array (for MCMC samplers).

    Args:
        theta: Array [Ne, log10_V_scale, phi_init, epsilon_corr]
        hyper: NoBoundaryHyperparams

    Returns:
        Log prior probability
    """
    params = NoBoundaryParams.from_array(theta)
    return log_prior_no_boundary(params, hyper)
