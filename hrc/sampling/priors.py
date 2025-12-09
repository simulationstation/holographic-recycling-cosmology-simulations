"""Prior distributions for HRC parameter inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np


class Prior(ABC):
    """Base class for prior distributions."""

    @abstractmethod
    def log_prob(self, value: float) -> float:
        """Compute log-probability of value under prior."""
        pass

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """Draw samples from prior."""
        pass

    @abstractmethod
    def in_bounds(self, value: float) -> bool:
        """Check if value is within prior support."""
        pass


@dataclass
class UniformPrior(Prior):
    """Uniform prior on [low, high]."""

    low: float
    high: float
    name: str = ""

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")
        self._log_prob = -np.log(self.high - self.low)

    def log_prob(self, value: float) -> float:
        if self.in_bounds(value):
            return self._log_prob
        return -np.inf

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size)

    def in_bounds(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class GaussianPrior(Prior):
    """Gaussian prior with mean μ and standard deviation σ."""

    mean: float
    sigma: float
    low: Optional[float] = None  # Optional lower truncation
    high: Optional[float] = None  # Optional upper truncation
    name: str = ""

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be > 0")
        self._log_norm = -0.5 * np.log(2 * np.pi) - np.log(self.sigma)

    def log_prob(self, value: float) -> float:
        if not self.in_bounds(value):
            return -np.inf
        z = (value - self.mean) / self.sigma
        return self._log_norm - 0.5 * z**2

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.normal(self.mean, self.sigma, size)
        if self.low is not None or self.high is not None:
            # Rejection sampling for truncated Gaussian
            while True:
                mask = np.ones(len(samples), dtype=bool)
                if self.low is not None:
                    mask &= samples >= self.low
                if self.high is not None:
                    mask &= samples <= self.high
                if np.all(mask):
                    break
                n_reject = np.sum(~mask)
                samples[~mask] = np.random.normal(self.mean, self.sigma, n_reject)
        return samples

    def in_bounds(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True


@dataclass
class LogUniformPrior(Prior):
    """Log-uniform (Jeffreys) prior on [low, high].

    p(x) ∝ 1/x for x ∈ [low, high]
    """

    low: float
    high: float
    name: str = ""

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError(f"low ({self.low}) must be > 0")
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")
        self._log_norm = -np.log(np.log(self.high / self.low))

    def log_prob(self, value: float) -> float:
        if not self.in_bounds(value):
            return -np.inf
        return self._log_norm - np.log(value)

    def sample(self, size: int = 1) -> np.ndarray:
        log_low = np.log(self.low)
        log_high = np.log(self.high)
        return np.exp(np.random.uniform(log_low, log_high, size))

    def in_bounds(self, value: float) -> bool:
        return self.low <= value <= self.high


class PriorSet:
    """Collection of priors for multiple parameters."""

    def __init__(
        self,
        priors: Optional[Dict[str, Prior]] = None,
    ):
        """Initialize prior set.

        Args:
            priors: Dictionary mapping parameter names to Prior objects
        """
        self.priors = priors or {}
        self.param_names = list(self.priors.keys())
        self.n_params = len(self.param_names)

    def add(self, name: str, prior: Prior) -> None:
        """Add a prior."""
        self.priors[name] = prior
        self.param_names = list(self.priors.keys())
        self.n_params = len(self.param_names)

    def log_prob(self, params: Dict[str, float]) -> float:
        """Compute total log-prior probability."""
        total = 0.0
        for name, value in params.items():
            if name in self.priors:
                lp = self.priors[name].log_prob(value)
                if lp == -np.inf:
                    return -np.inf
                total += lp
        return total

    def log_prob_vector(self, theta: np.ndarray) -> float:
        """Compute log-prior from parameter vector."""
        params = dict(zip(self.param_names, theta))
        return self.log_prob(params)

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw samples from all priors.

        Returns:
            Array of shape (size, n_params)
        """
        samples = np.zeros((size, self.n_params))
        for i, name in enumerate(self.param_names):
            samples[:, i] = self.priors[name].sample(size)
        return samples

    def in_bounds(self, params: Dict[str, float]) -> bool:
        """Check if all parameters are within prior bounds."""
        for name, value in params.items():
            if name in self.priors:
                if not self.priors[name].in_bounds(value):
                    return False
        return True

    def in_bounds_vector(self, theta: np.ndarray) -> bool:
        """Check bounds from parameter vector."""
        params = dict(zip(self.param_names, theta))
        return self.in_bounds(params)

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for all parameters."""
        bounds = []
        for name in self.param_names:
            prior = self.priors[name]
            if hasattr(prior, 'low') and hasattr(prior, 'high'):
                bounds.append((prior.low, prior.high))
            else:
                bounds.append((-np.inf, np.inf))
        return bounds


def default_hrc_priors() -> PriorSet:
    """Return default priors for HRC parameters."""
    priors = PriorSet()

    # Core HRC parameters
    priors.add("xi", LogUniformPrior(0.001, 0.5, name="xi"))
    priors.add("phi_0", UniformPrior(0.01, 0.5, name="phi_0"))
    priors.add("m_phi", LogUniformPrior(0.1, 10.0, name="m_phi"))

    # Remnant parameters
    priors.add("f_rem", UniformPrior(0.0, 1.0, name="f_rem"))

    # Cosmological parameters
    priors.add("h", GaussianPrior(0.7, 0.05, low=0.5, high=0.9, name="h"))
    priors.add("Omega_m", GaussianPrior(0.315, 0.02, low=0.1, high=0.5, name="Omega_m"))

    return priors
