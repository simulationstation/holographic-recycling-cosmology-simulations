"""MCMC sampler for HRC parameter inference.

Implements both a simple Metropolis-Hastings sampler and an
interface to the emcee affine-invariant ensemble sampler.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import warnings

from .priors import PriorSet, default_hrc_priors
from ..utils.config import HRCParameters
from ..background import BackgroundCosmology, BackgroundSolution


@dataclass
class MCMCResult:
    """Result of MCMC sampling."""

    chains: NDArray[np.floating]  # Shape: (n_walkers, n_steps, n_params)
    log_prob: NDArray[np.floating]  # Shape: (n_walkers, n_steps)
    param_names: List[str]
    n_walkers: int
    n_steps: int
    n_params: int
    acceptance_fraction: float

    # Convergence diagnostics
    autocorr_time: Optional[NDArray[np.floating]] = None
    gelman_rubin: Optional[NDArray[np.floating]] = None

    def get_flat_samples(
        self,
        burn: int = 0,
        thin: int = 1,
    ) -> NDArray[np.floating]:
        """Get flattened chain samples.

        Args:
            burn: Number of steps to discard as burn-in
            thin: Thinning factor

        Returns:
            Array of shape (n_samples, n_params)
        """
        return self.chains[:, burn::thin, :].reshape(-1, self.n_params)

    def get_param_samples(
        self,
        param: str,
        burn: int = 0,
        thin: int = 1,
    ) -> NDArray[np.floating]:
        """Get samples for a single parameter."""
        idx = self.param_names.index(param)
        return self.get_flat_samples(burn, thin)[:, idx]

    def summary(self, burn: int = 0) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all parameters."""
        samples = self.get_flat_samples(burn)

        summary = {}
        for i, name in enumerate(self.param_names):
            param_samples = samples[:, i]
            summary[name] = {
                "mean": float(np.mean(param_samples)),
                "std": float(np.std(param_samples)),
                "median": float(np.median(param_samples)),
                "q16": float(np.percentile(param_samples, 16)),
                "q84": float(np.percentile(param_samples, 84)),
            }
        return summary


class MetropolisHastings:
    """Simple Metropolis-Hastings sampler."""

    def __init__(
        self,
        log_prob_fn: Callable[[NDArray[np.floating]], float],
        priors: PriorSet,
        proposal_scale: Optional[NDArray[np.floating]] = None,
    ):
        """Initialize MH sampler.

        Args:
            log_prob_fn: Function computing log-posterior (log-likelihood + log-prior)
            priors: Prior distributions
            proposal_scale: Proposal step sizes (default: prior width / 10)
        """
        self.log_prob_fn = log_prob_fn
        self.priors = priors
        self.n_params = priors.n_params

        if proposal_scale is None:
            # Default: use prior width / 10
            bounds = priors.get_bounds()
            proposal_scale = np.array([
                (high - low) / 10 if np.isfinite(high - low) else 0.1
                for low, high in bounds
            ])
        self.proposal_scale = proposal_scale

    def _propose(self, current: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate proposal from current position."""
        return current + self.proposal_scale * np.random.randn(self.n_params)

    def run(
        self,
        initial: NDArray[np.floating],
        n_steps: int,
        progress: bool = True,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], float]:
        """Run MH sampler.

        Args:
            initial: Initial parameter vector
            n_steps: Number of steps
            progress: Show progress

        Returns:
            Tuple of (chain, log_prob, acceptance_fraction)
        """
        chain = np.zeros((n_steps, self.n_params))
        log_probs = np.zeros(n_steps)
        n_accept = 0

        current = initial.copy()
        current_log_prob = self.log_prob_fn(current)

        for i in range(n_steps):
            # Propose
            proposal = self._propose(current)

            # Check prior bounds
            if not self.priors.in_bounds_vector(proposal):
                proposal_log_prob = -np.inf
            else:
                proposal_log_prob = self.log_prob_fn(proposal)

            # Accept/reject
            log_alpha = proposal_log_prob - current_log_prob

            if np.log(np.random.rand()) < log_alpha:
                current = proposal
                current_log_prob = proposal_log_prob
                n_accept += 1

            chain[i] = current
            log_probs[i] = current_log_prob

            if progress and (i + 1) % (n_steps // 10) == 0:
                print(f"Step {i+1}/{n_steps}, acceptance: {n_accept/(i+1):.2%}")

        return chain, log_probs, n_accept / n_steps


class MCMCSampler:
    """MCMC sampler with emcee backend (if available) or MH fallback."""

    def __init__(
        self,
        log_likelihood_fn: Callable[[Dict[str, float]], float],
        priors: Optional[PriorSet] = None,
        use_emcee: bool = True,
    ):
        """Initialize MCMC sampler.

        Args:
            log_likelihood_fn: Function computing log-likelihood from param dict
            priors: Prior distributions (default: HRC defaults)
            use_emcee: Try to use emcee if available
        """
        self.log_likelihood_fn = log_likelihood_fn
        self.priors = priors or default_hrc_priors()

        self._emcee_available = False
        if use_emcee:
            try:
                import emcee
                self._emcee_available = True
            except ImportError:
                warnings.warn(
                    "emcee not installed, using Metropolis-Hastings. "
                    "Install with: pip install emcee"
                )

    def _log_prob(self, theta: NDArray[np.floating]) -> float:
        """Compute log-posterior (prior + likelihood)."""
        # Check priors
        log_prior = self.priors.log_prob_vector(theta)
        if log_prior == -np.inf:
            return -np.inf

        # Convert to parameter dict
        params = dict(zip(self.priors.param_names, theta))

        # Compute likelihood
        try:
            log_like = self.log_likelihood_fn(params)
        except Exception:
            return -np.inf

        if np.isnan(log_like):
            return -np.inf

        return log_prior + log_like

    def run(
        self,
        n_steps: int = 1000,
        n_walkers: Optional[int] = None,
        initial: Optional[NDArray[np.floating]] = None,
        burn: int = 100,
        thin: int = 1,
        progress: bool = True,
    ) -> MCMCResult:
        """Run MCMC sampling.

        Args:
            n_steps: Number of steps per walker
            n_walkers: Number of walkers (default: 2 * n_params)
            initial: Initial positions (shape: n_walkers × n_params)
            burn: Burn-in steps (stored but can be discarded later)
            thin: Thinning factor
            progress: Show progress bar

        Returns:
            MCMCResult with chains and diagnostics
        """
        n_params = self.priors.n_params

        if n_walkers is None:
            n_walkers = max(2 * n_params, 32)

        if initial is None:
            # Sample from priors
            initial = self.priors.sample(n_walkers)

        if self._emcee_available:
            return self._run_emcee(n_steps, n_walkers, initial, progress)
        else:
            return self._run_mh(n_steps, n_walkers, initial, progress)

    def _run_emcee(
        self,
        n_steps: int,
        n_walkers: int,
        initial: NDArray[np.floating],
        progress: bool,
    ) -> MCMCResult:
        """Run with emcee."""
        import emcee

        n_params = self.priors.n_params
        sampler = emcee.EnsembleSampler(n_walkers, n_params, self._log_prob)

        # Run sampler
        sampler.run_mcmc(initial, n_steps, progress=progress)

        # Get chains
        chains = sampler.get_chain()  # (n_steps, n_walkers, n_params)
        chains = np.transpose(chains, (1, 0, 2))  # (n_walkers, n_steps, n_params)
        log_prob = sampler.get_log_prob().T  # (n_walkers, n_steps)

        # Compute autocorrelation time
        try:
            autocorr = sampler.get_autocorr_time(quiet=True)
        except Exception:
            autocorr = None

        return MCMCResult(
            chains=chains,
            log_prob=log_prob,
            param_names=self.priors.param_names,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_params=n_params,
            acceptance_fraction=np.mean(sampler.acceptance_fraction),
            autocorr_time=autocorr,
        )

    def _run_mh(
        self,
        n_steps: int,
        n_walkers: int,
        initial: NDArray[np.floating],
        progress: bool,
    ) -> MCMCResult:
        """Run with Metropolis-Hastings."""
        n_params = self.priors.n_params

        chains = np.zeros((n_walkers, n_steps, n_params))
        log_probs = np.zeros((n_walkers, n_steps))
        acceptance_fractions = []

        mh = MetropolisHastings(self._log_prob, self.priors)

        for i in range(n_walkers):
            if progress:
                print(f"Walker {i+1}/{n_walkers}")
            chain, lp, acc = mh.run(initial[i], n_steps, progress=False)
            chains[i] = chain
            log_probs[i] = lp
            acceptance_fractions.append(acc)

        return MCMCResult(
            chains=chains,
            log_prob=log_probs,
            param_names=self.priors.param_names,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_params=n_params,
            acceptance_fraction=np.mean(acceptance_fractions),
        )


def run_mcmc(
    log_likelihood_fn: Callable[[Dict[str, float]], float],
    priors: Optional[PriorSet] = None,
    n_steps: int = 1000,
    n_walkers: Optional[int] = None,
    progress: bool = True,
) -> MCMCResult:
    """Convenience function to run MCMC.

    Args:
        log_likelihood_fn: Log-likelihood function
        priors: Prior distributions
        n_steps: Number of steps
        n_walkers: Number of walkers
        progress: Show progress

    Returns:
        MCMCResult
    """
    sampler = MCMCSampler(log_likelihood_fn, priors)
    return sampler.run(n_steps, n_walkers, progress=progress)


def compute_gelman_rubin(chains: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute Gelman-Rubin R-hat convergence diagnostic.

    Args:
        chains: Shape (n_walkers, n_steps, n_params)

    Returns:
        R-hat values for each parameter
    """
    n_walkers, n_steps, n_params = chains.shape

    # Use second half of chains
    chains = chains[:, n_steps // 2:, :]
    n_steps = chains.shape[1]

    R_hat = np.zeros(n_params)

    for i in range(n_params):
        # Between-chain variance
        chain_means = np.mean(chains[:, :, i], axis=1)
        B = n_steps * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(chains[:, :, i], axis=1, ddof=1)
        W = np.mean(chain_vars)

        # Pooled variance estimate
        var_hat = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B

        # R-hat
        R_hat[i] = np.sqrt(var_hat / W) if W > 0 else np.inf

    return R_hat
