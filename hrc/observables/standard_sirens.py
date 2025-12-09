"""Standard siren likelihoods for HRC.

Gravitational wave sources with electromagnetic counterparts provide
"standard sirens" - direct measurements of luminosity distance without
the cosmic distance ladder.

In HRC, standard sirens probe:
- Luminosity distance (depends on H(z) evolution)
- GW propagation (modified if G_eff evolves)
- H₀ directly (for low-z events like GW170817)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..background import BackgroundSolution
from .distances import DistanceCalculator


@dataclass
class GWEvent:
    """A gravitational wave event with distance measurement."""

    name: str
    z: float  # Redshift
    z_err: float  # Redshift uncertainty
    d_L: float  # Luminosity distance [Mpc]
    d_L_err_low: float  # Lower 1σ error [Mpc]
    d_L_err_high: float  # Upper 1σ error [Mpc]
    has_EM_counterpart: bool = True
    event_type: str = "BNS"  # BNS, NSBH, BBH


# Confirmed standard sirens
STANDARD_SIREN_EVENTS: List[GWEvent] = [
    GWEvent(
        name="GW170817",
        z=0.0099,
        z_err=0.0003,
        d_L=40.0,
        d_L_err_low=8.0,
        d_L_err_high=14.0,
        has_EM_counterpart=True,
        event_type="BNS",
    ),
    # Statistical standard sirens (dark sirens)
    GWEvent(
        name="GW190814",
        z=0.053,
        z_err=0.010,
        d_L=241.0,
        d_L_err_low=41.0,
        d_L_err_high=45.0,
        has_EM_counterpart=False,
        event_type="NSBH",
    ),
]


@dataclass
class SirenH0Result:
    """Result of H₀ inference from standard sirens."""

    H0: float  # Best-fit H₀
    H0_err_low: float  # Lower 1σ error
    H0_err_high: float  # Upper 1σ error
    n_events: int
    chi2: float


class StandardSirenLikelihood:
    """Standard siren likelihood for H₀ and cosmology.

    Computes log-likelihood from GW distance measurements.
    """

    def __init__(
        self,
        events: Optional[List[GWEvent]] = None,
        only_bright_sirens: bool = True,
    ):
        """Initialize standard siren likelihood.

        Args:
            events: List of GW events (default: confirmed sirens)
            only_bright_sirens: Only use events with EM counterparts
        """
        if events is None:
            events = STANDARD_SIREN_EVENTS

        if only_bright_sirens:
            events = [e for e in events if e.has_EM_counterpart]

        self.events = events

    def log_likelihood_H0(
        self,
        H0: float,
        Omega_m: float = 0.3,
    ) -> float:
        """Compute log-likelihood for H₀ from standard sirens.

        Uses simple flat ΛCDM to compute expected distances.

        Args:
            H0: Hubble constant [km/s/Mpc]
            Omega_m: Matter density parameter

        Returns:
            Log-likelihood
        """
        total_logL = 0.0

        for event in self.events:
            # Compute expected d_L
            from .distances import luminosity_distance
            d_L_pred = luminosity_distance(
                event.z, H0=H0, Omega_m=Omega_m, Omega_Lambda=1-Omega_m
            )

            # Asymmetric Gaussian likelihood
            if d_L_pred > event.d_L:
                sigma = event.d_L_err_high
            else:
                sigma = event.d_L_err_low

            # Add redshift uncertainty contribution
            # δd_L/d_L ≈ δz/z for low z
            sigma_z_contrib = event.d_L * event.z_err / max(event.z, 0.001)
            sigma_total = np.sqrt(sigma**2 + sigma_z_contrib**2)

            chi2 = ((d_L_pred - event.d_L) / sigma_total) ** 2
            total_logL -= 0.5 * chi2

        return total_logL

    def log_likelihood(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
    ) -> float:
        """Compute log-likelihood from HRC model.

        Args:
            params: HRC parameters
            background: Background solution

        Returns:
            Log-likelihood
        """
        calc = DistanceCalculator(params, background)

        total_logL = 0.0

        for event in self.events:
            d_L_pred = calc.luminosity_distance(event.z)

            if d_L_pred > event.d_L:
                sigma = event.d_L_err_high
            else:
                sigma = event.d_L_err_low

            sigma_z_contrib = event.d_L * event.z_err / max(event.z, 0.001)
            sigma_total = np.sqrt(sigma**2 + sigma_z_contrib**2)

            chi2 = ((d_L_pred - event.d_L) / sigma_total) ** 2
            total_logL -= 0.5 * chi2

        return total_logL

    def infer_H0(
        self,
        Omega_m: float = 0.3,
        H0_prior: Tuple[float, float] = (40.0, 120.0),
        n_samples: int = 1000,
    ) -> SirenH0Result:
        """Infer H₀ from standard sirens.

        Args:
            Omega_m: Fixed matter density
            H0_prior: Uniform prior range on H₀
            n_samples: Number of samples for integration

        Returns:
            SirenH0Result with H₀ constraints
        """
        H0_grid = np.linspace(H0_prior[0], H0_prior[1], n_samples)
        logL = np.array([self.log_likelihood_H0(H0, Omega_m) for H0 in H0_grid])

        # Convert to probability
        logL_max = np.max(logL)
        prob = np.exp(logL - logL_max)
        prob /= np.trapz(prob, H0_grid)

        # Compute statistics
        H0_mean = np.trapz(H0_grid * prob, H0_grid)
        H0_var = np.trapz((H0_grid - H0_mean)**2 * prob, H0_grid)
        H0_std = np.sqrt(H0_var)

        # Find percentiles
        cdf = np.cumsum(prob)
        cdf /= cdf[-1]

        H0_median = H0_grid[np.searchsorted(cdf, 0.5)]
        H0_16 = H0_grid[np.searchsorted(cdf, 0.16)]
        H0_84 = H0_grid[np.searchsorted(cdf, 0.84)]

        # Chi2 at best fit
        H0_best = H0_grid[np.argmax(prob)]
        chi2 = -2 * self.log_likelihood_H0(H0_best, Omega_m)

        return SirenH0Result(
            H0=H0_median,
            H0_err_low=H0_median - H0_16,
            H0_err_high=H0_84 - H0_median,
            n_events=len(self.events),
            chi2=chi2,
        )


def simulate_future_sirens(
    n_events: int = 50,
    z_max: float = 0.5,
    d_L_err_frac: float = 0.10,
    H0_true: float = 70.0,
    Omega_m: float = 0.3,
) -> List[GWEvent]:
    """Simulate future standard siren events.

    Args:
        n_events: Number of events
        z_max: Maximum redshift
        d_L_err_frac: Fractional distance error
        H0_true: True H₀ for simulation
        Omega_m: Matter density

    Returns:
        List of simulated GWEvent objects
    """
    from .distances import luminosity_distance

    np.random.seed(42)  # Reproducible

    events = []
    for i in range(n_events):
        # Uniform in comoving volume ∝ d_C² × (1+z)⁻¹
        # Approximate: uniform in z for simplicity
        z = np.random.uniform(0.01, z_max)
        z_err = 0.001 * (1 + z)  # Typical spec-z error

        d_L_true = luminosity_distance(z, H0_true, Omega_m, 1-Omega_m)

        # Add scatter
        d_L_err = d_L_err_frac * d_L_true
        d_L_obs = d_L_true + np.random.normal(0, d_L_err)

        events.append(GWEvent(
            name=f"SIM_{i:03d}",
            z=z,
            z_err=z_err,
            d_L=d_L_obs,
            d_L_err_low=d_L_err,
            d_L_err_high=d_L_err,
            has_EM_counterpart=True,
            event_type="BNS",
        ))

    return events


def forecast_h0_precision(
    n_events: int = 50,
    z_max: float = 0.5,
    d_L_err_frac: float = 0.10,
) -> dict:
    """Forecast H₀ precision from future standard sirens.

    Args:
        n_events: Number of events
        z_max: Maximum redshift
        d_L_err_frac: Fractional distance error

    Returns:
        Dictionary with forecast results
    """
    events = simulate_future_sirens(n_events, z_max, d_L_err_frac)
    likelihood = StandardSirenLikelihood(events, only_bright_sirens=False)
    result = likelihood.infer_H0()

    # Expected precision
    # σ(H0) ∝ 1/√N approximately
    expected_sigma = result.H0_err_high  # Use actual scatter

    return {
        "n_events": n_events,
        "z_max": z_max,
        "d_L_err_frac": d_L_err_frac,
        "H0_inferred": result.H0,
        "H0_sigma_low": result.H0_err_low,
        "H0_sigma_high": result.H0_err_high,
        "chi2": result.chi2,
    }
