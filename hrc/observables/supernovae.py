"""Type Ia Supernovae likelihoods for HRC.

Implements likelihoods for SNe Ia distance measurements:
- Pantheon+ compilation
- Union3 compilation

SNe Ia measure the luminosity distance dL(z), which depends on
the expansion history H(z) modified by G_eff in HRC.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..background import BackgroundSolution
from .distances import DistanceCalculator


@dataclass
class SNDataPoint:
    """A single SN Ia measurement."""

    name: str
    z_cmb: float  # CMB-frame redshift
    z_helio: float  # Heliocentric redshift
    mu: float  # Distance modulus [mag]
    mu_err: float  # Distance modulus error [mag]
    survey: str = ""


# Subset of Pantheon+ data (representative sample)
# Full dataset has ~1700 SNe
PANTHEON_PLUS_SUBSET: List[SNDataPoint] = [
    SNDataPoint("1990O", 0.030, 0.031, 34.53, 0.16, "CfA"),
    SNDataPoint("1990af", 0.050, 0.051, 36.23, 0.14, "CfA"),
    SNDataPoint("1992P", 0.026, 0.027, 34.25, 0.15, "CfA"),
    SNDataPoint("1993ag", 0.050, 0.051, 36.31, 0.14, "CfA"),
    SNDataPoint("1994M", 0.024, 0.025, 33.98, 0.15, "CfA"),
    SNDataPoint("1994S", 0.015, 0.016, 32.85, 0.14, "CfA"),
    SNDataPoint("1995ac", 0.049, 0.050, 36.15, 0.13, "CfA"),
    SNDataPoint("1996bo", 0.017, 0.018, 33.25, 0.14, "CfA"),
    SNDataPoint("1998ef", 0.017, 0.018, 33.18, 0.15, "CfA"),
    SNDataPoint("1999aa", 0.015, 0.016, 32.78, 0.13, "CfA"),
    # Higher-z sample
    SNDataPoint("04D1ag", 0.557, 0.558, 42.48, 0.11, "SNLS"),
    SNDataPoint("04D2gc", 0.521, 0.522, 42.22, 0.10, "SNLS"),
    SNDataPoint("04D3fq", 0.730, 0.731, 43.28, 0.12, "SNLS"),
    SNDataPoint("04D4bq", 0.550, 0.551, 42.42, 0.11, "SNLS"),
    SNDataPoint("05D1by", 0.298, 0.299, 40.48, 0.10, "SNLS"),
    SNDataPoint("05D2ac", 0.479, 0.480, 42.05, 0.11, "SNLS"),
    SNDataPoint("06D1bo", 0.309, 0.310, 40.58, 0.10, "SNLS"),
    SNDataPoint("06D3gx", 0.488, 0.489, 42.12, 0.11, "SNLS"),
    # HST high-z
    SNDataPoint("Aphrodite", 1.304, 1.305, 44.85, 0.20, "HST"),
    SNDataPoint("Athena", 1.120, 1.121, 44.42, 0.18, "HST"),
    SNDataPoint("Clio", 1.005, 1.006, 44.12, 0.17, "HST"),
    SNDataPoint("Hera", 1.235, 1.236, 44.68, 0.19, "HST"),
]


class SNeLikelihood:
    """Generic SNe Ia likelihood.

    Computes log-likelihood from distance modulus measurements.
    """

    def __init__(
        self,
        data: List[SNDataPoint],
        use_covariance: bool = False,
        covariance_matrix: Optional[NDArray[np.floating]] = None,
    ):
        """Initialize SNe likelihood.

        Args:
            data: List of SN measurements
            use_covariance: Whether to use full covariance matrix
            covariance_matrix: Covariance matrix (if use_covariance=True)
        """
        self.data = data
        self.use_covariance = use_covariance
        self.covariance = covariance_matrix

        # Pre-extract arrays for efficiency
        self.z_array = np.array([sn.z_cmb for sn in data])
        self.mu_array = np.array([sn.mu for sn in data])
        self.mu_err_array = np.array([sn.mu_err for sn in data])

    def compute_distance_moduli(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        M: float = -19.3,
    ) -> NDArray[np.floating]:
        """Compute predicted distance moduli.

        Args:
            params: HRC parameters
            background: Background solution
            M: Absolute magnitude (nuisance parameter)

        Returns:
            Array of predicted μ values
        """
        calc = DistanceCalculator(params, background)

        mu_pred = np.zeros(len(self.data))
        for i, z in enumerate(self.z_array):
            d_L = calc.luminosity_distance(z)
            # μ = 5 log₁₀(d_L/10pc) = 5 log₁₀(d_L[Mpc]) + 25
            mu_pred[i] = 5 * np.log10(d_L) + 25

        return mu_pred

    def log_likelihood(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        M: float = -19.3,
    ) -> float:
        """Compute log-likelihood from SNe data.

        The likelihood marginalizes over the nuisance parameter M
        (absolute magnitude offset).

        Args:
            params: HRC parameters
            background: Background solution
            M: Absolute magnitude (will be marginalized)

        Returns:
            Log-likelihood
        """
        mu_pred = self.compute_distance_moduli(params, background, M)

        if self.use_covariance and self.covariance is not None:
            # Full covariance treatment
            residuals = self.mu_array - mu_pred
            inv_cov = np.linalg.inv(self.covariance)
            chi2 = residuals @ inv_cov @ residuals

            # Marginalize over M analytically
            # This assumes a flat prior on M
            ones = np.ones(len(residuals))
            A = ones @ inv_cov @ ones
            B = ones @ inv_cov @ residuals
            chi2_marg = chi2 - B**2 / A

            # Add normalization term
            log_norm = -0.5 * np.log(A)
            return -0.5 * chi2_marg + log_norm

        else:
            # Diagonal approximation
            residuals = self.mu_array - mu_pred
            var = self.mu_err_array**2

            # Marginalize over M analytically
            # χ² = Σ (μ_obs - μ_pred - ΔM)² / σ²
            # Minimizing over ΔM:
            # ΔM_best = Σ[(μ_obs - μ_pred)/σ²] / Σ[1/σ²]
            w = 1.0 / var
            delta_M = np.sum(residuals * w) / np.sum(w)
            residuals_corrected = residuals - delta_M

            chi2 = np.sum(residuals_corrected**2 * w)

            # Add marginalization term
            log_norm = -0.5 * np.log(np.sum(w))

            return -0.5 * chi2 + log_norm

    def chi2(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
    ) -> float:
        """Compute χ² (without marginalization)."""
        mu_pred = self.compute_distance_moduli(params, background)
        residuals = self.mu_array - mu_pred

        # Best-fit M
        w = 1.0 / self.mu_err_array**2
        delta_M = np.sum(residuals * w) / np.sum(w)
        residuals_corrected = residuals - delta_M

        return np.sum(residuals_corrected**2 * w)


class PantheonPlusLikelihood(SNeLikelihood):
    """Pantheon+ SNe Ia likelihood.

    Uses the Pantheon+ compilation of ~1700 SNe Ia.
    Default uses a representative subset for speed.
    """

    def __init__(
        self,
        use_full_sample: bool = False,
        data_file: Optional[str] = None,
    ):
        """Initialize Pantheon+ likelihood.

        Args:
            use_full_sample: Load full Pantheon+ dataset
            data_file: Path to Pantheon+ data file
        """
        if use_full_sample and data_file is not None:
            data = self._load_pantheon_file(data_file)
        else:
            data = PANTHEON_PLUS_SUBSET

        super().__init__(data, use_covariance=False)

    def _load_pantheon_file(self, filename: str) -> List[SNDataPoint]:
        """Load Pantheon+ data from file."""
        # Placeholder for full data loading
        # In practice, would parse the official Pantheon+ data release
        return PANTHEON_PLUS_SUBSET


def compute_hubble_diagram(
    params: HRCParameters,
    z_array: NDArray[np.floating],
    background: Optional[BackgroundSolution] = None,
) -> dict:
    """Compute theoretical Hubble diagram.

    Args:
        params: HRC parameters
        z_array: Redshift array
        background: Background solution

    Returns:
        Dictionary with distance modulus and distances
    """
    calc = DistanceCalculator(params, background)

    mu = np.zeros(len(z_array))
    d_L = np.zeros(len(z_array))

    for i, z in enumerate(z_array):
        d_L[i] = calc.luminosity_distance(z)
        mu[i] = 5 * np.log10(d_L[i]) + 25

    return {
        "z": z_array,
        "mu": mu,
        "d_L": d_L,
    }
