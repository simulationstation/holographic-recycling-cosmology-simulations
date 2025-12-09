"""H₀ likelihoods for HRC.

Implements likelihoods for various H₀ measurements:
- SH0ES (Cepheid distance ladder)
- TRGB (Tip of the Red Giant Branch)
- CMB distance priors (Planck)
- Time-delay cosmography (TDCOSMO)

In HRC, different probes can give different H₀ values due to
epoch-dependent G_eff.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ..utils.config import HRCParameters
from ..utils.constants import PLANCK_2018, SHOES_2024
from ..background import BackgroundSolution
from .distances import DistanceCalculator


@dataclass
class H0Measurement:
    """H₀ measurement from a specific probe."""

    name: str
    H0: float  # Central value [km/s/Mpc]
    sigma: float  # 1σ uncertainty
    z_effective: float  # Effective redshift probed
    reference: str = ""


# Standard measurements
SHOES_MEASUREMENT = H0Measurement(
    name="SH0ES",
    H0=73.04,
    sigma=1.04,
    z_effective=0.01,
    reference="Riess et al. 2024",
)

TRGB_MEASUREMENT = H0Measurement(
    name="TRGB",
    H0=69.8,
    sigma=1.7,
    z_effective=0.01,
    reference="Freedman et al. 2021",
)

PLANCK_CMB_MEASUREMENT = H0Measurement(
    name="Planck CMB",
    H0=67.36,
    sigma=0.54,
    z_effective=1089.0,
    reference="Planck Collaboration 2018",
)

TDCOSMO_MEASUREMENT = H0Measurement(
    name="TDCOSMO",
    H0=73.3,
    sigma=3.3,
    z_effective=0.5,
    reference="TDCOSMO Collaboration 2020",
)


class H0Likelihood(ABC):
    """Base class for H₀ likelihoods."""

    @abstractmethod
    def log_likelihood(
        self,
        H0_predicted: float,
    ) -> float:
        """Compute log-likelihood for predicted H₀."""
        pass

    @abstractmethod
    def chi2(
        self,
        H0_predicted: float,
    ) -> float:
        """Compute χ² contribution."""
        pass


class GaussianH0Likelihood(H0Likelihood):
    """Gaussian likelihood for H₀ measurement."""

    def __init__(self, measurement: H0Measurement):
        self.measurement = measurement
        self.H0_obs = measurement.H0
        self.sigma = measurement.sigma

    def log_likelihood(self, H0_predicted: float) -> float:
        """Compute Gaussian log-likelihood."""
        chi2 = self.chi2(H0_predicted)
        return -0.5 * chi2

    def chi2(self, H0_predicted: float) -> float:
        """Compute χ² = (H0_pred - H0_obs)² / σ²."""
        return ((H0_predicted - self.H0_obs) / self.sigma) ** 2


class SH0ESLikelihood(GaussianH0Likelihood):
    """SH0ES likelihood for local H₀."""

    def __init__(
        self,
        H0: float = SHOES_MEASUREMENT.H0,
        sigma: float = SHOES_MEASUREMENT.sigma,
    ):
        measurement = H0Measurement(
            name="SH0ES",
            H0=H0,
            sigma=sigma,
            z_effective=0.01,
        )
        super().__init__(measurement)


class TRGBLikelihood(GaussianH0Likelihood):
    """TRGB likelihood for local H₀."""

    def __init__(
        self,
        H0: float = TRGB_MEASUREMENT.H0,
        sigma: float = TRGB_MEASUREMENT.sigma,
    ):
        measurement = H0Measurement(
            name="TRGB",
            H0=H0,
            sigma=sigma,
            z_effective=0.01,
        )
        super().__init__(measurement)


class CMBDistanceLikelihood(H0Likelihood):
    """CMB distance prior likelihood.

    The CMB constrains the combination:
        θ_* = r_s(z_*) / D_A(z_*)

    where r_s is the sound horizon and D_A is the angular diameter
    distance to last scattering. This constrains a combination of
    H₀ and other parameters.
    """

    def __init__(
        self,
        theta_star: float = PLANCK_2018.theta_star,
        sigma_theta: float = PLANCK_2018.theta_star_sigma,
        z_star: float = PLANCK_2018.z_star,
        r_s: float = PLANCK_2018.r_s,
        sigma_rs: float = PLANCK_2018.r_s_sigma,
    ):
        self.theta_star_obs = theta_star
        self.sigma_theta = sigma_theta
        self.z_star = z_star
        self.r_s_obs = r_s
        self.sigma_rs = sigma_rs

    def compute_theta_star(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
    ) -> float:
        """Compute θ* from HRC parameters.

        θ* = r_s / D_M where D_M is the transverse comoving distance
        (equals comoving distance for flat universe).
        """
        calc = DistanceCalculator(params, background)
        D_M = calc.transverse_comoving_distance(self.z_star)

        # Use observed r_s (or compute from HRC if available)
        r_s = self.r_s_obs

        return r_s / D_M

    def log_likelihood(
        self,
        H0_predicted: float,
        params: Optional[HRCParameters] = None,
        theta_star_predicted: Optional[float] = None,
    ) -> float:
        """Compute CMB distance prior log-likelihood.

        Can use either predicted θ* directly or compute from H₀.
        """
        if theta_star_predicted is not None:
            theta = theta_star_predicted
        elif params is not None:
            theta = self.compute_theta_star(params)
        else:
            # Fall back to simple H₀ scaling
            # θ* ∝ H₀^(-0.4) approximately
            theta = self.theta_star_obs * (PLANCK_2018.H0 / H0_predicted) ** 0.4

        chi2 = ((theta - self.theta_star_obs) / self.sigma_theta) ** 2
        return -0.5 * chi2

    def chi2(
        self,
        H0_predicted: float,
        params: Optional[HRCParameters] = None,
    ) -> float:
        """Compute χ² contribution."""
        return -2 * self.log_likelihood(H0_predicted, params)


def compute_hrc_h0_predictions(
    params: HRCParameters,
    background: BackgroundSolution,
    H0_true: float = 70.0,
) -> dict:
    """Compute H₀ predictions for different probes in HRC.

    In HRC, different probes measure different effective H₀ values
    due to epoch-dependent G_eff.

    Args:
        params: HRC parameters
        background: Background cosmology solution
        H0_true: Fiducial true H₀

    Returns:
        Dictionary with H₀ predictions for each probe
    """
    # G_eff at different epochs
    G_eff_0 = background.G_eff_at(0.0)
    G_eff_cmb = background.G_eff_at(1089.0)
    G_eff_bao = background.G_eff_at(0.5)  # Typical BAO redshift

    # Local measurement: directly measures H(z≈0)
    # H² ∝ G_eff × ρ, so H ∝ √G_eff for fixed ρ
    H0_local = H0_true * np.sqrt(G_eff_0)

    # CMB inference: assumes constant G to infer H₀
    # More complex relation, see effective_gravity.py
    Delta_G = (G_eff_0 - G_eff_cmb) / G_eff_cmb
    H0_cmb = H0_true * (1 + 0.4 * Delta_G)

    # BAO: measures combination of H(z) and D_A(z)
    # Intermediate between local and CMB
    H0_bao = H0_true * np.sqrt(G_eff_bao)

    # Standard sirens: measure luminosity distance directly
    # dL ∝ ∫ dz/H(z), so depends on G_eff evolution
    H0_sirens = H0_local  # Similar to local for low-z sirens

    return {
        "H0_local": H0_local,
        "H0_cmb": H0_cmb,
        "H0_bao": H0_bao,
        "H0_sirens": H0_sirens,
        "Delta_H0": H0_local - H0_cmb,
        "G_eff_0": G_eff_0,
        "G_eff_cmb": G_eff_cmb,
    }


def joint_h0_likelihood(
    params: HRCParameters,
    background: BackgroundSolution,
    include_shoes: bool = True,
    include_trgb: bool = False,
    include_cmb: bool = True,
    H0_true: float = 70.0,
) -> float:
    """Compute joint log-likelihood from H₀ measurements.

    Args:
        params: HRC parameters
        background: Background solution
        include_shoes: Include SH0ES likelihood
        include_trgb: Include TRGB likelihood
        include_cmb: Include CMB distance prior
        H0_true: Fiducial true H₀

    Returns:
        Total log-likelihood
    """
    predictions = compute_hrc_h0_predictions(params, background, H0_true)

    total_logL = 0.0

    if include_shoes:
        shoes = SH0ESLikelihood()
        total_logL += shoes.log_likelihood(predictions["H0_local"])

    if include_trgb:
        trgb = TRGBLikelihood()
        total_logL += trgb.log_likelihood(predictions["H0_local"])

    if include_cmb:
        cmb = CMBDistanceLikelihood()
        total_logL += cmb.log_likelihood(predictions["H0_cmb"], params)

    return total_logL
