"""BAO (Baryon Acoustic Oscillation) likelihoods for HRC.

BAO provides a standard ruler through the sound horizon scale r_s.
Measurements constrain:
- D_V(z)/r_s: Volume-averaged distance
- D_A(z)/r_s: Angular diameter distance
- H(z)r_s: Expansion rate

In HRC, r_s and distances are modified by G_eff evolution.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..utils.constants import PLANCK_2018
from ..background import BackgroundSolution
from .distances import DistanceCalculator


@dataclass
class BAODataPoint:
    """A single BAO measurement."""

    z: float  # Effective redshift
    observable: str  # Type: 'DV_rs', 'DA_rs', 'DH_rs', 'DM_rs', 'Hz_rs'
    value: float  # Measured value
    sigma: float  # 1σ uncertainty
    survey: str = ""
    reference: str = ""


# DESI 2024 BAO measurements
DESI_BAO_DATA: List[BAODataPoint] = [
    BAODataPoint(z=0.51, observable="DM_rs", value=13.62, sigma=0.25, survey="DESI-LRG"),
    BAODataPoint(z=0.51, observable="DH_rs", value=20.98, sigma=0.61, survey="DESI-LRG"),
    BAODataPoint(z=0.71, observable="DM_rs", value=16.85, sigma=0.32, survey="DESI-LRG"),
    BAODataPoint(z=0.71, observable="DH_rs", value=20.08, sigma=0.60, survey="DESI-LRG"),
    BAODataPoint(z=0.93, observable="DM_rs", value=21.71, sigma=0.28, survey="DESI-LRG"),
    BAODataPoint(z=0.93, observable="DH_rs", value=17.88, sigma=0.35, survey="DESI-LRG"),
    BAODataPoint(z=1.32, observable="DM_rs", value=27.79, sigma=0.69, survey="DESI-QSO"),
    BAODataPoint(z=1.32, observable="DH_rs", value=13.82, sigma=0.42, survey="DESI-QSO"),
    BAODataPoint(z=2.33, observable="DM_rs", value=39.71, sigma=0.94, survey="DESI-Lya"),
    BAODataPoint(z=2.33, observable="DH_rs", value=8.52, sigma=0.17, survey="DESI-Lya"),
]

# BOSS DR12 BAO measurements
BOSS_BAO_DATA: List[BAODataPoint] = [
    BAODataPoint(z=0.38, observable="DM_rs", value=10.27, sigma=0.15, survey="BOSS-LOWZ"),
    BAODataPoint(z=0.38, observable="DH_rs", value=25.00, sigma=0.76, survey="BOSS-LOWZ"),
    BAODataPoint(z=0.51, observable="DM_rs", value=13.38, sigma=0.18, survey="BOSS-CMASS"),
    BAODataPoint(z=0.51, observable="DH_rs", value=22.33, sigma=0.58, survey="BOSS-CMASS"),
    BAODataPoint(z=0.61, observable="DM_rs", value=15.45, sigma=0.20, survey="BOSS-CMASS"),
    BAODataPoint(z=0.61, observable="DH_rs", value=20.65, sigma=0.52, survey="BOSS-CMASS"),
]


class BAOLikelihood:
    """BAO likelihood for HRC cosmology.

    Computes log-likelihood from BAO distance measurements,
    accounting for HRC modifications to both r_s and distances.
    """

    def __init__(
        self,
        data: Optional[List[BAODataPoint]] = None,
        r_s_fid: float = PLANCK_2018.r_s,
    ):
        """Initialize BAO likelihood.

        Args:
            data: List of BAO measurements (default: DESI + BOSS)
            r_s_fid: Fiducial sound horizon [Mpc]
        """
        if data is None:
            data = DESI_BAO_DATA + BOSS_BAO_DATA
        self.data = data
        self.r_s_fid = r_s_fid

    def compute_observable(
        self,
        data_point: BAODataPoint,
        calc: DistanceCalculator,
        r_s: float,
    ) -> float:
        """Compute predicted BAO observable.

        Args:
            data_point: BAO measurement
            calc: Distance calculator
            r_s: Sound horizon [Mpc]

        Returns:
            Predicted value of observable
        """
        z = data_point.z
        obs_type = data_point.observable

        if obs_type == "DM_rs":
            # Comoving distance / r_s
            d_M = calc.transverse_comoving_distance(z)
            return d_M / r_s

        elif obs_type == "DA_rs":
            # Angular diameter distance / r_s
            d_A = calc.angular_diameter_distance(z)
            return d_A / r_s

        elif obs_type == "DH_rs":
            # Hubble distance / r_s = c / (H(z) r_s)
            c_km_s = 299792.458
            H = calc.H(z)
            return c_km_s / (H * r_s)

        elif obs_type == "DV_rs":
            # Volume-averaged distance
            # D_V = [z D_H D_M²]^(1/3)
            d_M = calc.transverse_comoving_distance(z)
            c_km_s = 299792.458
            d_H = c_km_s / calc.H(z)
            d_V = (z * d_H * d_M**2) ** (1.0 / 3.0)
            return d_V / r_s

        elif obs_type == "Hz_rs":
            # H(z) × r_s
            return calc.H(z) * r_s

        else:
            raise ValueError(f"Unknown BAO observable: {obs_type}")

    def log_likelihood(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        r_s: Optional[float] = None,
    ) -> float:
        """Compute log-likelihood from BAO data.

        Args:
            params: HRC parameters
            background: Background solution (for H(z))
            r_s: Sound horizon (default: fiducial value)

        Returns:
            Log-likelihood
        """
        if r_s is None:
            r_s = self.r_s_fid

        calc = DistanceCalculator(params, background)

        chi2 = 0.0
        for point in self.data:
            pred = self.compute_observable(point, calc, r_s)
            chi2 += ((pred - point.value) / point.sigma) ** 2

        return -0.5 * chi2

    def chi2(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        r_s: Optional[float] = None,
    ) -> float:
        """Compute total χ² from BAO data."""
        return -2 * self.log_likelihood(params, background, r_s)

    def chi2_per_point(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        r_s: Optional[float] = None,
    ) -> List[Tuple[BAODataPoint, float, float]]:
        """Compute χ² for each data point.

        Returns:
            List of (data_point, predicted_value, chi2_contribution)
        """
        if r_s is None:
            r_s = self.r_s_fid

        calc = DistanceCalculator(params, background)

        results = []
        for point in self.data:
            pred = self.compute_observable(point, calc, r_s)
            chi2 = ((pred - point.value) / point.sigma) ** 2
            results.append((point, pred, chi2))

        return results


def compute_bao_observables(
    params: HRCParameters,
    z_array: NDArray[np.floating],
    background: Optional[BackgroundSolution] = None,
    r_s: float = PLANCK_2018.r_s,
) -> dict:
    """Compute BAO observables over redshift range.

    Args:
        params: HRC parameters
        z_array: Redshift array
        background: Background solution
        r_s: Sound horizon [Mpc]

    Returns:
        Dictionary with observable arrays
    """
    calc = DistanceCalculator(params, background)
    c_km_s = 299792.458

    n = len(z_array)
    DM_rs = np.zeros(n)
    DA_rs = np.zeros(n)
    DH_rs = np.zeros(n)
    DV_rs = np.zeros(n)
    Hz_rs = np.zeros(n)

    for i, z in enumerate(z_array):
        d_M = calc.transverse_comoving_distance(z)
        d_A = calc.angular_diameter_distance(z)
        H = calc.H(z)
        d_H = c_km_s / H

        DM_rs[i] = d_M / r_s
        DA_rs[i] = d_A / r_s
        DH_rs[i] = d_H / r_s
        DV_rs[i] = (z * d_H * d_M**2) ** (1.0 / 3.0) / r_s if z > 0 else 0
        Hz_rs[i] = H * r_s

    return {
        "z": z_array,
        "DM_rs": DM_rs,
        "DA_rs": DA_rs,
        "DH_rs": DH_rs,
        "DV_rs": DV_rs,
        "Hz_rs": Hz_rs,
        "r_s": r_s,
    }
