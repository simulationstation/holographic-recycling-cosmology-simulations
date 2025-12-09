"""Cosmological distance calculations for HRC.

Implements the standard cosmological distances with HRC modifications:
- Comoving distance
- Angular diameter distance
- Luminosity distance
- Sound horizon

All distances depend on H(z), which is modified in HRC due to G_eff(z).
"""

from dataclasses import dataclass
from typing import Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import interp1d

from ..utils.config import HRCParameters
from ..utils.constants import SI_UNITS
from ..background import BackgroundSolution


@dataclass
class CosmologicalDistances:
    """Collection of cosmological distances at a redshift."""

    z: float
    d_C: float  # Comoving distance [Mpc]
    d_A: float  # Angular diameter distance [Mpc]
    d_L: float  # Luminosity distance [Mpc]
    d_H: float  # Hubble distance c/H(z) [Mpc]
    H: float  # Hubble parameter [km/s/Mpc]


class DistanceCalculator:
    """Calculate cosmological distances in HRC.

    Distances are computed by integrating:
        d_C(z) = c ∫₀ᶻ dz' / H(z')

    where H(z) includes HRC modifications.
    """

    def __init__(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        H_func: Optional[Callable[[float], float]] = None,
    ):
        """Initialize distance calculator.

        Args:
            params: HRC parameters
            background: Background solution (preferred for H(z))
            H_func: Custom H(z) function (alternative to background)
        """
        self.params = params
        self.background = background

        if background is not None:
            self._H_func = lambda z: background.H_at(z) * params.H0
        elif H_func is not None:
            self._H_func = H_func
        else:
            self._H_func = self._H_LCDM

        # Speed of light in km/s
        self._c_km_s = SI_UNITS.c / 1e3

    def _H_LCDM(self, z: float) -> float:
        """ΛCDM Hubble parameter [km/s/Mpc]."""
        Om = self.params.Omega_m
        Or = self.params.Omega_r
        Ok = self.params.Omega_k
        OL = self.params.Omega_Lambda
        H0 = self.params.H0

        E_z = np.sqrt(
            Om * (1 + z) ** 3
            + Or * (1 + z) ** 4
            + Ok * (1 + z) ** 2
            + OL
        )
        return H0 * E_z

    def H(self, z: float) -> float:
        """Hubble parameter H(z) [km/s/Mpc]."""
        return self._H_func(z)

    def E(self, z: float) -> float:
        """Dimensionless Hubble parameter E(z) = H(z)/H₀."""
        return self.H(z) / self.params.H0

    def hubble_distance(self, z: float) -> float:
        """Hubble distance d_H(z) = c/H(z) [Mpc]."""
        return self._c_km_s / self.H(z)

    def comoving_distance(
        self,
        z: float,
        z_min: float = 0.0,
    ) -> float:
        """Comoving distance d_C(z) [Mpc].

        d_C(z) = c ∫_{z_min}^{z} dz' / H(z')

        Args:
            z: Redshift
            z_min: Starting redshift (default 0)

        Returns:
            Comoving distance in Mpc
        """
        if z <= z_min:
            return 0.0

        integrand = lambda zp: self._c_km_s / self.H(zp)
        result, _ = quad(integrand, z_min, z, limit=200)
        return result

    def transverse_comoving_distance(self, z: float) -> float:
        """Transverse comoving distance d_M(z) [Mpc].

        Accounts for spatial curvature:
        - d_M = d_C for flat universe
        - d_M = (c/H0)/sqrt(Ωk) sinh(sqrt(Ωk) H0 d_C/c) for open
        - d_M = (c/H0)/sqrt(-Ωk) sin(sqrt(-Ωk) H0 d_C/c) for closed
        """
        d_C = self.comoving_distance(z)
        Ok = self.params.Omega_k
        H0 = self.params.H0

        if abs(Ok) < 1e-6:
            return d_C

        d_H = self._c_km_s / H0  # Hubble distance at z=0

        if Ok > 0:  # Open
            return d_H / np.sqrt(Ok) * np.sinh(np.sqrt(Ok) * d_C / d_H)
        else:  # Closed
            return d_H / np.sqrt(-Ok) * np.sin(np.sqrt(-Ok) * d_C / d_H)

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance d_A(z) [Mpc].

        d_A = d_M / (1+z)
        """
        return self.transverse_comoving_distance(z) / (1 + z)

    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance d_L(z) [Mpc].

        d_L = d_M × (1+z)
        """
        return self.transverse_comoving_distance(z) * (1 + z)

    def distance_modulus(self, z: float) -> float:
        """Distance modulus μ(z) [mag].

        μ = 5 log₁₀(d_L/10pc) = 5 log₁₀(d_L[Mpc]) + 25
        """
        d_L = self.luminosity_distance(z)
        return 5 * np.log10(d_L) + 25

    def get_distances(self, z: float) -> CosmologicalDistances:
        """Get all distances at a redshift."""
        d_C = self.comoving_distance(z)
        d_A = self.angular_diameter_distance(z)
        d_L = self.luminosity_distance(z)
        d_H = self.hubble_distance(z)
        H = self.H(z)

        return CosmologicalDistances(
            z=z, d_C=d_C, d_A=d_A, d_L=d_L, d_H=d_H, H=H
        )

    def compute_distance_table(
        self,
        z_array: NDArray[np.floating],
    ) -> dict:
        """Compute distances at array of redshifts.

        Args:
            z_array: Array of redshifts

        Returns:
            Dictionary with distance arrays
        """
        n = len(z_array)
        d_C = np.zeros(n)
        d_A = np.zeros(n)
        d_L = np.zeros(n)
        d_H = np.zeros(n)
        H = np.zeros(n)

        for i, z in enumerate(z_array):
            dists = self.get_distances(z)
            d_C[i] = dists.d_C
            d_A[i] = dists.d_A
            d_L[i] = dists.d_L
            d_H[i] = dists.d_H
            H[i] = dists.H

        return {
            "z": z_array,
            "d_C": d_C,
            "d_A": d_A,
            "d_L": d_L,
            "d_H": d_H,
            "H": H,
        }


def comoving_distance(
    z: float,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_Lambda: float = 0.7,
) -> float:
    """Quick comoving distance calculation [Mpc].

    Uses flat ΛCDM approximation.
    """
    c_km_s = SI_UNITS.c / 1e3

    def integrand(zp):
        E = np.sqrt(Omega_m * (1 + zp) ** 3 + Omega_Lambda)
        return c_km_s / (H0 * E)

    result, _ = quad(integrand, 0, z)
    return result


def angular_diameter_distance(
    z: float,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_Lambda: float = 0.7,
) -> float:
    """Quick angular diameter distance calculation [Mpc]."""
    d_C = comoving_distance(z, H0, Omega_m, Omega_Lambda)
    return d_C / (1 + z)


def luminosity_distance(
    z: float,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_Lambda: float = 0.7,
) -> float:
    """Quick luminosity distance calculation [Mpc]."""
    d_C = comoving_distance(z, H0, Omega_m, Omega_Lambda)
    return d_C * (1 + z)


def sound_horizon(
    z_drag: float = 1059.94,
    Omega_b_h2: float = 0.02237,
    Omega_m_h2: float = 0.1430,
    h: float = 0.6736,
) -> float:
    """Compute sound horizon at drag epoch [Mpc].

    Uses fitting formula from Eisenstein & Hu (1998).

    Args:
        z_drag: Redshift at drag epoch
        Omega_b_h2: Physical baryon density
        Omega_m_h2: Physical matter density
        h: Dimensionless Hubble parameter

    Returns:
        Sound horizon r_s in Mpc
    """
    # Fitting formula parameters
    Omega_b = Omega_b_h2 / h**2
    Omega_m = Omega_m_h2 / h**2

    # Baryon-to-photon ratio parameter
    R_eq = 31500 * Omega_b_h2 * (2.725 / 2.7) ** (-4) / (1 + z_drag)

    # Sound speed at drag
    c_s_drag = 1.0 / np.sqrt(3 * (1 + R_eq))

    # Approximate sound horizon
    # r_s ≈ c_s × t_drag ≈ c_s × (age at drag)
    H0 = 100 * h  # km/s/Mpc

    # More accurate fitting formula
    z_eq = 2.5e4 * Omega_m_h2 * (2.725 / 2.7) ** (-4)

    k_eq = 7.46e-2 * Omega_m_h2 * (2.725 / 2.7) ** (-2)  # Mpc^-1

    # Sound horizon fitting formula
    r_s = (
        44.5
        * np.log(9.83 / Omega_m_h2)
        / np.sqrt(1 + 10 * Omega_b_h2**0.75)
    )

    return r_s
