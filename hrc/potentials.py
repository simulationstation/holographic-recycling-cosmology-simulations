"""Scalar field potentials for HRC.

This module provides a pluggable interface for scalar field potentials,
including several candidates for stabilizing the field at high redshift.

The key requirement is that the potential should prevent the scalar field
from evolving toward the critical value phi_c = 1/(8*pi*xi) where G_eff diverges.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal, Type, Dict
import numpy as np


class Potential(ABC):
    """Abstract base class for scalar field potentials.

    All potentials must implement V(phi) and dV_dphi(phi).
    """

    @abstractmethod
    def V(self, phi: float) -> float:
        """Evaluate the potential V(phi).

        Args:
            phi: Scalar field value

        Returns:
            Potential energy V(phi)
        """
        pass

    @abstractmethod
    def dV_dphi(self, phi: float) -> float:
        """Evaluate the potential derivative dV/dphi.

        Args:
            phi: Scalar field value

        Returns:
            First derivative dV/dphi
        """
        pass

    def d2V_dphi2(self, phi: float) -> float:
        """Evaluate the second derivative d²V/dphi² (numerical by default).

        Args:
            phi: Scalar field value

        Returns:
            Second derivative d²V/dphi²
        """
        dphi = 1e-5
        return (self.dV_dphi(phi + dphi) - self.dV_dphi(phi - dphi)) / (2 * dphi)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the potential name for identification."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class QuadraticPotential(Potential):
    """Quadratic (mass) potential: V(phi) = V0 + 0.5 * m^2 * phi^2

    This is the simplest potential, but tends to drive phi toward
    large values at high redshift due to the ξR coupling.
    """

    def __init__(self, V0: float = 0.0, m: float = 1.0):
        """Initialize quadratic potential.

        Args:
            V0: Constant offset (cosmological constant contribution)
            m: Mass parameter (in units of H0)
        """
        self.V0 = V0
        self.m = m

    def V(self, phi: float) -> float:
        return self.V0 + 0.5 * self.m**2 * phi**2

    def dV_dphi(self, phi: float) -> float:
        return self.m**2 * phi

    def d2V_dphi2(self, phi: float) -> float:
        return self.m**2

    @property
    def name(self) -> str:
        return "quadratic"

    def __repr__(self) -> str:
        return f"QuadraticPotential(V0={self.V0}, m={self.m})"


class PlateauPotential(Potential):
    """Plateau / freezing potential: V(phi) = V0 / (1 + (phi/M)^n)

    This potential flattens at large phi, which can help stabilize
    the field and prevent runaway behavior. At small phi, V ~ V0.
    At large phi, V -> 0.

    The derivative dV/dphi is always negative for phi > 0, pushing
    the field toward larger values, but with decreasing force.
    """

    def __init__(self, V0: float = 0.7, M: float = 1.0, n: float = 2.0):
        """Initialize plateau potential.

        Args:
            V0: Amplitude (sets dark energy scale)
            M: Scale parameter (field value where potential transitions)
            n: Power (larger n = sharper transition)
        """
        self.V0 = V0
        self.M = M
        self.n = n

    def V(self, phi: float) -> float:
        x = phi / self.M
        return self.V0 / (1.0 + np.abs(x)**self.n)

    def dV_dphi(self, phi: float) -> float:
        x = phi / self.M
        if abs(x) < 1e-10:
            return 0.0
        denom = (1.0 + np.abs(x)**self.n)**2
        # d/dphi [V0 / (1 + |x|^n)] = -V0 * n * |x|^(n-1) * sign(x) / M / denom
        return -self.V0 * self.n * np.abs(x)**(self.n - 1) * np.sign(phi) / self.M / denom

    @property
    def name(self) -> str:
        return "plateau"

    def __repr__(self) -> str:
        return f"PlateauPotential(V0={self.V0}, M={self.M}, n={self.n})"


class SymmetronPotential(Potential):
    """Symmetron / Higgs-like potential: V(phi) = V0 - 0.5*mu^2*phi^2 + 0.25*lambda*phi^4

    This potential has a Mexican hat shape with minima at phi = +/- mu/sqrt(lambda).
    The negative mass term allows the field to roll away from phi=0, but the
    quartic term prevents runaway.

    In the presence of the ξR coupling, the effective mass depends on the
    curvature, potentially creating a screening mechanism.
    """

    def __init__(
        self,
        V0: float = 0.7,
        mu2: float = 1.0,
        lambda_: float = 1.0
    ):
        """Initialize symmetron potential.

        Args:
            V0: Constant offset (ensures V > 0 at minimum)
            mu2: Negative mass squared coefficient (mu^2 > 0)
            lambda_: Quartic self-coupling (lambda > 0)
        """
        self.V0 = V0
        self.mu2 = mu2
        self.lambda_ = lambda_

        # Compute minimum location and value
        if lambda_ > 0:
            self.phi_min = np.sqrt(mu2 / lambda_)
            self.V_min = V0 - 0.25 * mu2**2 / lambda_
        else:
            self.phi_min = 0.0
            self.V_min = V0

    def V(self, phi: float) -> float:
        return self.V0 - 0.5 * self.mu2 * phi**2 + 0.25 * self.lambda_ * phi**4

    def dV_dphi(self, phi: float) -> float:
        return -self.mu2 * phi + self.lambda_ * phi**3

    def d2V_dphi2(self, phi: float) -> float:
        return -self.mu2 + 3.0 * self.lambda_ * phi**2

    @property
    def name(self) -> str:
        return "symmetron"

    def __repr__(self) -> str:
        return f"SymmetronPotential(V0={self.V0}, mu2={self.mu2}, lambda_={self.lambda_})"


class ExponentialPotential(Potential):
    """Exponential / tracker potential: V(phi) = V0 * exp(-lambda * phi / M)

    This potential supports tracker solutions where the scalar field
    energy density tracks the dominant component. The exponential form
    naturally arises in many theories (e.g., string moduli).

    For lambda < sqrt(2), the field can track matter/radiation.
    For lambda > sqrt(6), slow-roll inflation is not possible.
    """

    def __init__(self, V0: float = 0.7, lambda_: float = 1.0, M: float = 1.0):
        """Initialize exponential potential.

        Args:
            V0: Amplitude (sets dark energy scale)
            lambda_: Slope parameter (dimensionless)
            M: Mass scale (typically Planck mass = 1)
        """
        self.V0 = V0
        self.lambda_ = lambda_
        self.M = M

    def V(self, phi: float) -> float:
        return self.V0 * np.exp(-self.lambda_ * phi / self.M)

    def dV_dphi(self, phi: float) -> float:
        return -self.lambda_ / self.M * self.V0 * np.exp(-self.lambda_ * phi / self.M)

    def d2V_dphi2(self, phi: float) -> float:
        return (self.lambda_ / self.M)**2 * self.V0 * np.exp(-self.lambda_ * phi / self.M)

    @property
    def name(self) -> str:
        return "exponential"

    def __repr__(self) -> str:
        return f"ExponentialPotential(V0={self.V0}, lambda_={self.lambda_}, M={self.M})"


class DoubleExponentialPotential(Potential):
    """Double exponential potential: V = V1*exp(-l1*phi/M) + V2*exp(-l2*phi/M)

    This potential can exhibit more complex behavior including
    local minima and tracking to different attractor solutions.
    """

    def __init__(
        self,
        V1: float = 0.5,
        V2: float = 0.2,
        lambda1: float = 1.0,
        lambda2: float = 3.0,
        M: float = 1.0,
    ):
        """Initialize double exponential potential.

        Args:
            V1, V2: Amplitudes
            lambda1, lambda2: Slope parameters
            M: Mass scale
        """
        self.V1 = V1
        self.V2 = V2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.M = M

    def V(self, phi: float) -> float:
        return (
            self.V1 * np.exp(-self.lambda1 * phi / self.M) +
            self.V2 * np.exp(-self.lambda2 * phi / self.M)
        )

    def dV_dphi(self, phi: float) -> float:
        return (
            -self.lambda1 / self.M * self.V1 * np.exp(-self.lambda1 * phi / self.M) +
            -self.lambda2 / self.M * self.V2 * np.exp(-self.lambda2 * phi / self.M)
        )

    @property
    def name(self) -> str:
        return "double_exponential"


class InverseExponentialPotential(Potential):
    """Inverse exponential potential: V = V0 * (1 - exp(-phi^2/M^2))

    This potential has V(0) = 0 and asymptotes to V0 for large |phi|.
    It naturally stabilizes the field near phi=0 for small perturbations.
    """

    def __init__(self, V0: float = 0.7, M: float = 0.5):
        """Initialize inverse exponential potential.

        Args:
            V0: Asymptotic value
            M: Width scale
        """
        self.V0 = V0
        self.M = M

    def V(self, phi: float) -> float:
        return self.V0 * (1.0 - np.exp(-phi**2 / self.M**2))

    def dV_dphi(self, phi: float) -> float:
        return self.V0 * 2.0 * phi / self.M**2 * np.exp(-phi**2 / self.M**2)

    @property
    def name(self) -> str:
        return "inverse_exponential"


# Registry of available potentials
POTENTIAL_REGISTRY: Dict[str, Type[Potential]] = {
    "quadratic": QuadraticPotential,
    "plateau": PlateauPotential,
    "symmetron": SymmetronPotential,
    "exponential": ExponentialPotential,
    "double_exponential": DoubleExponentialPotential,
    "inverse_exponential": InverseExponentialPotential,
}


@dataclass
class PotentialParams:
    """Parameters for potential construction.

    This dataclass holds all possible potential parameters.
    Each potential type uses only the relevant subset.
    """

    potential_type: Literal[
        "quadratic", "plateau", "symmetron", "exponential",
        "double_exponential", "inverse_exponential"
    ] = "quadratic"

    # Common parameters
    V0: float = 0.0  # Offset / amplitude

    # Quadratic
    m: float = 1.0  # Mass

    # Plateau
    M: float = 1.0  # Scale
    n: float = 2.0  # Power

    # Symmetron
    mu2: float = 1.0  # Tachyonic mass squared
    lambda_: float = 1.0  # Quartic coupling

    # Exponential
    lambda_exp: float = 1.0  # Exponential slope

    # Double exponential
    V1: float = 0.5
    V2: float = 0.2
    lambda1: float = 1.0
    lambda2: float = 3.0


def get_potential(params: PotentialParams) -> Potential:
    """Factory function to create a potential from parameters.

    Args:
        params: PotentialParams with type and parameters

    Returns:
        Concrete Potential instance
    """
    ptype = params.potential_type

    if ptype == "quadratic":
        return QuadraticPotential(V0=params.V0, m=params.m)

    elif ptype == "plateau":
        return PlateauPotential(V0=params.V0, M=params.M, n=params.n)

    elif ptype == "symmetron":
        return SymmetronPotential(V0=params.V0, mu2=params.mu2, lambda_=params.lambda_)

    elif ptype == "exponential":
        return ExponentialPotential(V0=params.V0, lambda_=params.lambda_exp, M=params.M)

    elif ptype == "double_exponential":
        return DoubleExponentialPotential(
            V1=params.V1, V2=params.V2,
            lambda1=params.lambda1, lambda2=params.lambda2,
            M=params.M
        )

    elif ptype == "inverse_exponential":
        return InverseExponentialPotential(V0=params.V0, M=params.M)

    else:
        raise ValueError(f"Unknown potential type: {ptype}")


class PotentialAdapter:
    """Adapter to make new Potential interface compatible with old PotentialConfig.

    This allows gradual migration of the codebase.
    """

    def __init__(self, potential: Potential):
        self.potential = potential

    def V(self, phi: float) -> float:
        return self.potential.V(phi)

    def dV(self, phi: float) -> float:
        return self.potential.dV_dphi(phi)
