"""General scalar-tensor theory definitions for HRC 2.0.

This module defines the abstract framework for scalar-tensor gravity:
    S = integral d^4x sqrt(-g) [F(phi)R/2 - Z(phi)(dphi)^2/2 - V(phi)] + S_matter

Classes:
- CouplingFamily: Enum for F(phi) functional forms
- KineticFamily: Enum for Z(phi) functional forms
- PotentialType: Enum for V(phi) functional forms
- ScalarTensorModel: Abstract model with F, Z, V and their derivatives
- HRC2Parameters: Configuration dataclass for model parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import numpy as np


# Physical constants (Planck units where M_pl = 1)
M_PL = 1.0  # Reduced Planck mass
M_PL_SQUARED = M_PL ** 2


class CouplingFamily(Enum):
    """Non-minimal coupling F(phi) functional forms.

    LINEAR: F = M_pl^2 - alpha*phi
        - Reproduces HRC 1.x behavior
        - Critical value: phi_crit = M_pl^2/alpha

    QUADRATIC: F = M_pl^2 * (1 + xi*phi^2)
        - Common in Higgs inflation models
        - Always positive for xi > 0
        - F' = 2*xi*M_pl^2*phi

    EXPONENTIAL: F = M_pl^2 * exp(beta*phi/M_pl)
        - Natural in string theory compactifications
        - Always positive
        - F' = beta*M_pl*exp(beta*phi/M_pl)

    PLATEAU_EVAP: F = M_pl^2 * (1 + xi * (1 - exp(-(phi/mu)^2)))
        - Evaporated-boundary-inspired coupling
        - f(0) = 0 -> pure GR at vacuum (outer evaporated region)
        - f(|phi| >> mu) -> 1 -> shifted effective Planck mass inside
        - Always positive for xi > -1

    CUSTOM: User-defined F(phi)
    """
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    PLATEAU_EVAP = "plateau_evap"
    CUSTOM = "custom"


class KineticFamily(Enum):
    """Kinetic term Z(phi) functional forms.

    CANONICAL: Z = 1 (standard kinetic term)
    NONCANONICAL: Z(phi) != 1 (k-essence type, placeholder for future)
    """
    CANONICAL = "canonical"
    NONCANONICAL = "noncanonical"


class PotentialType(Enum):
    """Scalar field potential V(phi) types.

    QUADRATIC: V = V0 + m^2*phi^2/2
    PLATEAU: V = V0 * (1 - exp(-phi/M))^n
    EXPONENTIAL: V = V0 * exp(-lambda*phi/M_pl)
    TRACKER: V = V0 / phi^alpha (inverse power law)
    CUSTOM: User-defined V(phi)
    """
    QUADRATIC = "quadratic"
    PLATEAU = "plateau"
    EXPONENTIAL = "exponential"
    TRACKER = "tracker"
    CUSTOM = "custom"


class ScalarTensorModel(ABC):
    """Abstract base class for scalar-tensor models.

    Defines the interface for computing F(phi), Z(phi), V(phi)
    and their derivatives, which are needed for the field equations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass

    @abstractmethod
    def F(self, phi: float) -> float:
        """Non-minimal coupling F(phi).

        Appears in action as F(phi)*R/2.
        Must be positive to avoid ghost graviton.
        """
        pass

    @abstractmethod
    def dF_dphi(self, phi: float) -> float:
        """First derivative dF/dphi."""
        pass

    @abstractmethod
    def d2F_dphi2(self, phi: float) -> float:
        """Second derivative d^2F/dphi^2.

        Needed for Dolgov-Kawasaki stability check.
        """
        pass

    @abstractmethod
    def Z(self, phi: float) -> float:
        """Kinetic coefficient Z(phi).

        Appears in action as -Z(phi)*(dphi)^2/2.
        Must be positive to avoid scalar ghost.
        """
        pass

    @abstractmethod
    def dZ_dphi(self, phi: float) -> float:
        """First derivative dZ/dphi."""
        pass

    @abstractmethod
    def V(self, phi: float) -> float:
        """Scalar potential V(phi)."""
        pass

    @abstractmethod
    def dV_dphi(self, phi: float) -> float:
        """First derivative dV/dphi."""
        pass

    def is_valid(self, phi: float, safety_margin: float = 0.05) -> bool:
        """Check if model is valid at given phi.

        Args:
            phi: Field value
            safety_margin: Minimum allowed F/M_pl^2 ratio

        Returns:
            True if F > safety_margin * M_pl^2 and Z > 0
        """
        F_val = self.F(phi)
        Z_val = self.Z(phi)

        return F_val > safety_margin * M_PL_SQUARED and Z_val > 0


@dataclass
class HRC2Parameters:
    """Configuration parameters for HRC 2.0 models.

    Attributes:
        coupling_family: Type of F(phi) coupling
        kinetic_family: Type of Z(phi) kinetic term
        potential_type: Type of V(phi) potential

        # Coupling parameters
        xi: Quadratic coupling strength (for QUADRATIC)
        alpha: Linear coupling strength (for LINEAR)
        beta: Exponential coupling strength (for EXPONENTIAL)

        # Potential parameters
        V0: Vacuum energy / cosmological constant scale
        m_phi: Scalar field mass (for QUADRATIC potential)
        M_scale: Mass scale for plateau/exponential potentials
        lambda_exp: Exponential decay rate (for EXPONENTIAL potential)
        n_plateau: Power for plateau potential
        alpha_tracker: Power for tracker potential

        # Initial conditions
        phi_0: Initial field value at z=0
        phi_dot_0: Initial field velocity at z=0

        # Cosmological parameters
        Omega_m0: Present matter density parameter
        Omega_r0: Present radiation density parameter
        H0: Present Hubble parameter (km/s/Mpc)
    """
    # Model type selection
    coupling_family: CouplingFamily = CouplingFamily.QUADRATIC
    kinetic_family: KineticFamily = KineticFamily.CANONICAL
    potential_type: PotentialType = PotentialType.QUADRATIC

    # Coupling parameters
    xi: float = 1e-3  # Quadratic coupling / plateau_evap amplitude
    alpha: float = 0.1  # Linear coupling (in M_pl units)
    beta: float = 0.1  # Exponential coupling
    mu: float = 0.1  # Plateau_evap width scale (in M_pl units)

    # Potential parameters
    V0: float = 0.7  # Vacuum energy scale (in units of 3*H0^2*M_pl^2)
    m_phi: float = 1.0  # Scalar mass (in units of H0)
    M_scale: float = 1.0  # Mass scale for plateau/exp (in M_pl units)
    lambda_exp: float = 1.0  # Exponential decay rate
    n_plateau: float = 2.0  # Plateau power
    alpha_tracker: float = 2.0  # Tracker power

    # Initial conditions
    phi_0: float = 0.1  # Initial field value (in M_pl units)
    phi_dot_0: float = 0.0  # Initial velocity

    # Cosmological parameters
    Omega_m0: float = 0.3  # Matter density
    Omega_r0: float = 9e-5  # Radiation density
    H0: float = 70.0  # Hubble constant (km/s/Mpc)

    # Thermal horizon recycling parameter
    alpha_rec: float = 0.0  # Recycling fraction: 0 <= alpha_rec < 1

    # Horizon-driven effective potential parameter
    gamma_rec: float = 0.0  # V_eff = V0(phi) + gamma_rec * H^4 * (1/2) * phi^2

    # Early Dark Energy (EDE) fluid parameters
    f_EDE: float = 0.0      # approximate peak fractional contribution at z_c
    z_c: float = 3000.0     # characteristic EDE peak redshift
    sigma_ln_a: float = 0.5 # width of the EDE bump in ln(a)

    # Nonlocal horizon-memory parameters
    lambda_hor: float = 0.0  # amplitude of horizon-memory energy density
    tau_hor: float = 1.0     # memory timescale in ln(a)

    # Custom functions (for CUSTOM types)
    custom_F: Optional[Callable[[float], float]] = None
    custom_dF: Optional[Callable[[float], float]] = None
    custom_d2F: Optional[Callable[[float], float]] = None
    custom_V: Optional[Callable[[float], float]] = None
    custom_dV: Optional[Callable[[float], float]] = None
    custom_Z: Optional[Callable[[float], float]] = None
    custom_dZ: Optional[Callable[[float], float]] = None


# ============================================================================
# Concrete Model Implementations
# ============================================================================

class LinearCouplingModel(ScalarTensorModel):
    """Linear coupling: F(phi) = M_pl^2 - alpha*phi.

    This reproduces HRC 1.x behavior where F = M_pl^2(1 - 8*pi*xi*phi)
    with alpha = 8*pi*xi*M_pl^2.

    Critical value: phi_crit = M_pl^2/alpha where F -> 0.
    """

    def __init__(self, params: HRC2Parameters):
        self.params = params
        self.alpha = params.alpha
        self._setup_potential(params)

    def _setup_potential(self, params: HRC2Parameters):
        """Setup potential functions based on type."""
        self.V0 = params.V0
        self.m_phi = params.m_phi
        self.M_scale = params.M_scale
        self.lambda_exp = params.lambda_exp
        self.n_plateau = params.n_plateau
        self.alpha_tracker = params.alpha_tracker
        self.potential_type = params.potential_type

    @property
    def name(self) -> str:
        return f"linear_F_alpha={self.alpha:.2e}"

    def F(self, phi: float) -> float:
        return M_PL_SQUARED - self.alpha * phi

    def dF_dphi(self, phi: float) -> float:
        return -self.alpha

    def d2F_dphi2(self, phi: float) -> float:
        return 0.0

    def Z(self, phi: float) -> float:
        return 1.0  # Canonical kinetic term

    def dZ_dphi(self, phi: float) -> float:
        return 0.0

    def V(self, phi: float) -> float:
        return self._compute_V(phi)

    def dV_dphi(self, phi: float) -> float:
        return self._compute_dV(phi)

    def _compute_V(self, phi: float) -> float:
        """Compute potential based on type."""
        if self.potential_type == PotentialType.QUADRATIC:
            return self.V0 + 0.5 * self.m_phi**2 * phi**2
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            return self.V0 * (1 - np.exp(-x))**self.n_plateau
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return self.V0 * 1e10  # Large value for small phi
            return self.V0 / (abs(phi)**self.alpha_tracker)
        else:
            return self.V0

    def _compute_dV(self, phi: float) -> float:
        """Compute potential derivative based on type."""
        if self.potential_type == PotentialType.QUADRATIC:
            return self.m_phi**2 * phi
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            exp_term = np.exp(-x)
            return (self.V0 * self.n_plateau / self.M_scale *
                    (1 - exp_term)**(self.n_plateau - 1) * exp_term)
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return -self.lambda_exp / M_PL * self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return 0.0
            sign = 1 if phi > 0 else -1
            return -sign * self.alpha_tracker * self.V0 / (abs(phi)**(self.alpha_tracker + 1))
        else:
            return 0.0

    @property
    def phi_critical(self) -> float:
        """Critical field value where F -> 0."""
        if self.alpha > 0:
            return M_PL_SQUARED / self.alpha
        return float('inf')


class QuadraticCouplingModel(ScalarTensorModel):
    """Quadratic coupling: F(phi) = M_pl^2 * (1 + xi*phi^2).

    Common in Higgs inflation and similar models.
    Always positive for xi > 0, so no critical phi.

    Derivatives:
        F' = 2*xi*M_pl^2*phi
        F'' = 2*xi*M_pl^2
    """

    def __init__(self, params: HRC2Parameters):
        self.params = params
        self.xi = params.xi
        self._setup_potential(params)

    def _setup_potential(self, params: HRC2Parameters):
        """Setup potential functions based on type."""
        self.V0 = params.V0
        self.m_phi = params.m_phi
        self.M_scale = params.M_scale
        self.lambda_exp = params.lambda_exp
        self.n_plateau = params.n_plateau
        self.alpha_tracker = params.alpha_tracker
        self.potential_type = params.potential_type

    @property
    def name(self) -> str:
        return f"quadratic_F_xi={self.xi:.2e}"

    def F(self, phi: float) -> float:
        return M_PL_SQUARED * (1.0 + self.xi * phi**2)

    def dF_dphi(self, phi: float) -> float:
        return 2.0 * self.xi * M_PL_SQUARED * phi

    def d2F_dphi2(self, phi: float) -> float:
        return 2.0 * self.xi * M_PL_SQUARED

    def Z(self, phi: float) -> float:
        return 1.0

    def dZ_dphi(self, phi: float) -> float:
        return 0.0

    def V(self, phi: float) -> float:
        return self._compute_V(phi)

    def dV_dphi(self, phi: float) -> float:
        return self._compute_dV(phi)

    def _compute_V(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.V0 + 0.5 * self.m_phi**2 * phi**2
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            return self.V0 * (1 - np.exp(-x))**self.n_plateau
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return self.V0 * 1e10
            return self.V0 / (abs(phi)**self.alpha_tracker)
        else:
            return self.V0

    def _compute_dV(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.m_phi**2 * phi
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            exp_term = np.exp(-x)
            return (self.V0 * self.n_plateau / self.M_scale *
                    (1 - exp_term)**(self.n_plateau - 1) * exp_term)
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return -self.lambda_exp / M_PL * self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return 0.0
            sign = 1 if phi > 0 else -1
            return -sign * self.alpha_tracker * self.V0 / (abs(phi)**(self.alpha_tracker + 1))
        else:
            return 0.0


class ExponentialCouplingModel(ScalarTensorModel):
    """Exponential coupling: F(phi) = M_pl^2 * exp(beta*phi/M_pl).

    Natural in string theory compactifications and Brans-Dicke variants.
    Always positive for all phi.

    Derivatives:
        F' = beta*M_pl*exp(beta*phi/M_pl)
        F'' = beta^2*exp(beta*phi/M_pl)
    """

    def __init__(self, params: HRC2Parameters):
        self.params = params
        self.beta = params.beta
        self._setup_potential(params)

    def _setup_potential(self, params: HRC2Parameters):
        """Setup potential functions based on type."""
        self.V0 = params.V0
        self.m_phi = params.m_phi
        self.M_scale = params.M_scale
        self.lambda_exp = params.lambda_exp
        self.n_plateau = params.n_plateau
        self.alpha_tracker = params.alpha_tracker
        self.potential_type = params.potential_type

    @property
    def name(self) -> str:
        return f"exponential_F_beta={self.beta:.2e}"

    def F(self, phi: float) -> float:
        return M_PL_SQUARED * np.exp(self.beta * phi / M_PL)

    def dF_dphi(self, phi: float) -> float:
        return self.beta * M_PL * np.exp(self.beta * phi / M_PL)

    def d2F_dphi2(self, phi: float) -> float:
        return self.beta**2 * np.exp(self.beta * phi / M_PL)

    def Z(self, phi: float) -> float:
        return 1.0

    def dZ_dphi(self, phi: float) -> float:
        return 0.0

    def V(self, phi: float) -> float:
        return self._compute_V(phi)

    def dV_dphi(self, phi: float) -> float:
        return self._compute_dV(phi)

    def _compute_V(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.V0 + 0.5 * self.m_phi**2 * phi**2
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            return self.V0 * (1 - np.exp(-x))**self.n_plateau
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return self.V0 * 1e10
            return self.V0 / (abs(phi)**self.alpha_tracker)
        else:
            return self.V0

    def _compute_dV(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.m_phi**2 * phi
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            exp_term = np.exp(-x)
            return (self.V0 * self.n_plateau / self.M_scale *
                    (1 - exp_term)**(self.n_plateau - 1) * exp_term)
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return -self.lambda_exp / M_PL * self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return 0.0
            sign = 1 if phi > 0 else -1
            return -sign * self.alpha_tracker * self.V0 / (abs(phi)**(self.alpha_tracker + 1))
        else:
            return 0.0


class PlateauEvapCouplingModel(ScalarTensorModel):
    """Evaporated-boundary-inspired plateau coupling.

    F(phi) = M_pl^2 * (1 + xi * f(phi))
    where f(phi) = 1 - exp(-(phi/mu)^2)

    This models a scenario where:
    - f(0) = 0: At the vacuum (outer evaporated boundary), we have pure GR
    - f(|phi| >> mu) -> 1: Inside our patch, effective Planck mass is shifted
    - The transition scale mu controls how quickly we approach the plateau

    Derivatives:
        f'(phi) = (2*phi/mu^2) * exp(-(phi/mu)^2)
        f''(phi) = (2/mu^2) * exp(-(phi/mu)^2) * (1 - 2*(phi/mu)^2)

        F' = xi * M_pl^2 * f'(phi)
        F'' = xi * M_pl^2 * f''(phi)
    """

    def __init__(self, params: HRC2Parameters):
        self.params = params
        self.xi = params.xi
        self.mu = params.mu
        self._setup_potential(params)

    def _setup_potential(self, params: HRC2Parameters):
        """Setup potential functions based on type."""
        self.V0 = params.V0
        self.m_phi = params.m_phi
        self.M_scale = params.M_scale
        self.lambda_exp = params.lambda_exp
        self.n_plateau = params.n_plateau
        self.alpha_tracker = params.alpha_tracker
        self.potential_type = params.potential_type

    @property
    def name(self) -> str:
        return f"plateau_evap_xi={self.xi:.2e}_mu={self.mu:.2e}"

    def _f(self, phi: float) -> float:
        """Coupling shape function f(phi) = 1 - exp(-(phi/mu)^2)."""
        x = phi / self.mu
        return 1.0 - np.exp(-x**2)

    def _df(self, phi: float) -> float:
        """Derivative df/dphi = (2*phi/mu^2) * exp(-(phi/mu)^2)."""
        x = phi / self.mu
        return (2.0 * phi / self.mu**2) * np.exp(-x**2)

    def _d2f(self, phi: float) -> float:
        """Second derivative d^2f/dphi^2."""
        x = phi / self.mu
        return (2.0 / self.mu**2) * np.exp(-x**2) * (1.0 - 2.0 * x**2)

    def F(self, phi: float) -> float:
        return M_PL_SQUARED * (1.0 + self.xi * self._f(phi))

    def dF_dphi(self, phi: float) -> float:
        return self.xi * M_PL_SQUARED * self._df(phi)

    def d2F_dphi2(self, phi: float) -> float:
        return self.xi * M_PL_SQUARED * self._d2f(phi)

    def Z(self, phi: float) -> float:
        return 1.0

    def dZ_dphi(self, phi: float) -> float:
        return 0.0

    def V(self, phi: float) -> float:
        return self._compute_V(phi)

    def dV_dphi(self, phi: float) -> float:
        return self._compute_dV(phi)

    def _compute_V(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.V0 + 0.5 * self.m_phi**2 * phi**2
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            return self.V0 * (1 - np.exp(-x))**self.n_plateau
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return self.V0 * 1e10
            return self.V0 / (abs(phi)**self.alpha_tracker)
        else:
            return self.V0

    def _compute_dV(self, phi: float) -> float:
        if self.potential_type == PotentialType.QUADRATIC:
            return self.m_phi**2 * phi
        elif self.potential_type == PotentialType.PLATEAU:
            x = phi / self.M_scale
            exp_term = np.exp(-x)
            return (self.V0 * self.n_plateau / self.M_scale *
                    (1 - exp_term)**(self.n_plateau - 1) * exp_term)
        elif self.potential_type == PotentialType.EXPONENTIAL:
            return -self.lambda_exp / M_PL * self.V0 * np.exp(-self.lambda_exp * phi / M_PL)
        elif self.potential_type == PotentialType.TRACKER:
            if abs(phi) < 1e-10:
                return 0.0
            sign = 1 if phi > 0 else -1
            return -sign * self.alpha_tracker * self.V0 / (abs(phi)**(self.alpha_tracker + 1))
        else:
            return 0.0


class CustomCouplingModel(ScalarTensorModel):
    """Custom user-defined coupling model.

    All functions must be provided via HRC2Parameters.
    """

    def __init__(self, params: HRC2Parameters):
        self.params = params

        # Validate custom functions are provided
        if params.custom_F is None or params.custom_dF is None:
            raise ValueError("Custom F and dF functions must be provided")
        if params.custom_V is None or params.custom_dV is None:
            raise ValueError("Custom V and dV functions must be provided")

        self._F = params.custom_F
        self._dF = params.custom_dF
        self._d2F = params.custom_d2F or (lambda phi: 0.0)
        self._V = params.custom_V
        self._dV = params.custom_dV
        self._Z = params.custom_Z or (lambda phi: 1.0)
        self._dZ = params.custom_dZ or (lambda phi: 0.0)

    @property
    def name(self) -> str:
        return "custom_model"

    def F(self, phi: float) -> float:
        return self._F(phi)

    def dF_dphi(self, phi: float) -> float:
        return self._dF(phi)

    def d2F_dphi2(self, phi: float) -> float:
        return self._d2F(phi)

    def Z(self, phi: float) -> float:
        return self._Z(phi)

    def dZ_dphi(self, phi: float) -> float:
        return self._dZ(phi)

    def V(self, phi: float) -> float:
        return self._V(phi)

    def dV_dphi(self, phi: float) -> float:
        return self._dV(phi)


# ============================================================================
# Factory Function
# ============================================================================

def create_model(params: HRC2Parameters) -> ScalarTensorModel:
    """Factory function to create appropriate model from parameters.

    Args:
        params: HRC2Parameters configuration

    Returns:
        Concrete ScalarTensorModel instance

    Raises:
        ValueError: For unknown coupling family
    """
    if params.coupling_family == CouplingFamily.LINEAR:
        return LinearCouplingModel(params)
    elif params.coupling_family == CouplingFamily.QUADRATIC:
        return QuadraticCouplingModel(params)
    elif params.coupling_family == CouplingFamily.EXPONENTIAL:
        return ExponentialCouplingModel(params)
    elif params.coupling_family == CouplingFamily.PLATEAU_EVAP:
        return PlateauEvapCouplingModel(params)
    elif params.coupling_family == CouplingFamily.CUSTOM:
        return CustomCouplingModel(params)
    else:
        raise ValueError(f"Unknown coupling family: {params.coupling_family}")


# ============================================================================
# Utility Functions
# ============================================================================

def compute_effective_brans_dicke_omega(model: ScalarTensorModel, phi: float) -> float:
    """Compute effective Brans-Dicke parameter omega_BD.

    For scalar-tensor theories, the effective BD parameter is:
        omega_BD = F * Z / (dF/dphi)^2 - 3/2

    This is used for PPN constraints (gamma - 1 = -1/(2*omega_BD + 3)).

    Args:
        model: ScalarTensorModel instance
        phi: Field value

    Returns:
        omega_BD or inf if dF/dphi = 0
    """
    F = model.F(phi)
    Z = model.Z(phi)
    dF = model.dF_dphi(phi)

    if abs(dF) < 1e-15:
        return float('inf')

    return F * Z / (dF**2) - 1.5


def compute_ppn_gamma(model: ScalarTensorModel, phi: float) -> float:
    """Compute PPN parameter gamma.

    gamma = (omega_BD + 1) / (omega_BD + 2)

    GR limit: gamma = 1 (omega_BD -> inf)

    Args:
        model: ScalarTensorModel instance
        phi: Field value

    Returns:
        PPN gamma parameter
    """
    omega_BD = compute_effective_brans_dicke_omega(model, phi)

    if omega_BD == float('inf'):
        return 1.0

    return (omega_BD + 1) / (omega_BD + 2)
