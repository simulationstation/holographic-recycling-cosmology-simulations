"""Configuration and parameter classes for HRC."""

from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
import numpy as np


@dataclass
class HRCParameters:
    """HRC model parameters.

    Attributes:
        xi: Non-minimal coupling constant ξ in the action term ξφR
        phi_0: Present-day scalar field value φ₀ (in reduced Planck units)
        V0: Potential amplitude V₀ for V(φ) = V₀ + ½m²φ²
        m_phi: Scalar field mass (in units of H0)
        phi_dot_0: Present-day scalar field velocity dφ/dt|₀
        f_rem: Fraction of dark matter in remnants
        alpha_rem: Remnant-scalar coupling strength

    Physical constraints:
        - xi > 0 for enhanced gravity (G_eff > G)
        - |8πGξφ| < 1 to avoid G_eff divergence
        - m_phi ~ H0 for cosmologically relevant evolution
    """

    # Core HRC parameters
    xi: float = 0.03  # Non-minimal coupling
    phi_0: float = 0.2  # Present field value (Planck units)
    V0: float = 0.0  # Potential offset (cosmological constant contribution)
    m_phi: float = 1.0  # Scalar mass in units of H0
    phi_dot_0: float = 0.0  # Present field velocity

    # Remnant parameters
    f_rem: float = 0.2  # Remnant fraction of DM
    alpha_rem: float = 0.01  # Remnant-scalar coupling

    # Standard cosmological parameters
    h: float = 0.7  # Dimensionless Hubble parameter
    Omega_b: float = 0.05  # Baryon density
    Omega_c: float = 0.25  # Cold dark matter density
    Omega_r: float = 9.0e-5  # Radiation density
    Omega_k: float = 0.0  # Curvature

    @property
    def Omega_m(self) -> float:
        """Total matter density."""
        return self.Omega_b + self.Omega_c

    @property
    def Omega_Lambda(self) -> float:
        """Dark energy density (closure)."""
        return 1.0 - self.Omega_m - self.Omega_r - self.Omega_k

    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return 100.0 * self.h

    def validate(self) -> tuple[bool, list[str]]:
        """Validate parameter values.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check G_eff divergence condition
        critical_value = 1.0 / (8 * np.pi * self.xi) if self.xi > 0 else np.inf
        if self.phi_0 >= critical_value:
            errors.append(
                f"φ₀ = {self.phi_0} exceeds critical value {critical_value:.3f} "
                f"for ξ = {self.xi}; G_eff would diverge"
            )

        # Check positivity
        if self.xi < 0:
            errors.append("ξ < 0 leads to weakened gravity; typically ξ > 0 for HRC")

        if self.Omega_m < 0 or self.Omega_m > 1:
            errors.append(f"Unphysical Omega_m = {self.Omega_m}")

        if self.Omega_Lambda < 0:
            errors.append(f"Negative Omega_Lambda = {self.Omega_Lambda}")

        if self.h <= 0 or self.h > 2:
            errors.append(f"Unphysical h = {self.h}")

        if self.m_phi < 0:
            errors.append(f"Negative scalar mass m_phi = {self.m_phi}")

        return len(errors) == 0, errors


@dataclass
class PotentialConfig:
    """Scalar field potential configuration.

    Supports various potential forms:
        - 'quadratic': V(φ) = V₀ + ½m²φ²
        - 'quartic': V(φ) = V₀ + ½m²φ² + ¼λφ⁴
        - 'exponential': V(φ) = V₀ exp(-αφ)
        - 'custom': User-provided callable
    """

    form: Literal["quadratic", "quartic", "exponential", "custom"] = "quadratic"
    V0: float = 0.0  # Potential offset
    m: float = 1.0  # Mass parameter
    lambda_4: float = 0.0  # Quartic coupling
    alpha_exp: float = 1.0  # Exponential slope
    custom_V: Optional[Callable[[float], float]] = None
    custom_dV: Optional[Callable[[float], float]] = None

    def V(self, phi: float) -> float:
        """Evaluate potential V(φ)."""
        if self.form == "quadratic":
            return self.V0 + 0.5 * self.m**2 * phi**2
        elif self.form == "quartic":
            return self.V0 + 0.5 * self.m**2 * phi**2 + 0.25 * self.lambda_4 * phi**4
        elif self.form == "exponential":
            return self.V0 * np.exp(-self.alpha_exp * phi)
        elif self.form == "custom" and self.custom_V is not None:
            return self.custom_V(phi)
        else:
            raise ValueError(f"Unknown potential form: {self.form}")

    def dV(self, phi: float) -> float:
        """Evaluate potential derivative V'(φ)."""
        if self.form == "quadratic":
            return self.m**2 * phi
        elif self.form == "quartic":
            return self.m**2 * phi + self.lambda_4 * phi**3
        elif self.form == "exponential":
            return -self.alpha_exp * self.V0 * np.exp(-self.alpha_exp * phi)
        elif self.form == "custom" and self.custom_dV is not None:
            return self.custom_dV(phi)
        else:
            raise ValueError(f"Unknown potential form: {self.form}")


@dataclass
class HRCConfig:
    """Full HRC configuration combining parameters and settings."""

    params: HRCParameters = field(default_factory=HRCParameters)
    potential: PotentialConfig = field(default_factory=PotentialConfig)

    # Integration settings
    z_min: float = 0.0
    z_max: float = 1200.0  # Up to recombination
    z_points: int = 1000
    rtol: float = 1e-8
    atol: float = 1e-10

    # Output settings
    output_dir: str = "results"
    save_intermediates: bool = False

    def get_z_array(self) -> np.ndarray:
        """Return redshift array for integration."""
        return np.linspace(self.z_min, self.z_max, self.z_points)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate full configuration."""
        valid, errors = self.params.validate()

        if self.z_min < 0:
            errors.append(f"z_min = {self.z_min} must be >= 0")
            valid = False

        if self.z_max <= self.z_min:
            errors.append(f"z_max = {self.z_max} must be > z_min = {self.z_min}")
            valid = False

        if self.z_points < 10:
            errors.append(f"z_points = {self.z_points} too small")
            valid = False

        return valid, errors
