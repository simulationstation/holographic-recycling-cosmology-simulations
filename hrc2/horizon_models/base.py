"""Base classes for horizon-memory refinement models.

This module provides the abstract base class and common data structures
for all horizon-memory refinement implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Callable, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, quad


class RefinementType(Enum):
    """Horizon-memory refinement types."""
    BASELINE = "T06_baseline"
    ADAPTIVE_KERNEL = "T06A_adaptive_kernel"
    TWO_COMPONENT = "T06B_two_component"
    EARLY_SUPPRESSION = "T06C_early_suppression"
    DYNAMICAL_EOS = "T06D_dynamical_eos"


@dataclass
class HorizonMemoryParameters:
    """Configuration parameters for horizon-memory models.

    Base parameters (used by all refinements):
        lambda_hor: Amplitude of horizon-memory energy density
        tau_hor: Memory timescale in ln(a) (for baseline)

    Refinement A - Adaptive Memory Kernel:
        tau0: Base memory timescale
        p_hor: Scale-factor exponent (tau = tau0 * a^p_hor)

    Refinement B - Two-Component Memory:
        lambda1: Amplitude for M1 channel
        lambda2: Amplitude for M2 channel
        tau1: M1 relaxation timescale
        tau2: M2 relaxation timescale (M2 lags M1)

    Refinement C - Early-Time Suppression:
        a_supp: Suppression scale factor
        n_supp: Suppression power

    Refinement D - Dynamical EoS:
        delta_w: EoS shift amplitude
        a_w: EoS transition scale factor
        m_eos: EoS transition power

    Cosmological parameters:
        Omega_m0: Present matter density parameter
        Omega_r0: Present radiation density parameter
        H0: Present Hubble parameter (km/s/Mpc)
    """
    # Refinement type
    refinement_type: RefinementType = RefinementType.BASELINE

    # Base horizon-memory parameters
    lambda_hor: float = 0.2
    tau_hor: float = 0.1

    # Refinement A: Adaptive Memory Kernel
    tau0: float = 0.1
    p_hor: float = 0.0  # tau(a) = tau0 * a^p_hor

    # Refinement B: Two-Component Memory
    lambda1: float = 0.15
    lambda2: float = 0.05
    tau1: float = 0.05
    tau2: float = 0.2

    # Refinement C: Early-Time Suppression
    a_supp: float = 0.01  # Scale factor for suppression
    n_supp: float = 2.0   # Suppression power

    # Refinement D: Dynamical EoS
    delta_w: float = -0.2  # EoS shift (negative = more phantom)
    a_w: float = 0.3       # Transition scale factor
    m_eos: float = 2.0     # Transition power

    # Cosmological parameters
    Omega_m0: float = 0.3
    Omega_r0: float = 9e-5
    H0: float = 67.4  # km/s/Mpc

    def get_model_id(self) -> str:
        """Generate model ID string based on refinement type and key parameters."""
        if self.refinement_type == RefinementType.BASELINE:
            return f"baseline_{self.lambda_hor:.3f}_{self.tau_hor:.2f}"
        elif self.refinement_type == RefinementType.ADAPTIVE_KERNEL:
            return f"T06A_{self.tau0:.3f}_{self.p_hor:.2f}"
        elif self.refinement_type == RefinementType.TWO_COMPONENT:
            return f"T06B_{self.lambda1:.3f}_{self.tau1:.2f}_{self.tau2:.2f}"
        elif self.refinement_type == RefinementType.EARLY_SUPPRESSION:
            return f"T06C_{self.a_supp:.4f}_{self.n_supp:.1f}"
        elif self.refinement_type == RefinementType.DYNAMICAL_EOS:
            return f"T06D_{self.delta_w:.2f}_{self.a_w:.2f}"
        else:
            return f"unknown_{self.lambda_hor:.3f}"


@dataclass
class HorizonMemoryResult:
    """Result container for horizon-memory model evolution.

    Attributes:
        params: Model parameters
        success: Whether integration succeeded

        # Evolution arrays
        a: Scale factor array
        z: Redshift array
        H: Hubble parameter H(a) in km/s/Mpc
        M: Memory field(s) - shape depends on model
        rho_hor: Horizon-memory energy density (units of 3*H0^2)
        w_hor: Horizon-memory equation of state

        # Derived quantities
        H_ratio: H(z)/H_LCDM(z) ratio
        D_A: Angular diameter distance
        D_A_ratio: D_A/D_A_LCDM ratio

        # Diagnostics
        delta_H0_frac: Fractional H0 shift at z=0 (always 0 with self-consistent Lambda)
        late_time_H_effect: H(z=0.5) deviation from LCDM (percent)
        sn_distance_deviation: D_L deviation at z=0.5 (percent, for SN/BAO)
        cmb_distance_deviation: D_A deviation at z*
        message: Status message
    """
    params: HorizonMemoryParameters
    success: bool = True

    # Evolution arrays
    a: Optional[NDArray] = None
    z: Optional[NDArray] = None
    H: Optional[NDArray] = None
    M: Optional[NDArray] = None  # Memory field(s)
    rho_hor: Optional[NDArray] = None
    w_hor: Optional[NDArray] = None

    # Derived quantities
    H_ratio: Optional[NDArray] = None
    D_A: Optional[NDArray] = None
    D_A_ratio: Optional[NDArray] = None

    # Diagnostics
    delta_H0_frac: float = 0.0  # Always 0 with self-consistent Lambda
    late_time_H_effect: float = 0.0  # H deviation at z=0.5
    sn_distance_deviation: float = 0.0  # D_L deviation at z=0.5
    cmb_distance_deviation: float = 0.0
    Omega_hor0: float = 0.0
    Omega_L0_eff: float = 0.0

    message: str = "Success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.params.get_model_id(),
            "refinement_type": self.params.refinement_type.value,
            "success": self.success,
            "delta_H0_frac": float(self.delta_H0_frac),
            "late_time_H_effect": float(self.late_time_H_effect),
            "sn_distance_deviation": float(self.sn_distance_deviation),
            "cmb_distance_deviation": float(self.cmb_distance_deviation),
            "Omega_hor0": float(self.Omega_hor0),
            "Omega_L0_eff": float(self.Omega_L0_eff),
            "message": self.message,
            "parameters": {
                "lambda_hor": self.params.lambda_hor,
                "tau_hor": self.params.tau_hor,
                "tau0": self.params.tau0,
                "p_hor": self.params.p_hor,
                "lambda1": self.params.lambda1,
                "lambda2": self.params.lambda2,
                "tau1": self.params.tau1,
                "tau2": self.params.tau2,
                "a_supp": self.params.a_supp,
                "n_supp": self.params.n_supp,
                "delta_w": self.params.delta_w,
                "a_w": self.params.a_w,
                "m_eos": self.params.m_eos,
            }
        }


class HorizonMemoryModel(ABC):
    """Abstract base class for horizon-memory refinement models.

    All refinements must implement:
    - integrate_memory(): Solve the memory field ODE(s)
    - compute_rho_hor(): Compute horizon-memory energy density
    - compute_w_hor(): Compute equation of state
    - compute_H(): Compute Hubble parameter including horizon-memory
    """

    def __init__(self, params: HorizonMemoryParameters):
        """Initialize model with parameters."""
        self.params = params
        self.H0 = params.H0
        self.Omega_m0 = params.Omega_m0
        self.Omega_r0 = params.Omega_r0
        self.Omega_L0_base = 1.0 - params.Omega_m0 - params.Omega_r0

        # Will be set after integration
        self.Omega_hor0 = 0.0
        self.Omega_L0_eff = self.Omega_L0_base
        self._M_interp = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass

    @abstractmethod
    def integrate_memory(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate memory field ODE(s) from z=z_max to z=0.

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            Tuple of (ln_a array, M array) where M may be multi-dimensional
        """
        pass

    @abstractmethod
    def compute_rho_hor(self, a: float, M: Any) -> float:
        """Compute horizon-memory energy density at scale factor a.

        Args:
            a: Scale factor
            M: Memory field value(s)

        Returns:
            rho_hor in units of 3*H0^2
        """
        pass

    @abstractmethod
    def compute_w_hor(self, a: float, M: Any, eps: float = 1e-4) -> float:
        """Compute horizon-memory equation of state.

        w_hor = P_hor / rho_hor

        For standard memory: w = -1 - (1/3) * d ln(rho_hor) / d ln(a)

        Args:
            a: Scale factor
            M: Memory field value(s)
            eps: Numerical derivative step

        Returns:
            Equation of state w_hor
        """
        pass

    def S_norm(self, H: float) -> float:
        """Compute normalized horizon entropy proxy.

        S_norm = (H0 / H)^2

        Args:
            H: Hubble parameter

        Returns:
            Normalized horizon entropy
        """
        if H <= 0.0:
            return 0.0
        return (self.H0 / H) ** 2

    def H_GR(self, a: float) -> float:
        """Compute GR Hubble parameter (baseline LCDM).

        Args:
            a: Scale factor

        Returns:
            H in km/s/Mpc
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        rho_L = self.Omega_L0_base

        H_squared = rho_m + rho_r + rho_L
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def H_with_memory(self, a: float, M: Any) -> float:
        """Compute Hubble parameter including horizon-memory.

        Uses self-consistent Lambda: Omega_L0_eff = Omega_L0_base - Omega_hor0

        Args:
            a: Scale factor
            M: Memory field value(s)

        Returns:
            H in km/s/Mpc
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        rho_L = self.Omega_L0_eff
        rho_hor = self.compute_rho_hor(a, M)

        H_squared = rho_m + rho_r + rho_L + rho_hor
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def set_M_today(self, M_today: Any) -> None:
        """Set memory field at z=0 and compute self-consistent Lambda.

        Args:
            M_today: Memory field value(s) at z=0
        """
        self.Omega_hor0 = self.compute_rho_hor(1.0, M_today)
        self.Omega_L0_eff = self.Omega_L0_base - self.Omega_hor0

        if self.Omega_L0_eff < 0:
            import warnings
            warnings.warn(
                f"Effective Lambda is negative: Omega_L0_eff = {self.Omega_L0_eff:.4f}. "
                f"Omega_hor0 = {self.Omega_hor0:.4f} > Omega_L0_base = {self.Omega_L0_base:.4f}."
            )

    def solve(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> HorizonMemoryResult:
        """Solve the complete horizon-memory model.

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            HorizonMemoryResult with full evolution
        """
        try:
            # Integrate memory field
            ln_a_arr, M_arr = self.integrate_memory(z_max, n_points)

            if ln_a_arr is None or M_arr is None:
                return HorizonMemoryResult(
                    params=self.params,
                    success=False,
                    message="Memory field integration failed"
                )

            # Convert to scale factor and redshift
            a_arr = np.exp(ln_a_arr)
            z_arr = 1.0 / a_arr - 1.0

            # Get M at z=0 and set self-consistent Lambda
            M_today = self._get_M_at_ln_a(0.0, ln_a_arr, M_arr)
            self.set_M_today(M_today)

            # Compute H(z), rho_hor(z), w_hor(z)
            n = len(a_arr)
            H_arr = np.zeros(n)
            H_gr_arr = np.zeros(n)
            rho_hor_arr = np.zeros(n)
            w_hor_arr = np.zeros(n)

            for i, (a, ln_a) in enumerate(zip(a_arr, ln_a_arr)):
                M_i = self._get_M_at_ln_a(ln_a, ln_a_arr, M_arr)
                H_arr[i] = self.H_with_memory(a, M_i)
                H_gr_arr[i] = self.H_GR(a)
                rho_hor_arr[i] = self.compute_rho_hor(a, M_i)
                w_hor_arr[i] = self.compute_w_hor(a, M_i)

            # Compute H ratio
            H_ratio = H_arr / H_gr_arr

            # Compute delta_H0 at z=0 (always ~0 with self-consistent Lambda)
            delta_H0_frac = H_ratio[np.argmin(np.abs(z_arr))] - 1.0

            # Compute late-time H effect at z=0.5 (relevant for SNe)
            z_05_idx = np.argmin(np.abs(z_arr - 0.5))
            late_time_H_effect = (H_ratio[z_05_idx] - 1.0) * 100.0  # Percent

            # Compute SN distance deviation at z=0.5
            sn_dev = self._compute_sn_distance_deviation(ln_a_arr, M_arr, z_sn=0.5)

            # Compute CMB distance deviation
            cmb_dev = self._compute_cmb_distance_deviation(ln_a_arr, M_arr)

            return HorizonMemoryResult(
                params=self.params,
                success=True,
                a=a_arr,
                z=z_arr,
                H=H_arr,
                M=M_arr,
                rho_hor=rho_hor_arr,
                w_hor=w_hor_arr,
                H_ratio=H_ratio,
                delta_H0_frac=delta_H0_frac,
                late_time_H_effect=late_time_H_effect,
                sn_distance_deviation=sn_dev,
                cmb_distance_deviation=cmb_dev,
                Omega_hor0=self.Omega_hor0,
                Omega_L0_eff=self.Omega_L0_eff,
                message="Success"
            )

        except Exception as e:
            return HorizonMemoryResult(
                params=self.params,
                success=False,
                message=f"Error: {str(e)}"
            )

    def _get_M_at_ln_a(self, ln_a: float, ln_a_arr: NDArray, M_arr: NDArray) -> Any:
        """Interpolate memory field at given ln(a).

        Args:
            ln_a: Target ln(a)
            ln_a_arr: Array of ln(a) values
            M_arr: Memory field array

        Returns:
            Interpolated M value
        """
        if M_arr.ndim == 1:
            return np.interp(ln_a, ln_a_arr, M_arr)
        else:
            # Multi-component memory
            result = np.zeros(M_arr.shape[0])
            for j in range(M_arr.shape[0]):
                result[j] = np.interp(ln_a, ln_a_arr, M_arr[j])
            return result

    def _compute_cmb_distance_deviation(
        self,
        ln_a_arr: NDArray,
        M_arr: NDArray,
        z_star: float = 1089.0
    ) -> float:
        """Compute CMB angular diameter distance deviation.

        Args:
            ln_a_arr: ln(a) array from integration
            M_arr: Memory field array
            z_star: Redshift of last scattering

        Returns:
            Percent deviation |D_A_hm/D_A_lcdm - 1| * 100
        """
        def H_hm_func(z):
            a = 1.0 / (1.0 + z)
            ln_a = np.log(a)
            # Clamp ln_a to valid range
            ln_a = max(ln_a, ln_a_arr.min())
            ln_a = min(ln_a, ln_a_arr.max())
            M = self._get_M_at_ln_a(ln_a, ln_a_arr, M_arr)
            return self.H_with_memory(a, M) / self.H0

        def H_gr_func(z):
            a = 1.0 / (1.0 + z)
            return self.H_GR(a) / self.H0

        # Compute comoving distances
        try:
            D_star_hm, _ = quad(lambda z: 1.0 / H_hm_func(z), 0, z_star, limit=500)
            D_star_gr, _ = quad(lambda z: 1.0 / H_gr_func(z), 0, z_star, limit=500)
        except Exception:
            return np.nan

        if D_star_gr <= 0:
            return np.nan

        ratio = D_star_hm / D_star_gr
        return abs(ratio - 1.0) * 100.0  # Percent

    def _compute_sn_distance_deviation(
        self,
        ln_a_arr: NDArray,
        M_arr: NDArray,
        z_sn: float = 0.5
    ) -> float:
        """Compute luminosity distance deviation at SN-relevant redshift.

        Args:
            ln_a_arr: ln(a) array from integration
            M_arr: Memory field array
            z_sn: Redshift for SNe comparison (default z=0.5)

        Returns:
            Percent deviation |D_L_hm/D_L_lcdm - 1| * 100
        """
        def H_hm_func(z):
            a = 1.0 / (1.0 + z)
            ln_a = np.log(a)
            # Clamp ln_a to valid range
            ln_a = max(ln_a, ln_a_arr.min())
            ln_a = min(ln_a, ln_a_arr.max())
            M = self._get_M_at_ln_a(ln_a, ln_a_arr, M_arr)
            return self.H_with_memory(a, M) / self.H0

        def H_gr_func(z):
            a = 1.0 / (1.0 + z)
            return self.H_GR(a) / self.H0

        # Compute comoving distances to z_sn
        try:
            D_sn_hm, _ = quad(lambda z: 1.0 / H_hm_func(z), 0, z_sn, limit=100)
            D_sn_gr, _ = quad(lambda z: 1.0 / H_gr_func(z), 0, z_sn, limit=100)
        except Exception:
            return np.nan

        if D_sn_gr <= 0:
            return np.nan

        # Luminosity distance D_L = (1+z) * D_comoving
        # But ratio is same for D_L and D_comoving
        ratio = D_sn_hm / D_sn_gr
        return abs(ratio - 1.0) * 100.0  # Percent
