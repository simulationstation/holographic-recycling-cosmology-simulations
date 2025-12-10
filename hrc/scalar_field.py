"""Scalar field dynamics for HRC.

This module provides a standalone scalar field solver that can be used
independently or coupled with the background cosmology.

The scalar field equation of motion is:
    φ̈ + 3Hφ̇ + V'(φ) + ξR = 0

where R = 6(2H² + Ḣ) is the Ricci scalar.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .utils.config import HRCParameters, PotentialConfig
from .utils.numerics import check_divergence, DivergenceResult
from .potentials import Potential, QuadraticPotential


class PotentialInterface:
    """Unified interface for both old PotentialConfig and new Potential classes."""

    def __init__(self, potential: Union[PotentialConfig, Potential, None], params: HRCParameters):
        """Initialize with either PotentialConfig or Potential.

        Args:
            potential: Either PotentialConfig, Potential, or None
            params: HRC parameters for defaults
        """
        if potential is None:
            # Default to quadratic potential
            self._potential = QuadraticPotential(V0=params.V0, m=params.m_phi)
            self._is_new_interface = True
        elif isinstance(potential, Potential):
            self._potential = potential
            self._is_new_interface = True
        else:
            # It's a PotentialConfig
            self._potential = potential
            self._is_new_interface = False

    def V(self, phi: float) -> float:
        """Evaluate potential V(phi)."""
        return self._potential.V(phi)

    def dV(self, phi: float) -> float:
        """Evaluate potential derivative dV/dphi."""
        if self._is_new_interface:
            return self._potential.dV_dphi(phi)
        return self._potential.dV(phi)

    @property
    def form(self) -> str:
        """Return potential form for backward compatibility."""
        if self._is_new_interface:
            return self._potential.name
        return self._potential.form


@dataclass
class ScalarFieldState:
    """State of the scalar field at a given time."""

    t: float  # Time (in H0⁻¹ units)
    z: float  # Redshift
    phi: float  # Field value
    phi_dot: float  # dφ/dt
    phi_ddot: float  # d²φ/dt²
    V: float  # Potential V(φ)
    V_prime: float  # dV/dφ
    rho_phi: float  # Energy density ½φ̇² + V
    P_phi: float  # Pressure ½φ̇² - V
    w_phi: float  # Equation of state P/ρ


@dataclass
class ScalarFieldSolution:
    """Full scalar field solution."""

    t: NDArray[np.floating]  # Time array
    z: NDArray[np.floating]  # Redshift array
    phi: NDArray[np.floating]  # Field values
    phi_dot: NDArray[np.floating]  # Field velocities
    V: NDArray[np.floating]  # Potential values
    rho_phi: NDArray[np.floating]  # Energy density
    P_phi: NDArray[np.floating]  # Pressure
    w_phi: NDArray[np.floating]  # Equation of state

    success: bool = True
    message: str = ""

    _phi_interp: Optional[Callable[[float], float]] = None

    def phi_at(self, z: float) -> float:
        """Interpolate φ(z)."""
        if self._phi_interp is None:
            self._phi_interp = interp1d(
                self.z, self.phi, kind="cubic", fill_value="extrapolate"
            )
        return float(self._phi_interp(z))


class ScalarFieldSolver:
    """Solver for scalar field evolution in HRC.

    This solver integrates the Klein-Gordon equation with non-minimal
    coupling to gravity:

        φ̈ + 3Hφ̇ + V'(φ) + ξR = 0

    The Hubble parameter H(z) and Ricci scalar R(z) can be provided
    externally or computed from a background cosmology.
    """

    def __init__(
        self,
        params: HRCParameters,
        potential: Optional[Union[PotentialConfig, Potential]] = None,
        H_func: Optional[Callable[[float], float]] = None,
        R_func: Optional[Callable[[float], float]] = None,
    ):
        """Initialize scalar field solver.

        Args:
            params: HRC parameters
            potential: Potential configuration (default: quadratic)
                      Can be either PotentialConfig (old) or Potential (new)
            H_func: Function H(z) returning Hubble parameter (in H0 units)
                   If None, uses ΛCDM approximation
            R_func: Function R(z) returning Ricci scalar (in H0² units)
                   If None, computes from H(z)
        """
        self.params = params
        self.potential = PotentialInterface(potential, params)

        self._H_func = H_func or self._H_LCDM
        self._R_func = R_func

    def _H_LCDM(self, z: float) -> float:
        """ΛCDM Hubble parameter (in H0 units)."""
        Om = self.params.Omega_m
        Or = self.params.Omega_r
        OL = self.params.Omega_Lambda

        return np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL)

    def _H_dot_LCDM(self, z: float) -> float:
        """Time derivative of H in ΛCDM (in H0² units)."""
        Om = self.params.Omega_m
        Or = self.params.Omega_r

        H = self._H_LCDM(z)

        # dH/dt = -H² d(ln H)/d(ln a) = H² d(ln H)/d(ln(1+z))
        # Using dz/dt = -(1+z)H
        dH2_dz = 3 * Om * (1 + z) ** 2 + 4 * Or * (1 + z) ** 3

        # dH/dt = (1/2H) dH²/dz · dz/dt = -(1+z)/2 · dH²/dz
        # But we need to be careful with the sign
        return -0.5 * dH2_dz / H * (-(1 + z) * H)  # = -(1+z)/2 · dH²/dz

    def _compute_R(self, z: float) -> float:
        """Compute Ricci scalar R = 6(2H² + Ḣ)."""
        if self._R_func is not None:
            return self._R_func(z)

        H = self._H_func(z)
        H_dot = self._H_dot_LCDM(z)
        return 6.0 * (2.0 * H**2 + H_dot)

    def _z_from_t(self, t: float, z0: float = 0.0) -> float:
        """Compute z(t) approximately.

        For small t (near z=0): dz/dt ≈ -(1+z)H(z)
        """
        # Simple approximation: for early times, z ≈ H0 * t
        # More accurate would require integrating dz/dt = -(1+z)H
        return z0 + t  # Very rough approximation

    def _ode_system_t(
        self,
        t: float,
        y: NDArray[np.floating],
        z_of_t: Callable[[float], float],
    ) -> NDArray[np.floating]:
        """ODE system in time coordinates.

        State: y = [φ, φ̇]

        Returns: [φ̇, φ̈]
        """
        phi, phi_dot = y

        z = z_of_t(t)
        H = self._H_func(z)
        R = self._compute_R(z)

        V_prime = self.potential.dV(phi)

        # φ̈ + 3Hφ̇ + V'(φ) + ξR = 0
        phi_ddot = -3.0 * H * phi_dot - V_prime - self.params.xi * R

        return np.array([phi_dot, phi_ddot])

    def _ode_system_z(
        self,
        z: float,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """ODE system in redshift coordinates.

        State: y = [φ, dφ/dz]

        Using dz/dt = -(1+z)H, we have:
        dφ/dz = dφ/dt · dt/dz = -φ̇/[(1+z)H]
        d²φ/dz² = -d(φ̇)/dz / [(1+z)H] + φ̇ d[1/((1+z)H)]/dz

        Returns: [dφ/dz, d²φ/dz²]
        """
        phi, dphi_dz = y

        H = self._H_func(z)

        # Reconstruct φ̇ from dφ/dz
        # dφ/dz = -φ̇/[(1+z)H] => φ̇ = -(1+z)H · dφ/dz
        phi_dot = -(1 + z) * H * dphi_dz

        R = self._compute_R(z)
        V_prime = self.potential.dV(phi)

        # φ̈ from EOM
        phi_ddot = -3.0 * H * phi_dot - V_prime - self.params.xi * R

        # Convert φ̈ to d²φ/dz²
        # φ̈ = d(φ̇)/dt = d(φ̇)/dz · dz/dt = -(1+z)H · d(φ̇)/dz
        # So d(φ̇)/dz = -φ̈/[(1+z)H]
        dphi_dot_dz = -phi_ddot / ((1 + z) * H) if abs(H) > 1e-30 else 0.0

        # dφ̇/dz = d[-(1+z)H · dφ/dz]/dz
        # = -H·dφ/dz - (1+z)·dH/dz·dφ/dz - (1+z)H·d²φ/dz²
        # Solving for d²φ/dz²:
        # d²φ/dz² = -[dφ̇/dz + H·dφ/dz + (1+z)·dH/dz·dφ/dz] / [(1+z)H]

        # Numerical approximation for dH/dz
        dz = 1e-4
        dH_dz = (self._H_func(z + dz) - self._H_func(z - dz)) / (2 * dz)

        if abs(H) > 1e-30:
            d2phi_dz2 = -(
                dphi_dot_dz + H * dphi_dz + (1 + z) * dH_dz * dphi_dz
            ) / ((1 + z) * H)
        else:
            d2phi_dz2 = 0.0

        return np.array([dphi_dz, d2phi_dz2])

    def solve_vs_z(
        self,
        z_max: float = 1100.0,
        z_points: int = 1000,
        phi_0: Optional[float] = None,
        phi_dot_0: Optional[float] = None,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> ScalarFieldSolution:
        """Solve scalar field evolution as function of redshift.

        Args:
            z_max: Maximum redshift
            z_points: Number of output points
            phi_0: Initial field value (default: from params)
            phi_dot_0: Initial velocity (default: from params)
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            ScalarFieldSolution
        """
        phi_0 = phi_0 if phi_0 is not None else self.params.phi_0
        phi_dot_0 = phi_dot_0 if phi_dot_0 is not None else self.params.phi_dot_0

        # Convert φ̇₀ to dφ/dz|₀
        H_0 = self._H_func(0.0)
        dphi_dz_0 = -phi_dot_0 / H_0 if abs(H_0) > 1e-30 else 0.0

        y0 = np.array([phi_0, dphi_dz_0])

        z_eval = np.linspace(0, z_max, z_points)

        try:
            sol = solve_ivp(
                self._ode_system_z,
                t_span=(0, z_max),
                y0=y0,
                method="RK45",
                t_eval=z_eval,
                rtol=rtol,
                atol=atol,
            )

            if not sol.success:
                return self._empty_solution(z_eval, phi_0, f"Failed: {sol.message}")

            phi = sol.y[0]
            dphi_dz = sol.y[1]

        except Exception as e:
            return self._empty_solution(z_eval, phi_0, f"Error: {str(e)}")

        # Compute derived quantities
        n = len(z_eval)
        phi_dot = np.zeros(n)
        V = np.zeros(n)
        rho_phi = np.zeros(n)
        P_phi = np.zeros(n)
        w_phi = np.zeros(n)

        for i in range(n):
            z = z_eval[i]
            H = self._H_func(z)

            phi_dot[i] = -(1 + z) * H * dphi_dz[i]
            V[i] = self.potential.V(phi[i])
            rho_phi[i] = 0.5 * phi_dot[i] ** 2 + V[i]
            P_phi[i] = 0.5 * phi_dot[i] ** 2 - V[i]
            w_phi[i] = P_phi[i] / rho_phi[i] if abs(rho_phi[i]) > 1e-30 else -1.0

        # Time array (approximate)
        # dt/dz = -1/[(1+z)H]
        t = np.zeros(n)
        for i in range(1, n):
            dz = z_eval[i] - z_eval[i - 1]
            H = self._H_func(z_eval[i])
            t[i] = t[i - 1] - dz / ((1 + z_eval[i]) * H)

        return ScalarFieldSolution(
            t=t,
            z=z_eval,
            phi=phi,
            phi_dot=phi_dot,
            V=V,
            rho_phi=rho_phi,
            P_phi=P_phi,
            w_phi=w_phi,
            success=True,
            message="Integration successful",
        )

    def _empty_solution(
        self,
        z_eval: NDArray[np.floating],
        phi_0: float,
        message: str,
    ) -> ScalarFieldSolution:
        """Create empty solution for failed integration."""
        n = len(z_eval)
        return ScalarFieldSolution(
            t=np.zeros(n),
            z=z_eval,
            phi=np.full(n, phi_0),
            phi_dot=np.zeros(n),
            V=np.full(n, self.potential.V(phi_0)),
            rho_phi=np.full(n, self.potential.V(phi_0)),
            P_phi=np.full(n, -self.potential.V(phi_0)),
            w_phi=np.full(n, -1.0),
            success=False,
            message=message,
        )


def compute_slow_roll_parameters(
    phi: float,
    potential: Union[PotentialConfig, Potential, PotentialInterface],
    params: HRCParameters,
) -> Tuple[float, float]:
    """Compute slow-roll parameters ε and η.

    ε = (M_Pl²/2)(V'/V)²
    η = M_Pl²(V''/V)

    Args:
        phi: Field value
        potential: Potential configuration (PotentialConfig, Potential, or PotentialInterface)
        params: HRC parameters

    Returns:
        Tuple of (ε, η)
    """
    # Wrap in interface if needed
    if isinstance(potential, PotentialInterface):
        pot_interface = potential
    else:
        pot_interface = PotentialInterface(potential, params)

    V = pot_interface.V(phi)
    V_prime = pot_interface.dV(phi)

    if abs(V) < 1e-30:
        return np.inf, np.inf

    # In Planck units, M_Pl = 1
    epsilon = 0.5 * (V_prime / V) ** 2

    # V'' - use numerical derivative for generality
    dphi = 1e-5
    V_dprime = (pot_interface.dV(phi + dphi) - pot_interface.dV(phi - dphi)) / (2 * dphi)

    eta = V_dprime / V

    return epsilon, eta


def is_slow_roll_valid(
    phi: float,
    potential: Union[PotentialConfig, Potential, PotentialInterface],
    params: HRCParameters,
    epsilon_max: float = 1.0,
    eta_max: float = 1.0,
) -> Tuple[bool, str]:
    """Check if slow-roll approximation is valid.

    Args:
        phi: Field value
        potential: Potential configuration (PotentialConfig, Potential, or PotentialInterface)
        params: HRC parameters
        epsilon_max: Maximum allowed ε
        eta_max: Maximum allowed |η|

    Returns:
        Tuple of (is_valid, message)
    """
    epsilon, eta = compute_slow_roll_parameters(phi, potential, params)

    if epsilon > epsilon_max:
        return False, f"ε = {epsilon:.3f} > {epsilon_max} (slow roll violated)"
    if abs(eta) > eta_max:
        return False, f"|η| = {abs(eta):.3f} > {eta_max} (slow roll violated)"

    return True, f"Slow roll valid: ε = {epsilon:.3e}, η = {eta:.3e}"
