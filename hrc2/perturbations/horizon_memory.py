"""Horizon-memory perturbation model.

This module implements a fluid-based effective perturbation treatment for the
horizon-memory dark energy component. The horizon-memory model modifies the
Friedmann equation with an effective energy density rho_hor(a) that evolves
as a memory integral of the horizon entropy.

Since this is a fluid-based DE model (not a scalar field), we treat it using
the standard perturbed fluid equations in synchronous gauge:

    delta'_hor = -(1+w_hor)(theta_hor + h'/2)
                 - 3*H*(c_s^2 - w_hor)*delta_hor
                 - 9*H^2*(1+w_hor)*(c_s^2 - c_a^2)*theta_hor/k^2

    theta'_hor = -H*(1 - 3*c_s^2)*theta_hor
                 + c_s^2/(1+w_hor) * k^2 * delta_hor
                 + (source terms from w_hor' if needed)

where:
    - delta_hor = perturbation in horizon-memory density
    - theta_hor = velocity divergence of horizon-memory component
    - w_hor(z) = effective equation of state from background calculation
    - c_s^2 = sound speed squared (we use c_s^2 = 1 for smooth DE)
    - c_a^2 = adiabatic sound speed (= w_hor for adiabatic fluids)
    - h = trace of metric perturbation in synchronous gauge

The c_s^2 = 1 choice makes the DE smooth on subhorizon scales, which is
appropriate for a non-clustering dark energy model.

References:
- Ma & Bertschinger (1995) for synchronous gauge formalism
- Hu & Eisenstein (1999) for DE perturbations
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# Safety constants
W_MIN = -2.5  # Minimum allowed w_hor (phantom limit)
W_MAX = 0.5   # Maximum allowed w_hor
W_SINGULARITY_EPS = 1e-6  # Epsilon for avoiding 1+w -> 0 singularity


@dataclass
class PerturbationState:
    """State of perturbation variables at a given time."""
    a: float  # Scale factor
    z: float  # Redshift
    delta: float  # Density perturbation
    theta: float  # Velocity divergence

    @property
    def is_valid(self) -> bool:
        """Check if state is physically valid."""
        return (np.isfinite(self.delta) and
                np.isfinite(self.theta) and
                np.abs(self.delta) < 100.0)  # Cap unreasonable growth


@dataclass
class PerturbationResult:
    """Full perturbation evolution result."""
    k: float  # Wavenumber [h/Mpc]
    a: NDArray[np.floating]  # Scale factors
    z: NDArray[np.floating]  # Redshifts
    delta: NDArray[np.floating]  # Density perturbations
    theta: NDArray[np.floating]  # Velocity divergences

    success: bool = True
    message: str = "Success"

    def delta_at(self, a_target: float) -> float:
        """Interpolate delta at given scale factor."""
        if a_target < self.a.min() or a_target > self.a.max():
            return np.nan
        return np.interp(a_target, self.a, self.delta)

    def theta_at(self, a_target: float) -> float:
        """Interpolate theta at given scale factor."""
        if a_target < self.a.min() or a_target > self.a.max():
            return np.nan
        return np.interp(a_target, self.a, self.theta)


class HorizonMemoryPerturbations:
    """Perturbation solver for horizon-memory dark energy.

    This class implements the effective fluid perturbation equations for the
    horizon-memory component, treating it as a smooth dark energy fluid with
    time-varying equation of state w_hor(z).

    The key feature is using c_s^2 = 1 which makes the DE perturbations
    oscillate and decay on subhorizon scales, effectively making the DE
    smooth (non-clustering).

    Attributes:
        w_hor_func: Callable w_hor(z) from background analysis
        H_func: Callable H(z) in units of H0
        rho_hor_func: Callable rho_hor(z) / rho_crit0
        c_s_squared: Sound speed squared (default 1.0)
        H0: Hubble constant in km/s/Mpc
    """

    def __init__(
        self,
        w_hor_func: Callable[[float], float],
        H_func: Callable[[float], float],
        rho_hor_func: Optional[Callable[[float], float]] = None,
        c_s_squared: float = 1.0,
        H0: float = 67.4,
    ):
        """Initialize perturbation solver.

        Args:
            w_hor_func: Function w_hor(z) returning equation of state
            H_func: Function H(z) returning Hubble parameter in units of H0
            rho_hor_func: Optional function rho_hor(z)/rho_crit0
            c_s_squared: Sound speed squared (default 1.0 for smooth DE)
            H0: Hubble constant [km/s/Mpc]
        """
        self.w_hor_func = w_hor_func
        self.H_func = H_func
        self.rho_hor_func = rho_hor_func
        self.c_s_squared = c_s_squared
        self.H0 = H0

        # Storage for evolution history (single k mode)
        self._history: List[PerturbationState] = []
        self._current_k: Optional[float] = None

        # Precompute w_hor derivative if needed
        self._dw_dz_func: Optional[Callable] = None

    def _safe_w_hor(self, z: float) -> float:
        """Get w_hor with safety bounds and singularity avoidance."""
        w = self.w_hor_func(z)

        # Clamp to physical range
        w = np.clip(w, W_MIN, W_MAX)

        # Avoid exact w = -1 (singularity in perturbation equations)
        if np.abs(w + 1.0) < W_SINGULARITY_EPS:
            w = -1.0 + W_SINGULARITY_EPS * np.sign(w + 1.0 + 1e-10)

        return w

    def _one_plus_w(self, z: float) -> float:
        """Compute 1+w with safety bounds."""
        w = self._safe_w_hor(z)
        opw = 1.0 + w

        # Ensure we don't divide by zero
        if np.abs(opw) < W_SINGULARITY_EPS:
            opw = W_SINGULARITY_EPS * np.sign(opw + 1e-10)

        return opw

    def _compute_dw_dz(self, z: float, eps: float = 0.01) -> float:
        """Compute dw_hor/dz numerically."""
        if z < eps:
            w_plus = self._safe_w_hor(z + eps)
            w_0 = self._safe_w_hor(z)
            return (w_plus - w_0) / eps
        else:
            w_plus = self._safe_w_hor(z + eps)
            w_minus = self._safe_w_hor(z - eps)
            return (w_plus - w_minus) / (2 * eps)

    def _H_conformal(self, z: float) -> float:
        """Compute conformal Hubble parameter aH in units of H0."""
        a = 1.0 / (1.0 + z)
        H = self.H_func(z)
        return a * H

    def initialize(
        self,
        a_init: float,
        delta_init: float = 0.0,
        theta_init: float = 0.0,
        k: float = 0.1,
    ) -> PerturbationState:
        """Initialize perturbation evolution.

        For smooth DE (c_s^2 = 1), we typically start with small or zero
        perturbations. The initial conditions are:
            delta(a_init) = delta_init (typically 0)
            theta(a_init) = theta_init (typically 0)

        Args:
            a_init: Initial scale factor
            delta_init: Initial density perturbation (default 0)
            theta_init: Initial velocity divergence (default 0)
            k: Wavenumber [h/Mpc]

        Returns:
            Initial PerturbationState
        """
        z_init = 1.0 / a_init - 1.0

        self._history = []
        self._current_k = k

        state = PerturbationState(
            a=a_init,
            z=z_init,
            delta=delta_init,
            theta=theta_init,
        )
        self._history.append(state)

        return state

    def step(
        self,
        a: float,
        delta: float,
        theta: float,
        h_prime: float = 0.0,
    ) -> Tuple[float, float]:
        """Compute one step of perturbation evolution.

        This returns the derivatives d(delta)/d(ln a) and d(theta)/d(ln a).

        Equations (in synchronous gauge, using conformal time derivative ' = d/d(ln a)):

            delta' = -(1+w)(theta/H + h'/(2H))
                     - 3(c_s^2 - w)*delta
                     - 9H(1+w)(c_s^2 - c_a^2)*theta/(k^2)

            theta' = -(1 - 3c_s^2)*theta
                     + (c_s^2/(1+w)) * (k/H)^2 * delta

        For c_s^2 = 1 and assuming c_a^2 = w (adiabatic):
            The last term in delta' equation becomes -9H(1+w)(1-w)*theta/k^2

        Args:
            a: Current scale factor
            delta: Current density perturbation
            theta: Current velocity divergence
            h_prime: Metric perturbation derivative (typically small, set to 0)

        Returns:
            Tuple (d_delta/d_ln_a, d_theta/d_ln_a)
        """
        if self._current_k is None:
            raise RuntimeError("Must call initialize() before step()")

        z = 1.0 / a - 1.0
        k = self._current_k

        # Background quantities
        w = self._safe_w_hor(z)
        opw = self._one_plus_w(z)
        H = self.H_func(z)  # In units of H0

        c_s2 = self.c_s_squared
        c_a2 = w  # Adiabatic sound speed for fluid

        # For numerical stability, cap perturbation magnitudes
        delta = np.clip(delta, -50.0, 50.0)
        theta = np.clip(theta, -1e6, 1e6)

        # Convert k from h/Mpc to dimensionless k/H0
        # k_phys [Mpc^-1] = k [h/Mpc] * h
        # k/H = k_phys / H = k * h / (H * H0/c) where H0/c ~ 1/(3000 Mpc/h)
        # So k/H ~ k * 3000/H (with H in units of H0)
        # Let's use: k_over_H = k * (c/H0) / H = k * 2997.9 / (H * H0)
        # But H is already in H0 units, so k_over_H = k * 2997.9 / H
        # Actually, we need (k/aH)^2 for subhorizon modes

        # Use k in units where H0 = 100 km/s/Mpc = 1/(2997.9 Mpc/h)
        # k [h/Mpc] / H [in H0 units] / (H0 in Mpc^-1)
        # k/(aH) in Mpc^-1 units: k_eff = k * h / (a * H * H0/c)
        # With H0 = 100h km/s/Mpc: H0/c = h/(2997.9 Mpc)
        # So k/(aH) = k * 2997.9 / (a * H) in dimensionless units

        h = self.H0 / 100.0  # h = H0/(100 km/s/Mpc)
        k_over_aH = k * 2997.9 * h / (a * H * self.H0) if H > 0 else 0.0
        k_over_aH_sq = k_over_aH ** 2

        # Perturbation equations
        # delta' = -(1+w)*(theta/(aH) + h'/(2*aH)) - 3*(c_s^2 - w)*delta
        #          - 9*(aH)*(1+w)*(c_s^2 - c_a^2)*theta/k^2
        #
        # For simplicity, we'll work in conformal Hubble units and assume h' ~ 0

        aH = a * H  # Conformal Hubble in H0 units

        # Term 1: density perturbation source from velocity
        # In the gauge where h' is small, this simplifies
        term1 = -opw * theta / aH if aH > 0 else 0.0

        # Term 2: pressure perturbation contribution
        term2 = -3.0 * (c_s2 - w) * delta

        # Term 3: non-adiabatic pressure (vanishes for c_s^2 = c_a^2)
        # This term is typically small for smooth DE
        if k > 0 and np.abs(c_s2 - c_a2) > 1e-10:
            term3 = -9.0 * aH * opw * (c_s2 - c_a2) * theta / (k * k)
        else:
            term3 = 0.0

        d_delta_dlna = term1 + term2 + term3

        # Theta equation
        # theta' = -(1 - 3*c_s^2)*theta + (c_s^2/(1+w)) * k^2/(aH)^2 * (aH) * delta
        #        = -(1 - 3*c_s^2)*theta + c_s^2/(1+w) * (k/aH)^2 * aH * delta

        # Actually the standard form is:
        # d(theta)/d(ln a) = -H'/(H)*theta - (1-3c_s^2)*theta + c_s^2/(1+w) * (k/(aH))^2 * delta * aH
        # where H'/H = d ln H / d ln a

        # Simpler form assuming approximately de Sitter (H ~ const):
        theta_friction = -(1.0 - 3.0 * c_s2) * theta

        # Pressure gradient term
        if np.abs(opw) > W_SINGULARITY_EPS:
            theta_pressure = c_s2 / opw * k_over_aH_sq * delta * aH
        else:
            theta_pressure = 0.0

        d_theta_dlna = theta_friction + theta_pressure

        # Safety caps on derivatives
        d_delta_dlna = np.clip(d_delta_dlna, -100.0, 100.0)
        d_theta_dlna = np.clip(d_theta_dlna, -1e7, 1e7)

        return d_delta_dlna, d_theta_dlna

    def solve(
        self,
        k: float,
        a_init: float = 1e-3,
        a_final: float = 1.0,
        n_points: int = 200,
        delta_init: float = 0.0,
        theta_init: float = 0.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> PerturbationResult:
        """Solve the full perturbation evolution.

        Args:
            k: Wavenumber [h/Mpc]
            a_init: Initial scale factor
            a_final: Final scale factor
            n_points: Number of output points
            delta_init: Initial density perturbation
            theta_init: Initial velocity divergence
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            PerturbationResult with full evolution history
        """
        self._current_k = k

        # Output points in ln(a)
        ln_a_init = np.log(a_init)
        ln_a_final = np.log(a_final)
        ln_a_eval = np.linspace(ln_a_init, ln_a_final, n_points)

        def ode_system(ln_a: float, y: np.ndarray) -> np.ndarray:
            """ODE system for perturbation evolution in ln(a)."""
            delta, theta = y
            a = np.exp(ln_a)

            d_delta, d_theta = self.step(a, delta, theta)

            return np.array([d_delta, d_theta])

        # Initial conditions
        y0 = np.array([delta_init, theta_init])

        # Solve ODE
        try:
            sol = solve_ivp(
                ode_system,
                (ln_a_init, ln_a_final),
                y0,
                method='RK45',
                t_eval=ln_a_eval,
                rtol=rtol,
                atol=atol,
            )

            if not sol.success:
                return PerturbationResult(
                    k=k,
                    a=np.exp(ln_a_eval),
                    z=1.0 / np.exp(ln_a_eval) - 1.0,
                    delta=np.full(n_points, np.nan),
                    theta=np.full(n_points, np.nan),
                    success=False,
                    message=f"ODE solver failed: {sol.message}",
                )

            a_arr = np.exp(sol.t)
            z_arr = 1.0 / a_arr - 1.0

            return PerturbationResult(
                k=k,
                a=a_arr,
                z=z_arr,
                delta=sol.y[0],
                theta=sol.y[1],
                success=True,
                message="Success",
            )

        except Exception as e:
            return PerturbationResult(
                k=k,
                a=np.exp(ln_a_eval),
                z=1.0 / np.exp(ln_a_eval) - 1.0,
                delta=np.full(n_points, np.nan),
                theta=np.full(n_points, np.nan),
                success=False,
                message=f"Integration error: {str(e)}",
            )

    def get_delta(self, a: float) -> float:
        """Get density perturbation at given scale factor.

        This uses the most recently solved perturbation history.

        Args:
            a: Scale factor

        Returns:
            Density perturbation delta(a)
        """
        if not self._history:
            return 0.0

        # Simple linear interpolation from history
        a_arr = np.array([s.a for s in self._history])
        delta_arr = np.array([s.delta for s in self._history])

        return np.interp(a, a_arr, delta_arr)

    def get_theta(self, a: float) -> float:
        """Get velocity divergence at given scale factor.

        Args:
            a: Scale factor

        Returns:
            Velocity divergence theta(a)
        """
        if not self._history:
            return 0.0

        a_arr = np.array([s.a for s in self._history])
        theta_arr = np.array([s.theta for s in self._history])

        return np.interp(a, a_arr, theta_arr)

    def solve_multiple_k(
        self,
        k_values: List[float],
        a_init: float = 1e-3,
        a_final: float = 1.0,
        n_points: int = 200,
        delta_init: float = 0.0,
        theta_init: float = 0.0,
    ) -> Dict[float, PerturbationResult]:
        """Solve perturbation evolution for multiple k modes.

        Args:
            k_values: List of wavenumbers [h/Mpc]
            a_init: Initial scale factor
            a_final: Final scale factor
            n_points: Number of output points
            delta_init: Initial density perturbation
            theta_init: Initial velocity divergence

        Returns:
            Dictionary mapping k -> PerturbationResult
        """
        results = {}

        for k in k_values:
            results[k] = self.solve(
                k=k,
                a_init=a_init,
                a_final=a_final,
                n_points=n_points,
                delta_init=delta_init,
                theta_init=theta_init,
            )

        return results


def create_perturbations_from_background(
    z_array: np.ndarray,
    w_hor_array: np.ndarray,
    H_array: np.ndarray,
    rho_hor_array: Optional[np.ndarray] = None,
    c_s_squared: float = 1.0,
    H0: float = 67.4,
) -> HorizonMemoryPerturbations:
    """Create perturbation solver from background evolution arrays.

    This is a convenience function for creating a HorizonMemoryPerturbations
    instance from arrays computed by the background analysis.

    Args:
        z_array: Redshift array
        w_hor_array: w_hor(z) array
        H_array: H(z) array in units of H0
        rho_hor_array: Optional rho_hor(z)/rho_crit0 array
        c_s_squared: Sound speed squared
        H0: Hubble constant [km/s/Mpc]

    Returns:
        HorizonMemoryPerturbations instance
    """
    # Create interpolators
    # Handle NaN values in w_hor
    valid_mask = ~np.isnan(w_hor_array)
    if np.sum(valid_mask) < 2:
        # Not enough valid points, use constant w = -1
        def w_func(z):
            return -1.0
    else:
        z_valid = z_array[valid_mask]
        w_valid = w_hor_array[valid_mask]
        w_interp = interp1d(z_valid, w_valid, kind='linear',
                           bounds_error=False, fill_value=(w_valid[0], w_valid[-1]))
        def w_func(z):
            return float(w_interp(z))

    H_interp = interp1d(z_array, H_array, kind='linear',
                       bounds_error=False, fill_value=(H_array[0], H_array[-1]))
    def H_func(z):
        return float(H_interp(z))

    rho_func = None
    if rho_hor_array is not None:
        rho_interp = interp1d(z_array, rho_hor_array, kind='linear',
                             bounds_error=False, fill_value=(rho_hor_array[0], rho_hor_array[-1]))
        def rho_func(z):
            return float(rho_interp(z))

    return HorizonMemoryPerturbations(
        w_hor_func=w_func,
        H_func=H_func,
        rho_hor_func=rho_func,
        c_s_squared=c_s_squared,
        H0=H0,
    )
