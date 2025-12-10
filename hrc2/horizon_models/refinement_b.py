"""Refinement B: Two-Component Memory Fluid.

This refinement introduces two coupled memory channels:
    dM1/d(ln a) = (S_norm - M1) / τ1
    dM2/d(ln a) = (M1 - M2) / τ2

Energy density:
    ρ_hor = λ1 * M1 + λ2 * M2

Key physics:
- M1 directly tracks horizon entropy S_norm with timescale τ1
- M2 tracks M1 with a lag (τ2), providing smoother evolution
- The two channels can have different amplitudes (λ1, λ2)
- This allows decoupling early-time and late-time effects:
  - Fast τ1: M1 responds quickly at early times
  - Slow τ2: M2 provides smoothed late-time contribution

Parameters:
    lambda1: Amplitude for M1 channel
    lambda2: Amplitude for M2 channel
    tau1: M1 relaxation timescale (tracks S_norm)
    tau2: M2 relaxation timescale (tracks M1)

Goal: Independent control of early vs late-time behavior
"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import HorizonMemoryModel, HorizonMemoryParameters, RefinementType


class TwoComponentMemory(HorizonMemoryModel):
    """Two-component memory fluid with coupled M1 and M2 channels.

    M1 directly tracks S_norm, M2 tracks M1 with a lag.
    This provides independent control over early-time (fast M1)
    and late-time (smoothed M2) contributions.
    """

    def __init__(self, params: HorizonMemoryParameters):
        """Initialize with parameters."""
        # Ensure correct refinement type
        params.refinement_type = RefinementType.TWO_COMPONENT
        super().__init__(params)

        self.lambda1 = params.lambda1
        self.lambda2 = params.lambda2
        self.tau1 = params.tau1
        self.tau2 = params.tau2

    @property
    def name(self) -> str:
        return f"T06B_2comp_l1={self.lambda1:.3f}_l2={self.lambda2:.3f}_t1={self.tau1:.2f}_t2={self.tau2:.2f}"

    def integrate_memory(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate two-component memory field ODEs.

        Solves:
            dM1/d(ln a) = (S_norm(a) - M1) / τ1
            dM2/d(ln a) = (M1 - M2) / τ2

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            Tuple of (ln_a array, M array) where M has shape (2, n_points)
        """
        a_start = 1.0 / (1.0 + z_max)
        a_end = 1.0
        ln_a_start = np.log(a_start)
        ln_a_end = np.log(a_end)

        # Output array
        ln_a_eval = np.linspace(ln_a_start, ln_a_end, n_points)

        def memory_ode(ln_a, y):
            M1, M2 = y
            a = np.exp(ln_a)

            # Compute H from GR (for S_norm)
            H = self.H_GR(a)
            S_n = self.S_norm(H)

            # M1 tracks S_norm directly
            dM1_dlna = (S_n - M1) / self.tau1

            # M2 tracks M1 with a lag
            dM2_dlna = (M1 - M2) / self.tau2

            return [dM1_dlna, dM2_dlna]

        # Initial conditions: both start at 0
        M1_init = 0.0
        M2_init = 0.0

        try:
            sol = solve_ivp(
                memory_ode,
                (ln_a_start, ln_a_end),
                [M1_init, M2_init],
                method='RK45',
                t_eval=ln_a_eval,
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                return None, None

            # Return M as (2, n_points) array
            return sol.t, sol.y

        except Exception:
            return None, None

    def compute_rho_hor(self, a: float, M: Any) -> float:
        """Compute horizon-memory energy density.

        ρ_hor = λ1 * M1 + λ2 * M2

        Args:
            a: Scale factor
            M: Memory field values [M1, M2]

        Returns:
            Energy density in units of 3*H0^2
        """
        if hasattr(M, '__len__') and len(M) >= 2:
            M1, M2 = float(M[0]), float(M[1])
        else:
            # Fallback: treat as single component
            M1 = float(M) if np.isscalar(M) else float(M[0])
            M2 = 0.0

        return self.lambda1 * max(M1, 0.0) + self.lambda2 * max(M2, 0.0)

    def compute_w_hor(self, a: float, M: Any, eps: float = 1e-4) -> float:
        """Compute horizon-memory equation of state.

        For two-component: w is computed from the combined ρ_hor evolution.

        w = -1 - (1/3) * d ln(ρ_hor) / d ln(a)

        Using the ODEs:
            d(λ1*M1 + λ2*M2)/d ln(a) = λ1*(S_norm - M1)/τ1 + λ2*(M1 - M2)/τ2

        Args:
            a: Scale factor
            M: Memory field values [M1, M2]
            eps: Not used here

        Returns:
            Equation of state w_hor
        """
        if hasattr(M, '__len__') and len(M) >= 2:
            M1, M2 = float(M[0]), float(M[1])
        else:
            M1 = float(M) if np.isscalar(M) else float(M[0])
            M2 = 0.0

        rho = self.lambda1 * M1 + self.lambda2 * M2

        if rho <= 1e-15:
            return -1.0

        # Get S_norm
        H = self.H_GR(a)
        S_n = self.S_norm(H)

        # d(ρ_hor)/d ln(a) = λ1 * dM1/d ln(a) + λ2 * dM2/d ln(a)
        dM1_dlna = (S_n - M1) / self.tau1
        dM2_dlna = (M1 - M2) / self.tau2

        drho_dlna = self.lambda1 * dM1_dlna + self.lambda2 * dM2_dlna

        # d ln(ρ)/d ln(a) = (1/ρ) * dρ/d ln(a)
        d_ln_rho = drho_dlna / rho

        # w = -1 - (1/3) * d ln(ρ)/d ln(a)
        w = -1.0 - d_ln_rho / 3.0

        # Safety bounds
        return np.clip(w, -2.5, 0.5)


def create_two_component_model(
    lambda1: float = 0.15,
    lambda2: float = 0.05,
    tau1: float = 0.05,
    tau2: float = 0.2,
    **kwargs
) -> TwoComponentMemory:
    """Factory function to create a two-component memory model.

    Args:
        lambda1: Amplitude for M1 (direct S_norm tracker)
        lambda2: Amplitude for M2 (M1 follower with lag)
        tau1: M1 relaxation timescale
        tau2: M2 relaxation timescale
        **kwargs: Additional HorizonMemoryParameters fields

    Returns:
        TwoComponentMemory model instance
    """
    params = HorizonMemoryParameters(
        refinement_type=RefinementType.TWO_COMPONENT,
        lambda1=lambda1,
        lambda2=lambda2,
        tau1=tau1,
        tau2=tau2,
        **kwargs
    )
    return TwoComponentMemory(params)


def scan_two_component_parameters(
    tau1_range: Tuple[float, float] = (0.01, 0.2),
    tau2_range: Tuple[float, float] = (0.1, 1.0),
    n_tau1: int = 20,
    n_tau2: int = 20,
    lambda1: float = 0.15,
    lambda2: float = 0.05,
    z_max: float = 1200.0,
) -> dict:
    """Perform 2D parameter scan for two-component memory.

    Scans over tau1 and tau2 with fixed lambda1, lambda2.

    Args:
        tau1_range: (min, max) for tau1
        tau2_range: (min, max) for tau2
        n_tau1: Number of tau1 grid points
        n_tau2: Number of tau2 grid points
        lambda1: Fixed lambda1 value
        lambda2: Fixed lambda2 value
        z_max: Maximum redshift for integration

    Returns:
        Dictionary with scan results
    """
    tau1_vals = np.linspace(tau1_range[0], tau1_range[1], n_tau1)
    tau2_vals = np.linspace(tau2_range[0], tau2_range[1], n_tau2)

    # Result arrays
    delta_H0 = np.zeros((n_tau1, n_tau2))
    cmb_dev = np.zeros((n_tau1, n_tau2))
    success_mask = np.zeros((n_tau1, n_tau2), dtype=bool)

    for i, tau1 in enumerate(tau1_vals):
        for j, tau2 in enumerate(tau2_vals):
            try:
                model = create_two_component_model(
                    lambda1=lambda1,
                    lambda2=lambda2,
                    tau1=tau1,
                    tau2=tau2,
                )
                result = model.solve(z_max=z_max)

                if result.success:
                    delta_H0[i, j] = result.delta_H0_frac * 100  # Percent
                    cmb_dev[i, j] = result.cmb_distance_deviation
                    success_mask[i, j] = True
                else:
                    delta_H0[i, j] = np.nan
                    cmb_dev[i, j] = np.nan

            except Exception:
                delta_H0[i, j] = np.nan
                cmb_dev[i, j] = np.nan

    return {
        "refinement": "T06B",
        "tau1_vals": tau1_vals,
        "tau2_vals": tau2_vals,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "delta_H0_percent": delta_H0,
        "cmb_deviation_percent": cmb_dev,
        "success_mask": success_mask,
    }
