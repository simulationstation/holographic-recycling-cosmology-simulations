"""Refinement A: Adaptive Memory Kernel.

This refinement implements a scale-factor dependent memory relaxation rate:
    τ_hor(a) = τ0 * (a / 1)^p

The memory field evolution becomes:
    dM/d(ln a) = (S_norm(a) - M) / τ_hor(a)

Key physics:
- At early times (a << 1), if p > 0: τ_hor is small -> fast relaxation
- At late times (a ~ 1), τ_hor = τ0 -> normal relaxation
- This allows M to track S_norm more closely at early times,
  reducing the CMB distance deviation while keeping late-time H0 effect

Parameters:
    tau0: Base memory timescale at a=1
    p_hor: Power-law exponent for scale factor dependence

Goal: Reduce D_A error at z* while maintaining ≥5% H0 effect at z=0
"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import HorizonMemoryModel, HorizonMemoryParameters, RefinementType


class AdaptiveMemoryKernel(HorizonMemoryModel):
    """Adaptive memory kernel: τ_hor(a) = τ0 * a^p_hor.

    This refinement modifies the memory relaxation timescale to be
    scale-factor dependent, allowing faster relaxation at early times
    (reducing CMB distance error) while maintaining normal relaxation
    at late times (preserving H0 modification).
    """

    def __init__(self, params: HorizonMemoryParameters):
        """Initialize with parameters."""
        # Ensure correct refinement type
        params.refinement_type = RefinementType.ADAPTIVE_KERNEL
        super().__init__(params)

        self.tau0 = params.tau0
        self.p_hor = params.p_hor
        self.lambda_hor = params.lambda_hor

    @property
    def name(self) -> str:
        return f"T06A_adaptive_tau0={self.tau0:.3f}_p={self.p_hor:.2f}"

    def tau_hor(self, a: float) -> float:
        """Compute scale-factor dependent memory timescale.

        τ_hor(a) = τ0 * a^p_hor

        Args:
            a: Scale factor

        Returns:
            Memory relaxation timescale
        """
        # Safety: ensure tau > 0
        tau = self.tau0 * (a ** self.p_hor)
        return max(tau, 1e-6)

    def integrate_memory(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate adaptive memory field ODE.

        Solves: dM/d(ln a) = (S_norm(a) - M) / τ_hor(a)

        where τ_hor(a) = τ0 * a^p_hor

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            Tuple of (ln_a array, M array)
        """
        a_start = 1.0 / (1.0 + z_max)
        a_end = 1.0
        ln_a_start = np.log(a_start)
        ln_a_end = np.log(a_end)

        # Output array
        ln_a_eval = np.linspace(ln_a_start, ln_a_end, n_points)

        def memory_ode(ln_a, y):
            M = y[0]
            a = np.exp(ln_a)

            # Compute H from GR (for S_norm)
            H = self.H_GR(a)
            S_n = self.S_norm(H)

            # Adaptive timescale
            tau = self.tau_hor(a)

            # Memory evolution
            dM_dlna = (S_n - M) / tau
            return [dM_dlna]

        # Initial condition: M starts at 0 (or small value)
        M_init = 0.0

        try:
            sol = solve_ivp(
                memory_ode,
                (ln_a_start, ln_a_end),
                [M_init],
                method='RK45',
                t_eval=ln_a_eval,
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                return None, None

            return sol.t, sol.y[0]

        except Exception:
            return None, None

    def compute_rho_hor(self, a: float, M: Any) -> float:
        """Compute horizon-memory energy density.

        ρ_hor(a) = λ_hor * M(a)

        Args:
            a: Scale factor
            M: Memory field value

        Returns:
            Energy density in units of 3*H0^2
        """
        M_val = float(M) if np.isscalar(M) else float(M[0]) if hasattr(M, '__len__') else float(M)
        return self.lambda_hor * max(M_val, 0.0)

    def compute_w_hor(self, a: float, M: Any, eps: float = 1e-4) -> float:
        """Compute horizon-memory equation of state.

        For memory model: w = -1 - (1/3) * d ln(ρ_hor) / d ln(a)
                        = -1 - (1/3) * d ln(M) / d ln(a)

        Using the ODE: dM/d ln(a) = (S_norm - M) / τ
        => d ln(M)/d ln(a) = (S_norm/M - 1) / τ
        => w = -1 - (1/3) * (S_norm/M - 1) / τ

        Args:
            a: Scale factor
            M: Memory field value
            eps: Not used here (analytical formula)

        Returns:
            Equation of state w_hor
        """
        M_val = float(M) if np.isscalar(M) else float(M[0]) if hasattr(M, '__len__') else float(M)

        if M_val <= 1e-15:
            return -1.0

        H = self.H_GR(a)
        S_n = self.S_norm(H)
        tau = self.tau_hor(a)

        # From the ODE: d ln(M)/d ln(a) = (S_norm/M - 1) / τ
        d_ln_M = (S_n / M_val - 1.0) / tau

        # w = -1 - (1/3) * d ln(ρ)/d ln(a)
        w = -1.0 - d_ln_M / 3.0

        # Safety bounds
        return np.clip(w, -2.5, 0.5)


def create_adaptive_kernel_model(
    tau0: float = 0.1,
    p_hor: float = 1.0,
    lambda_hor: float = 0.2,
    **kwargs
) -> AdaptiveMemoryKernel:
    """Factory function to create an adaptive memory kernel model.

    Args:
        tau0: Base memory timescale at a=1
        p_hor: Power-law exponent for τ(a) = τ0 * a^p
        lambda_hor: Amplitude of horizon-memory density
        **kwargs: Additional HorizonMemoryParameters fields

    Returns:
        AdaptiveMemoryKernel model instance
    """
    params = HorizonMemoryParameters(
        refinement_type=RefinementType.ADAPTIVE_KERNEL,
        lambda_hor=lambda_hor,
        tau0=tau0,
        p_hor=p_hor,
        **kwargs
    )
    return AdaptiveMemoryKernel(params)


def scan_adaptive_kernel_parameters(
    tau0_range: Tuple[float, float] = (0.01, 0.5),
    p_hor_range: Tuple[float, float] = (-1.0, 3.0),
    n_tau0: int = 20,
    n_p: int = 20,
    lambda_hor: float = 0.2,
    z_max: float = 1200.0,
) -> dict:
    """Perform 2D parameter scan for adaptive memory kernel.

    Args:
        tau0_range: (min, max) for tau0
        p_hor_range: (min, max) for p_hor
        n_tau0: Number of tau0 grid points
        n_p: Number of p_hor grid points
        lambda_hor: Fixed lambda_hor value
        z_max: Maximum redshift for integration

    Returns:
        Dictionary with scan results
    """
    tau0_vals = np.linspace(tau0_range[0], tau0_range[1], n_tau0)
    p_hor_vals = np.linspace(p_hor_range[0], p_hor_range[1], n_p)

    # Result arrays
    delta_H0 = np.zeros((n_tau0, n_p))
    cmb_dev = np.zeros((n_tau0, n_p))
    success_mask = np.zeros((n_tau0, n_p), dtype=bool)

    for i, tau0 in enumerate(tau0_vals):
        for j, p_hor in enumerate(p_hor_vals):
            try:
                model = create_adaptive_kernel_model(
                    tau0=tau0,
                    p_hor=p_hor,
                    lambda_hor=lambda_hor,
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
        "refinement": "T06A",
        "tau0_vals": tau0_vals,
        "p_hor_vals": p_hor_vals,
        "lambda_hor": lambda_hor,
        "delta_H0_percent": delta_H0,
        "cmb_deviation_percent": cmb_dev,
        "success_mask": success_mask,
    }
