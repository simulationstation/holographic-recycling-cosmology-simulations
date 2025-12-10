"""Refinement C: Early-Time Suppression Window.

This refinement applies a suppression factor to the horizon-memory density
at early times:

    ρ_hor(a) → ρ_hor(a) * f_supp(a)

where:
    f_supp(a) = 1 - exp(-(a / a_supp)^n_supp)

Key physics:
- At a << a_supp: f_supp → 0, no horizon-memory contribution
- At a >> a_supp: f_supp → 1, full horizon-memory contribution
- This removes the early-time ρ_hor that causes CMB distance deviation
- The transition is controlled by a_supp (scale) and n_supp (sharpness)

Parameters:
    a_supp: Suppression scale factor (transition center)
    n_supp: Suppression power (controls transition sharpness)
    lambda_hor: Base amplitude
    tau_hor: Memory timescale

Goal: Zero out early-time contribution while preserving late-time H0 effect
"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import HorizonMemoryModel, HorizonMemoryParameters, RefinementType


class EarlyTimeSuppression(HorizonMemoryModel):
    """Early-time suppression window for horizon-memory density.

    Applies a smooth suppression factor:
        f_supp(a) = 1 - exp(-(a/a_supp)^n_supp)

    This turns off the memory contribution at early times (a << a_supp)
    while preserving it at late times (a ~ 1).
    """

    def __init__(self, params: HorizonMemoryParameters):
        """Initialize with parameters."""
        # Ensure correct refinement type
        params.refinement_type = RefinementType.EARLY_SUPPRESSION
        super().__init__(params)

        self.a_supp = params.a_supp
        self.n_supp = params.n_supp
        self.lambda_hor = params.lambda_hor
        self.tau_hor = params.tau_hor

    @property
    def name(self) -> str:
        return f"T06C_supp_a={self.a_supp:.4f}_n={self.n_supp:.1f}"

    def f_suppression(self, a: float) -> float:
        """Compute suppression factor at scale factor a.

        f_supp(a) = 1 - exp(-(a/a_supp)^n_supp)

        Args:
            a: Scale factor

        Returns:
            Suppression factor in [0, 1]
        """
        if a <= 0:
            return 0.0
        x = a / self.a_supp
        return 1.0 - np.exp(-(x ** self.n_supp))

    def df_suppression_dlna(self, a: float) -> float:
        """Compute d(f_supp)/d(ln a) for EoS calculation.

        df/d(ln a) = a * df/da = n * (a/a_supp)^n * exp(-(a/a_supp)^n)

        Args:
            a: Scale factor

        Returns:
            Derivative of suppression factor wrt ln(a)
        """
        if a <= 0:
            return 0.0
        x = a / self.a_supp
        return self.n_supp * (x ** self.n_supp) * np.exp(-(x ** self.n_supp))

    def integrate_memory(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate standard memory field ODE.

        The suppression is applied to rho_hor, not to the memory field itself.
        This keeps M(a) evolution unchanged, but ρ_hor is modulated.

        Solves: dM/d(ln a) = (S_norm(a) - M) / τ_hor

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

            # Standard memory evolution (suppression applied to rho, not M)
            dM_dlna = (S_n - M) / self.tau_hor
            return [dM_dlna]

        # Initial condition
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
        """Compute suppressed horizon-memory energy density.

        ρ_hor(a) = λ_hor * M(a) * f_supp(a)

        Args:
            a: Scale factor
            M: Memory field value

        Returns:
            Suppressed energy density in units of 3*H0^2
        """
        M_val = float(M) if np.isscalar(M) else float(M[0]) if hasattr(M, '__len__') else float(M)

        # Apply suppression factor
        f_supp = self.f_suppression(a)
        return self.lambda_hor * max(M_val, 0.0) * f_supp

    def compute_w_hor(self, a: float, M: Any, eps: float = 1e-4) -> float:
        """Compute horizon-memory equation of state with suppression.

        For suppressed model:
            ρ_hor = λ * M * f_supp
            w = -1 - (1/3) * d ln(ρ_hor) / d ln(a)
              = -1 - (1/3) * [d ln(M)/d ln(a) + d ln(f_supp)/d ln(a)]

        Args:
            a: Scale factor
            M: Memory field value
            eps: Not used

        Returns:
            Equation of state w_hor
        """
        M_val = float(M) if np.isscalar(M) else float(M[0]) if hasattr(M, '__len__') else float(M)

        f_supp = self.f_suppression(a)
        rho = self.lambda_hor * M_val * f_supp

        if rho <= 1e-15 or M_val <= 1e-15 or f_supp <= 1e-15:
            return -1.0

        # Get S_norm
        H = self.H_GR(a)
        S_n = self.S_norm(H)

        # d ln(M)/d ln(a) from the ODE
        d_ln_M = (S_n / M_val - 1.0) / self.tau_hor

        # d ln(f_supp)/d ln(a) = (1/f_supp) * df/d ln(a)
        df_dlna = self.df_suppression_dlna(a)
        d_ln_f = df_dlna / f_supp

        # Total d ln(ρ)/d ln(a)
        d_ln_rho = d_ln_M + d_ln_f

        # w = -1 - (1/3) * d ln(ρ)/d ln(a)
        w = -1.0 - d_ln_rho / 3.0

        # Safety bounds
        return np.clip(w, -2.5, 0.5)


def create_early_suppression_model(
    a_supp: float = 0.01,
    n_supp: float = 2.0,
    lambda_hor: float = 0.2,
    tau_hor: float = 0.1,
    **kwargs
) -> EarlyTimeSuppression:
    """Factory function to create an early-time suppression model.

    Args:
        a_supp: Suppression scale factor (transition center)
        n_supp: Suppression power (sharpness)
        lambda_hor: Amplitude of horizon-memory density
        tau_hor: Memory timescale
        **kwargs: Additional HorizonMemoryParameters fields

    Returns:
        EarlyTimeSuppression model instance
    """
    params = HorizonMemoryParameters(
        refinement_type=RefinementType.EARLY_SUPPRESSION,
        a_supp=a_supp,
        n_supp=n_supp,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        **kwargs
    )
    return EarlyTimeSuppression(params)


def scan_early_suppression_parameters(
    a_supp_range: Tuple[float, float] = (0.001, 0.1),
    n_supp_range: Tuple[float, float] = (1.0, 5.0),
    n_a_supp: int = 20,
    n_n_supp: int = 20,
    lambda_hor: float = 0.2,
    tau_hor: float = 0.1,
    z_max: float = 1200.0,
) -> dict:
    """Perform 2D parameter scan for early-time suppression.

    Args:
        a_supp_range: (min, max) for a_supp (log scale recommended)
        n_supp_range: (min, max) for n_supp
        n_a_supp: Number of a_supp grid points
        n_n_supp: Number of n_supp grid points
        lambda_hor: Fixed lambda_hor value
        tau_hor: Fixed tau_hor value
        z_max: Maximum redshift for integration

    Returns:
        Dictionary with scan results
    """
    # Use log spacing for a_supp
    a_supp_vals = np.logspace(
        np.log10(a_supp_range[0]),
        np.log10(a_supp_range[1]),
        n_a_supp
    )
    n_supp_vals = np.linspace(n_supp_range[0], n_supp_range[1], n_n_supp)

    # Result arrays
    delta_H0 = np.zeros((n_a_supp, n_n_supp))
    cmb_dev = np.zeros((n_a_supp, n_n_supp))
    success_mask = np.zeros((n_a_supp, n_n_supp), dtype=bool)

    for i, a_supp in enumerate(a_supp_vals):
        for j, n_supp in enumerate(n_supp_vals):
            try:
                model = create_early_suppression_model(
                    a_supp=a_supp,
                    n_supp=n_supp,
                    lambda_hor=lambda_hor,
                    tau_hor=tau_hor,
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
        "refinement": "T06C",
        "a_supp_vals": a_supp_vals,
        "n_supp_vals": n_supp_vals,
        "lambda_hor": lambda_hor,
        "tau_hor": tau_hor,
        "delta_H0_percent": delta_H0,
        "cmb_deviation_percent": cmb_dev,
        "success_mask": success_mask,
    }
