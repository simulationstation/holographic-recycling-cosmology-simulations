"""Refinement D: Dynamical Equation-of-State Modifier.

This refinement modifies the effective equation of state w_hor(a) using a
transition function:

    w_eff(a) = w_base(a) + Δw / (1 + (a/a_w)^m)

Key physics:
- At early times (a << a_w): w_eff ≈ w_base + Δw
- At late times (a >> a_w): w_eff ≈ w_base
- This allows phantom-like behavior (w < -1) to be concentrated at late times
- The integral ∫ dz/H(z) is tuned for CMB distance

Parameters:
    delta_w: EoS shift amplitude (negative = more phantom)
    a_w: Transition scale factor
    m_eos: Transition power (controls sharpness)
    lambda_hor: Base amplitude
    tau_hor: Memory timescale

Goal: Tune the w(a) profile to fix CMB distance while maintaining H0 effect
"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import HorizonMemoryModel, HorizonMemoryParameters, RefinementType


class DynamicalEoSModifier(HorizonMemoryModel):
    """Dynamical equation-of-state modifier for horizon-memory.

    Modifies w_hor(a) using:
        w_eff(a) = w_base(a) + Δw / (1 + (a/a_w)^m)

    This concentrates phantom behavior at specific redshifts,
    allowing fine-tuning of the CMB distance integral.
    """

    def __init__(self, params: HorizonMemoryParameters):
        """Initialize with parameters."""
        # Ensure correct refinement type
        params.refinement_type = RefinementType.DYNAMICAL_EOS
        super().__init__(params)

        self.delta_w = params.delta_w
        self.a_w = params.a_w
        self.m_eos = params.m_eos
        self.lambda_hor = params.lambda_hor
        self.tau_hor = params.tau_hor

    @property
    def name(self) -> str:
        return f"T06D_eos_dw={self.delta_w:.2f}_aw={self.a_w:.2f}_m={self.m_eos:.1f}"

    def w_modifier(self, a: float) -> float:
        """Compute EoS modification factor.

        Δw_eff(a) = Δw / (1 + (a/a_w)^m)

        Args:
            a: Scale factor

        Returns:
            EoS modification to add to base w
        """
        if a <= 0:
            return self.delta_w
        x = a / self.a_w
        return self.delta_w / (1.0 + x ** self.m_eos)

    def dw_modifier_dlna(self, a: float) -> float:
        """Compute d(Δw_eff)/d(ln a) for consistency checks.

        Let f(a) = Δw / (1 + (a/a_w)^m)
        df/d(ln a) = a * df/da = -Δw * m * (a/a_w)^m / (1 + (a/a_w)^m)^2

        Args:
            a: Scale factor

        Returns:
            Derivative of EoS modifier wrt ln(a)
        """
        if a <= 0:
            return 0.0
        x = a / self.a_w
        numer = -self.delta_w * self.m_eos * (x ** self.m_eos)
        denom = (1.0 + x ** self.m_eos) ** 2
        return numer / denom

    def integrate_memory(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate memory field with modified dynamics.

        The EoS modification affects how ρ_hor evolves through:
            d ln(ρ_hor)/d ln(a) = -3(1 + w_eff)

        We integrate ρ_hor directly using the modified w.

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            Tuple of (ln_a array, combined state array)
        """
        a_start = 1.0 / (1.0 + z_max)
        a_end = 1.0
        ln_a_start = np.log(a_start)
        ln_a_end = np.log(a_end)

        # Output array
        ln_a_eval = np.linspace(ln_a_start, ln_a_end, n_points)

        def combined_ode(ln_a, y):
            """Evolve both M and ρ_hor simultaneously."""
            M, ln_rho = y
            a = np.exp(ln_a)

            # Compute H from GR (for S_norm)
            H = self.H_GR(a)
            S_n = self.S_norm(H)

            # Memory evolution (standard)
            dM_dlna = (S_n - M) / self.tau_hor

            # Compute base w from memory
            if M > 1e-15:
                d_ln_M = (S_n / M - 1.0) / self.tau_hor
                w_base = -1.0 - d_ln_M / 3.0
            else:
                w_base = -1.0

            # Apply EoS modification
            w_eff = w_base + self.w_modifier(a)

            # Clamp w to reasonable range
            w_eff = np.clip(w_eff, -3.0, 1.0)

            # ρ_hor evolution: d ln(ρ)/d ln(a) = -3(1 + w)
            d_ln_rho = -3.0 * (1.0 + w_eff)

            return [dM_dlna, d_ln_rho]

        # Initial conditions
        M_init = 1e-10  # Small positive value
        # Initial ρ_hor at high z is very small
        rho_init = self.lambda_hor * M_init
        ln_rho_init = np.log(max(rho_init, 1e-30))

        try:
            sol = solve_ivp(
                combined_ode,
                (ln_a_start, ln_a_end),
                [M_init, ln_rho_init],
                method='RK45',
                t_eval=ln_a_eval,
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                return None, None

            # Return combined state: [M, ln_rho]
            return sol.t, sol.y

        except Exception:
            return None, None

    def compute_rho_hor(self, a: float, M: Any) -> float:
        """Compute horizon-memory energy density.

        For this refinement, M contains [M_field, ln_rho].
        We use the evolved ln_rho directly.

        Args:
            a: Scale factor
            M: State array [M_field, ln_rho] or scalar M

        Returns:
            Energy density in units of 3*H0^2
        """
        if hasattr(M, '__len__') and len(M) >= 2:
            # Use evolved rho
            ln_rho = float(M[1])
            return np.exp(ln_rho)
        else:
            # Fallback to standard formula
            M_val = float(M) if np.isscalar(M) else float(M[0])
            return self.lambda_hor * max(M_val, 0.0)

    def compute_w_hor(self, a: float, M: Any, eps: float = 1e-4) -> float:
        """Compute modified horizon-memory equation of state.

        w_eff(a) = w_base(a) + Δw / (1 + (a/a_w)^m)

        Args:
            a: Scale factor
            M: State array [M_field, ln_rho] or scalar M
            eps: Not used

        Returns:
            Modified equation of state w_eff
        """
        if hasattr(M, '__len__') and len(M) >= 2:
            M_val = float(M[0])
        else:
            M_val = float(M) if np.isscalar(M) else float(M[0]) if hasattr(M, '__len__') else float(M)

        if M_val <= 1e-15:
            w_base = -1.0
        else:
            # Get S_norm for base w calculation
            H = self.H_GR(a)
            S_n = self.S_norm(H)
            d_ln_M = (S_n / M_val - 1.0) / self.tau_hor
            w_base = -1.0 - d_ln_M / 3.0

        # Apply modification
        w_eff = w_base + self.w_modifier(a)

        # Safety bounds
        return np.clip(w_eff, -3.0, 1.0)

    def solve(
        self,
        z_max: float = 1200.0,
        n_points: int = 500,
    ) -> 'HorizonMemoryResult':
        """Solve the dynamical EoS model.

        Overrides base to handle the combined state array.

        Args:
            z_max: Maximum redshift
            n_points: Number of output points

        Returns:
            HorizonMemoryResult
        """
        from .base import HorizonMemoryResult

        try:
            # Integrate combined system
            ln_a_arr, state_arr = self.integrate_memory(z_max, n_points)

            if ln_a_arr is None or state_arr is None:
                return HorizonMemoryResult(
                    params=self.params,
                    success=False,
                    message="Integration failed"
                )

            # Extract components
            M_arr = state_arr[0]  # Memory field
            ln_rho_arr = state_arr[1]  # ln(rho_hor)
            rho_hor_arr = np.exp(ln_rho_arr)

            # Convert to scale factor and redshift
            a_arr = np.exp(ln_a_arr)
            z_arr = 1.0 / a_arr - 1.0

            # Get values at z=0
            idx_z0 = np.argmin(np.abs(z_arr))
            rho_hor_today = rho_hor_arr[idx_z0]

            # Set self-consistent Lambda
            self.Omega_hor0 = rho_hor_today
            self.Omega_L0_eff = self.Omega_L0_base - self.Omega_hor0

            # Compute H(z) and other quantities
            n = len(a_arr)
            H_arr = np.zeros(n)
            H_gr_arr = np.zeros(n)
            w_hor_arr = np.zeros(n)

            for i, (a, ln_a) in enumerate(zip(a_arr, ln_a_arr)):
                # For H calculation, use evolved rho directly
                z = 1.0 / a - 1.0
                rho_m = self.Omega_m0 * (1 + z)**3
                rho_r = self.Omega_r0 * (1 + z)**4
                rho_L = self.Omega_L0_eff
                rho_hor = rho_hor_arr[i]

                H_squared = rho_m + rho_r + rho_L + rho_hor
                H_arr[i] = np.sqrt(max(H_squared, 0)) * self.H0
                H_gr_arr[i] = self.H_GR(a)

                # Compute w
                w_hor_arr[i] = self.compute_w_hor(a, [M_arr[i], ln_rho_arr[i]])

            # Compute H ratio
            H_ratio = H_arr / H_gr_arr

            # delta_H0 at z=0
            delta_H0_frac = H_ratio[idx_z0] - 1.0

            # CMB distance deviation
            cmb_dev = self._compute_cmb_distance_deviation_direct(
                a_arr, H_arr, H_gr_arr
            )

            return HorizonMemoryResult(
                params=self.params,
                success=True,
                a=a_arr,
                z=z_arr,
                H=H_arr,
                M=state_arr,  # Combined state
                rho_hor=rho_hor_arr,
                w_hor=w_hor_arr,
                H_ratio=H_ratio,
                delta_H0_frac=delta_H0_frac,
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

    def _compute_cmb_distance_deviation_direct(
        self,
        a_arr: NDArray,
        H_arr: NDArray,
        H_gr_arr: NDArray,
        z_star: float = 1089.0,
    ) -> float:
        """Compute CMB distance deviation using pre-computed H arrays.

        Args:
            a_arr: Scale factor array
            H_arr: H(a) with horizon-memory
            H_gr_arr: H(a) GR baseline
            z_star: Redshift of last scattering

        Returns:
            Percent deviation
        """
        from scipy.integrate import quad

        z_arr = 1.0 / a_arr - 1.0

        def H_hm_interp(z):
            return np.interp(z, z_arr[::-1], H_arr[::-1]) / self.H0

        def H_gr_interp(z):
            return np.interp(z, z_arr[::-1], H_gr_arr[::-1]) / self.H0

        try:
            D_star_hm, _ = quad(lambda z: 1.0 / H_hm_interp(z), 0, z_star, limit=500)
            D_star_gr, _ = quad(lambda z: 1.0 / H_gr_interp(z), 0, z_star, limit=500)
        except Exception:
            return np.nan

        if D_star_gr <= 0:
            return np.nan

        ratio = D_star_hm / D_star_gr
        return abs(ratio - 1.0) * 100.0


def create_dynamical_eos_model(
    delta_w: float = -0.2,
    a_w: float = 0.3,
    m_eos: float = 2.0,
    lambda_hor: float = 0.2,
    tau_hor: float = 0.1,
    **kwargs
) -> DynamicalEoSModifier:
    """Factory function to create a dynamical EoS modifier model.

    Args:
        delta_w: EoS shift amplitude (negative = more phantom at early times)
        a_w: Transition scale factor
        m_eos: Transition power
        lambda_hor: Amplitude of horizon-memory density
        tau_hor: Memory timescale
        **kwargs: Additional HorizonMemoryParameters fields

    Returns:
        DynamicalEoSModifier model instance
    """
    params = HorizonMemoryParameters(
        refinement_type=RefinementType.DYNAMICAL_EOS,
        delta_w=delta_w,
        a_w=a_w,
        m_eos=m_eos,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        **kwargs
    )
    return DynamicalEoSModifier(params)


def scan_dynamical_eos_parameters(
    delta_w_range: Tuple[float, float] = (-0.5, 0.1),
    a_w_range: Tuple[float, float] = (0.1, 0.5),
    n_delta_w: int = 20,
    n_a_w: int = 20,
    m_eos: float = 2.0,
    lambda_hor: float = 0.2,
    tau_hor: float = 0.1,
    z_max: float = 1200.0,
) -> dict:
    """Perform 2D parameter scan for dynamical EoS modifier.

    Args:
        delta_w_range: (min, max) for delta_w
        a_w_range: (min, max) for a_w
        n_delta_w: Number of delta_w grid points
        n_a_w: Number of a_w grid points
        m_eos: Fixed transition power
        lambda_hor: Fixed lambda_hor value
        tau_hor: Fixed tau_hor value
        z_max: Maximum redshift for integration

    Returns:
        Dictionary with scan results
    """
    delta_w_vals = np.linspace(delta_w_range[0], delta_w_range[1], n_delta_w)
    a_w_vals = np.linspace(a_w_range[0], a_w_range[1], n_a_w)

    # Result arrays
    delta_H0 = np.zeros((n_delta_w, n_a_w))
    cmb_dev = np.zeros((n_delta_w, n_a_w))
    success_mask = np.zeros((n_delta_w, n_a_w), dtype=bool)

    for i, delta_w in enumerate(delta_w_vals):
        for j, a_w in enumerate(a_w_vals):
            try:
                model = create_dynamical_eos_model(
                    delta_w=delta_w,
                    a_w=a_w,
                    m_eos=m_eos,
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
        "refinement": "T06D",
        "delta_w_vals": delta_w_vals,
        "a_w_vals": a_w_vals,
        "m_eos": m_eos,
        "lambda_hor": lambda_hor,
        "tau_hor": tau_hor,
        "delta_H0_percent": delta_H0,
        "cmb_deviation_percent": cmb_dev,
        "success_mask": success_mask,
    }
