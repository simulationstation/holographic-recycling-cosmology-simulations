"""Background cosmology for HRC 2.0.

This module implements the background evolution for general scalar-tensor
cosmology with action:
    S = integral d^4x sqrt(-g) [F(phi)R/2 - Z(phi)(dphi)^2/2 - V(phi)] + S_matter

The evolution equations are:
1. Friedmann equation:
    3*F*H^2 = rho_m + rho_r + rho_phi_eff

2. Raychaudhuri equation:
    2*F*H_dot = -rho_m - (4/3)*rho_r - Z*phi_dot^2 + 2*H*F'*phi_dot + F''*phi_dot^2

3. Scalar field equation:
    Z*(phi_ddot + 3*H*phi_dot) + dV/dphi - 3*F'*(H_dot + 2*H^2) = 0

We evolve in redshift z with dy/dz = -(1+z)^{-1} * H^{-1} * dy/dt
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .theory import ScalarTensorModel, HRC2Parameters, create_model, M_PL_SQUARED
from .effective_gravity import compute_Geff_ratio, is_Geff_valid
from .scalar_field import (
    compute_scalar_energy_density,
    compute_effective_scalar_density,
    compute_effective_eos,
)


# Safety margins
F_SAFETY_MARGIN = 0.05  # Minimum F/M_pl^2
GEFF_MAX_RATIO = 20.0   # Maximum G_eff/G_N


@dataclass
class BackgroundSolution:
    """Solution of background cosmological evolution.

    Attributes:
        z: Redshift array
        H: Hubble parameter H(z) in units of H0
        phi: Scalar field phi(z) in M_pl units
        phi_dot: Field velocity dphi/dt(z)
        G_eff_ratio: G_eff(z)/G_N
        rho_m: Matter density (normalized to critical)
        rho_r: Radiation density
        rho_phi: Scalar field density
        w_phi: Scalar field equation of state

        success: Whether integration succeeded
        geff_valid: Whether G_eff stayed in valid range
        geff_divergence_z: Redshift where G_eff diverged (if any)
        stability_valid: Whether stability conditions held
        message: Status message
    """
    z: NDArray[np.floating]
    H: NDArray[np.floating]
    phi: NDArray[np.floating]
    phi_dot: NDArray[np.floating]
    G_eff_ratio: NDArray[np.floating]
    rho_m: NDArray[np.floating]
    rho_r: NDArray[np.floating]
    rho_phi: NDArray[np.floating]
    w_phi: NDArray[np.floating]

    success: bool = True
    geff_valid: bool = True
    geff_divergence_z: Optional[float] = None
    stability_valid: bool = True
    message: str = "Success"

    def G_eff_at(self, z_target: float) -> float:
        """Interpolate G_eff/G_N at given redshift."""
        if z_target < self.z.min() or z_target > self.z.max():
            return np.nan
        return np.interp(z_target, self.z, self.G_eff_ratio)

    def phi_at(self, z_target: float) -> float:
        """Interpolate phi at given redshift."""
        if z_target < self.z.min() or z_target > self.z.max():
            return np.nan
        return np.interp(z_target, self.z, self.phi)

    def H_at(self, z_target: float) -> float:
        """Interpolate H at given redshift."""
        if z_target < self.z.min() or z_target > self.z.max():
            return np.nan
        return np.interp(z_target, self.z, self.H)

    @property
    def delta_G_over_G(self) -> float:
        """Compute (G_eff(z=0) - G_eff(z_max)) / G_eff(z_max)."""
        if not self.geff_valid:
            return np.nan
        G0 = self.G_eff_ratio[0]
        Gmax = self.G_eff_ratio[-1]
        if Gmax == 0:
            return np.nan
        return (G0 - Gmax) / Gmax


class BackgroundCosmology:
    """Solver for background cosmological evolution in HRC 2.0.

    Integrates the Friedmann + scalar field equations from z=0 to z=z_max.
    """

    def __init__(
        self,
        params: HRC2Parameters,
        model: Optional[ScalarTensorModel] = None,
    ):
        """Initialize background solver.

        Args:
            params: HRC2Parameters configuration
            model: Optional pre-created model (will create from params if None)
        """
        self.params = params
        self.model = model if model is not None else create_model(params)

        # Cosmological parameters
        self.Omega_m0 = params.Omega_m0
        self.Omega_r0 = params.Omega_r0
        self.H0 = params.H0

        # Initial conditions at z=0
        self.phi_0 = params.phi_0
        self.phi_dot_0 = params.phi_dot_0

    def solve(
        self,
        z_max: float = 1100.0,
        z_points: int = 500,
        method: str = 'RK45',
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> BackgroundSolution:
        """Integrate background equations from z=0 to z_max.

        Args:
            z_max: Maximum redshift
            z_points: Number of output points
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            BackgroundSolution with evolution history
        """
        # Output redshift array
        z_eval = np.linspace(0, z_max, z_points)

        # Initial state: [phi, phi_dot_z, H_normalized]
        # phi_dot_z = dphi/dz = -(1+z)^{-1} * H^{-1} * dphi/dt
        # We'll track phi and dphi/dz directly

        # Compute initial H from Friedmann
        H_init = self._compute_H_from_friedmann(0, self.phi_0, self.phi_dot_0)
        if H_init is None or H_init <= 0:
            return self._invalid_solution(z_eval, "Invalid initial Hubble parameter")

        # Convert phi_dot (d/dt) to phi_prime (d/dz)
        # dphi/dz = dphi/dt * dt/dz = phi_dot / (-(1+z)*H)
        phi_prime_0 = -self.phi_dot_0 / H_init if H_init != 0 else 0

        y0 = np.array([self.phi_0, phi_prime_0])

        # Track validity
        self._geff_valid = True
        self._stability_valid = True
        self._divergence_z = None

        # Solve ODE
        try:
            sol = solve_ivp(
                self._ode_system,
                (0, z_max),
                y0,
                method=method,
                t_eval=z_eval,
                rtol=rtol,
                atol=atol,
                events=[self._geff_divergence_event],
            )

            if not sol.success:
                # Check if stopped due to event
                if sol.t_events[0].size > 0:
                    self._divergence_z = sol.t_events[0][0]
                    return self._partial_solution(sol, z_eval, "G_eff divergence")
                return self._invalid_solution(z_eval, f"Integration failed: {sol.message}")

        except Exception as e:
            return self._invalid_solution(z_eval, f"Integration error: {str(e)}")

        # Extract solution
        z = sol.t
        phi = sol.y[0]
        phi_prime = sol.y[1]  # dphi/dz

        # Reconstruct other quantities
        H = np.zeros_like(z)
        phi_dot = np.zeros_like(z)
        G_eff_ratio = np.zeros_like(z)
        rho_m = np.zeros_like(z)
        rho_r = np.zeros_like(z)
        rho_phi = np.zeros_like(z)
        w_phi = np.zeros_like(z)

        for i, (zi, phii, phi_prime_i) in enumerate(zip(z, phi, phi_prime)):
            # Matter and radiation densities (in units of 3*H0^2)
            rho_m[i] = self.Omega_m0 * (1 + zi)**3
            rho_r[i] = self.Omega_r0 * (1 + zi)**4

            # Compute H from constraint
            Hi = self._compute_H_from_friedmann_with_phi_prime(zi, phii, phi_prime_i)
            if Hi is None or Hi <= 0:
                Hi = 1.0  # Fallback

            H[i] = Hi

            # Convert phi_prime to phi_dot
            phi_dot[i] = -phi_prime_i * (1 + zi) * Hi

            # G_eff ratio
            G_eff_ratio[i] = compute_Geff_ratio(phii, self.model)

            # Scalar field density and EOS
            rho_phi[i] = compute_effective_scalar_density(phii, phi_dot[i], Hi, self.model)
            w_phi[i] = compute_effective_eos(phii, phi_dot[i], self.model)

        # Check G_eff validity
        geff_valid = np.all(np.isfinite(G_eff_ratio)) and np.all(G_eff_ratio > F_SAFETY_MARGIN)

        return BackgroundSolution(
            z=z,
            H=H,
            phi=phi,
            phi_dot=phi_dot,
            G_eff_ratio=G_eff_ratio,
            rho_m=rho_m,
            rho_r=rho_r,
            rho_phi=rho_phi,
            w_phi=w_phi,
            success=True,
            geff_valid=geff_valid and self._geff_valid,
            geff_divergence_z=self._divergence_z,
            stability_valid=self._stability_valid,
            message="Success",
        )

    def _ode_system(self, z: float, y: NDArray) -> NDArray:
        """ODE system for background evolution.

        State variables: y = [phi, phi_prime]
        where phi_prime = dphi/dz

        Evolution equations in redshift:
            dphi/dz = phi_prime
            d(phi_prime)/dz = ... (from scalar field equation)

        Args:
            z: Redshift
            y: State vector [phi, phi_prime]

        Returns:
            dy/dz
        """
        phi, phi_prime = y

        # Check validity
        if not self.model.is_valid(phi, F_SAFETY_MARGIN):
            self._geff_valid = False
            return np.array([0.0, 0.0])

        # Matter and radiation densities
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4

        # Compute H from Friedmann constraint
        H = self._compute_H_from_friedmann_with_phi_prime(z, phi, phi_prime)
        if H is None or H <= 0:
            self._geff_valid = False
            return np.array([0.0, 0.0])

        # Convert phi_prime to phi_dot
        # phi_dot = dphi/dt = -phi_prime * (1+z) * H
        phi_dot = -phi_prime * (1 + z) * H

        # Model functions
        F = self.model.F(phi)
        dF = self.model.dF_dphi(phi)
        d2F = self.model.d2F_dphi2(phi)
        Z = self.model.Z(phi)
        dZ = self.model.dZ_dphi(phi)
        dV = self.model.dV_dphi(phi)

        if F <= 0 or Z <= 0:
            self._geff_valid = False
            return np.array([0.0, 0.0])

        # Compute H_dot from Raychaudhuri equation:
        # 2*F*H_dot = -rho_m - (4/3)*rho_r - Z*phi_dot^2 + 2*H*dF*phi_dot + d2F*phi_dot^2
        H_dot_source = (-rho_m - (4.0/3.0) * rho_r
                       - Z * phi_dot**2
                       + 2.0 * H * dF * phi_dot
                       + d2F * phi_dot**2)
        H_dot = H_dot_source / (2.0 * F)

        # Scalar field equation (for phi_ddot):
        # Z*(phi_ddot + 3*H*phi_dot) + Z'*phi_dot^2/2 + dV - 3*dF*(H_dot + 2*H^2) = 0
        # Note: including Z' term for completeness

        source = 3.0 * dF * (H_dot + 2.0 * H**2)
        friction = 3.0 * H * phi_dot * Z
        kinetic_Z = 0.5 * dZ * phi_dot**2  # Often zero for canonical

        phi_ddot = (source - friction - dV - kinetic_Z) / Z

        # Convert phi_ddot to d(phi_prime)/dz
        # phi_prime = -phi_dot / ((1+z)*H)
        # d(phi_prime)/dz = d/dz[-phi_dot / ((1+z)*H)]
        #
        # Using chain rule:
        # d(phi_prime)/dz = -[phi_ddot/(1+z)/H - phi_dot/(1+z)^2/H - phi_dot*H'/((1+z)*H^2)] * dz/dz
        # where H' = dH/dz = -H_dot / ((1+z)*H)

        dH_dz = -H_dot / ((1 + z) * H) if H != 0 else 0

        d_phi_prime_dz = (-phi_ddot / ((1 + z) * H)
                         + phi_dot / ((1 + z)**2 * H)
                         + phi_dot * dH_dz / ((1 + z) * H))

        return np.array([phi_prime, d_phi_prime_dz])

    def _compute_H_from_friedmann(
        self,
        z: float,
        phi: float,
        phi_dot: float,
    ) -> Optional[float]:
        """Compute H from Friedmann equation.

        3*F*H^2 = rho_m + rho_r + rho_phi_eff

        This requires solving iteratively since rho_phi_eff depends on H.
        """
        F = self.model.F(phi)
        if F <= 0:
            return None

        # Matter and radiation
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4

        # Standard scalar contribution (without H-dependent part)
        rho_phi_std = compute_scalar_energy_density(phi, phi_dot, self.model)

        # F' contribution: -3*H*F'*phi_dot
        # Full equation: 3*F*H^2 = rho_m + rho_r + rho_phi_std - 3*H*F'*phi_dot
        # Rewrite: 3*F*H^2 + 3*F'*phi_dot*H = rho_m + rho_r + rho_phi_std

        dF = self.model.dF_dphi(phi)

        # This is quadratic in H: 3*F*H^2 + 3*dF*phi_dot*H - rho_total = 0
        # H = (-3*dF*phi_dot + sqrt(9*dF^2*phi_dot^2 + 12*F*rho_total)) / (6*F)

        rho_total = rho_m + rho_r + rho_phi_std

        a_coef = 3.0 * F
        b_coef = 3.0 * dF * phi_dot
        c_coef = -rho_total

        discriminant = b_coef**2 - 4 * a_coef * c_coef

        if discriminant < 0:
            return None

        H = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

        return H if H > 0 else None

    def _compute_H_from_friedmann_with_phi_prime(
        self,
        z: float,
        phi: float,
        phi_prime: float,
    ) -> Optional[float]:
        """Compute H from Friedmann equation using phi_prime = dphi/dz.

        Since phi_dot = -phi_prime * (1+z) * H, the equation becomes:
        3*F*H^2 = rho_m + rho_r + Z*phi_prime^2*(1+z)^2*H^2/2 + V + 3*H*dF*phi_prime*(1+z)*H
        """
        F = self.model.F(phi)
        if F <= 0:
            return None

        Z = self.model.Z(phi)
        V = self.model.V(phi)
        dF = self.model.dF_dphi(phi)

        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4

        # Rewrite equation collecting H^2 terms:
        # 3*F*H^2 - Z*phi_prime^2*(1+z)^2*H^2/2 - 3*dF*phi_prime*(1+z)*H^2 = rho_m + rho_r + V
        #
        # Factor: H^2 * [3*F - Z*phi_prime^2*(1+z)^2/2 - 3*dF*phi_prime*(1+z)] = rho_m + rho_r + V

        factor = (3.0 * F
                  - 0.5 * Z * phi_prime**2 * (1 + z)**2
                  - 3.0 * dF * phi_prime * (1 + z))

        rho_total = rho_m + rho_r + V

        if factor <= 0 or rho_total < 0:
            return None

        H_squared = rho_total / factor

        return np.sqrt(H_squared) if H_squared > 0 else None

    def _geff_divergence_event(self, z: float, y: NDArray) -> float:
        """Event function for G_eff divergence detection.

        Returns negative when G_eff exceeds threshold.
        """
        phi = y[0]
        F = self.model.F(phi)

        # Return positive when valid, negative when invalid
        # F > safety_margin * M_pl^2 should be maintained
        return F - F_SAFETY_MARGIN * M_PL_SQUARED

    _geff_divergence_event.terminal = True
    _geff_divergence_event.direction = -1

    def _invalid_solution(
        self,
        z_eval: NDArray,
        message: str,
    ) -> BackgroundSolution:
        """Create an invalid solution placeholder."""
        n = len(z_eval)
        return BackgroundSolution(
            z=z_eval,
            H=np.full(n, np.nan),
            phi=np.full(n, np.nan),
            phi_dot=np.full(n, np.nan),
            G_eff_ratio=np.full(n, np.nan),
            rho_m=np.full(n, np.nan),
            rho_r=np.full(n, np.nan),
            rho_phi=np.full(n, np.nan),
            w_phi=np.full(n, np.nan),
            success=False,
            geff_valid=False,
            geff_divergence_z=self._divergence_z,
            stability_valid=False,
            message=message,
        )

    def _partial_solution(
        self,
        sol,
        z_eval: NDArray,
        message: str,
    ) -> BackgroundSolution:
        """Create partial solution up to divergence point."""
        # Use only the computed portion
        z = sol.t
        phi = sol.y[0]
        phi_prime = sol.y[1]

        n = len(z)
        H = np.zeros(n)
        phi_dot = np.zeros(n)
        G_eff_ratio = np.zeros(n)
        rho_m = np.zeros(n)
        rho_r = np.zeros(n)
        rho_phi = np.zeros(n)
        w_phi = np.zeros(n)

        for i, (zi, phii, phi_prime_i) in enumerate(zip(z, phi, phi_prime)):
            rho_m[i] = self.Omega_m0 * (1 + zi)**3
            rho_r[i] = self.Omega_r0 * (1 + zi)**4

            Hi = self._compute_H_from_friedmann_with_phi_prime(zi, phii, phi_prime_i)
            if Hi is None or Hi <= 0:
                Hi = 1.0
            H[i] = Hi

            phi_dot[i] = -phi_prime_i * (1 + zi) * Hi
            G_eff_ratio[i] = compute_Geff_ratio(phii, self.model)
            rho_phi[i] = compute_effective_scalar_density(phii, phi_dot[i], Hi, self.model)
            w_phi[i] = compute_effective_eos(phii, phi_dot[i], self.model)

        return BackgroundSolution(
            z=z,
            H=H,
            phi=phi,
            phi_dot=phi_dot,
            G_eff_ratio=G_eff_ratio,
            rho_m=rho_m,
            rho_r=rho_r,
            rho_phi=rho_phi,
            w_phi=w_phi,
            success=False,
            geff_valid=False,
            geff_divergence_z=self._divergence_z,
            stability_valid=False,
            message=message,
        )
