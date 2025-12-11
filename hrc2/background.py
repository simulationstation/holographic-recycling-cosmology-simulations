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
import time
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


def apply_recycling_correction(H_base: float, alpha_rec: float) -> float:
    """Apply thermal horizon recycling correction to Hubble parameter.

    The recycling hypothesis posits that information at the cosmic horizon
    is recycled back into the universe, effectively reducing the Hubble
    friction. This modifies:
        H^2 = H_base^2 / (1 - alpha_rec)

    Args:
        H_base: Base Hubble parameter from standard Friedmann equation
        alpha_rec: Recycling fraction, must satisfy 0 <= alpha_rec < 1

    Returns:
        Corrected Hubble parameter H

    Raises:
        ValueError: if alpha_rec >= 1 (would cause divergence)
    """
    if alpha_rec >= 1.0:
        raise ValueError(f"alpha_rec must be < 1, got {alpha_rec}")
    if alpha_rec <= 0.0:
        return H_base
    # H^2 = H_base^2 / (1 - alpha_rec)
    # H = H_base / sqrt(1 - alpha_rec)
    return H_base / np.sqrt(1.0 - alpha_rec)


def horizon_potential_force(phi: float, H: float, gamma_rec: float) -> float:
    """Compute horizon-driven effective potential force term.

    The horizon-driven effective potential adds an H-dependent mass term:
        V_hor(phi, H) = gamma_rec * H^4 * (1/2) * phi^2

    The force (derivative) is:
        dV_hor/dphi = gamma_rec * H^4 * phi

    This term couples the scalar field evolution to the Hubble rate,
    providing an additional restoring force that depends on cosmic expansion.

    Args:
        phi: Scalar field value
        H: Hubble parameter
        gamma_rec: Horizon-driven potential strength (dimensionless)

    Returns:
        Force contribution gamma_rec * H^4 * phi
    """
    if gamma_rec == 0.0:
        return 0.0
    return gamma_rec * (H**4) * phi


def compute_rho_EDE(
    a: float,
    f_EDE: float,
    z_c: float,
    sigma_ln_a: float,
    Omega_m0: float,
    Omega_r0: float,
) -> float:
    """Compute phenomenological Early Dark Energy (EDE) density.

    This implements a simple 'bump' in ln(a) centered at a_c = 1/(1+z_c),
    with width sigma_ln_a, and rough normalization using f_EDE.

    This is NOT a full EDE model, just a toy fluid to explore how
    a few-percent early contribution behaves.

    The EDE density is:
        rho_EDE(a) = amp * (a/a_c)^(-3(1+w_early)) * exp(-(ln(a/a_c)/sigma)^2)

    where amp is chosen so that rho_EDE(a_c) ~ f_EDE * rho_tot(a_c).

    Args:
        a: Scale factor
        f_EDE: Peak fractional contribution at z_c
        z_c: Characteristic EDE peak redshift
        sigma_ln_a: Width of the EDE bump in ln(a)
        Omega_m0: Present matter density parameter
        Omega_r0: Present radiation density parameter

    Returns:
        EDE energy density (in units of 3*H0^2, same as Omega_i*(1+z)^n)
    """
    if f_EDE <= 0.0:
        return 0.0

    a_c = 1.0 / (1.0 + z_c)

    # Rough early-time equation of state (stiff-ish / scalar-like)
    w_early = 1.0

    # Bump in ln(a)
    x = np.log(a) - np.log(a_c)
    bump = np.exp(-(x / sigma_ln_a)**2)

    # Very rough normalization:
    # set rho_EDE(a_c) ~ f_EDE * rho_tot_approx(a_c),
    # where rho_tot_approx(a_c) ~ Omega_m * a_c^-3 + Omega_r * a_c^-4
    # (ignoring Lambda and EDE itself for this first toy)
    rho_m_ac = Omega_m0 * a_c**(-3.0)
    rho_r_ac = Omega_r0 * a_c**(-4.0)
    rho_tot_ac_approx = rho_m_ac + rho_r_ac

    # Amplitude so that rho_EDE(a_c) ~ f_EDE * rho_tot_ac_approx
    # (since bump(a_c) ~ 1 and (a_c/a_c)^(-3(1+w)) = 1)
    amp = f_EDE * rho_tot_ac_approx

    # Redshift scaling relative to a_c with w_early
    scaling = (a / a_c)**(-3.0 * (1.0 + w_early))

    return amp * scaling * bump


def apply_epsilon_corr(
    H_base: float,
    z: float,
    epsilon_corr: float,
    z_transition: float = 3000.0,
    transition_width: float = 500.0,
) -> float:
    """Apply early-time epsilon correction to Hubble parameter.

    This correction models deviations from standard LCDM at high redshift,
    potentially from primordial physics (e.g., no-boundary initial conditions).

    The correction is:
        H(z) = H_base * (1 + epsilon_corr * f(z))

    where f(z) is a smooth transition function:
        f(z) = 0 for z << z_transition
        f(z) -> 1 for z >> z_transition

    Using a tanh transition:
        f(z) = 0.5 * (1 + tanh((z - z_transition) / transition_width))

    Args:
        H_base: Base Hubble parameter from standard Friedmann equation
        z: Redshift
        epsilon_corr: Fractional correction (can be positive or negative)
        z_transition: Redshift where correction turns on (default: 3000)
        transition_width: Width of the transition (default: 500)

    Returns:
        Corrected Hubble parameter H(z)

    Notes:
        - epsilon_corr > 0: H(z) larger at high-z, smaller sound horizon, higher inferred H0
        - epsilon_corr < 0: H(z) smaller at high-z, larger sound horizon, lower inferred H0
        - Typical values: |epsilon_corr| < 0.05 to remain consistent with CMB
    """
    if abs(epsilon_corr) < 1e-12:
        return H_base

    # Smooth transition function using tanh
    f_z = 0.5 * (1.0 + np.tanh((z - z_transition) / transition_width))

    # Apply correction
    return H_base * (1.0 + epsilon_corr * f_z)


def compute_sound_horizon_epsilon_effect(
    epsilon_corr: float,
    z_transition: float = 3000.0,
    z_drag: float = 1059.94,
) -> float:
    """Estimate fractional change in sound horizon due to epsilon correction.

    The sound horizon integral is r_s = integral c_s/H dz from z_drag to infinity.
    If H is increased by a fraction epsilon at high z, r_s decreases roughly as:
        delta_r_s / r_s ~ -epsilon_corr * (effective fraction of integral at z > z_transition)

    This is a rough estimate; the actual effect depends on the full integral.

    Args:
        epsilon_corr: Fractional H(z) correction
        z_transition: Transition redshift
        z_drag: Drag epoch redshift

    Returns:
        Approximate fractional change in sound horizon
    """
    # Rough estimate: about 20% of the sound horizon integral comes from z > 3000
    # (the exact fraction depends on cosmological parameters)
    high_z_fraction = 0.20

    if z_transition > z_drag * 2:
        # If transition is well above drag epoch, most effect is on high-z tail
        effective_fraction = high_z_fraction * 0.5
    else:
        effective_fraction = high_z_fraction

    # delta_r_s / r_s ~ -epsilon * effective_fraction
    # (negative because larger H means smaller r_s)
    return -epsilon_corr * effective_fraction


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

        # Thermal horizon recycling
        self.alpha_rec = params.alpha_rec

        # Horizon-driven effective potential
        self.gamma_rec = params.gamma_rec

        # Early Dark Energy (EDE) fluid parameters
        self.f_EDE = params.f_EDE
        self.z_c = params.z_c
        self.sigma_ln_a = params.sigma_ln_a

        # Nonlocal horizon-memory parameters
        self.lambda_hor = params.lambda_hor
        self.tau_hor = params.tau_hor

        # Critical density at z=0 (in units of 3*H0^2, so rho_crit0 = 1)
        self.rho_crit0 = 1.0

        # Self-consistent horizon-memory Lambda tracking
        # Omega_L0_base is the "baseline" Lambda = 1 - Omega_m0 - Omega_r0
        # Omega_L0_eff is the effective Lambda when horizon-memory is on
        self.Omega_L0_base = 1.0 - self.Omega_m0 - self.Omega_r0
        self.Omega_hor0 = 0.0  # Will be set by set_M_today()
        self.Omega_L0_eff = self.Omega_L0_base  # Will be updated by set_M_today()
        self._M_today = None  # Cached M_today value

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
        timeout: Optional[float] = None,
    ) -> BackgroundSolution:
        """Integrate background equations from z=0 to z_max.

        Args:
            z_max: Maximum redshift
            z_points: Number of output points
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            timeout: Maximum wall-clock time in seconds (None = no limit)

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

        # Timeout tracking
        self._start_time = time.time()
        self._timeout = timeout
        self._rhs_call_count = 0

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

        # Timeout check (every 100 RHS calls to avoid overhead)
        self._rhs_call_count += 1
        if self._timeout is not None and self._rhs_call_count % 100 == 0:
            elapsed = time.time() - self._start_time
            if elapsed > self._timeout:
                raise RuntimeError(f"Integration timeout after {elapsed:.1f}s")

        # Early exit: F(phi) below safety threshold
        F = self.model.F(phi)
        if F <= F_SAFETY_MARGIN * M_PL_SQUARED:
            self._geff_valid = False
            raise RuntimeError("F(phi) below safety threshold")

        # Early exit: G_eff out of safe range
        G_eff = compute_Geff_ratio(phi, self.model)
        if not (0.05 < G_eff < GEFF_MAX_RATIO):
            self._geff_valid = False
            raise RuntimeError("G_eff out of safe range")

        # Check validity
        if not self.model.is_valid(phi, F_SAFETY_MARGIN):
            self._geff_valid = False
            raise RuntimeError("Model validity check failed")

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
        # Z*(phi_ddot + 3*H*phi_dot) + Z'*phi_dot^2/2 + dV + dV_hor - 3*dF*(H_dot + 2*H^2) = 0
        # Note: including Z' term for completeness
        # dV_hor = gamma_rec * H^4 * phi is the horizon-driven effective potential force

        source = 3.0 * dF * (H_dot + 2.0 * H**2)
        friction = 3.0 * H * phi_dot * Z
        kinetic_Z = 0.5 * dZ * phi_dot**2  # Often zero for canonical

        # Horizon-driven effective potential contribution
        hor_force = horizon_potential_force(phi, H, self.gamma_rec)

        phi_ddot = (source - friction - dV - kinetic_Z - hor_force) / Z

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

        # EDE contribution
        a = 1.0 / (1.0 + z)
        rho_ede = compute_rho_EDE(
            a, self.f_EDE, self.z_c, self.sigma_ln_a,
            self.Omega_m0, self.Omega_r0
        )

        # Standard scalar contribution (without H-dependent part)
        rho_phi_std = compute_scalar_energy_density(phi, phi_dot, self.model)

        # F' contribution: -3*H*F'*phi_dot
        # Full equation: 3*F*H^2 = rho_m + rho_r + rho_phi_std + rho_ede - 3*H*F'*phi_dot
        # Rewrite: 3*F*H^2 + 3*F'*phi_dot*H = rho_m + rho_r + rho_phi_std + rho_ede

        dF = self.model.dF_dphi(phi)

        # This is quadratic in H: 3*F*H^2 + 3*dF*phi_dot*H - rho_total = 0
        # H = (-3*dF*phi_dot + sqrt(9*dF^2*phi_dot^2 + 12*F*rho_total)) / (6*F)

        rho_total = rho_m + rho_r + rho_phi_std + rho_ede

        a_coef = 3.0 * F
        b_coef = 3.0 * dF * phi_dot
        c_coef = -rho_total

        discriminant = b_coef**2 - 4 * a_coef * c_coef

        if discriminant < 0:
            return None

        H = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

        if H <= 0:
            return None

        # Apply thermal horizon recycling correction
        return apply_recycling_correction(H, self.alpha_rec)

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

        # EDE contribution
        a = 1.0 / (1.0 + z)
        rho_ede = compute_rho_EDE(
            a, self.f_EDE, self.z_c, self.sigma_ln_a,
            self.Omega_m0, self.Omega_r0
        )

        # Rewrite equation collecting H^2 terms:
        # 3*F*H^2 - Z*phi_prime^2*(1+z)^2*H^2/2 - 3*dF*phi_prime*(1+z)*H^2 = rho_m + rho_r + V + rho_ede
        #
        # Factor: H^2 * [3*F - Z*phi_prime^2*(1+z)^2/2 - 3*dF*phi_prime*(1+z)] = rho_m + rho_r + V + rho_ede

        factor = (3.0 * F
                  - 0.5 * Z * phi_prime**2 * (1 + z)**2
                  - 3.0 * dF * phi_prime * (1 + z))

        rho_total = rho_m + rho_r + V + rho_ede

        if factor <= 0 or rho_total < 0:
            return None

        H_squared = rho_total / factor

        if H_squared <= 0:
            return None

        H = np.sqrt(H_squared)

        # Apply thermal horizon recycling correction
        return apply_recycling_correction(H, self.alpha_rec)

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

    # =========================================================================
    # Helper methods for EDE analysis
    # =========================================================================

    def radiation_density(self, a: float) -> float:
        """Compute radiation density at scale factor a.

        Args:
            a: Scale factor

        Returns:
            rho_r in units of 3*H0^2
        """
        z = 1.0 / a - 1.0
        return self.Omega_r0 * (1 + z)**4

    def matter_density(self, a: float) -> float:
        """Compute matter density at scale factor a.

        Args:
            a: Scale factor

        Returns:
            rho_m in units of 3*H0^2
        """
        z = 1.0 / a - 1.0
        return self.Omega_m0 * (1 + z)**3

    def rho_EDE_component(self, a: float) -> float:
        """Compute EDE density at scale factor a.

        Args:
            a: Scale factor

        Returns:
            rho_EDE in units of 3*H0^2
        """
        return compute_rho_EDE(
            a, self.f_EDE, self.z_c, self.sigma_ln_a,
            self.Omega_m0, self.Omega_r0
        )

    def total_density(self, a: float) -> float:
        """Compute total energy density at scale factor a (pure GR case).

        This is a simplified calculation for pure GR (xi=0, phi0=0)
        used for EDE analysis. It includes matter, radiation, Lambda, and EDE.

        Args:
            a: Scale factor

        Returns:
            rho_tot in units of 3*H0^2
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4

        # Cosmological constant (Omega_Lambda ~ 1 - Omega_m - Omega_r)
        Omega_L = 1.0 - self.Omega_m0 - self.Omega_r0
        rho_L = Omega_L

        # EDE component
        rho_ede = self.rho_EDE_component(a)

        return rho_m + rho_r + rho_L + rho_ede

    # =========================================================================
    # Helper methods for horizon-memory analysis
    # =========================================================================

    def S_norm(self, H: float) -> float:
        """Compute normalized horizon entropy proxy.

        S_norm = (H0 / H)^2

        At present (a=1, H=H0), S_norm = 1.
        At early times when H >> H0, S_norm << 1.

        Args:
            H: Hubble parameter

        Returns:
            Normalized horizon entropy proxy
        """
        if H <= 0.0:
            return 0.0
        return (self.H0 / H) ** 2

    def memory_derivative(self, ln_a: float, M: float, H: float) -> float:
        """Compute derivative of memory field M with respect to ln(a).

        dM/d(ln a) = (S_norm(a) - M) / tau_hor

        This makes M relax toward S_norm with timescale tau_hor in ln(a).

        Args:
            ln_a: Natural log of scale factor
            M: Current memory value
            H: Current Hubble parameter

        Returns:
            dM/d(ln a)
        """
        if self.lambda_hor == 0.0:
            return 0.0
        S_n = self.S_norm(H)
        return (S_n - M) / self.tau_hor

    def rho_horizon_memory(self, M: float) -> float:
        """Compute horizon-memory effective energy density.

        rho_hor(a) = lambda_hor * M(a) * rho_crit0

        With M starting at 0 and relaxing toward S_norm = (H0/H)^2,
        this gives positive rho_hor that grows as M catches up with S_norm.

        Args:
            M: Memory field value

        Returns:
            Horizon-memory energy density in units of 3*H0^2
        """
        if self.lambda_hor == 0.0:
            return 0.0
        return self.lambda_hor * M * self.rho_crit0

    def set_M_today(self, M_today: float) -> None:
        """Set the memory field value at z=0 and compute self-consistent Lambda.

        This enforces flatness at z=0 by having horizon-memory replace part of Lambda:
            Omega_hor0 = lambda_hor * M_today
            Omega_L0_eff = 1 - Omega_m0 - Omega_r0 - Omega_hor0 = Omega_L0_base - Omega_hor0

        After calling this, H_of_a_selfconsistent() will use the effective Lambda.

        Args:
            M_today: Memory field value at z=0
        """
        self._M_today = M_today
        self.Omega_hor0 = self.lambda_hor * M_today
        self.Omega_L0_eff = self.Omega_L0_base - self.Omega_hor0

        # Warn if effective Lambda is negative (unphysical)
        if self.Omega_L0_eff < 0:
            import warnings
            warnings.warn(
                f"Effective Lambda is negative: Omega_L0_eff = {self.Omega_L0_eff:.4f}. "
                f"This indicates Omega_hor0 = {self.Omega_hor0:.4f} > Omega_L0_base = {self.Omega_L0_base:.4f}."
            )

    def H_of_a_selfconsistent(self, a: float, M: float) -> float:
        """Compute Hubble parameter with self-consistent Lambda + horizon-memory.

        This uses the effective Lambda (reduced by Omega_hor0) so that at z=0:
            Omega_m0 + Omega_r0 + Omega_L0_eff + Omega_hor0 = 1

        At arbitrary z, the evolution is:
            H^2/H0^2 = Omega_m0*(1+z)^3 + Omega_r0*(1+z)^4 + Omega_L0_eff + lambda_hor*M

        Args:
            a: Scale factor
            M: Memory field value at this scale factor

        Returns:
            H in units of H0
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        rho_L = self.Omega_L0_eff  # Use effective Lambda
        rho_hor = self.lambda_hor * M  # Horizon-memory contribution

        H_squared = rho_m + rho_r + rho_L + rho_hor
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def H_of_a_gr_baseline(self, a: float) -> float:
        """Compute Hubble parameter at scale factor a using baseline Lambda.

        This is the pure GR reference with Omega_L = Omega_L0_base (no horizon-memory).
        Used for comparing H(z) with and without horizon-memory.

        Args:
            a: Scale factor

        Returns:
            H in units of H0
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        rho_L = self.Omega_L0_base

        H_squared = rho_m + rho_r + rho_L
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def H_of_a_gr(self, a: float) -> float:
        """Compute Hubble parameter at scale factor a assuming pure GR.

        This is a simplified calculation for pure GR (no scalar field)
        used for horizon-memory analysis.

        Args:
            a: Scale factor

        Returns:
            H in units of H0
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        Omega_L = 1.0 - self.Omega_m0 - self.Omega_r0
        rho_L = Omega_L

        # H^2 / H0^2 = rho_tot / rho_crit0 = rho_m + rho_r + rho_L
        H_squared = rho_m + rho_r + rho_L
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def rho_horizon_memory_component(self, a: float) -> float:
        """Compute horizon-memory density at scale factor a.

        For a first approximation, we compute M(a) by approximating
        M ~ S_norm(a) (ignoring memory relaxation). This gives an
        order-of-magnitude estimate for how big rho_hor can be.

        A more accurate version would integrate the M ODE properly.

        Args:
            a: Scale factor

        Returns:
            rho_hor in units of 3*H0^2
        """
        if self.lambda_hor == 0.0:
            return 0.0

        H = self.H_of_a_gr(a)
        # Approximate M(a) ~ S_norm(a) (crude approximation ignoring true memory)
        M = self.S_norm(H)
        return self.rho_horizon_memory(M)

    def H_of_a(self, a: float, M: float) -> float:
        """Compute Hubble parameter including horizon-memory contribution.

        Friedmann equation with horizon memory:
            H^2 = H_GR^2 + (rho_hor / 3)
        where rho_hor = lambda_hor * M * rho_crit0

        In our units where rho_crit0 = 1 (units of 3*H0^2):
            H^2 / H0^2 = rho_m + rho_r + rho_L + lambda_hor * M

        Args:
            a: Scale factor
            M: Memory field value

        Returns:
            H in units of H0
        """
        z = 1.0 / a - 1.0
        rho_m = self.Omega_m0 * (1 + z)**3
        rho_r = self.Omega_r0 * (1 + z)**4
        Omega_L = 1.0 - self.Omega_m0 - self.Omega_r0
        rho_L = Omega_L

        # Add horizon-memory contribution
        rho_hor = self.rho_horizon_memory(M)

        # H^2 / H0^2 = rho_tot
        H_squared = rho_m + rho_r + rho_L + rho_hor
        if H_squared <= 0:
            return 0.0
        return np.sqrt(H_squared) * self.H0

    def delta_H0_proxy(self, M_today: float) -> float:
        """Estimate fractional shift in inferred H0 from horizon memory.

        If the memory field at z=0 is M_today, then there's an additional
        contribution to the energy density:
            rho_hor(z=0) = lambda_hor * M_today * rho_crit0

        This modifies H(z=0):
            H_new^2 = H_GR^2 + rho_hor
            H_new^2 / H0_GR^2 = 1 + lambda_hor * M_today

        So:
            H_new / H0_GR = sqrt(1 + lambda_hor * M_today)
            delta_H0 / H0 = sqrt(1 + lambda_hor * M_today) - 1

        For small corrections:
            delta_H0 / H0 ~ lambda_hor * M_today / 2

        Args:
            M_today: Memory field value at z=0

        Returns:
            Fractional shift (H_new - H0_GR) / H0_GR
        """
        if self.lambda_hor == 0.0:
            return 0.0

        # At z=0, GR gives H = H0 with rho_tot = 1 (in our units)
        # Adding rho_hor = lambda_hor * M_today gives:
        # H_new^2 / H0^2 = 1 + lambda_hor * M_today
        x = self.lambda_hor * M_today
        if x < -1.0:
            return -1.0  # Can't have negative H^2
        return np.sqrt(1.0 + x) - 1.0

    def compute_delta_H0(
        self,
        M_interp: callable,
        z_calibration: float = 0.5,
    ) -> dict:
        """Compute fractional H0 shift from actual H(z) comparison.

        This compares:
        1. H(z) with horizon-memory (self-consistent Lambda + rho_hor)
        2. H(z) with pure baseline GR (no horizon-memory)

        At z_calibration, we set H_baseline = H_hm (matching mid-z data).
        Then at z=0, the ratio gives the "inferred" H0 shift:
            delta_H0/H0 = H_hm(z=0)/H_baseline(z=0) - 1

        Args:
            M_interp: Callable M(ln_a) returning memory field at ln(a)
            z_calibration: Redshift for calibration (default 0.5)

        Returns:
            Dictionary with:
                - delta_H0_frac: Fractional H0 shift (H_hm(0)/H_baseline(0) - 1)
                - delta_H0_kmsMpc: Shift in km/s/Mpc (using H0 = 67.4)
                - H_ratio_z0: H_hm(z=0) / H_baseline(z=0)
                - H_ratio_z_cal: H_hm(z_cal) / H_baseline(z_cal)
                - Omega_hor0: Horizon-memory density fraction at z=0
                - Omega_L0_eff: Effective Lambda density fraction
        """
        if self.lambda_hor == 0.0 or self._M_today is None:
            return {
                "delta_H0_frac": 0.0,
                "delta_H0_kmsMpc": 0.0,
                "H_ratio_z0": 1.0,
                "H_ratio_z_cal": 1.0,
                "Omega_hor0": 0.0,
                "Omega_L0_eff": self.Omega_L0_base,
            }

        # Scale factors
        a_0 = 1.0
        a_cal = 1.0 / (1.0 + z_calibration)

        # Get memory field values (extract scalar from array if needed)
        ln_a_0 = 0.0
        ln_a_cal = np.log(a_cal)
        M_0_raw = M_interp(ln_a_0)
        M_cal_raw = M_interp(ln_a_cal)
        M_0 = float(M_0_raw[0]) if hasattr(M_0_raw, '__len__') else float(M_0_raw)
        M_cal = float(M_cal_raw[0]) if hasattr(M_cal_raw, '__len__') else float(M_cal_raw)

        # Compute H(z) with self-consistent horizon-memory
        H_hm_0 = self.H_of_a_selfconsistent(a_0, M_0)
        H_hm_cal = self.H_of_a_selfconsistent(a_cal, M_cal)

        # Compute H(z) with pure baseline GR
        H_gr_0 = self.H_of_a_gr_baseline(a_0)
        H_gr_cal = self.H_of_a_gr_baseline(a_cal)

        # Ratios
        H_ratio_z0 = H_hm_0 / H_gr_0 if H_gr_0 > 0 else 1.0
        H_ratio_z_cal = H_hm_cal / H_gr_cal if H_gr_cal > 0 else 1.0

        # The key quantity: if we calibrate to match at z_cal, what H0 do we infer?
        # H0_inferred / H0_baseline = (H_hm(0)/H_hm(z_cal)) / (H_gr(0)/H_gr(z_cal))
        #                           = H_ratio_z0 / H_ratio_z_cal
        # But in self-consistent case with proper flatness, we compare directly:
        delta_H0_frac = H_ratio_z0 - 1.0

        # Convert to km/s/Mpc (H0 = 67.4 km/s/Mpc from Planck)
        H0_Planck = 67.4  # km/s/Mpc
        delta_H0_kmsMpc = delta_H0_frac * H0_Planck

        return {
            "delta_H0_frac": delta_H0_frac,
            "delta_H0_kmsMpc": delta_H0_kmsMpc,
            "H_ratio_z0": H_ratio_z0,
            "H_ratio_z_cal": H_ratio_z_cal,
            "Omega_hor0": self.Omega_hor0,
            "Omega_L0_eff": self.Omega_L0_eff,
        }
