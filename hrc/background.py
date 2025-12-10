"""Background cosmology evolution for HRC.

This module implements the background Friedmann equations including
the scalar field and effective gravitational coupling modifications.

The key equations are:
    H² = (8πG_eff/3)(ρ_m + ρ_r + ρ_φ) + Λ/3
    Ḣ = -4πG_eff(ρ + P) - (1/2)Ġ_eff/G_eff · H

where G_eff = G/(1 - 8πGξφ) and the scalar field contributes:
    ρ_φ = ½φ̇² + V(φ)
    P_φ = ½φ̇² - V(φ)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Protocol
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .utils.config import HRCParameters, HRCConfig, PotentialConfig
from .utils.constants import SI_UNITS, convert_H0_to_si
from .utils.numerics import (
    check_divergence,
    check_positivity,
    safe_divide,
    DivergenceResult,
    GeffDivergenceError,
    GeffValidityResult,
    compute_critical_phi,
    check_geff_validity,
)


class EffectiveGravityProtocol(Protocol):
    """Protocol for effective gravity computation."""

    def G_eff(self, phi: float) -> float:
        """Compute G_eff given scalar field value."""
        ...

    def G_eff_ratio(self, phi: float) -> float:
        """Compute G_eff/G given scalar field value."""
        ...


@dataclass
class BackgroundState:
    """State of the background cosmology at a given time/redshift."""

    z: float  # Redshift
    a: float  # Scale factor
    H: float  # Hubble parameter (in H0 units)
    phi: float  # Scalar field value
    phi_dot: float  # Scalar field derivative dφ/dt
    G_eff_ratio: float  # G_eff/G
    rho_m: float  # Matter density (in H0² units)
    rho_r: float  # Radiation density
    rho_phi: float  # Scalar field density
    R: float  # Ricci scalar (in H0² units)


@dataclass
class BackgroundSolution:
    """Full background solution over redshift range."""

    z: NDArray[np.floating]
    a: NDArray[np.floating]
    H: NDArray[np.floating]  # In H0 units
    phi: NDArray[np.floating]
    phi_dot: NDArray[np.floating]
    G_eff_ratio: NDArray[np.floating]
    rho_m: NDArray[np.floating]
    rho_r: NDArray[np.floating]
    rho_phi: NDArray[np.floating]
    R: NDArray[np.floating]

    # Integration metadata
    success: bool = True
    message: str = ""
    divergence_info: Optional[DivergenceResult] = None

    # G_eff validity tracking
    geff_valid: bool = True
    geff_divergence_z: Optional[float] = None  # Redshift where G_eff diverges
    phi_critical: Optional[float] = None  # Critical phi value

    # Interpolators (created lazily)
    _H_interp: Optional[Callable[[float], float]] = field(default=None, repr=False)
    _phi_interp: Optional[Callable[[float], float]] = field(default=None, repr=False)
    _G_eff_interp: Optional[Callable[[float], float]] = field(default=None, repr=False)

    def H_at(self, z: float) -> float:
        """Interpolate H(z)."""
        if self._H_interp is None:
            self._H_interp = interp1d(
                self.z, self.H, kind="cubic", fill_value="extrapolate"
            )
        return float(self._H_interp(z))

    def phi_at(self, z: float) -> float:
        """Interpolate φ(z)."""
        if self._phi_interp is None:
            self._phi_interp = interp1d(
                self.z, self.phi, kind="cubic", fill_value="extrapolate"
            )
        return float(self._phi_interp(z))

    def G_eff_at(self, z: float) -> float:
        """Interpolate G_eff(z)/G."""
        if self._G_eff_interp is None:
            self._G_eff_interp = interp1d(
                self.z, self.G_eff_ratio, kind="cubic", fill_value="extrapolate"
            )
        return float(self._G_eff_interp(z))


class BackgroundCosmology:
    """Solver for HRC background cosmology.

    Integrates the coupled system of Friedmann + scalar field equations
    from z=0 backwards in time (increasing z).

    The system of ODEs (with t → ln(a) = -ln(1+z) as time variable):

    d(φ)/d(ln a) = φ'/H
    d(φ')/d(ln a) = -3φ' - (1/H²)[V'(φ) + ξR]

    where ' denotes d/dt and R = 6(2H² + Ḣ).
    """

    def __init__(
        self,
        params: HRCParameters,
        potential: Optional[PotentialConfig] = None,
        geff_epsilon: float = 0.01,
    ):
        """Initialize background solver.

        Args:
            params: HRC model parameters
            potential: Scalar field potential configuration
            geff_epsilon: Safety margin for G_eff divergence (default 1%)
        """
        self.params = params
        self.potential = potential or PotentialConfig(m=params.m_phi)
        self.geff_epsilon = geff_epsilon

        # Validate parameters
        valid, errors = params.validate()
        if not valid:
            raise ValueError(f"Invalid parameters: {errors}")

        # Precompute derived quantities
        self._8piG_xi = 8 * np.pi * params.xi  # For G_eff computation

        # Compute critical phi value
        self._phi_critical = compute_critical_phi(params.xi)
        self._phi_threshold = self._phi_critical * (1.0 - geff_epsilon)

    @property
    def phi_critical(self) -> float:
        """Critical scalar field value where G_eff diverges."""
        return self._phi_critical

    def check_phi_validity(self, phi: float, z: Optional[float] = None) -> GeffValidityResult:
        """Check if phi is within safe bounds.

        Args:
            phi: Scalar field value
            z: Optional redshift for error reporting

        Returns:
            GeffValidityResult with validity status
        """
        return check_geff_validity(phi, self.params.xi, self.geff_epsilon, z)

    def G_eff_ratio(self, phi: float) -> float:
        """Compute G_eff/G at given scalar field value.

        G_eff = G / (1 - 8πGξφ)

        Args:
            phi: Scalar field value

        Returns:
            Ratio G_eff/G
        """
        denominator = 1.0 - self._8piG_xi * phi
        if abs(denominator) < 1e-10:
            raise ValueError(
                f"G_eff divergence: 1 - 8πξφ = {denominator:.3e} for φ = {phi}"
            )
        return 1.0 / denominator

    def _rho_matter(self, z: float) -> float:
        """Matter density in units of ρ_crit,0 = 3H₀²/(8πG)."""
        return self.params.Omega_m * (1 + z) ** 3

    def _rho_radiation(self, z: float) -> float:
        """Radiation density in units of ρ_crit,0."""
        return self.params.Omega_r * (1 + z) ** 4

    def _rho_Lambda(self) -> float:
        """Dark energy density in units of ρ_crit,0."""
        return self.params.Omega_Lambda

    def _rho_scalar(self, phi: float, phi_dot: float, H: float) -> float:
        """Scalar field density.

        ρ_φ = ½φ̇² + V(φ)

        Args:
            phi: Scalar field value
            phi_dot: dφ/dt (in H0 units)
            H: Hubble parameter (in H0 units)

        Returns:
            Scalar field density (in ρ_crit,0 units)
        """
        kinetic = 0.5 * phi_dot**2
        potential = self.potential.V(phi)
        return kinetic + potential

    def _pressure_scalar(self, phi: float, phi_dot: float) -> float:
        """Scalar field pressure.

        P_φ = ½φ̇² - V(φ)
        """
        kinetic = 0.5 * phi_dot**2
        potential = self.potential.V(phi)
        return kinetic - potential

    def _compute_H_squared(
        self,
        z: float,
        phi: float,
        phi_dot: float,
    ) -> float:
        """Compute H² from Friedmann equation.

        H² = (8πG_eff/3) · ρ_tot + Λ/3

        In our units (H0 = 1), this becomes:
        (H/H0)² = G_eff/G · (Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_φ) + Ω_Λ

        Returns:
            H² in units of H₀²
        """
        G_ratio = self.G_eff_ratio(phi)

        rho_m = self._rho_matter(z)
        rho_r = self._rho_radiation(z)
        rho_phi = self._rho_scalar(phi, phi_dot, 1.0)  # H unused here
        rho_Lambda = self._rho_Lambda()

        # Modified Friedmann equation
        H_squared = G_ratio * (rho_m + rho_r + rho_phi) + rho_Lambda

        return H_squared

    def _compute_H_dot(
        self,
        z: float,
        H: float,
        phi: float,
        phi_dot: float,
    ) -> float:
        """Compute dH/dt from acceleration equation.

        Ḣ = -4πG_eff(ρ + P) - ½(Ġ_eff/G_eff)H

        The G_eff time derivative contributes when φ evolves.
        """
        G_ratio = self.G_eff_ratio(phi)

        # Matter: P = 0
        rho_m = self._rho_matter(z)
        rhoP_m = rho_m

        # Radiation: P = ρ/3
        rho_r = self._rho_radiation(z)
        rhoP_r = (4.0 / 3.0) * rho_r

        # Scalar field
        rho_phi = self._rho_scalar(phi, phi_dot, H)
        P_phi = self._pressure_scalar(phi, phi_dot)
        rhoP_phi = rho_phi + P_phi  # = φ̇²

        # Standard acceleration term
        H_dot = -0.5 * G_ratio * (rhoP_m + rhoP_r + rhoP_phi)

        # G_eff evolution contribution
        # Ġ_eff/G_eff = 8πξφ̇ / (1 - 8πξφ)
        if abs(phi_dot) > 1e-30:
            G_eff_dot_ratio = self._8piG_xi * phi_dot * G_ratio
            H_dot -= 0.5 * G_eff_dot_ratio * H

        return H_dot

    def _compute_Ricci(self, H: float, H_dot: float) -> float:
        """Compute Ricci scalar.

        R = 6(2H² + Ḣ)

        In an expanding universe with our conventions.
        """
        return 6.0 * (2.0 * H**2 + H_dot)

    def _scalar_field_eom(
        self,
        phi: float,
        phi_dot: float,
        H: float,
        R: float,
    ) -> float:
        """Scalar field equation of motion.

        φ̈ + 3Hφ̇ + V'(φ) + ξR = 0

        Returns:
            φ̈ (second time derivative)
        """
        V_prime = self.potential.dV(phi)
        phi_ddot = -3.0 * H * phi_dot - V_prime - self.params.xi * R
        return phi_ddot

    def _ode_system(
        self,
        lna: float,  # ln(a) = -ln(1+z), our "time" variable
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """ODE system for background evolution.

        State vector y = [φ, dφ/d(ln a)]

        We integrate in ln(a) rather than t for numerical stability.
        dφ/d(ln a) = (1/H) dφ/dt = φ̇/H

        The equation becomes:
        d²φ/d(ln a)² + (3 + Ḣ/H²)dφ/d(ln a) + (V' + ξR)/H² = 0
        """
        phi, dphi_dlna = y

        # Reconstruct redshift
        z = np.exp(-lna) - 1.0
        if z < 0:
            z = 0.0

        # Check if phi is approaching critical value
        if self._phi_critical != float('inf') and abs(phi) >= self._phi_threshold:
            # Store the divergence info (only record the first occurrence)
            if not self._hit_divergence:
                self._hit_divergence = True
                self._divergence_z = z
                self._divergence_phi = phi
            # Return zero derivatives to halt meaningful evolution
            return np.array([0.0, 0.0])

        # First, estimate H from Friedmann equation
        # φ̇ = H · dφ/d(ln a)
        # We need to solve self-consistently
        try:
            # Initial guess for H
            H_squared = self._compute_H_squared(z, phi, 0.0)
            if H_squared <= 0:
                return np.array([0.0, 0.0])
            H = np.sqrt(H_squared)

            # Refine with actual phi_dot
            phi_dot = H * dphi_dlna
            H_squared = self._compute_H_squared(z, phi, phi_dot)
            if H_squared <= 0:
                return np.array([0.0, 0.0])
            H = np.sqrt(H_squared)
            phi_dot = H * dphi_dlna

            # Compute Ḣ
            H_dot = self._compute_H_dot(z, H, phi, phi_dot)

            # Compute Ricci scalar
            R = self._compute_Ricci(H, H_dot)

            # Scalar field EOM: φ̈ + 3Hφ̇ + V'(φ) + ξR = 0
            # In ln(a) coordinates:
            # d²φ/d(ln a)² = (φ̈/H² - (Ḣ/H²)(dφ/d ln a))
            phi_ddot = self._scalar_field_eom(phi, phi_dot, H, R)

            # Convert to ln(a) derivative
            d2phi_dlna2 = phi_ddot / H**2 - (H_dot / H**2) * dphi_dlna

            return np.array([dphi_dlna, d2phi_dlna2])

        except (ValueError, ZeroDivisionError):
            # Return zero derivatives if we hit a singularity
            return np.array([0.0, 0.0])

    def solve(
        self,
        z_max: float = 1100.0,
        z_points: int = 1000,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        method: str = "RK45",
    ) -> BackgroundSolution:
        """Solve background evolution from z=0 to z=z_max.

        Args:
            z_max: Maximum redshift
            z_points: Number of output points
            rtol: Relative tolerance
            atol: Absolute tolerance
            method: Integration method ('RK45', 'DOP853', 'Radau', 'BDF')

        Returns:
            BackgroundSolution with full evolution history
        """
        # Initialize divergence tracking
        self._hit_divergence = False
        self._divergence_z = None
        self._divergence_phi = None

        # Check initial phi validity
        initial_validity = self.check_phi_validity(self.params.phi_0, z=0.0)
        if not initial_validity.valid:
            return BackgroundSolution(
                z=np.array([0.0]),
                a=np.array([1.0]),
                H=np.array([1.0]),
                phi=np.array([self.params.phi_0]),
                phi_dot=np.array([self.params.phi_dot_0]),
                G_eff_ratio=np.array([np.nan]),
                rho_m=np.array([self.params.Omega_m]),
                rho_r=np.array([self.params.Omega_r]),
                rho_phi=np.array([0.0]),
                R=np.array([0.0]),
                success=False,
                message=f"Initial phi invalid: {initial_validity.message}",
                geff_valid=False,
                geff_divergence_z=0.0,
                phi_critical=self._phi_critical,
            )

        # Initial conditions at z=0
        phi_0 = self.params.phi_0
        phi_dot_0 = self.params.phi_dot_0

        # Convert to ln(a) derivative
        H_0 = 1.0  # H₀ in our units
        dphi_dlna_0 = phi_dot_0 / H_0 if abs(H_0) > 1e-30 else 0.0

        y0 = np.array([phi_0, dphi_dlna_0])

        # Integration limits (ln(a) decreases as z increases)
        lna_0 = 0.0  # z = 0
        lna_max = -np.log(1.0 + z_max)  # z = z_max

        # Output points
        z_eval = np.linspace(0, z_max, z_points)
        lna_eval = -np.log(1.0 + z_eval)

        # Solve ODE
        try:
            sol = solve_ivp(
                self._ode_system,
                t_span=(lna_0, lna_max),
                y0=y0,
                method=method,
                t_eval=lna_eval,
                rtol=rtol,
                atol=atol,
                dense_output=True,
            )

            if not sol.success:
                return BackgroundSolution(
                    z=z_eval,
                    a=1.0 / (1.0 + z_eval),
                    H=np.ones_like(z_eval),
                    phi=np.full_like(z_eval, phi_0),
                    phi_dot=np.zeros_like(z_eval),
                    G_eff_ratio=np.ones_like(z_eval),
                    rho_m=self.params.Omega_m * (1 + z_eval) ** 3,
                    rho_r=self.params.Omega_r * (1 + z_eval) ** 4,
                    rho_phi=np.zeros_like(z_eval),
                    R=np.zeros_like(z_eval),
                    success=False,
                    message=f"Integration failed: {sol.message}",
                    geff_valid=not self._hit_divergence,
                    geff_divergence_z=self._divergence_z,
                    phi_critical=self._phi_critical,
                )

            # Extract solution
            phi = sol.y[0]
            dphi_dlna = sol.y[1]

        except Exception as e:
            return BackgroundSolution(
                z=z_eval,
                a=1.0 / (1.0 + z_eval),
                H=np.ones_like(z_eval),
                phi=np.full_like(z_eval, phi_0),
                phi_dot=np.zeros_like(z_eval),
                G_eff_ratio=np.ones_like(z_eval),
                rho_m=self.params.Omega_m * (1 + z_eval) ** 3,
                rho_r=self.params.Omega_r * (1 + z_eval) ** 4,
                rho_phi=np.zeros_like(z_eval),
                R=np.zeros_like(z_eval),
                success=False,
                message=f"Integration error: {str(e)}",
                geff_valid=not self._hit_divergence,
                geff_divergence_z=self._divergence_z,
                phi_critical=self._phi_critical,
            )

        # Reconstruct all quantities from the solution
        n = len(z_eval)
        H = np.zeros(n)
        phi_dot = np.zeros(n)
        G_eff_ratio = np.zeros(n)
        rho_m = np.zeros(n)
        rho_r = np.zeros(n)
        rho_phi = np.zeros(n)
        R = np.zeros(n)

        for i in range(n):
            z = z_eval[i]
            phi_i = phi[i]

            try:
                G_eff_ratio[i] = self.G_eff_ratio(phi_i)
            except ValueError:
                G_eff_ratio[i] = np.nan

            rho_m[i] = self._rho_matter(z)
            rho_r[i] = self._rho_radiation(z)

            # Compute H
            H_sq = self._compute_H_squared(z, phi_i, 0.0)
            if H_sq > 0:
                H[i] = np.sqrt(H_sq)
                phi_dot[i] = H[i] * dphi_dlna[i]

                # Refine H with actual phi_dot
                H_sq = self._compute_H_squared(z, phi_i, phi_dot[i])
                if H_sq > 0:
                    H[i] = np.sqrt(H_sq)
                    phi_dot[i] = H[i] * dphi_dlna[i]

                    rho_phi[i] = self._rho_scalar(phi_i, phi_dot[i], H[i])
                    H_dot = self._compute_H_dot(z, H[i], phi_i, phi_dot[i])
                    R[i] = self._compute_Ricci(H[i], H_dot)
            else:
                H[i] = np.nan
                phi_dot[i] = np.nan
                rho_phi[i] = np.nan
                R[i] = np.nan

        # Check for divergences
        div_check = check_divergence(G_eff_ratio)

        # Check if we hit G_eff divergence during integration
        geff_valid = not self._hit_divergence
        if self._hit_divergence:
            message = (
                f"G_eff divergence: phi approaches critical value "
                f"phi_c={self._phi_critical:.6f} at z={self._divergence_z:.2f}"
            )
        else:
            message = "Integration successful"

        return BackgroundSolution(
            z=z_eval,
            a=1.0 / (1.0 + z_eval),
            H=H,
            phi=phi,
            phi_dot=phi_dot,
            G_eff_ratio=G_eff_ratio,
            rho_m=rho_m,
            rho_r=rho_r,
            rho_phi=rho_phi,
            R=R,
            success=geff_valid,  # Mark as failed if G_eff diverged
            message=message,
            divergence_info=div_check if div_check.has_divergence else None,
            geff_valid=geff_valid,
            geff_divergence_z=self._divergence_z,
            phi_critical=self._phi_critical,
        )

    def get_state(self, z: float, solution: BackgroundSolution) -> BackgroundState:
        """Get interpolated state at specific redshift."""
        H = solution.H_at(z)
        phi = solution.phi_at(z)
        G_eff = solution.G_eff_at(z)

        # Interpolate other quantities
        from scipy.interpolate import interp1d

        phi_dot_interp = interp1d(solution.z, solution.phi_dot, kind="cubic")
        rho_m_interp = interp1d(solution.z, solution.rho_m, kind="cubic")
        rho_r_interp = interp1d(solution.z, solution.rho_r, kind="cubic")
        rho_phi_interp = interp1d(solution.z, solution.rho_phi, kind="cubic")
        R_interp = interp1d(solution.z, solution.R, kind="cubic")

        return BackgroundState(
            z=z,
            a=1.0 / (1.0 + z),
            H=H,
            phi=phi,
            phi_dot=float(phi_dot_interp(z)),
            G_eff_ratio=G_eff,
            rho_m=float(rho_m_interp(z)),
            rho_r=float(rho_r_interp(z)),
            rho_phi=float(rho_phi_interp(z)),
            R=float(R_interp(z)),
        )
