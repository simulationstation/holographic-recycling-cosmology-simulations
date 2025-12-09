"""Structure growth constraints on HRC.

The growth of cosmic structure provides constraints on modified gravity
through the growth rate f(z) = d ln δ / d ln a and the combination fσ₈(z).

In GR with ΛCDM:
    f(z) ≈ Ω_m(z)^γ with γ ≈ 0.55

In HRC with G_eff(z):
    - Growth is enhanced when G_eff > G
    - The growth index γ is modified
    - Tension with fσ₈ measurements can constrain HRC

Key observables:
    - fσ₈(z) from redshift-space distortions (RSD)
    - Weak lensing measurements
    - CMB lensing
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ..utils.config import HRCParameters
from ..background import BackgroundSolution


@dataclass
class GrowthConstraint:
    """Result of structure growth constraint check."""

    name: str
    allowed: bool
    value: float  # Computed fσ₈
    observed: float  # Observed value
    sigma: float  # Observational uncertainty
    z: float  # Redshift of measurement
    chi2: float  # Chi-squared contribution
    message: str


@dataclass
class GrowthSolution:
    """Solution for growth factor evolution."""

    z: NDArray[np.floating]
    D: NDArray[np.floating]  # Growth factor D(z), normalized to D(0)=1
    f: NDArray[np.floating]  # Growth rate f = d ln D / d ln a
    sigma8: NDArray[np.floating]  # σ₈(z)
    fsigma8: NDArray[np.floating]  # fσ₈(z)

    success: bool = True
    message: str = ""

    def D_at(self, z: float) -> float:
        """Interpolate D(z)."""
        interp = interp1d(self.z, self.D, kind="cubic", fill_value="extrapolate")
        return float(interp(z))

    def fsigma8_at(self, z: float) -> float:
        """Interpolate fσ₈(z)."""
        interp = interp1d(self.z, self.fsigma8, kind="cubic", fill_value="extrapolate")
        return float(interp(z))


# Observational fσ₈ measurements
FSIGMA8_DATA = [
    # (z, fσ₈, σ, survey)
    (0.02, 0.398, 0.065, "6dFGS"),
    (0.10, 0.370, 0.130, "SDSS-DR7"),
    (0.15, 0.490, 0.145, "SDSS-DR7"),
    (0.32, 0.384, 0.095, "BOSS-LOWZ"),
    (0.38, 0.497, 0.045, "BOSS-DR12"),
    (0.51, 0.458, 0.038, "BOSS-DR12"),
    (0.61, 0.436, 0.034, "BOSS-DR12"),
    (0.70, 0.448, 0.043, "eBOSS-LRG"),
    (0.85, 0.315, 0.095, "eBOSS-ELG"),
    (1.48, 0.462, 0.045, "eBOSS-QSO"),
]


class GrowthCalculator:
    """Calculate structure growth in HRC.

    The growth factor D(a) satisfies:
        D'' + (2 + H'/H) D' - (3/2) Ω_m(a) G_eff(a)/G D = 0

    where ' denotes d/d(ln a).
    """

    def __init__(
        self,
        params: HRCParameters,
        background: Optional[BackgroundSolution] = None,
        sigma8_0: float = 0.811,  # Planck 2018 value
    ):
        """Initialize growth calculator.

        Args:
            params: HRC parameters
            background: Background solution (for H(z) and G_eff(z))
            sigma8_0: Present-day σ₈ normalization
        """
        self.params = params
        self.background = background
        self.sigma8_0 = sigma8_0

        # Set up interpolators if background provided
        if background is not None:
            self._H_interp = interp1d(
                background.z, background.H, kind="cubic", fill_value="extrapolate"
            )
            self._G_eff_interp = interp1d(
                background.z,
                background.G_eff_ratio,
                kind="cubic",
                fill_value="extrapolate",
            )
        else:
            self._H_interp = None
            self._G_eff_interp = None

    def _H(self, z: float) -> float:
        """Hubble parameter H(z)/H₀."""
        if self._H_interp is not None:
            return float(self._H_interp(z))
        # ΛCDM approximation
        Om = self.params.Omega_m
        OL = self.params.Omega_Lambda
        return np.sqrt(Om * (1 + z) ** 3 + OL)

    def _G_eff(self, z: float) -> float:
        """Effective gravitational coupling G_eff(z)/G."""
        if self._G_eff_interp is not None:
            return float(self._G_eff_interp(z))
        # Constant G if no background
        return 1.0

    def _Omega_m(self, z: float) -> float:
        """Matter density parameter Ω_m(z)."""
        Om0 = self.params.Omega_m
        H = self._H(z)
        return Om0 * (1 + z) ** 3 / H**2

    def _growth_ode(
        self,
        lna: float,  # ln(a) = -ln(1+z)
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """ODE for growth factor.

        State: y = [D, dD/d(ln a)]

        Equation:
        D'' + (2 + H'/H) D' - (3/2) Ω_m G_eff/G D = 0

        where ' = d/d(ln a)
        """
        D, Dprime = y

        z = np.exp(-lna) - 1
        if z < 0:
            z = 0

        H = self._H(z)
        G_eff = self._G_eff(z)
        Omega_m = self._Omega_m(z)

        # Compute H'/H numerically
        dlna = 0.01
        H_plus = self._H(np.exp(-(lna + dlna)) - 1)
        H_minus = self._H(np.exp(-(lna - dlna)) - 1)
        Hprime_over_H = (H_plus - H_minus) / (2 * dlna * H)

        # Growth equation
        Dprimeprime = -(2 + Hprime_over_H) * Dprime + 1.5 * Omega_m * G_eff * D

        return np.array([Dprime, Dprimeprime])

    def solve(
        self,
        z_max: float = 10.0,
        z_points: int = 200,
    ) -> GrowthSolution:
        """Solve for growth factor evolution.

        Args:
            z_max: Maximum redshift
            z_points: Number of output points

        Returns:
            GrowthSolution with D(z), f(z), σ₈(z), fσ₈(z)
        """
        # Initial conditions at high z (matter-dominated approximation)
        # D ∝ a in matter domination, so D' = D
        lna_start = -np.log(1 + z_max)
        D0 = 1.0 / (1 + z_max)  # D ∝ a
        Dprime0 = D0  # D' = dD/d(ln a) = D in matter domination

        y0 = np.array([D0, Dprime0])

        # Integration points
        z_eval = np.linspace(z_max, 0, z_points)
        lna_eval = -np.log(1 + z_eval)

        try:
            sol = solve_ivp(
                self._growth_ode,
                t_span=(lna_start, 0),
                y0=y0,
                method="RK45",
                t_eval=lna_eval,
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                return GrowthSolution(
                    z=z_eval,
                    D=np.ones(z_points),
                    f=np.full(z_points, 0.55),
                    sigma8=np.full(z_points, self.sigma8_0),
                    fsigma8=np.full(z_points, 0.45),
                    success=False,
                    message=f"Integration failed: {sol.message}",
                )

            D = sol.y[0]
            Dprime = sol.y[1]

        except Exception as e:
            return GrowthSolution(
                z=z_eval,
                D=np.ones(z_points),
                f=np.full(z_points, 0.55),
                sigma8=np.full(z_points, self.sigma8_0),
                fsigma8=np.full(z_points, 0.45),
                success=False,
                message=f"Error: {str(e)}",
            )

        # Normalize D(z=0) = 1
        D = D / D[-1]
        Dprime = Dprime / D[-1] if D[-1] != 0 else Dprime

        # Growth rate f = d ln D / d ln a = D'/D
        f = Dprime / D
        f = np.clip(f, 0, 2)  # Physical bounds

        # σ₈(z) = σ₈(0) × D(z)
        sigma8 = self.sigma8_0 * D

        # fσ₈(z)
        fsigma8 = f * sigma8

        return GrowthSolution(
            z=z_eval,
            D=D,
            f=f,
            sigma8=sigma8,
            fsigma8=fsigma8,
            success=True,
            message="Growth computation successful",
        )


def check_fsigma8_point(
    growth: GrowthSolution,
    z_obs: float,
    fsigma8_obs: float,
    sigma_obs: float,
    survey: str = "",
) -> GrowthConstraint:
    """Check single fσ₈ measurement.

    Args:
        growth: Growth solution
        z_obs: Observed redshift
        fsigma8_obs: Observed fσ₈
        sigma_obs: Observational uncertainty
        survey: Survey name

    Returns:
        GrowthConstraint result
    """
    fsigma8_pred = growth.fsigma8_at(z_obs)

    residual = fsigma8_pred - fsigma8_obs
    chi2 = (residual / sigma_obs) ** 2
    n_sigma = abs(residual) / sigma_obs

    allowed = n_sigma < 3.0  # 3σ threshold

    if allowed:
        message = f"z={z_obs:.2f}: fσ₈ = {fsigma8_pred:.3f} vs {fsigma8_obs:.3f}±{sigma_obs:.3f} ({n_sigma:.1f}σ)"
    else:
        message = f"z={z_obs:.2f}: fσ₈ = {fsigma8_pred:.3f} vs {fsigma8_obs:.3f}±{sigma_obs:.3f} ({n_sigma:.1f}σ) TENSION"

    return GrowthConstraint(
        name=f"fsigma8_{survey}" if survey else f"fsigma8_z{z_obs:.2f}",
        allowed=allowed,
        value=fsigma8_pred,
        observed=fsigma8_obs,
        sigma=sigma_obs,
        z=z_obs,
        chi2=chi2,
        message=message,
    )


def check_growth_constraints(
    growth: GrowthSolution,
    data: Optional[List[Tuple[float, float, float, str]]] = None,
    verbose: bool = False,
) -> Tuple[bool, List[GrowthConstraint], float]:
    """Check all fσ₈ constraints.

    Args:
        growth: Growth solution
        data: List of (z, fσ₈, σ, survey) tuples (default: FSIGMA8_DATA)
        verbose: Print results

    Returns:
        Tuple of (all_passed, list of GrowthConstraint, total χ²)
    """
    if data is None:
        data = FSIGMA8_DATA

    results = []
    total_chi2 = 0.0

    for z_obs, fsigma8_obs, sigma_obs, survey in data:
        if z_obs > growth.z[0]:  # Beyond solution range
            continue

        constraint = check_fsigma8_point(growth, z_obs, fsigma8_obs, sigma_obs, survey)
        results.append(constraint)
        total_chi2 += constraint.chi2

    all_passed = all(r.allowed for r in results)
    n_points = len(results)
    chi2_per_dof = total_chi2 / n_points if n_points > 0 else 0

    if verbose:
        print("\n=== Structure Growth Constraint Checks ===")
        for r in results:
            status = "✓" if r.allowed else "✗"
            print(f"{status} {r.message}")
        print(f"\nTotal χ²/dof = {chi2_per_dof:.2f} ({n_points} points)")

    return all_passed, results, total_chi2
