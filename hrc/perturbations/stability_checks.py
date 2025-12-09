"""Stability checks for scalar-tensor perturbations in HRC.

This module implements the theoretical consistency conditions that must
be satisfied for a healthy scalar-tensor theory:

1. No-ghost condition: Kinetic term has correct sign
2. Gradient stability: Sound speed squared is positive (c_s² > 0)
3. Tensor stability: Graviton mass squared is non-negative
4. Positive effective Planck mass

For HRC with action:
    S = ∫d⁴x√(-g)[R/(16πG) - ½(∂φ)² - V(φ) - ξφR]

The effective Planck mass is M_eff² = 1/(8πG_eff) = (1 - 8πGξφ)/(8πG)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Protocol
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters, PotentialConfig
from ..background import BackgroundSolution


@dataclass
class StabilityResult:
    """Result of a stability check."""

    name: str
    passed: bool
    value: float  # The computed quantity
    bound: float  # The constraint bound
    margin: float  # How far from violation (in units of bound)
    message: str

    @property
    def critical(self) -> bool:
        """Return True if close to instability (margin < 0.1)."""
        return self.margin < 0.1


class StabilityProtocol(Protocol):
    """Protocol for stability checkers."""

    def check(
        self,
        phi: float,
        phi_dot: float,
        H: float,
        params: HRCParameters,
    ) -> StabilityResult:
        """Check stability condition."""
        ...


def compute_effective_planck_mass_squared(
    phi: float,
    params: HRCParameters,
) -> float:
    """Compute effective Planck mass squared.

    M_eff² = M_Pl² (1 - 8πξφ) = (1 - 8πξφ)/(8πG)

    In Planck units (G = 1):
        M_eff² = (1 - 8πξφ)/(8π)

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        M_eff² in Planck units
    """
    return (1.0 - 8 * np.pi * params.xi * phi) / (8 * np.pi)


def check_no_ghost(
    phi: float,
    phi_dot: float,
    H: float,
    params: HRCParameters,
    potential: Optional[PotentialConfig] = None,
) -> StabilityResult:
    """Check no-ghost condition for scalar perturbations.

    The kinetic term for scalar perturbations must have the correct sign.
    For our theory, this requires:
        Q_s = M_eff² + (kinetic corrections) > 0

    In the simplest case (minimal kinetic term):
        Q_s = M_eff² > 0  ⟹  1 - 8πξφ > 0

    Args:
        phi: Scalar field value
        phi_dot: Scalar field velocity
        H: Hubble parameter
        params: HRC parameters
        potential: Potential configuration

    Returns:
        StabilityResult for no-ghost condition
    """
    # Effective Planck mass squared
    M_eff_sq = compute_effective_planck_mass_squared(phi, params)

    # Additional kinetic contribution from scalar field
    # In Horndeski, this gets more complex, but for our simple case:
    Q_s = M_eff_sq

    # Ghost-free requires Q_s > 0
    passed = Q_s > 0
    margin = Q_s / abs(Q_s) if abs(Q_s) > 1e-30 else 0.0

    if passed:
        message = f"No ghost: Q_s = {Q_s:.4f} > 0"
    else:
        message = f"GHOST INSTABILITY: Q_s = {Q_s:.4f} < 0"

    return StabilityResult(
        name="no_ghost",
        passed=passed,
        value=Q_s,
        bound=0.0,
        margin=margin,
        message=message,
    )


def check_gradient_stability(
    phi: float,
    phi_dot: float,
    H: float,
    params: HRCParameters,
    potential: Optional[PotentialConfig] = None,
) -> StabilityResult:
    """Check gradient stability (positive sound speed squared).

    For scalar perturbations, we need c_s² > 0 to avoid gradient
    instabilities (exponential growth of short-wavelength modes).

    For our scalar-tensor theory:
        c_s² = 1 + (corrections from non-minimal coupling)

    In the simplest approximation, c_s² ≈ 1 for a canonical scalar.

    Args:
        phi: Scalar field value
        phi_dot: Scalar field velocity
        H: Hubble parameter
        params: HRC parameters
        potential: Potential configuration

    Returns:
        StabilityResult for gradient stability
    """
    M_eff_sq = compute_effective_planck_mass_squared(phi, params)

    # For a canonical scalar with non-minimal coupling,
    # the sound speed squared is approximately:
    # c_s² ≈ 1 - (corrections)

    # Leading correction from time-varying M_eff:
    # c_s² ≈ 1 + O(ξ φ̇ / H)

    if M_eff_sq <= 0:
        # Already have ghost, gradient stability is moot
        return StabilityResult(
            name="gradient_stability",
            passed=False,
            value=0.0,
            bound=0.0,
            margin=0.0,
            message="Gradient stability undefined (ghost present)",
        )

    # Compute effective sound speed squared
    # c_s² = P_s / Q_s where P_s is the spatial kinetic coefficient
    # For minimal coupling in the scalar sector: c_s² = 1

    # Time derivative of M_eff²: dM_eff²/dt = -8πξφ̇/(8π) = -ξφ̇
    dM_eff_sq_dt = -params.xi * phi_dot

    # Correction to sound speed (approximate)
    if abs(H) > 1e-30 and abs(M_eff_sq) > 1e-30:
        correction = -0.5 * dM_eff_sq_dt / (H * M_eff_sq)
    else:
        correction = 0.0

    c_s_squared = 1.0 + correction

    passed = c_s_squared > 0
    margin = c_s_squared if passed else c_s_squared / abs(c_s_squared)

    if passed:
        message = f"Gradient stable: c_s² = {c_s_squared:.4f} > 0"
    else:
        message = f"GRADIENT INSTABILITY: c_s² = {c_s_squared:.4f} < 0"

    return StabilityResult(
        name="gradient_stability",
        passed=passed,
        value=c_s_squared,
        bound=0.0,
        margin=margin,
        message=message,
    )


def check_tensor_stability(
    phi: float,
    phi_dot: float,
    H: float,
    params: HRCParameters,
) -> StabilityResult:
    """Check tensor perturbation stability.

    For tensor modes (gravitational waves), we need:
    1. No tensor ghost: G_T > 0
    2. Positive tensor sound speed: c_T² > 0
    3. Non-negative tensor mass: m_T² ≥ 0

    For our theory:
        G_T = M_eff² > 0 (same as scalar no-ghost)
        c_T² = 1 (luminal GW propagation)
        m_T² = 0 (massless graviton)

    Args:
        phi: Scalar field value
        phi_dot: Scalar field velocity
        H: Hubble parameter
        params: HRC parameters

    Returns:
        StabilityResult for tensor stability
    """
    M_eff_sq = compute_effective_planck_mass_squared(phi, params)

    # Tensor kinetic coefficient
    G_T = M_eff_sq

    # Tensor sound speed (GW propagation speed)
    c_T_squared = 1.0  # Exactly luminal in our theory

    # Tensor mass (should be zero for GR-like theory)
    m_T_squared = 0.0

    # Combined stability
    tensor_stable = G_T > 0 and c_T_squared > 0 and m_T_squared >= 0

    if tensor_stable:
        message = f"Tensor stable: G_T = {G_T:.4f}, c_T² = {c_T_squared:.4f}"
    else:
        message = f"TENSOR INSTABILITY: G_T = {G_T:.4f}"

    return StabilityResult(
        name="tensor_stability",
        passed=tensor_stable,
        value=G_T,
        bound=0.0,
        margin=G_T if G_T > 0 else G_T / abs(G_T),
        message=message,
    )


def check_effective_planck_mass(
    phi: float,
    params: HRCParameters,
) -> StabilityResult:
    """Check positivity of effective Planck mass.

    M_eff² > 0 is required for gravity to be attractive.

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        StabilityResult for Planck mass positivity
    """
    M_eff_sq = compute_effective_planck_mass_squared(phi, params)

    passed = M_eff_sq > 0

    if passed:
        M_eff = np.sqrt(M_eff_sq)
        message = f"M_eff² = {M_eff_sq:.4f} > 0 (M_eff/M_Pl = {M_eff:.4f})"
    else:
        message = f"NEGATIVE M_eff²: {M_eff_sq:.4f} (antigravity!)"

    return StabilityResult(
        name="positive_planck_mass",
        passed=passed,
        value=M_eff_sq,
        bound=0.0,
        margin=M_eff_sq if passed else M_eff_sq / abs(M_eff_sq),
        message=message,
    )


def check_all_stability(
    phi: float,
    phi_dot: float,
    H: float,
    params: HRCParameters,
    potential: Optional[PotentialConfig] = None,
    abort_on_failure: bool = False,
) -> Tuple[bool, List[StabilityResult]]:
    """Run all stability checks.

    Args:
        phi: Scalar field value
        phi_dot: Scalar field velocity
        H: Hubble parameter
        params: HRC parameters
        potential: Potential configuration
        abort_on_failure: Raise exception if any check fails

    Returns:
        Tuple of (all_passed, list of StabilityResults)

    Raises:
        ValueError: If abort_on_failure and any check fails
    """
    results = [
        check_effective_planck_mass(phi, params),
        check_no_ghost(phi, phi_dot, H, params, potential),
        check_gradient_stability(phi, phi_dot, H, params, potential),
        check_tensor_stability(phi, phi_dot, H, params),
    ]

    all_passed = all(r.passed for r in results)

    if abort_on_failure and not all_passed:
        failed = [r for r in results if not r.passed]
        messages = "; ".join(r.message for r in failed)
        raise ValueError(f"Stability check failed: {messages}")

    return all_passed, results


class StabilityChecker:
    """Class-based stability checker for use over evolution history."""

    def __init__(
        self,
        params: HRCParameters,
        potential: Optional[PotentialConfig] = None,
    ):
        self.params = params
        self.potential = potential
        self.history: List[Tuple[float, List[StabilityResult]]] = []

    def check_at_z(
        self,
        z: float,
        phi: float,
        phi_dot: float,
        H: float,
    ) -> Tuple[bool, List[StabilityResult]]:
        """Check stability at given redshift."""
        all_passed, results = check_all_stability(
            phi, phi_dot, H, self.params, self.potential
        )
        self.history.append((z, results))
        return all_passed, results

    def check_solution(
        self,
        solution: BackgroundSolution,
    ) -> Tuple[bool, NDArray[np.bool_], List[str]]:
        """Check stability over full solution.

        Args:
            solution: Background cosmology solution

        Returns:
            Tuple of:
                - all_stable: True if stable everywhere
                - stable_mask: Boolean array of stability at each z
                - messages: List of warning messages
        """
        n = len(solution.z)
        stable_mask = np.ones(n, dtype=bool)
        messages = []

        self.history = []

        for i in range(n):
            z = solution.z[i]
            phi = solution.phi[i]
            phi_dot = solution.phi_dot[i]
            H = solution.H[i]

            if np.isnan(phi) or np.isnan(phi_dot) or np.isnan(H):
                stable_mask[i] = False
                continue

            all_passed, results = self.check_at_z(z, phi, phi_dot, H)
            stable_mask[i] = all_passed

            if not all_passed:
                failed = [r.name for r in results if not r.passed]
                messages.append(f"z = {z:.2f}: Failed {', '.join(failed)}")

        all_stable = np.all(stable_mask)

        if not all_stable:
            first_unstable = np.argmin(stable_mask)
            messages.insert(
                0,
                f"First instability at z = {solution.z[first_unstable]:.2f}",
            )

        return all_stable, stable_mask, messages

    def get_stability_summary(self) -> dict:
        """Get summary of stability checks over history."""
        if not self.history:
            return {"checked": False}

        n_checks = len(self.history)
        z_values = [h[0] for h in self.history]
        all_results = [h[1] for h in self.history]

        # Count failures by type
        failure_counts = {}
        for z, results in self.history:
            for r in results:
                if not r.passed:
                    failure_counts[r.name] = failure_counts.get(r.name, 0) + 1

        return {
            "checked": True,
            "n_points": n_checks,
            "z_range": (min(z_values), max(z_values)),
            "all_stable": all(
                all(r.passed for r in results) for _, results in self.history
            ),
            "failure_counts": failure_counts,
        }
