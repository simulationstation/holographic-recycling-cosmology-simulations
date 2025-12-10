"""Numerical utilities for HRC computations."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, TypeVar
import numpy as np
from numpy.typing import NDArray


T = TypeVar("T", float, NDArray[np.floating])


class GeffDivergenceError(Exception):
    """Raised when G_eff approaches divergence (phi -> phi_critical).

    This occurs when the scalar field approaches the critical value
    phi_c = 1/(8*pi*xi) where the effective gravitational coupling diverges.
    """

    def __init__(
        self,
        phi: float,
        phi_critical: float,
        z: Optional[float] = None,
        message: Optional[str] = None,
    ):
        self.phi = phi
        self.phi_critical = phi_critical
        self.z = z
        self.fraction = phi / phi_critical if phi_critical != 0 else float('inf')

        if message is None:
            z_str = f" at z={z:.2f}" if z is not None else ""
            message = (
                f"G_eff divergence: phi={phi:.6f} approaches critical value "
                f"phi_c={phi_critical:.6f} (phi/phi_c={self.fraction:.4f}){z_str}"
            )

        super().__init__(message)


@dataclass
class GeffValidityResult:
    """Result of G_eff validity check."""

    valid: bool
    phi: float
    phi_critical: float
    G_eff_ratio: Optional[float] = None
    z: Optional[float] = None
    message: str = ""

    @property
    def fraction_of_critical(self) -> float:
        """Fraction phi/phi_critical."""
        if self.phi_critical == 0:
            return float('inf')
        return abs(self.phi) / self.phi_critical


def compute_critical_phi(xi: float) -> float:
    """Compute the critical scalar field value where G_eff diverges.

    phi_c = 1 / (8 * pi * xi)

    Args:
        xi: Non-minimal coupling constant

    Returns:
        Critical phi value (inf if xi <= 0)
    """
    if xi <= 0:
        return float('inf')
    return 1.0 / (8.0 * np.pi * xi)


def check_geff_validity(
    phi: float,
    xi: float,
    epsilon: float = 0.01,
    z: Optional[float] = None,
) -> GeffValidityResult:
    """Check if phi is within safe bounds for G_eff computation.

    G_eff = G / (1 - 8*pi*xi*phi) diverges when phi -> phi_c = 1/(8*pi*xi).
    We flag as invalid if |phi| >= phi_c * (1 - epsilon).

    Args:
        phi: Scalar field value
        xi: Non-minimal coupling constant
        epsilon: Safety margin (default 0.01 = 1%)
        z: Optional redshift for error reporting

    Returns:
        GeffValidityResult with validity status and G_eff if valid
    """
    phi_c = compute_critical_phi(xi)

    if phi_c == float('inf'):
        # xi <= 0: no divergence possible
        G_eff_ratio = 1.0 / (1.0 - 8.0 * np.pi * xi * phi)
        return GeffValidityResult(
            valid=True,
            phi=phi,
            phi_critical=phi_c,
            G_eff_ratio=G_eff_ratio,
            z=z,
            message="No divergence (xi <= 0)",
        )

    # Check if phi is too close to critical value
    threshold = phi_c * (1.0 - epsilon)

    if abs(phi) >= threshold:
        return GeffValidityResult(
            valid=False,
            phi=phi,
            phi_critical=phi_c,
            G_eff_ratio=None,
            z=z,
            message=f"phi={phi:.6f} exceeds {(1-epsilon)*100:.1f}% of phi_c={phi_c:.6f}",
        )

    # Compute G_eff
    denominator = 1.0 - 8.0 * np.pi * xi * phi
    if abs(denominator) < 1e-10:
        return GeffValidityResult(
            valid=False,
            phi=phi,
            phi_critical=phi_c,
            G_eff_ratio=None,
            z=z,
            message=f"Denominator too small: {denominator:.3e}",
        )

    G_eff_ratio = 1.0 / denominator

    # Check for negative G_eff
    if G_eff_ratio < 0:
        return GeffValidityResult(
            valid=False,
            phi=phi,
            phi_critical=phi_c,
            G_eff_ratio=G_eff_ratio,
            z=z,
            message=f"G_eff/G = {G_eff_ratio:.4f} < 0 (negative gravity)",
        )

    # Check for unreasonably large G_eff
    if G_eff_ratio > 10.0:
        return GeffValidityResult(
            valid=False,
            phi=phi,
            phi_critical=phi_c,
            G_eff_ratio=G_eff_ratio,
            z=z,
            message=f"G_eff/G = {G_eff_ratio:.4f} > 10 (too large)",
        )

    return GeffValidityResult(
        valid=True,
        phi=phi,
        phi_critical=phi_c,
        G_eff_ratio=G_eff_ratio,
        z=z,
        message=f"Valid: G_eff/G = {G_eff_ratio:.4f}, phi/phi_c = {phi/phi_c:.4f}",
    )


@dataclass
class NumericalConfig:
    """Configuration for numerical integration."""

    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 0.1
    min_step: float = 1e-12
    max_iter: int = 10000

    # Divergence detection
    divergence_threshold: float = 1e10
    zero_threshold: float = 1e-30

    # Interpolation
    interp_kind: str = "cubic"


def safe_divide(
    numerator: T,
    denominator: T,
    default: float = 0.0,
    threshold: float = 1e-30,
) -> T:
    """Safely divide, returning default where denominator is near zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value when division would be singular
        threshold: Threshold below which denominator is considered zero

    Returns:
        Result of division, with default where singular
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(
            np.abs(denominator) > threshold,
            numerator / np.where(np.abs(denominator) > threshold, denominator, 1.0),
            default,
        )
        return result
    else:
        if abs(denominator) > threshold:
            return numerator / denominator
        return default


@dataclass
class DivergenceResult:
    """Result of divergence check."""

    has_divergence: bool
    divergence_indices: Optional[NDArray[np.intp]] = None
    divergence_values: Optional[NDArray[np.floating]] = None
    message: str = ""


def check_divergence(
    values: NDArray[np.floating],
    threshold: float = 1e10,
    check_nan: bool = True,
    check_inf: bool = True,
) -> DivergenceResult:
    """Check array for divergences.

    Args:
        values: Array to check
        threshold: Value magnitude threshold for divergence
        check_nan: Whether to flag NaN as divergence
        check_inf: Whether to flag Inf as divergence

    Returns:
        DivergenceResult with divergence information
    """
    divergent = np.abs(values) > threshold

    if check_nan:
        divergent |= np.isnan(values)
    if check_inf:
        divergent |= np.isinf(values)

    if np.any(divergent):
        indices = np.where(divergent)[0]
        return DivergenceResult(
            has_divergence=True,
            divergence_indices=indices,
            divergence_values=values[divergent],
            message=f"Divergence detected at {len(indices)} points; "
            f"first at index {indices[0]}, value = {values[indices[0]]:.3e}",
        )

    return DivergenceResult(has_divergence=False, message="No divergence detected")


def check_positivity(
    values: NDArray[np.floating],
    name: str = "quantity",
    threshold: float = -1e-10,
) -> Tuple[bool, str]:
    """Check that values are positive (or within numerical tolerance of zero).

    Args:
        values: Array to check
        name: Name of quantity for error message
        threshold: Values below this are considered negative

    Returns:
        Tuple of (all_positive, error_message)
    """
    min_val = np.min(values)
    if min_val < threshold:
        idx = np.argmin(values)
        return False, f"{name} becomes negative: min = {min_val:.3e} at index {idx}"
    return True, ""


def numerical_derivative(
    f: NDArray[np.floating],
    x: NDArray[np.floating],
    order: int = 1,
) -> NDArray[np.floating]:
    """Compute numerical derivative using finite differences.

    Uses second-order central differences in the interior
    and first-order forward/backward differences at boundaries.

    Args:
        f: Function values
        x: Independent variable values
        order: Order of derivative (1 or 2)

    Returns:
        Derivative values at each x
    """
    if order == 1:
        df = np.gradient(f, x, edge_order=2)
    elif order == 2:
        df1 = np.gradient(f, x, edge_order=2)
        df = np.gradient(df1, x, edge_order=2)
    else:
        raise ValueError(f"Order {order} not supported; use 1 or 2")

    return df


def interpolate_solution(
    x_data: NDArray[np.floating],
    y_data: NDArray[np.floating],
    x_query: Union[float, NDArray[np.floating]],
    kind: str = "cubic",
) -> Union[float, NDArray[np.floating]]:
    """Interpolate solution at query points.

    Args:
        x_data: Known x values (must be sorted)
        y_data: Known y values
        x_query: Points at which to interpolate
        kind: Interpolation method ('linear', 'cubic', 'quadratic')

    Returns:
        Interpolated y values at x_query
    """
    from scipy.interpolate import interp1d

    interp = interp1d(
        x_data,
        y_data,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp(x_query)


def solve_with_retry(
    solver_func,
    y0: NDArray[np.floating],
    t_span: Tuple[float, float],
    t_eval: Optional[NDArray[np.floating]] = None,
    rtol_sequence: Tuple[float, ...] = (1e-8, 1e-6, 1e-4),
    atol_sequence: Tuple[float, ...] = (1e-10, 1e-8, 1e-6),
) -> Tuple[bool, Optional[object], str]:
    """Attempt to solve ODE with progressively relaxed tolerances.

    Args:
        solver_func: Callable that takes (y0, t_span, t_eval, rtol, atol)
                    and returns scipy solve_ivp result
        y0: Initial conditions
        t_span: Integration interval
        t_eval: Evaluation points
        rtol_sequence: Sequence of relative tolerances to try
        atol_sequence: Sequence of absolute tolerances to try

    Returns:
        Tuple of (success, solution, message)
    """
    for rtol, atol in zip(rtol_sequence, atol_sequence):
        try:
            result = solver_func(y0, t_span, t_eval, rtol, atol)
            if result.success:
                return True, result, f"Converged with rtol={rtol}, atol={atol}"
        except Exception as e:
            continue

    return False, None, "Failed to converge with all tolerance settings"


class AdaptiveIntegrator:
    """Adaptive step size integrator for stiff cosmological ODEs."""

    def __init__(
        self,
        config: Optional[NumericalConfig] = None,
    ):
        self.config = config or NumericalConfig()
        self.step_history: list[float] = []
        self.rejection_count: int = 0

    def reset(self) -> None:
        """Reset integrator state."""
        self.step_history = []
        self.rejection_count = 0

    def estimate_initial_step(
        self,
        f: callable,
        t0: float,
        y0: NDArray[np.floating],
    ) -> float:
        """Estimate appropriate initial step size.

        Uses the algorithm from Hairer, Norsett, Wanner (1993).
        """
        d0 = np.linalg.norm(y0)
        f0 = f(t0, y0)
        d1 = np.linalg.norm(f0)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        # Limit step size
        h0 = min(h0, self.config.max_step)
        h0 = max(h0, self.config.min_step)

        return h0
