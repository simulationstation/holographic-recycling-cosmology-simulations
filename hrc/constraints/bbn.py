"""Big Bang Nucleosynthesis constraints on G_eff.

BBN provides one of the strongest constraints on variations of Newton's
constant in the early universe. The primordial abundances of light elements
(D, ³He, ⁴He, ⁷Li) depend sensitively on the expansion rate H ∝ √G during
nucleosynthesis (z ~ 10⁸ - 10⁹, T ~ 0.1-1 MeV).

Key constraint:
    |ΔG/G|_BBN < 10% (conservative)
    |ΔG/G|_BBN < 5% (from ⁴He abundance)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ..utils.config import HRCParameters
from ..background import BackgroundSolution


@dataclass
class BBNConstraint:
    """Result of BBN constraint check."""

    allowed: bool
    value: float  # |ΔG_eff/G|_BBN
    bound: float  # Constraint bound
    sigma_margin: float  # Number of sigma from bound
    z_bbn: float  # Redshift used for BBN
    G_eff_bbn: float  # G_eff/G at BBN
    G_eff_today: float  # G_eff/G today
    message: str


def _bbn_redshift() -> float:
    """Return characteristic BBN redshift.

    BBN occurs at T ~ 0.1 - 1 MeV, corresponding to z ~ 10⁸ - 10⁹.
    We use z = 4 × 10⁸ as a characteristic value (T ~ 0.3 MeV,
    corresponding to deuterium formation).
    """
    return 4e8


def _bbn_G_eff_bound(
    constraint_level: str = "conservative",
) -> Tuple[float, float]:
    """Return BBN bound on |ΔG/G|.

    Args:
        constraint_level: 'conservative', 'moderate', or 'strict'

    Returns:
        Tuple of (bound, 1σ uncertainty)
    """
    if constraint_level == "strict":
        # From ⁴He abundance + D/H
        return 0.05, 0.02
    elif constraint_level == "moderate":
        return 0.08, 0.03
    else:  # conservative
        return 0.10, 0.05


def check_bbn_constraint(
    solution: Optional[BackgroundSolution] = None,
    G_eff_bbn: Optional[float] = None,
    G_eff_today: Optional[float] = None,
    params: Optional[HRCParameters] = None,
    constraint_level: str = "conservative",
) -> BBNConstraint:
    """Check BBN constraint on G_eff variation.

    The constraint is on the relative change in G_eff between
    BBN epoch and today:
        |G_eff(z_BBN)/G_eff(0) - 1| < bound

    Args:
        solution: Background solution (preferred)
        G_eff_bbn: G_eff/G at BBN (alternative to solution)
        G_eff_today: G_eff/G today (alternative to solution)
        params: HRC parameters (for computing G_eff if needed)
        constraint_level: 'conservative', 'moderate', or 'strict'

    Returns:
        BBNConstraint result
    """
    z_bbn = _bbn_redshift()
    bound, sigma = _bbn_G_eff_bound(constraint_level)

    # Get G_eff values
    if solution is not None:
        # Interpolate from solution
        G_eff_today = solution.G_eff_at(0.0)

        # BBN redshift may be beyond solution range
        if z_bbn > solution.z[-1]:
            # Extrapolate or use endpoint
            G_eff_bbn = solution.G_eff_ratio[-1]
            z_bbn = solution.z[-1]
        else:
            G_eff_bbn = solution.G_eff_at(z_bbn)
    elif G_eff_bbn is None or G_eff_today is None:
        if params is None:
            raise ValueError("Must provide solution, G_eff values, or params")

        # Compute G_eff directly
        xi = params.xi
        phi_0 = params.phi_0

        # Assume simple evolution: φ(z) ≈ φ₀ (slow evolution during BBN)
        # This is an approximation; full solution should be used
        G_eff_today = 1.0 / (1.0 - 8 * np.pi * xi * phi_0)
        G_eff_bbn = G_eff_today  # Approximation for slow-roll

    # Check for NaN or invalid values
    if np.isnan(G_eff_bbn) or np.isnan(G_eff_today):
        return BBNConstraint(
            allowed=False,
            value=np.inf,
            bound=bound,
            sigma_margin=-np.inf,
            z_bbn=z_bbn,
            G_eff_bbn=G_eff_bbn,
            G_eff_today=G_eff_today,
            message="G_eff is NaN",
        )

    # Compute relative change
    if abs(G_eff_today) < 1e-10:
        Delta_G = np.inf
    else:
        Delta_G = abs(G_eff_bbn / G_eff_today - 1.0)

    # Check constraint
    allowed = Delta_G < bound
    sigma_margin = (bound - Delta_G) / sigma if sigma > 0 else np.inf

    if allowed:
        message = (
            f"BBN constraint satisfied: |ΔG/G| = {Delta_G:.3f} < {bound:.2f} "
            f"({sigma_margin:.1f}σ margin)"
        )
    else:
        message = (
            f"BBN constraint VIOLATED: |ΔG/G| = {Delta_G:.3f} > {bound:.2f} "
            f"({-sigma_margin:.1f}σ excess)"
        )

    return BBNConstraint(
        allowed=allowed,
        value=Delta_G,
        bound=bound,
        sigma_margin=sigma_margin,
        z_bbn=z_bbn,
        G_eff_bbn=G_eff_bbn,
        G_eff_today=G_eff_today,
        message=message,
    )


def compute_bbn_abundances(
    G_eff_ratio: float,
    Omega_b_h2: float = 0.02237,
) -> dict:
    """Compute approximate primordial abundances with modified G_eff.

    This uses simple scaling relations. For accurate predictions,
    use a full BBN code like AlterBBN or PArthENoPE.

    Args:
        G_eff_ratio: G_eff/G at BBN
        Omega_b_h2: Baryon density parameter

    Returns:
        Dictionary with abundance predictions
    """
    # Standard BBN values (Planck 2018 + theory)
    Yp_std = 0.2471  # ⁴He mass fraction
    DH_std = 2.57e-5  # D/H ratio
    He3H_std = 1.0e-5  # ³He/H ratio
    Li7H_std = 5.0e-10  # ⁷Li/H ratio

    # Scaling with G_eff
    # Yp increases with expansion rate (less time for n→p conversion)
    # Yp ∝ 1 + 0.4 * ΔG/G approximately
    Delta_G = G_eff_ratio - 1.0

    Yp = Yp_std * (1 + 0.4 * Delta_G)

    # D/H decreases with faster expansion (less time for D burning)
    DH = DH_std * (1 - 0.6 * Delta_G)

    # ³He/H approximately constant
    He3H = He3H_std

    # ⁷Li/H more complex dependence
    Li7H = Li7H_std * (1 - 0.3 * Delta_G)

    return {
        "Yp": Yp,
        "D/H": DH,
        "3He/H": He3H,
        "7Li/H": Li7H,
        "Delta_G": Delta_G,
        "standard_Yp": Yp_std,
        "standard_D/H": DH_std,
    }
