"""Observational constraints for HRC 2.0.

This module adapts BBN, PPN, and stellar constraints for general scalar-tensor
theories. It provides thin wrappers around the HRC 1.x constraint modules
with appropriate mappings for the generalized coupling F(phi).

Constraints:
1. BBN: |Delta G/G| < 10% at z ~ 4e8 (conservative)
2. PPN: gamma - 1 constrained by Cassini, Lunar Laser Ranging
3. Stellar: Helioseismology, white dwarf cooling, globular clusters
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..theory import ScalarTensorModel, HRC2Parameters
    from ..background import BackgroundSolution


# BBN constraint: |Delta G/G| at z_BBN
Z_BBN = 4e8  # BBN redshift

# BBN bounds on |Delta G/G|
BBN_BOUNDS = {
    "conservative": 0.10,  # 10%
    "moderate": 0.08,      # 8%
    "strict": 0.05,        # 5%
}

# PPN gamma constraint from Cassini: |gamma - 1| < 2.3e-5
PPN_GAMMA_BOUND = 2.3e-5

# G_dot/G constraint from Lunar Laser Ranging: |G_dot/G| < 1e-13 / yr
GDOT_G_BOUND = 1e-13  # per year


@dataclass
class HRC2ConstraintResult:
    """Result of constraint analysis for HRC 2.0.

    Attributes:
        bbn_allowed: Passes BBN constraint
        ppn_allowed: Passes PPN constraints
        stellar_allowed: Passes stellar constraints
        all_allowed: Passes all constraints

        delta_G_bbn: |Delta G/G| at BBN epoch
        ppn_gamma: PPN gamma parameter at z=0
        Gdot_G: |dG/dt|/G at z=0

        bbn_bound: BBN bound used
        constraint_level: 'conservative', 'moderate', or 'strict'
    """
    bbn_allowed: bool
    ppn_allowed: bool
    stellar_allowed: bool
    all_allowed: bool

    delta_G_bbn: float = 0.0
    ppn_gamma: float = 1.0
    Gdot_G: float = 0.0

    bbn_bound: float = 0.10
    constraint_level: str = "conservative"


def check_bbn_constraint_hrc2(
    solution: "BackgroundSolution",
    model: "ScalarTensorModel",
    constraint_level: str = "conservative",
) -> Tuple[bool, float]:
    """Check BBN constraint on G_eff variation.

    BBN requires |G_eff(z_BBN) - G_N| / G_N < bound

    Since G_eff/G_N = 1/F(phi), this becomes:
        |1/F(phi_BBN) - 1| < bound
        |F(phi_BBN) - 1| / F(phi_BBN) < bound (approximately)

    For small deviations:
        |F - 1| < bound (in M_pl^2 = 1 units)

    More precisely, we check the fractional change:
        |G_eff(z_BBN)/G_eff(z=0) - 1| < bound

    Args:
        solution: BackgroundSolution from integration
        model: ScalarTensorModel instance
        constraint_level: 'conservative', 'moderate', or 'strict'

    Returns:
        Tuple of (is_allowed, delta_G_value)
    """
    bound = BBN_BOUNDS.get(constraint_level, 0.10)

    if not solution.success or not solution.geff_valid:
        return False, float('inf')

    # Get G_eff at z=0 and z_max (proxy for high z)
    # Ideally we'd extrapolate to z_BBN, but for now use the available range
    G_eff_0 = solution.G_eff_ratio[0]  # z=0
    G_eff_max = solution.G_eff_ratio[-1]  # z=z_max

    if G_eff_0 <= 0 or np.isnan(G_eff_0):
        return False, float('inf')
    if G_eff_max <= 0 or np.isnan(G_eff_max):
        return False, float('inf')

    # Fractional change: |G_eff(z_max) - G_eff(0)| / G_eff(0)
    delta_G = abs(G_eff_max - G_eff_0) / G_eff_0

    # Also check absolute deviation from 1
    delta_from_unity = max(abs(G_eff_0 - 1.0), abs(G_eff_max - 1.0))

    # Use the larger of the two as the constraint measure
    delta_G_effective = max(delta_G, delta_from_unity)

    is_allowed = delta_G_effective < bound

    return is_allowed, delta_G_effective


def check_ppn_constraints_hrc2(
    solution: "BackgroundSolution",
    model: "ScalarTensorModel",
    params: Optional["HRC2Parameters"] = None,
) -> Tuple[bool, float, float]:
    """Check PPN constraints on scalar-tensor theory.

    The PPN parameter gamma for scalar-tensor theories is:
        gamma = (omega_BD + 1) / (omega_BD + 2)

    where omega_BD = F * Z / (dF/dphi)^2 - 3/2

    Cassini constraint: |gamma - 1| < 2.3e-5

    Also check G_dot/G from Lunar Laser Ranging.

    Args:
        solution: BackgroundSolution from integration
        model: ScalarTensorModel instance
        params: Optional HRC2Parameters

    Returns:
        Tuple of (is_allowed, gamma_value, Gdot_G_value)
    """
    if not solution.success or not solution.geff_valid:
        return False, 0.0, float('inf')

    # Get phi at z=0
    phi_0 = solution.phi[0]
    phi_dot_0 = solution.phi_dot[0]

    # Compute effective Brans-Dicke parameter
    F = model.F(phi_0)
    Z = model.Z(phi_0)
    dF = model.dF_dphi(phi_0)

    if F <= 0 or abs(dF) < 1e-15:
        # dF = 0 means GR limit, gamma = 1
        gamma = 1.0
        omega_BD = float('inf')
    else:
        omega_BD = F * Z / (dF**2) - 1.5
        if omega_BD == float('inf') or omega_BD > 1e10:
            gamma = 1.0
        else:
            gamma = (omega_BD + 1) / (omega_BD + 2)

    # Check gamma constraint
    gamma_ok = abs(gamma - 1.0) < PPN_GAMMA_BOUND

    # Compute G_dot/G
    # G_eff ~ 1/F, so G_dot/G = -F_dot/F = -dF/dphi * phi_dot / F
    if F > 0:
        Gdot_G = abs(-dF * phi_dot_0 / F)
    else:
        Gdot_G = float('inf')

    # Convert to per year units (very rough - need proper time conversion)
    # For now, use dimensionless comparison
    # H0 ~ 70 km/s/Mpc ~ 2.3e-18 / s ~ 7e-11 / yr
    # Gdot_G in code units is per Hubble time ~ H0
    # To convert to per year: Gdot_G * H0 * (yr in s) / (s in Hubble^-1)
    # Approximate: Gdot_G_per_year ~ Gdot_G * 1e-10

    Gdot_G_per_year = Gdot_G * 1e-10  # Very rough conversion

    Gdot_ok = Gdot_G_per_year < GDOT_G_BOUND

    is_allowed = gamma_ok and Gdot_ok

    return is_allowed, gamma, Gdot_G_per_year


def check_stellar_constraints_hrc2(
    solution: "BackgroundSolution",
    model: "ScalarTensorModel",
) -> Tuple[bool, float]:
    """Check stellar constraints on G variation.

    Stellar constraints come from:
    1. Helioseismology: |Delta G/G| < 0.04 over solar lifetime (~4.5 Gyr)
    2. White dwarf cooling: consistent G over ~10 Gyr
    3. Globular cluster ages: G didn't vary much over ~12 Gyr

    We use a conservative bound on |G_eff/G_N - 1| at z=0.

    Args:
        solution: BackgroundSolution from integration
        model: ScalarTensorModel instance

    Returns:
        Tuple of (is_allowed, delta_G_stellar)
    """
    if not solution.success or not solution.geff_valid:
        return False, float('inf')

    # Stellar constraint: G_eff at z=0 should be close to G_N
    G_eff_0 = solution.G_eff_ratio[0]

    if G_eff_0 <= 0 or np.isnan(G_eff_0):
        return False, float('inf')

    # |G_eff/G_N - 1| should be small
    delta_G_stellar = abs(G_eff_0 - 1.0)

    # Conservative bound: 10% deviation from G_N
    stellar_bound = 0.10

    is_allowed = delta_G_stellar < stellar_bound

    return is_allowed, delta_G_stellar


def check_all_constraints_hrc2(
    solution: "BackgroundSolution",
    model: "ScalarTensorModel",
    params: Optional["HRC2Parameters"] = None,
    constraint_level: str = "conservative",
) -> HRC2ConstraintResult:
    """Check all observational constraints.

    Args:
        solution: BackgroundSolution from integration
        model: ScalarTensorModel instance
        params: Optional HRC2Parameters
        constraint_level: 'conservative', 'moderate', or 'strict'

    Returns:
        HRC2ConstraintResult with all constraint checks
    """
    # BBN constraint
    bbn_allowed, delta_G_bbn = check_bbn_constraint_hrc2(
        solution, model, constraint_level
    )

    # PPN constraints
    ppn_allowed, gamma, Gdot_G = check_ppn_constraints_hrc2(
        solution, model, params
    )

    # Stellar constraints
    stellar_allowed, delta_G_stellar = check_stellar_constraints_hrc2(
        solution, model
    )

    # Combined
    all_allowed = bbn_allowed and ppn_allowed and stellar_allowed

    return HRC2ConstraintResult(
        bbn_allowed=bbn_allowed,
        ppn_allowed=ppn_allowed,
        stellar_allowed=stellar_allowed,
        all_allowed=all_allowed,
        delta_G_bbn=delta_G_bbn,
        ppn_gamma=gamma,
        Gdot_G=Gdot_G,
        bbn_bound=BBN_BOUNDS.get(constraint_level, 0.10),
        constraint_level=constraint_level,
    )


def estimate_delta_H0(delta_G_over_G: float) -> float:
    """Estimate Hubble tension contribution from G_eff variation.

    The Hubble parameter scales as H^2 ~ G_eff * rho, so:
        Delta H / H ~ (1/2) * Delta G / G

    For the Hubble tension (~5-7 km/s/Mpc out of ~70 km/s/Mpc):
        Delta H / H ~ 7-10%

    This requires Delta G / G ~ 14-20%.

    Rough conversion:
        Delta H0 (km/s/Mpc) ~ 35 * |Delta G/G|

    Args:
        delta_G_over_G: Fractional G_eff variation

    Returns:
        Estimated Delta H0 in km/s/Mpc
    """
    return 35.0 * abs(delta_G_over_G)
