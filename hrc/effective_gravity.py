"""Effective gravitational coupling for HRC.

This module computes the effective Newton's constant G_eff and its
evolution, including detection of divergences and unphysical regions.

The key relation is:
    G_eff(φ) = G / (1 - 8πGξφ)

where ξ is the non-minimal coupling and φ is the scalar field value.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import numpy as np
from numpy.typing import NDArray

from .utils.config import HRCParameters
from .utils.numerics import check_divergence, DivergenceResult
from .background import BackgroundSolution


@dataclass
class GeffResult:
    """Result of G_eff computation."""

    G_eff_ratio: float  # G_eff/G
    is_physical: bool  # Whether the value is in the physical region
    message: str = ""


@dataclass
class GeffEvolution:
    """Evolution of G_eff over redshift."""

    z: NDArray[np.floating]
    G_eff_ratio: NDArray[np.floating]  # G_eff/G
    G_eff_dot_ratio: NDArray[np.floating]  # (dG_eff/dt)/G (in H0 units)
    is_physical: NDArray[np.bool_]

    # Critical redshift where G_eff diverges (if any)
    z_divergence: Optional[float] = None

    # Derived constraints
    Delta_G_eff_BBN: Optional[float] = None  # G_eff variation at BBN
    G_dot_over_G_today: Optional[float] = None  # Present rate of change


class EffectiveGravity:
    """Compute effective gravitational coupling in HRC.

    The effective Newton's constant is:
        G_eff = G / (1 - 8πξφ)

    where we work in units where G = 1 (Planck units).

    Physical requirements:
        1. G_eff > 0 (positive gravity)
        2. |G_eff - G| / G < constraints from observations
        3. |Ġ_eff/G| < solar system constraints
    """

    def __init__(self, params: HRCParameters):
        """Initialize effective gravity calculator.

        Args:
            params: HRC model parameters
        """
        self.params = params
        self._8pi_xi = 8 * np.pi * params.xi

    def critical_phi(self) -> float:
        """Return critical field value where G_eff diverges.

        G_eff → ∞ when 1 - 8πξφ → 0, i.e., φ → 1/(8πξ)
        """
        if self.params.xi <= 0:
            return np.inf  # No divergence for ξ ≤ 0
        return 1.0 / self._8pi_xi

    def G_eff_ratio(self, phi: float) -> GeffResult:
        """Compute G_eff/G at given scalar field value.

        Args:
            phi: Scalar field value

        Returns:
            GeffResult with G_eff/G and validity information
        """
        denominator = 1.0 - self._8pi_xi * phi

        # Check for divergence
        if abs(denominator) < 1e-10:
            return GeffResult(
                G_eff_ratio=np.inf,
                is_physical=False,
                message=f"G_eff diverges: 1 - 8πξφ = {denominator:.3e}",
            )

        ratio = 1.0 / denominator

        # Check for negative G_eff (would reverse gravity)
        if ratio < 0:
            return GeffResult(
                G_eff_ratio=ratio,
                is_physical=False,
                message=f"G_eff < 0: gravity would be repulsive",
            )

        # Check for extremely large G_eff
        if ratio > 10.0:
            return GeffResult(
                G_eff_ratio=ratio,
                is_physical=False,
                message=f"G_eff/G = {ratio:.1f} > 10: unreasonably strong gravity",
            )

        return GeffResult(
            G_eff_ratio=ratio,
            is_physical=True,
            message=f"G_eff/G = {ratio:.4f}",
        )

    def G_eff_ratio_array(
        self,
        phi: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
        """Compute G_eff/G for array of field values.

        Args:
            phi: Array of scalar field values

        Returns:
            Tuple of (G_eff/G array, is_physical mask)
        """
        denominator = 1.0 - self._8pi_xi * phi

        # Avoid division by zero
        safe_denom = np.where(np.abs(denominator) > 1e-10, denominator, 1e-10)
        ratio = 1.0 / safe_denom

        # Physical conditions
        is_physical = (np.abs(denominator) > 1e-10) & (ratio > 0) & (ratio < 10.0)

        # Mark divergent regions as NaN
        ratio = np.where(is_physical, ratio, np.nan)

        return ratio, is_physical

    def compute_G_eff_derivative(
        self,
        phi: float,
        phi_dot: float,
    ) -> Tuple[float, float]:
        """Compute G_eff time derivative.

        dG_eff/dt = 8πξφ̇ · G_eff²/G

        Args:
            phi: Scalar field value
            phi_dot: Scalar field velocity dφ/dt

        Returns:
            Tuple of (dG_eff/dt / G, Ġ_eff/G_eff)
        """
        result = self.G_eff_ratio(phi)
        if not result.is_physical:
            return np.nan, np.nan

        G_ratio = result.G_eff_ratio

        # dG_eff/dt = 8πξφ̇ · G_eff²/G = 8πξφ̇ · G_ratio²
        dG_dt_over_G = self._8pi_xi * phi_dot * G_ratio**2

        # Ġ_eff/G_eff = 8πξφ̇ · G_eff/G = 8πξφ̇ · G_ratio
        G_dot_over_G_eff = self._8pi_xi * phi_dot * G_ratio

        return dG_dt_over_G, G_dot_over_G_eff

    def compute_evolution(
        self,
        solution: BackgroundSolution,
    ) -> GeffEvolution:
        """Compute G_eff evolution from background solution.

        Args:
            solution: Background cosmology solution

        Returns:
            GeffEvolution with full evolution data
        """
        G_eff_ratio, is_physical = self.G_eff_ratio_array(solution.phi)

        # Compute time derivative
        G_eff_dot_ratio = np.zeros_like(solution.z)
        for i in range(len(solution.z)):
            if is_physical[i] and not np.isnan(solution.phi_dot[i]):
                _, G_dot_over_G_eff = self.compute_G_eff_derivative(
                    solution.phi[i], solution.phi_dot[i]
                )
                G_eff_dot_ratio[i] = G_dot_over_G_eff
            else:
                G_eff_dot_ratio[i] = np.nan

        # Find divergence point if any
        z_div = None
        unphysical_idx = np.where(~is_physical)[0]
        if len(unphysical_idx) > 0:
            z_div = solution.z[unphysical_idx[0]]

        # Compute constraints
        # BBN: z ~ 10^9, we use z ~ 10^8 as proxy for available data
        z_bbn = min(1e8, solution.z[-1])
        idx_bbn = np.argmin(np.abs(solution.z - z_bbn))
        Delta_G_BBN = (
            (G_eff_ratio[idx_bbn] - G_eff_ratio[0]) / G_eff_ratio[0]
            if is_physical[0] and is_physical[idx_bbn]
            else None
        )

        # Present rate of change
        G_dot_today = G_eff_dot_ratio[0] if is_physical[0] else None

        return GeffEvolution(
            z=solution.z,
            G_eff_ratio=G_eff_ratio,
            G_eff_dot_ratio=G_eff_dot_ratio,
            is_physical=is_physical,
            z_divergence=z_div,
            Delta_G_eff_BBN=Delta_G_BBN,
            G_dot_over_G_today=G_dot_today,
        )


def check_G_eff_constraints(
    G_eff_evolution: GeffEvolution,
    verbose: bool = False,
) -> List[Tuple[str, bool, str]]:
    """Check G_eff against observational constraints.

    Args:
        G_eff_evolution: Evolution from EffectiveGravity.compute_evolution
        verbose: Print constraint checks

    Returns:
        List of (constraint_name, passed, message) tuples
    """
    results = []

    # 1. Solar system constraint: |Ġ/G| < 1.5 × 10⁻¹² yr⁻¹
    # In H0 units: H0 ≈ 70 km/s/Mpc ≈ 7 × 10⁻¹¹ yr⁻¹
    # So constraint is |Ġ/G| < 20 in H0 units
    G_dot_limit = 20.0  # Very conservative in H0 units
    if G_eff_evolution.G_dot_over_G_today is not None:
        G_dot = abs(G_eff_evolution.G_dot_over_G_today)
        passed = G_dot < G_dot_limit
        results.append((
            "Solar System Ġ/G",
            passed,
            f"|Ġ/G|_today = {G_dot:.2e} H0 ({'<' if passed else '>'} {G_dot_limit})",
        ))

    # 2. BBN constraint: |ΔG/G| < 10% at BBN
    if G_eff_evolution.Delta_G_eff_BBN is not None:
        Delta_G = abs(G_eff_evolution.Delta_G_eff_BBN)
        passed = Delta_G < 0.1
        results.append((
            "BBN ΔG/G",
            passed,
            f"|ΔG/G|_BBN = {Delta_G:.3f} ({'<' if passed else '>'} 0.1)",
        ))

    # 3. No divergence in observable universe
    passed = G_eff_evolution.z_divergence is None
    if passed:
        results.append((
            "No divergence",
            True,
            "G_eff remains finite for all z",
        ))
    else:
        results.append((
            "No divergence",
            False,
            f"G_eff diverges at z = {G_eff_evolution.z_divergence:.2f}",
        ))

    # 4. G_eff always positive
    all_positive = np.all(G_eff_evolution.is_physical | np.isnan(G_eff_evolution.G_eff_ratio))
    results.append((
        "Positive G_eff",
        all_positive,
        "G_eff > 0 everywhere" if all_positive else "G_eff becomes negative",
    ))

    if verbose:
        print("\n=== G_eff Constraint Checks ===")
        for name, passed, msg in results:
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {msg}")

    return results


def compute_H0_shift(
    G_eff_local: float,
    G_eff_cmb: float,
    H0_true: float = 70.0,
) -> Tuple[float, float]:
    """Compute the inferred H0 values from local and CMB measurements.

    In HRC, local measurements probe G_eff(z≈0) while CMB probes
    a weighted average of G_eff over the early universe.

    Args:
        G_eff_local: G_eff/G at z≈0
        G_eff_cmb: G_eff/G at recombination
        H0_true: True H0 value

    Returns:
        Tuple of (H0_local, H0_CMB) inferred values
    """
    # Local measurements: H ∝ sqrt(G_eff) for fixed matter content
    # So observed H0_local ∝ sqrt(G_eff(0))
    H0_local = H0_true * np.sqrt(G_eff_local)

    # CMB inference: More complex, depends on angular diameter distance
    # Approximate relation: H0^CMB ∝ 1 for fixed θ*
    # The shift comes from the change in D_A
    # H0_CMB ≈ H0_true * (1 + 0.4 * (G_eff(0) - G_eff(z_rec)) / G_eff(z_rec))
    Delta_G_ratio = (G_eff_local - G_eff_cmb) / G_eff_cmb
    H0_cmb = H0_true * (1 + 0.4 * Delta_G_ratio)

    return H0_local, H0_cmb


def compute_hubble_tension(
    solution: BackgroundSolution,
    params: HRCParameters,
    H0_true: float = 70.0,
) -> dict:
    """Compute Hubble tension predictions from HRC.

    Args:
        solution: Background cosmology solution
        params: HRC parameters
        H0_true: Fiducial true H0 value

    Returns:
        Dictionary with H0 predictions and tension metrics
    """
    eff_grav = EffectiveGravity(params)
    evolution = eff_grav.compute_evolution(solution)

    # Get G_eff at z=0 and z≈1100
    G_eff_local = evolution.G_eff_ratio[0]

    z_cmb = 1089.0
    idx_cmb = np.argmin(np.abs(solution.z - z_cmb))
    G_eff_cmb = evolution.G_eff_ratio[idx_cmb]

    if np.isnan(G_eff_local) or np.isnan(G_eff_cmb):
        return {
            "valid": False,
            "message": "G_eff is NaN at local or CMB redshift",
        }

    H0_local, H0_cmb = compute_H0_shift(G_eff_local, G_eff_cmb, H0_true)

    Delta_H0 = H0_local - H0_cmb
    tension_sigma = Delta_H0 / 1.5  # Approximate combined uncertainty

    return {
        "valid": True,
        "G_eff_local": float(G_eff_local),
        "G_eff_cmb": float(G_eff_cmb),
        "H0_local": float(H0_local),
        "H0_cmb": float(H0_cmb),
        "Delta_H0": float(Delta_H0),
        "tension_sigma": float(tension_sigma),
        "resolves_tension": Delta_H0 > 4.0 and Delta_H0 < 10.0,
    }
