"""Parameter space scanning for HRC model.

This module provides systematic exploration of the (xi, phi_0) parameter space,
classifying each point based on G_eff validity and Hubble tension resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..utils.numerics import compute_critical_phi, check_geff_validity
from ..background import BackgroundCosmology, BackgroundSolution
from ..effective_gravity import compute_hubble_tension
from ..potentials import (
    Potential,
    QuadraticPotential,
    PlateauPotential,
    SymmetronPotential,
    ExponentialPotential,
    PotentialParams,
    get_potential,
    POTENTIAL_REGISTRY,
)


class PointClassification(Enum):
    """Classification of a parameter space point."""

    INVALID = "invalid"  # G_eff diverges before z_max
    VALID_NO_TENSION = "valid_but_no_tension"  # G_eff valid, but doesn't resolve tension
    VALID_RESOLVES = "valid_and_resolves_tension"  # G_eff valid and resolves tension


@dataclass
class ScanPoint:
    """Result for a single parameter space point."""

    xi: float
    phi_0: float
    classification: PointClassification

    # Details
    geff_valid: bool
    geff_divergence_z: Optional[float] = None
    phi_critical: Optional[float] = None

    # Hubble tension (only computed if G_eff is valid)
    H0_local: Optional[float] = None
    H0_cmb: Optional[float] = None
    Delta_H0: Optional[float] = None
    resolves_tension: Optional[bool] = None

    # G_eff values
    G_eff_0: Optional[float] = None  # G_eff/G at z=0
    G_eff_cmb: Optional[float] = None  # G_eff/G at z=1089


@dataclass
class ParameterScanResult:
    """Full result of a parameter space scan."""

    xi_grid: NDArray[np.floating]
    phi_0_grid: NDArray[np.floating]
    classification: NDArray  # 2D array of PointClassification values

    # 2D arrays for detailed results
    geff_valid: NDArray[np.bool_]
    geff_divergence_z: NDArray[np.floating]
    Delta_H0: NDArray[np.floating]
    G_eff_0: NDArray[np.floating]
    G_eff_cmb: NDArray[np.floating]

    # Summary statistics
    n_invalid: int = 0
    n_valid_no_tension: int = 0
    n_valid_resolves: int = 0

    # All individual scan points
    points: List[ScanPoint] = field(default_factory=list)

    def get_valid_region(self) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get (xi, phi_0) coordinates of valid points."""
        valid_mask = self.geff_valid
        xi_mesh, phi_mesh = np.meshgrid(self.xi_grid, self.phi_0_grid, indexing='ij')
        return xi_mesh[valid_mask], phi_mesh[valid_mask]

    def get_tension_resolving_region(
        self,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get (xi, phi_0) coordinates of tension-resolving points."""
        resolves_mask = np.array(
            [[p == PointClassification.VALID_RESOLVES for p in row]
             for row in self.classification]
        )
        xi_mesh, phi_mesh = np.meshgrid(self.xi_grid, self.phi_0_grid, indexing='ij')
        return xi_mesh[resolves_mask], phi_mesh[resolves_mask]


def scan_parameter_space(
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 20,
    n_phi_0: int = 20,
    z_max: float = 1100.0,
    z_points: int = 500,
    tension_threshold: float = 3.0,
    verbose: bool = True,
    h: float = 0.7,
    geff_epsilon: float = 0.01,
    potential: Optional[Union[Potential, PotentialParams, str]] = None,
) -> ParameterScanResult:
    """Scan the (xi, phi_0) parameter space systematically.

    For each point, the function:
    1. Constructs HRCParameters
    2. Integrates background cosmology to z_max
    3. Checks if G_eff stays valid throughout
    4. Computes Hubble tension resolution if valid

    Args:
        xi_range: (min, max) for non-minimal coupling
        phi_0_range: (min, max) for initial scalar field
        n_xi: Number of xi grid points
        n_phi_0: Number of phi_0 grid points
        z_max: Maximum redshift for integration
        z_points: Number of points for integration
        tension_threshold: Minimum Delta_H0 (km/s/Mpc) to count as resolving tension
        verbose: Print progress
        h: Hubble constant parameter (H0 = 100*h km/s/Mpc)
        geff_epsilon: Safety margin for G_eff divergence
        potential: Scalar field potential. Can be:
            - None: uses default QuadraticPotential
            - str: potential type name (e.g., "quadratic", "plateau", "symmetron", "exponential")
            - PotentialParams: parameters for get_potential()
            - Potential: a concrete Potential instance

    Returns:
        ParameterScanResult with full classification results
    """
    # Resolve potential
    resolved_potential = _resolve_potential(potential)

    xi_grid = np.linspace(xi_range[0], xi_range[1], n_xi)
    phi_0_grid = np.linspace(phi_0_range[0], phi_0_range[1], n_phi_0)

    # Initialize result arrays
    classification = np.empty((n_xi, n_phi_0), dtype=object)
    geff_valid = np.zeros((n_xi, n_phi_0), dtype=bool)
    geff_divergence_z = np.full((n_xi, n_phi_0), np.nan)
    Delta_H0 = np.full((n_xi, n_phi_0), np.nan)
    G_eff_0 = np.full((n_xi, n_phi_0), np.nan)
    G_eff_cmb = np.full((n_xi, n_phi_0), np.nan)

    points = []
    n_invalid = 0
    n_valid_no_tension = 0
    n_valid_resolves = 0

    total_points = n_xi * n_phi_0
    processed = 0

    for i, xi in enumerate(xi_grid):
        for j, phi_0 in enumerate(phi_0_grid):
            processed += 1

            if verbose and processed % max(1, total_points // 20) == 0:
                pct = 100 * processed / total_points
                print(f"  Scanning: {pct:.0f}% ({processed}/{total_points})")

            point = _scan_single_point(
                xi=xi,
                phi_0=phi_0,
                z_max=z_max,
                z_points=z_points,
                tension_threshold=tension_threshold,
                h=h,
                geff_epsilon=geff_epsilon,
                potential=resolved_potential,
            )

            # Store results
            classification[i, j] = point.classification
            geff_valid[i, j] = point.geff_valid

            if point.geff_divergence_z is not None:
                geff_divergence_z[i, j] = point.geff_divergence_z

            if point.Delta_H0 is not None:
                Delta_H0[i, j] = point.Delta_H0

            if point.G_eff_0 is not None:
                G_eff_0[i, j] = point.G_eff_0

            if point.G_eff_cmb is not None:
                G_eff_cmb[i, j] = point.G_eff_cmb

            points.append(point)

            # Count classifications
            if point.classification == PointClassification.INVALID:
                n_invalid += 1
            elif point.classification == PointClassification.VALID_NO_TENSION:
                n_valid_no_tension += 1
            else:
                n_valid_resolves += 1

    if verbose:
        print(f"\nScan complete!")
        print(f"  Invalid (G_eff diverges): {n_invalid} ({100*n_invalid/total_points:.1f}%)")
        print(f"  Valid but no tension: {n_valid_no_tension} ({100*n_valid_no_tension/total_points:.1f}%)")
        print(f"  Valid and resolves tension: {n_valid_resolves} ({100*n_valid_resolves/total_points:.1f}%)")

    return ParameterScanResult(
        xi_grid=xi_grid,
        phi_0_grid=phi_0_grid,
        classification=classification,
        geff_valid=geff_valid,
        geff_divergence_z=geff_divergence_z,
        Delta_H0=Delta_H0,
        G_eff_0=G_eff_0,
        G_eff_cmb=G_eff_cmb,
        n_invalid=n_invalid,
        n_valid_no_tension=n_valid_no_tension,
        n_valid_resolves=n_valid_resolves,
        points=points,
    )


def _scan_single_point(
    xi: float,
    phi_0: float,
    z_max: float,
    z_points: int,
    tension_threshold: float,
    h: float,
    geff_epsilon: float,
    potential: Optional[Potential] = None,
) -> ScanPoint:
    """Scan a single parameter space point.

    Args:
        xi: Non-minimal coupling
        phi_0: Initial scalar field value
        z_max: Maximum redshift for integration
        z_points: Number of integration points
        tension_threshold: Delta_H0 threshold for tension resolution
        h: Hubble parameter
        geff_epsilon: Safety margin for G_eff divergence
        potential: Scalar field potential (or None for default)

    Returns:
        ScanPoint with classification and details
    """
    phi_critical = compute_critical_phi(xi)

    # Quick pre-check: is initial phi already too close to critical?
    initial_check = check_geff_validity(phi_0, xi, epsilon=geff_epsilon)
    if not initial_check.valid:
        return ScanPoint(
            xi=xi,
            phi_0=phi_0,
            classification=PointClassification.INVALID,
            geff_valid=False,
            geff_divergence_z=0.0,
            phi_critical=phi_critical,
        )

    # Try to construct and solve
    try:
        params = HRCParameters(xi=xi, phi_0=phi_0, h=h)
        valid, errors = params.validate()
        if not valid:
            return ScanPoint(
                xi=xi,
                phi_0=phi_0,
                classification=PointClassification.INVALID,
                geff_valid=False,
                geff_divergence_z=0.0,
                phi_critical=phi_critical,
            )

        cosmo = BackgroundCosmology(params, potential=potential, geff_epsilon=geff_epsilon)
        solution = cosmo.solve(z_max=z_max, z_points=z_points)

        if not solution.geff_valid:
            return ScanPoint(
                xi=xi,
                phi_0=phi_0,
                classification=PointClassification.INVALID,
                geff_valid=False,
                geff_divergence_z=solution.geff_divergence_z,
                phi_critical=phi_critical,
            )

        # G_eff is valid - compute Hubble tension
        tension = compute_hubble_tension(solution, params)

        G_eff_0 = solution.G_eff_at(0.0) if solution.success else None
        G_eff_cmb = solution.G_eff_at(1089.0) if solution.success and z_max >= 1089 else None

        if tension["valid"]:
            H0_local = tension["H0_local"]
            H0_cmb = tension["H0_cmb"]
            Delta_H0 = tension["Delta_H0"]
            resolves = Delta_H0 >= tension_threshold

            classification = (
                PointClassification.VALID_RESOLVES
                if resolves
                else PointClassification.VALID_NO_TENSION
            )

            return ScanPoint(
                xi=xi,
                phi_0=phi_0,
                classification=classification,
                geff_valid=True,
                phi_critical=phi_critical,
                H0_local=H0_local,
                H0_cmb=H0_cmb,
                Delta_H0=Delta_H0,
                resolves_tension=resolves,
                G_eff_0=G_eff_0,
                G_eff_cmb=G_eff_cmb,
            )
        else:
            # Valid G_eff but couldn't compute tension
            return ScanPoint(
                xi=xi,
                phi_0=phi_0,
                classification=PointClassification.VALID_NO_TENSION,
                geff_valid=True,
                phi_critical=phi_critical,
                G_eff_0=G_eff_0,
                G_eff_cmb=G_eff_cmb,
            )

    except Exception as e:
        # Any error means invalid
        return ScanPoint(
            xi=xi,
            phi_0=phi_0,
            classification=PointClassification.INVALID,
            geff_valid=False,
            phi_critical=phi_critical,
        )


def _resolve_potential(
    potential: Optional[Union[Potential, PotentialParams, str]]
) -> Optional[Potential]:
    """Resolve potential argument to a Potential instance.

    Args:
        potential: Can be None, string name, PotentialParams, or Potential

    Returns:
        Resolved Potential instance or None for default
    """
    if potential is None:
        return None
    elif isinstance(potential, str):
        # String name - create with default params
        if potential not in POTENTIAL_REGISTRY:
            raise ValueError(
                f"Unknown potential type: {potential}. "
                f"Available: {list(POTENTIAL_REGISTRY.keys())}"
            )
        return POTENTIAL_REGISTRY[potential]()
    elif isinstance(potential, PotentialParams):
        return get_potential(potential)
    elif isinstance(potential, Potential):
        return potential
    else:
        raise TypeError(f"Invalid potential type: {type(potential)}")


def compute_validity_boundary(
    xi_range: Tuple[float, float] = (0.01, 0.1),
    n_xi: int = 50,
    z_max: float = 1100.0,
    h: float = 0.7,
    geff_epsilon: float = 0.01,
    verbose: bool = True,
    potential: Optional[Union[Potential, PotentialParams, str]] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the boundary curve between valid and invalid regions.

    For each xi value, finds the maximum phi_0 that keeps G_eff valid.

    Args:
        xi_range: Range of xi values to scan
        n_xi: Number of xi points
        z_max: Maximum redshift for integration
        h: Hubble parameter
        geff_epsilon: Safety margin for G_eff
        verbose: Print progress
        potential: Scalar field potential (or None for default)

    Returns:
        Tuple of (xi_values, max_phi_0_values) for the boundary
    """
    resolved_potential = _resolve_potential(potential)

    xi_values = np.linspace(xi_range[0], xi_range[1], n_xi)
    max_phi_0 = np.zeros(n_xi)

    for i, xi in enumerate(xi_values):
        if verbose and i % max(1, n_xi // 10) == 0:
            print(f"  Computing boundary: {100*i/n_xi:.0f}%")

        # Binary search for maximum valid phi_0
        phi_critical = compute_critical_phi(xi)

        # Start with a reasonable range
        phi_low = 0.0
        phi_high = min(phi_critical * 0.9, 1.0)  # Don't go above critical

        for _ in range(20):  # Binary search iterations
            phi_mid = (phi_low + phi_high) / 2

            if phi_mid < 1e-6:
                break

            point = _scan_single_point(
                xi=xi,
                phi_0=phi_mid,
                z_max=z_max,
                z_points=200,
                tension_threshold=0.0,
                h=h,
                geff_epsilon=geff_epsilon,
                potential=resolved_potential,
            )

            if point.geff_valid:
                phi_low = phi_mid
            else:
                phi_high = phi_mid

            if abs(phi_high - phi_low) < 0.001:
                break

        max_phi_0[i] = phi_low

    if verbose:
        print("  Boundary computation complete!")

    return xi_values, max_phi_0
