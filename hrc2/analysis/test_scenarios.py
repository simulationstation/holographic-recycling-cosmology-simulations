"""Test scenario registry for HRC 2.0 coupling family scans.

This module defines standardized test scenarios for exploring different
coupling functions and their ability to address the Hubble tension.

Each scenario specifies:
- Coupling function type and parameters
- Parameter ranges for xi and phi0
- Grid size (keep small, 5x5 for fast testing)
- Maximum redshift z_max
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

from ..theory import CouplingFamily, PotentialType


@dataclass
class TestScenario:
    """Configuration for a single test scenario.

    Attributes:
        id: Unique identifier (e.g., "T01_evap_boundary_plateau")
        description: Human-readable description
        coupling_family: CouplingFamily enum value
        potential_type: PotentialType enum value
        coupling_params: Additional params (mu for plateau_evap, etc.)
        z_max: Maximum redshift for integration
        xi_range: (min, max) for coupling strength
        phi0_range: (min, max) for initial field value
        nx: Number of xi grid points
        nphi: Number of phi0 grid points
    """
    id: str
    description: str
    coupling_family: CouplingFamily
    potential_type: PotentialType = PotentialType.QUADRATIC
    coupling_params: Dict[str, Any] = field(default_factory=dict)
    z_max: float = 1100.0
    xi_range: Tuple[float, float] = (1e-5, 5e-3)
    phi0_range: Tuple[float, float] = (0.0, 0.3)
    nx: int = 5
    nphi: int = 5


# ============================================================================
# Test Scenario Registry
# ============================================================================

TEST_SCENARIOS: List[TestScenario] = [
    # T01: Evaporated-boundary-inspired plateau coupling (PRIORITY)
    TestScenario(
        id="T01_evap_boundary_plateau",
        description="Evaporated-boundary-inspired coupling f(phi) = 1 - exp(-(phi/mu)^2)",
        coupling_family=CouplingFamily.PLATEAU_EVAP,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={"mu": 0.1},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),

    # T02: Standard quadratic coupling baseline
    TestScenario(
        id="T02_quadratic_baseline",
        description="Standard non-minimal coupling f(phi) = phi^2 baseline",
        coupling_family=CouplingFamily.QUADRATIC,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),

    # T03: Exponential coupling
    TestScenario(
        id="T03_exponential_coupling",
        description="Exponential coupling f(phi) = exp(beta*phi/M_pl)",
        coupling_family=CouplingFamily.EXPONENTIAL,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),

    # T04: Linear coupling (HRC 1.x style)
    TestScenario(
        id="T04_linear_coupling",
        description="Linear coupling f(phi) = M_pl^2 - alpha*phi (HRC 1.x style)",
        coupling_family=CouplingFamily.LINEAR,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),

    # T05: Plateau_evap with larger mu (wider transition)
    TestScenario(
        id="T05_evap_wide_transition",
        description="Evaporated-boundary with wider transition (mu=0.3)",
        coupling_family=CouplingFamily.PLATEAU_EVAP,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={"mu": 0.3},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),

    # T06: Plateau_evap with narrower mu (sharper transition)
    TestScenario(
        id="T06_evap_narrow_transition",
        description="Evaporated-boundary with narrower transition (mu=0.05)",
        coupling_family=CouplingFamily.PLATEAU_EVAP,
        potential_type=PotentialType.QUADRATIC,
        coupling_params={"mu": 0.05},
        z_max=1100.0,
        xi_range=(1e-5, 5e-3),
        phi0_range=(0.0, 0.3),
        nx=5,
        nphi=5,
    ),
]


def get_scenario_by_id(scenario_id: str) -> TestScenario:
    """Look up a scenario by its ID.

    Args:
        scenario_id: The scenario ID string

    Returns:
        TestScenario if found

    Raises:
        ValueError: If scenario not found
    """
    for scenario in TEST_SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario ID: {scenario_id}")


def list_scenario_ids() -> List[str]:
    """Return list of all scenario IDs."""
    return [s.id for s in TEST_SCENARIOS]
