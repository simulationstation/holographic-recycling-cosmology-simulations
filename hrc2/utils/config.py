"""Performance configuration for HRC 2.0 parallel scans."""

from dataclasses import dataclass


@dataclass
class PerformanceConfig:
    """Configuration for parallel model evaluation.

    Attributes:
        n_workers: Number of parallel workers for ProcessPoolExecutor
        atol: Absolute tolerance for ODE solver
        rtol: Relative tolerance for ODE solver
        max_time_per_model: Maximum time in seconds per model evaluation
        coarse_grid_factor: Factor to reduce grid resolution (1 = full)
    """
    n_workers: int = 10
    atol: float = 1e-8
    rtol: float = 1e-6
    max_time_per_model: float = 30.0
    coarse_grid_factor: int = 1
