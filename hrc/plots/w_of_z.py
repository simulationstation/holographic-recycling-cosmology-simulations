"""Plotting functions for effective dark energy equation of state."""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..background import BackgroundSolution


def compute_effective_w(
    solution: BackgroundSolution,
    params: HRCParameters,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute effective dark energy equation of state w(z).

    An observer assuming ΛCDM would infer an effective dark energy
    equation of state from the modified expansion history.

    w_eff(z) = -1 + (2/3)(1+z) d ln E / dz

    where E(z) = H(z)/H₀.

    Args:
        solution: Background cosmology solution
        params: HRC parameters

    Returns:
        Tuple of (z, w_eff)
    """
    z = solution.z
    H = solution.H

    # E(z) = H(z)/H(0)
    E = H / H[0]

    # d ln E / dz = (1/E) dE/dz
    dE_dz = np.gradient(E, z)
    dlnE_dz = dE_dz / E

    # w_eff(z) = -1 + (2/3)(1+z) d ln E / dz
    w_eff = -1 + (2.0 / 3.0) * (1 + z) * dlnE_dz

    return z, w_eff


def compute_w0_wa(
    solution: BackgroundSolution,
    params: HRCParameters,
) -> Tuple[float, float]:
    """Compute w₀-wₐ parametrization.

    Fits w(a) = w₀ + wₐ(1-a) to the effective w(z).

    Args:
        solution: Background cosmology solution
        params: HRC parameters

    Returns:
        Tuple of (w₀, wₐ)
    """
    z, w_eff = compute_effective_w(solution, params)

    # Fit in low-z regime where CPL is good approximation
    mask = (z > 0.01) & (z < 2.0) & np.isfinite(w_eff)
    if np.sum(mask) < 5:
        return -1.0, 0.0

    z_fit = z[mask]
    w_fit = w_eff[mask]
    a_fit = 1.0 / (1.0 + z_fit)

    # Linear regression: w = w₀ + wₐ(1-a)
    # w = w₀ + wₐ - wₐ a = (w₀ + wₐ) - wₐ a
    X = np.vstack([np.ones_like(a_fit), 1 - a_fit]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, w_fit, rcond=None)
        w0 = coeffs[0]
        wa = coeffs[1]
    except:
        w0, wa = -1.0, 0.0

    return float(w0), float(wa)


def plot_effective_w(
    solution: BackgroundSolution,
    params: HRCParameters,
    ax=None,
    z_max: float = 3.0,
    show_cpl_fit: bool = True,
    figsize: Tuple[float, float] = (10, 6),
):
    """Plot effective dark energy equation of state w(z).

    Args:
        solution: Background cosmology solution
        params: HRC parameters
        ax: Matplotlib axes
        z_max: Maximum redshift to plot
        show_cpl_fit: Show CPL (w₀-wₐ) fit
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    z, w_eff = compute_effective_w(solution, params)

    # Filter to z_max
    mask = z <= z_max
    z_plot = z[mask]
    w_plot = w_eff[mask]

    # Main plot
    ax.plot(z_plot, w_plot, 'b-', lw=2, label=r'HRC $w_{\rm eff}(z)$')

    # ΛCDM reference
    ax.axhline(y=-1.0, color='k', ls='--', lw=1, label=r'$\Lambda$CDM ($w = -1$)')

    # CPL fit
    if show_cpl_fit:
        w0, wa = compute_w0_wa(solution, params)
        a_plot = 1.0 / (1.0 + z_plot)
        w_cpl = w0 + wa * (1 - a_plot)
        ax.plot(z_plot, w_cpl, 'r:', lw=2,
               label=f'CPL fit: $w_0 = {w0:.2f}$, $w_a = {wa:.2f}$')

    # DESI constraints
    ax.fill_between(z_plot, -0.89, -0.77, color='orange', alpha=0.2,
                   label=r'DESI 1$\sigma$ ($w_0$)')

    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$w_{\rm eff}(z)$', fontsize=12)
    ax.set_title('Effective Dark Energy Equation of State', fontsize=12)
    ax.set_xlim(0, z_max)
    ax.set_ylim(-1.5, -0.5)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_w_comparison(
    solutions: List[BackgroundSolution],
    params_list: List[HRCParameters],
    labels: Optional[List[str]] = None,
    ax=None,
    z_max: float = 3.0,
    figsize: Tuple[float, float] = (10, 6),
):
    """Compare w(z) for multiple parameter sets.

    Args:
        solutions: List of background solutions
        params_list: List of HRC parameters
        labels: Labels for each solution
        ax: Matplotlib axes
        z_max: Maximum redshift
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(solutions))]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(solutions)))

    for solution, params, label, color in zip(solutions, params_list, labels, colors):
        z, w_eff = compute_effective_w(solution, params)
        mask = z <= z_max
        ax.plot(z[mask], w_eff[mask], color=color, lw=2, label=label)

    ax.axhline(y=-1.0, color='k', ls='--', lw=1, label=r'$\Lambda$CDM')
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$w_{\rm eff}(z)$', fontsize=12)
    ax.set_title('Effective EoS Comparison', fontsize=12)
    ax.set_xlim(0, z_max)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax
