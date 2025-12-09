"""Phase diagram and stability region plotting for HRC."""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..background import BackgroundSolution
from ..perturbations.stability_checks import (
    check_effective_planck_mass,
    check_no_ghost,
    check_gradient_stability,
)


def plot_phase_diagram(
    solution: BackgroundSolution,
    params: HRCParameters,
    ax=None,
    show_stability: bool = True,
    figsize: Tuple[float, float] = (10, 8),
):
    """Plot phase space trajectory (φ, φ̇).

    Args:
        solution: Background cosmology solution
        params: HRC parameters
        ax: Matplotlib axes
        show_stability: Color by stability
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    phi = solution.phi
    phi_dot = solution.phi_dot
    z = solution.z

    if show_stability:
        # Color by stability
        stable = np.ones(len(phi), dtype=bool)
        for i in range(len(phi)):
            result = check_effective_planck_mass(phi[i], params)
            stable[i] = result.passed

        # Plot stable and unstable regions
        ax.scatter(phi[stable], phi_dot[stable], c=z[stable], cmap='viridis',
                  s=20, alpha=0.6, label='Stable')
        ax.scatter(phi[~stable], phi_dot[~stable], c='red',
                  s=40, alpha=0.8, marker='x', label='Unstable')
    else:
        sc = ax.scatter(phi, phi_dot, c=z, cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Redshift $z$')

    # Mark special points
    ax.plot(phi[0], phi_dot[0], 'ko', ms=10, label='Today (z=0)')
    ax.plot(phi[-1], phi_dot[-1], 'k^', ms=10, label=f'z={z[-1]:.0f}')

    # Critical line: φ_crit = 1/(8πξ)
    phi_crit = 1.0 / (8 * np.pi * params.xi)
    ax.axvline(x=phi_crit, color='red', ls='--', lw=2,
              label=f'$G_{{\\rm eff}}$ divergence ($\\phi = {phi_crit:.2f}$)')

    ax.set_xlabel(r'$\phi$ (scalar field)', fontsize=12)
    ax.set_ylabel(r'$\dot{\phi}$ (field velocity)', fontsize=12)
    ax.set_title(f'Phase Space Trajectory (ξ={params.xi:.3f})', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_stability_region(
    xi_range: Tuple[float, float] = (0.001, 0.2),
    phi_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 100,
    params_template: Optional[HRCParameters] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
):
    """Plot stability region in (ξ, φ) parameter space.

    Args:
        xi_range: Range of ξ values
        phi_range: Range of φ values
        n_points: Grid resolution
        params_template: Template parameters
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if params_template is None:
        params_template = HRCParameters()

    xi_arr = np.linspace(xi_range[0], xi_range[1], n_points)
    phi_arr = np.linspace(phi_range[0], phi_range[1], n_points)

    XI, PHI = np.meshgrid(xi_arr, phi_arr)
    stable = np.zeros_like(XI, dtype=bool)
    M_eff_sq = np.zeros_like(XI)

    for i in range(n_points):
        for j in range(n_points):
            xi = XI[i, j]
            phi = PHI[i, j]

            # Create modified params
            params = HRCParameters(
                xi=xi,
                phi_0=phi,
                h=params_template.h,
                Omega_m=params_template.Omega_m,
            )

            # Check stability
            M_eff_sq[i, j] = (1 - 8 * np.pi * xi * phi) / (8 * np.pi)
            result = check_effective_planck_mass(phi, params)
            stable[i, j] = result.passed

    # Plot stability region
    ax.contourf(XI, PHI, stable.astype(float), levels=[0, 0.5, 1],
               colors=['red', 'green'], alpha=0.3)
    ax.contour(XI, PHI, stable.astype(float), levels=[0.5],
              colors=['black'], linewidths=2)

    # Contours of M_eff²
    cs = ax.contour(XI, PHI, M_eff_sq, levels=[0.01, 0.05, 0.1, 0.2],
                   colors='blue', linestyles='--', alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt=r'$M_{\rm eff}^2 = %.2f$')

    # Divergence line
    phi_crit = 1.0 / (8 * np.pi * xi_arr)
    ax.plot(xi_arr, phi_crit, 'r-', lw=2, label=r'$G_{\rm eff} \to \infty$')

    # Fiducial point
    ax.plot(params_template.xi, params_template.phi_0, 'ko', ms=10,
           label=f'Fiducial (ξ={params_template.xi}, φ₀={params_template.phi_0})')

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi$ (scalar field value)', fontsize=12)
    ax.set_title('Stability Region in Parameter Space', fontsize=12)
    ax.set_xlim(xi_range)
    ax.set_ylim(phi_range)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text labels
    ax.text(0.02, 0.8, 'STABLE\n(attractive gravity)',
           transform=ax.transAxes, fontsize=12, color='green')
    ax.text(0.7, 0.2, 'UNSTABLE\n(repulsive/divergent)',
           transform=ax.transAxes, fontsize=12, color='red')

    return ax


def plot_evolution_stability(
    solution: BackgroundSolution,
    params: HRCParameters,
    ax=None,
    figsize: Tuple[float, float] = (12, 4),
):
    """Plot stability metrics vs redshift.

    Args:
        solution: Background solution
        params: HRC parameters
        ax: Matplotlib axes (or list of 3 axes)
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        axes = ax

    z = solution.z
    n = len(z)

    M_eff_sq = np.zeros(n)
    Q_s = np.zeros(n)
    c_s_sq = np.zeros(n)

    for i in range(n):
        M_eff_sq[i] = (1 - 8 * np.pi * params.xi * solution.phi[i]) / (8 * np.pi)
        Q_s[i] = M_eff_sq[i]  # Simplified
        c_s_sq[i] = 1.0  # Approximate

    # Panel 1: M_eff²
    axes[0].semilogy(z, np.abs(M_eff_sq), 'b-', lw=2)
    axes[0].axhline(y=0, color='r', ls='--')
    axes[0].set_xlabel('Redshift $z$')
    axes[0].set_ylabel(r'$|M_{\rm eff}^2|$')
    axes[0].set_title('Effective Planck Mass')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: G_eff/G
    axes[1].plot(z, solution.G_eff_ratio, 'g-', lw=2)
    axes[1].axhline(y=1, color='k', ls='--')
    axes[1].set_xlabel('Redshift $z$')
    axes[1].set_ylabel(r'$G_{\rm eff}/G$')
    axes[1].set_title('Effective G')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: φ evolution
    axes[2].plot(z, solution.phi, 'm-', lw=2)
    phi_crit = 1.0 / (8 * np.pi * params.xi)
    axes[2].axhline(y=phi_crit, color='r', ls='--', label=r'$\phi_{\rm crit}$')
    axes[2].set_xlabel('Redshift $z$')
    axes[2].set_ylabel(r'$\phi$')
    axes[2].set_title('Scalar Field')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return axes
