"""Plotting functions for G_eff evolution in HRC."""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from ..utils.config import HRCParameters
from ..background import BackgroundSolution
from ..effective_gravity import EffectiveGravity, GeffEvolution


def plot_geff_evolution(
    solution: BackgroundSolution,
    params: HRCParameters,
    ax=None,
    show_constraints: bool = True,
    z_max: float = 1200.0,
    figsize: Tuple[float, float] = (10, 6),
):
    """Plot G_eff/G evolution with redshift.

    Args:
        solution: Background cosmology solution
        params: HRC parameters
        ax: Matplotlib axes (creates new figure if None)
        show_constraints: Show BBN and solar system constraints
        z_max: Maximum redshift to plot
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Compute G_eff evolution
    eff_grav = EffectiveGravity(params)
    evolution = eff_grav.compute_evolution(solution)

    # Filter to z_max
    mask = solution.z <= z_max
    z = solution.z[mask]
    G_eff = evolution.G_eff_ratio[mask]

    # Main plot
    ax.plot(z, G_eff, 'b-', lw=2, label=r'$G_{\rm eff}/G$')
    ax.axhline(y=1.0, color='k', ls='--', lw=1, alpha=0.5, label=r'GR ($G_{\rm eff} = G$)')

    # Constraint regions
    if show_constraints:
        # Solar system: |ΔG/G| < 0.02
        ax.fill_between([0, 0.01], 0.98, 1.02, color='green', alpha=0.2,
                       label='Solar system constraint')

        # BBN: z ~ 10^8-10^9, |ΔG/G| < 0.1
        if z_max > 1e6:
            ax.axvspan(1e8, 1e9, color='orange', alpha=0.1, label='BBN epoch')

        # Recombination
        ax.axvline(x=1089, color='red', ls=':', lw=1, alpha=0.7, label='Recombination')

    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$G_{\rm eff}/G$', fontsize=12)
    ax.set_title(f'Effective Gravitational Coupling (ξ={params.xi:.3f}, φ₀={params.phi_0:.2f})',
                fontsize=12)

    if z_max > 100:
        ax.set_xscale('log')
    ax.set_xlim(max(0.01, z[0]), z_max)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_hubble_tension_resolution(
    solution: BackgroundSolution,
    params: HRCParameters,
    ax=None,
    H0_true: float = 70.0,
    figsize: Tuple[float, float] = (10, 6),
):
    """Plot H₀ predictions from different probes.

    Args:
        solution: Background cosmology solution
        params: HRC parameters
        ax: Matplotlib axes
        H0_true: Fiducial true H₀
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get G_eff values
    G_eff_0 = solution.G_eff_at(0.0)
    G_eff_cmb = solution.G_eff_at(1089.0)

    # Compute H₀ predictions
    H0_local = H0_true * np.sqrt(G_eff_0)
    Delta_G = (G_eff_0 - G_eff_cmb) / G_eff_cmb
    H0_cmb = H0_true * (1 + 0.4 * Delta_G)

    # Observational data
    probes = ['SH0ES\n(local)', 'TRGB', 'Planck\n(CMB)', 'DESI\nBAO', 'HRC\nLocal', 'HRC\nCMB']
    H0_obs = [73.04, 69.8, 67.36, 67.8, H0_local, H0_cmb]
    H0_err = [1.04, 1.7, 0.54, 1.3, 0, 0]  # No error for predictions

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    x = np.arange(len(probes))
    bars = ax.bar(x, H0_obs, yerr=H0_err, capsize=5, color=colors, alpha=0.7)

    # Add horizontal bands for tension
    ax.axhspan(66.8, 68.0, color='blue', alpha=0.1, label='Planck 1σ')
    ax.axhspan(72.0, 74.1, color='red', alpha=0.1, label='SH0ES 1σ')

    ax.set_xticks(x)
    ax.set_xticklabels(probes, fontsize=10)
    ax.set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_title('Hubble Tension Resolution in HRC', fontsize=12)

    # Annotate HRC predictions
    ax.annotate(f'{H0_local:.1f}', (4, H0_local + 1), ha='center', fontsize=9)
    ax.annotate(f'{H0_cmb:.1f}', (5, H0_cmb + 1), ha='center', fontsize=9)

    ax.set_ylim(60, 80)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def plot_geff_parameter_space(
    xi_range: Tuple[float, float] = (0.001, 0.1),
    phi0_range: Tuple[float, float] = (0.05, 0.5),
    n_points: int = 50,
    target_Delta_H0: float = 6.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
):
    """Plot parameter space for Hubble tension resolution.

    Args:
        xi_range: Range of ξ values
        phi0_range: Range of φ₀ values
        n_points: Number of grid points per dimension
        target_Delta_H0: Target H₀ difference
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xi_arr = np.logspace(np.log10(xi_range[0]), np.log10(xi_range[1]), n_points)
    phi0_arr = np.linspace(phi0_range[0], phi0_range[1], n_points)

    XI, PHI0 = np.meshgrid(xi_arr, phi0_arr)
    Delta_H0 = np.zeros_like(XI)

    H0_true = 70.0

    for i in range(n_points):
        for j in range(n_points):
            xi = XI[i, j]
            phi0 = PHI0[i, j]

            # G_eff at z=0
            G_eff_0 = 1.0 / (1.0 - 8 * np.pi * xi * phi0)

            # G_eff at CMB (assume slow evolution)
            G_eff_cmb = G_eff_0 * 0.99  # Approximate

            if G_eff_0 > 0 and G_eff_cmb > 0:
                H0_local = H0_true * np.sqrt(G_eff_0)
                dG = (G_eff_0 - G_eff_cmb) / G_eff_cmb
                H0_cmb = H0_true * (1 + 0.4 * dG)
                Delta_H0[i, j] = H0_local - H0_cmb
            else:
                Delta_H0[i, j] = np.nan

    # Plot
    levels = np.linspace(0, 15, 16)
    cs = ax.contourf(XI, PHI0, Delta_H0, levels=levels, cmap='RdYlBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label=r'$\Delta H_0$ [km/s/Mpc]')

    # Target contour
    ax.contour(XI, PHI0, Delta_H0, levels=[5, 6, 7], colors='k', linewidths=2)

    # Stability boundary: 8πξφ₀ < 1
    phi_crit = 1.0 / (8 * np.pi * xi_arr)
    ax.plot(xi_arr, phi_crit, 'r--', lw=2, label=r'$G_{\rm eff}$ divergence')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (field value today)', fontsize=12)
    ax.set_title('Parameter Space for Hubble Tension Resolution', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    return ax
