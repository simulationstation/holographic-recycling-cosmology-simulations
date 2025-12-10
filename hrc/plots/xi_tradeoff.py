"""Plots for xi-stability-effect tradeoff analysis.

These plots visualize the fundamental tradeoff in HRC:
- Smaller xi -> more stable but smaller effect
- Larger xi -> larger effect but unstable
"""

from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from ..analysis.xi_tradeoff import XiTradeoffResult, scan_xi_tradeoff, find_critical_xi
from ..potentials import QuadraticPotential, PlateauPotential


def plot_max_delta_G_vs_xi(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    show_unstable: bool = True,
    title: Optional[str] = None,
):
    """Plot maximum |ΔG/G| vs xi for stable configurations.

    This is the key diagnostic plot showing the stability-effect tradeoff.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        show_unstable: Mark xi values with no stable configurations
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xi = result.xi_values
    max_dg = result.max_delta_G
    stable_frac = result.stable_fraction

    # Plot max |ΔG/G| for xi values with stable solutions
    has_stable = stable_frac > 0
    ax.plot(xi[has_stable], max_dg[has_stable], 'bo-', lw=2, ms=8,
            label='Max |ΔG/G| (stable solutions)')

    # Mark unstable xi values
    if show_unstable and np.any(~has_stable):
        ax.scatter(xi[~has_stable], np.zeros(np.sum(~has_stable)),
                  marker='x', s=100, c='red', lw=2,
                  label='No stable solutions', zorder=5)

    # Find and mark critical xi
    xi_crit, max_dg_stable = find_critical_xi(result)
    if xi_crit > 0:
        ax.axvline(xi_crit, color='orange', ls='--', lw=2,
                  label=f'$\\xi_{{crit}} \\approx {xi_crit:.1e}$')

    # Reference lines
    ax.axhline(0.1, color='gray', ls=':', alpha=0.5, label='|ΔG/G| = 0.1')
    ax.axhline(0.01, color='gray', ls=':', alpha=0.3)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$ among stable solutions', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Stability-Effect Tradeoff ({result.potential_name} potential)', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.01)

    return ax


def plot_stability_map(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'RdYlGn',
):
    """Plot 2D map of stability and ΔG/G in (xi, phi0) space.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid for plotting
    XI, PHI = np.meshgrid(result.xi_values, result.phi0_values, indexing='ij')

    # Plot ΔG/G with NaN for unstable points
    delta_g = result.delta_G_over_G.copy()

    # Determine color scale
    valid_mask = ~np.isnan(delta_g)
    if np.any(valid_mask):
        vmax = np.nanmax(np.abs(delta_g))
        vmin = -vmax
    else:
        vmin, vmax = -0.1, 0.1

    im = ax.pcolormesh(XI, PHI, delta_g, cmap=cmap, shading='auto',
                       vmin=vmin, vmax=vmax)

    # Mark unstable region boundary
    ax.contour(XI, PHI, result.stable_mask.astype(float),
              levels=[0.5], colors='black', linewidths=2, linestyles='--')

    plt.colorbar(im, ax=ax, label=r'$\Delta G/G = G_{eff}(0) - G_{eff}(z_{rec})$')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_title(f'Stability Map ({result.potential_name} potential)\n'
                f'White/gray = unstable (G_eff diverges before z={result.z_max:.0f})',
                fontsize=12)

    return ax


def plot_delta_G_vs_phi0(
    result: Optional[XiTradeoffResult] = None,
    xi_indices: Optional[list] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
):
    """Plot ΔG/G vs phi0 for selected xi values.

    Args:
        result: XiTradeoffResult (will compute if None)
        xi_indices: Indices of xi values to plot (default: select a few)
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Select xi indices to show
    if xi_indices is None:
        # Pick a few representative xi values
        n_xi = len(result.xi_values)
        xi_indices = [0, n_xi//4, n_xi//2, 3*n_xi//4, n_xi-1]
        xi_indices = [i for i in xi_indices if i < n_xi]

    colors = plt.cm.viridis(np.linspace(0, 1, len(xi_indices)))

    for idx, color in zip(xi_indices, colors):
        xi = result.xi_values[idx]
        delta_g = result.delta_G_over_G[idx, :]
        stable = result.stable_mask[idx, :]

        # Plot stable points
        ax.plot(result.phi0_values[stable], delta_g[stable],
               'o-', color=color, lw=2, ms=6,
               label=f'$\\xi = {xi:.1e}$')

        # Mark unstable points
        if np.any(~stable):
            ax.scatter(result.phi0_values[~stable],
                      np.zeros(np.sum(~stable)),
                      marker='x', color=color, s=50, alpha=0.5)

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_ylabel(r'$\Delta G/G$', fontsize=12)
    ax.set_title(f'Effect Size vs Initial Field ({result.potential_name} potential)\n'
                f'x marks = unstable configurations', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_stable_fraction_vs_xi(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
):
    """Plot fraction of phi0 values that are stable vs xi.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(result.xi_values, 100 * result.stable_fraction, 'go-', lw=2, ms=8)

    # Find critical xi
    xi_crit, _ = find_critical_xi(result)
    if xi_crit > 0:
        ax.axvline(xi_crit, color='orange', ls='--', lw=2,
                  label=f'$\\xi_{{crit}} \\approx {xi_crit:.1e}$')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Fraction of $\phi_0$ values that are stable (%)', fontsize=12)
    ax.set_title(f'Stability vs Coupling Strength ({result.potential_name} potential)',
                fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    return ax


def create_xi_tradeoff_summary_figure(
    result: Optional[XiTradeoffResult] = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
):
    """Create a 4-panel summary figure for xi-tradeoff analysis.

    Args:
        result: XiTradeoffResult (will compute if None)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Max |ΔG/G| vs xi (key plot)
    plot_max_delta_G_vs_xi(result, ax=axes[0, 0])

    # Panel 2: Stable fraction vs xi
    plot_stable_fraction_vs_xi(result, ax=axes[0, 1])

    # Panel 3: 2D stability map
    plot_stability_map(result, ax=axes[1, 0])

    # Panel 4: ΔG/G vs phi0 for selected xi
    plot_delta_G_vs_phi0(result, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_potential_comparison(
    results: Optional[Dict[str, XiTradeoffResult]] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
):
    """Compare max |ΔG/G| vs xi for multiple potentials.

    Args:
        results: Dict of potential_name -> XiTradeoffResult
        ax: Matplotlib axes
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    from ..analysis.xi_tradeoff import compare_potentials_xi_tradeoff

    if results is None:
        print("Running comparison scan...")
        results = compare_potentials_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for (name, result), color, marker in zip(results.items(), colors, markers):
        xi = result.xi_values
        max_dg = result.max_delta_G
        has_stable = result.stable_fraction > 0

        # Plot line for stable region
        if np.any(has_stable):
            ax.plot(xi[has_stable], max_dg[has_stable],
                   marker=marker, color=color, lw=2, ms=8,
                   label=name)

        # Mark where stability ends
        if np.any(~has_stable):
            # Find transition point
            trans_idx = np.where(has_stable)[0]
            if len(trans_idx) > 0:
                last_stable = trans_idx[-1]
                if last_stable < len(xi) - 1:
                    ax.axvline(xi[last_stable], color=color, ls=':', alpha=0.5)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$ among stable solutions', fontsize=12)
    ax.set_title('Stability-Effect Tradeoff: Potential Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return ax
