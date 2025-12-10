"""Plots for xi-stability-effect tradeoff analysis.

These plots visualize the fundamental tradeoff in HRC:
- Smaller xi -> more stable but smaller effect
- Larger xi -> larger effect but unstable

Now includes constraint information (BBN, PPN, stellar).
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
    show_constraints: bool = True,
    title: Optional[str] = None,
):
    """Plot maximum |ΔG/G| vs xi for stable and allowed configurations.

    This is the key diagnostic plot showing the stability-effect tradeoff.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        show_unstable: Mark xi values with no stable configurations
        show_constraints: Show both stable-only and constrained curves
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
    max_dg_stable = result.max_delta_G_stable
    max_dg_allowed = result.max_delta_G_allowed
    stable_frac = result.stable_fraction
    allowed_frac = result.obs_allowed_fraction

    # Plot max |ΔG/G| for stable-only (dashed)
    has_stable = stable_frac > 0
    if show_constraints:
        ax.plot(xi[has_stable], max_dg_stable[has_stable], 'b--', lw=2, ms=6,
                marker='o', markerfacecolor='none',
                label='Stable only (no constraints)')

    # Plot max |ΔG/G| for allowed (solid)
    has_allowed = allowed_frac > 0
    if np.any(has_allowed):
        ax.plot(xi[has_allowed], max_dg_allowed[has_allowed], 'g-', lw=2, ms=8,
                marker='s', label='Stable + allowed (with constraints)')

    # Mark unstable xi values
    if show_unstable and np.any(~has_stable):
        ax.scatter(xi[~has_stable], np.zeros(np.sum(~has_stable)),
                  marker='x', s=100, c='red', lw=2,
                  label='No stable solutions', zorder=5)

    # Mark stable but not allowed
    if show_constraints:
        stable_not_allowed = has_stable & ~has_allowed
        if np.any(stable_not_allowed):
            ax.scatter(xi[stable_not_allowed],
                      max_dg_stable[stable_not_allowed],
                      marker='o', s=80, c='orange', lw=2,
                      label='Stable but excluded by constraints', zorder=4)

    # Find and mark critical xi values
    xi_crit_stable, max_dg_crit_stable = find_critical_xi(result, use_constraints=False)
    xi_crit_allowed, max_dg_crit_allowed = find_critical_xi(result, use_constraints=True)

    if xi_crit_stable > 0:
        ax.axvline(xi_crit_stable, color='blue', ls=':', lw=1.5, alpha=0.7)

    if xi_crit_allowed > 0 and show_constraints:
        ax.axvline(xi_crit_allowed, color='green', ls='--', lw=1.5, alpha=0.7,
                  label=f'$\\xi_{{crit}}^{{allowed}} \\approx {xi_crit_allowed:.1e}$')

    # Reference lines
    ax.axhline(0.1, color='gray', ls=':', alpha=0.5, label='|ΔG/G| = 0.1')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        constraint_str = f" ({result.constraint_level} BBN)" if show_constraints else ""
        ax.set_title(f'Stability-Effect Tradeoff ({result.potential_name} potential){constraint_str}',
                    fontsize=14)

    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    return ax


def plot_max_delta_G_with_constraints(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (12, 7),
    save_path: Optional[str] = None,
):
    """Plot max |ΔG/G| showing both stable-only and constrained curves.

    This is the main comparison plot requested in TASK 2.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan with constraints...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xi = result.xi_values

    # Stable-only curve
    has_stable = result.stable_fraction > 0
    ax.plot(xi[has_stable], result.max_delta_G_stable[has_stable],
           'b-o', lw=2, ms=8, markerfacecolor='lightblue',
           label='Dynamically stable only')

    # Constrained curve
    has_allowed = result.obs_allowed_fraction > 0
    if np.any(has_allowed):
        ax.plot(xi[has_allowed], result.max_delta_G_allowed[has_allowed],
               'g-s', lw=2.5, ms=10, markerfacecolor='lightgreen',
               label=f'Stable + constraints ({result.constraint_level} BBN + PPN + stellar)')

    # Mark points that are stable but excluded
    stable_excluded = has_stable & ~has_allowed
    if np.any(stable_excluded):
        ax.scatter(xi[stable_excluded], result.max_delta_G_stable[stable_excluded],
                  marker='x', s=120, c='orange', lw=3, zorder=5,
                  label='Stable but excluded by constraints')

    # Mark completely unstable
    unstable = ~has_stable
    if np.any(unstable):
        ax.scatter(xi[unstable], np.zeros(np.sum(unstable)),
                  marker='X', s=100, c='red', lw=2, zorder=5,
                  label='Dynamically unstable')

    # Critical xi annotations
    xi_crit_stable, max_dg_stable = find_critical_xi(result, use_constraints=False)
    xi_crit_allowed, max_dg_allowed = find_critical_xi(result, use_constraints=True)

    if xi_crit_stable > 0:
        ax.axvline(xi_crit_stable, color='blue', ls='--', lw=1.5, alpha=0.6)
        ax.annotate(f'$\\xi_{{crit}}^{{stable}} = {xi_crit_stable:.1e}$',
                   xy=(xi_crit_stable, max_dg_stable * 0.9),
                   fontsize=10, color='blue')

    if xi_crit_allowed > 0:
        ax.axvline(xi_crit_allowed, color='green', ls='--', lw=1.5, alpha=0.6)

    # Hubble tension reference
    # ΔH0 ~ 5 km/s/Mpc corresponds to |ΔG/G| ~ 0.14
    ax.axhline(0.14, color='purple', ls=':', lw=2, alpha=0.7,
              label=r'$|\Delta G/G| = 0.14$ ($\Delta H_0 \approx 5$ km/s/Mpc)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=13)
    ax.set_ylabel(r'Max $|\Delta G/G|$ among valid configurations', fontsize=13)
    ax.set_title(f'Stability-Effect Tradeoff with Observational Constraints\n'
                f'({result.potential_name} potential, z_max = {result.z_max:.0f})',
                fontsize=14)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return ax


def plot_stability_map(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    show_constraints: bool = True,
):
    """Plot 2D map of stability and ΔG/G in (xi, phi0) space.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        show_constraints: Show constraint boundaries

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

    im = ax.pcolormesh(XI, PHI, delta_g, cmap='RdYlGn', shading='auto',
                       vmin=vmin, vmax=vmax)

    # Mark stable boundary
    ax.contour(XI, PHI, result.stable_mask.astype(float),
              levels=[0.5], colors='blue', linewidths=2, linestyles='--',
              label='Stable boundary')

    # Mark allowed boundary
    if show_constraints:
        ax.contour(XI, PHI, result.obs_allowed_mask.astype(float),
                  levels=[0.5], colors='green', linewidths=2.5, linestyles='-')

    plt.colorbar(im, ax=ax, label=r'$\Delta G/G = G_{eff}(0) - G_{eff}(z_{rec})$')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)

    legend_text = 'Blue dashed = stability boundary'
    if show_constraints:
        legend_text += ', Green = constraint boundary'
    ax.set_title(f'Stability Map ({result.potential_name} potential)\n{legend_text}',
                fontsize=12)

    return ax


def plot_constraint_breakdown(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (12, 6),
):
    """Plot breakdown of which constraints exclude stable points.

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

    xi = result.xi_values
    n_phi0 = len(result.phi0_values)

    # Compute fractions
    stable_frac = result.stable_fraction
    bbn_frac = np.array([result.bbn_allowed[i, :].sum() / n_phi0 for i in range(len(xi))])
    ppn_frac = np.array([result.ppn_allowed[i, :].sum() / n_phi0 for i in range(len(xi))])
    stellar_frac = np.array([(result.stellar_allowed[i, :] & result.stable_mask[i, :]).sum() / n_phi0
                            for i in range(len(xi))])
    allowed_frac = result.obs_allowed_fraction

    # Plot stacked areas or lines
    ax.plot(xi, 100 * stable_frac, 'b-o', lw=2, ms=6, label='Stable')
    ax.plot(xi, 100 * bbn_frac, 'r--^', lw=2, ms=6, label='BBN allowed')
    ax.plot(xi, 100 * ppn_frac, 'm-.v', lw=2, ms=6, label='PPN allowed')
    ax.plot(xi, 100 * stellar_frac, 'c:s', lw=2, ms=6, label='Stellar allowed')
    ax.plot(xi, 100 * allowed_frac, 'g-D', lw=2.5, ms=8, label='All constraints')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Fraction of $\phi_0$ values passing (%)', fontsize=12)
    ax.set_title(f'Constraint Breakdown ({result.potential_name} potential)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    return ax


def plot_delta_G_vs_phi0(
    result: Optional[XiTradeoffResult] = None,
    xi_indices: Optional[list] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    show_constraints: bool = True,
):
    """Plot ΔG/G vs phi0 for selected xi values, marking allowed regions.

    Args:
        result: XiTradeoffResult (will compute if None)
        xi_indices: Indices of xi values to plot (default: select a few)
        ax: Matplotlib axes
        figsize: Figure size
        show_constraints: Mark allowed vs excluded points

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
        n_xi = len(result.xi_values)
        xi_indices = [0, n_xi//4, n_xi//2, 3*n_xi//4, n_xi-1]
        xi_indices = [i for i in xi_indices if i < n_xi]

    colors = plt.cm.viridis(np.linspace(0, 1, len(xi_indices)))

    for idx, color in zip(xi_indices, colors):
        xi = result.xi_values[idx]
        delta_g = result.delta_G_over_G[idx, :]
        stable = result.stable_mask[idx, :]
        allowed = result.obs_allowed_mask[idx, :]

        # Plot allowed points (filled)
        if show_constraints and np.any(allowed):
            ax.plot(result.phi0_values[allowed], delta_g[allowed],
                   'o', color=color, ms=8, label=f'$\\xi = {xi:.1e}$ (allowed)')

        # Plot stable but not allowed (open circles)
        stable_not_allowed = stable & ~allowed
        if show_constraints and np.any(stable_not_allowed):
            ax.plot(result.phi0_values[stable_not_allowed], delta_g[stable_not_allowed],
                   'o', color=color, ms=8, markerfacecolor='none', lw=1.5)

        # If not showing constraints, just show all stable
        if not show_constraints and np.any(stable):
            ax.plot(result.phi0_values[stable], delta_g[stable],
                   'o-', color=color, lw=2, ms=6, label=f'$\\xi = {xi:.1e}$')

        # Mark unstable points
        if np.any(~stable):
            ax.scatter(result.phi0_values[~stable],
                      np.zeros(np.sum(~stable)),
                      marker='x', color=color, s=50, alpha=0.3)

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_ylabel(r'$\Delta G/G$', fontsize=12)

    title = f'Effect Size vs Initial Field ({result.potential_name} potential)'
    if show_constraints:
        title += '\nFilled = allowed, Open = stable but excluded'
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    return ax


def plot_stable_fraction_vs_xi(
    result: Optional[XiTradeoffResult] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    show_constraints: bool = True,
):
    """Plot fraction of phi0 values that are stable/allowed vs xi.

    Args:
        result: XiTradeoffResult (will compute if None)
        ax: Matplotlib axes
        figsize: Figure size
        show_constraints: Show both stable and allowed fractions

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if result is None:
        print("Running xi-tradeoff scan...")
        result = scan_xi_tradeoff(verbose=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(result.xi_values, 100 * result.stable_fraction, 'b-o', lw=2, ms=8,
           label='Dynamically stable')

    if show_constraints:
        ax.plot(result.xi_values, 100 * result.obs_allowed_fraction, 'g-s', lw=2, ms=8,
               label='Stable + all constraints')

    # Critical xi markers
    xi_crit_stable, _ = find_critical_xi(result, use_constraints=False)
    xi_crit_allowed, _ = find_critical_xi(result, use_constraints=True)

    if xi_crit_stable > 0:
        ax.axvline(xi_crit_stable, color='blue', ls='--', lw=1.5, alpha=0.6)

    if xi_crit_allowed > 0 and show_constraints:
        ax.axvline(xi_crit_allowed, color='green', ls='--', lw=1.5, alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Fraction of $\phi_0$ values (%)', fontsize=12)
    ax.set_title(f'Stability vs Coupling Strength ({result.potential_name} potential)',
                fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    return ax


def create_xi_tradeoff_summary_figure(
    result: Optional[XiTradeoffResult] = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
):
    """Create a 4-panel summary figure for xi-tradeoff analysis with constraints.

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

    # Panel 1: Max |ΔG/G| vs xi with constraints (key plot)
    plot_max_delta_G_with_constraints(result, ax=axes[0, 0])

    # Panel 2: Stable fraction vs xi
    plot_stable_fraction_vs_xi(result, ax=axes[0, 1])

    # Panel 3: Constraint breakdown
    plot_constraint_breakdown(result, ax=axes[1, 0])

    # Panel 4: 2D stability map
    plot_stability_map(result, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_potential_comparison(
    results: Optional[Dict[str, XiTradeoffResult]] = None,
    ax=None,
    figsize: Tuple[float, float] = (12, 7),
    use_constraints: bool = True,
    save_path: Optional[str] = None,
):
    """Compare max |ΔG/G| vs xi for multiple potentials.

    Args:
        results: Dict of potential_name -> XiTradeoffResult
        ax: Matplotlib axes
        figsize: Figure size
        use_constraints: If True, plot constrained values; else stable-only
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

        if use_constraints:
            max_dg = result.max_delta_G_allowed
            has_solutions = result.obs_allowed_fraction > 0
            label_suffix = " (constrained)"
        else:
            max_dg = result.max_delta_G_stable
            has_solutions = result.stable_fraction > 0
            label_suffix = " (stable only)"

        # Plot line for valid region
        if np.any(has_solutions):
            ax.plot(xi[has_solutions], max_dg[has_solutions],
                   marker=marker, color=color, lw=2, ms=8,
                   label=name + label_suffix)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$', fontsize=12)

    title = 'Stability-Effect Tradeoff: Potential Comparison'
    if use_constraints:
        title += '\n(with observational constraints)'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return ax
