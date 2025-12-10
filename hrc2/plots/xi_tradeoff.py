"""Plotting functions for HRC 2.0 xi-tradeoff analysis.

This module provides visualization tools for exploring the parameter
space of general scalar-tensor models.
"""

from typing import Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from ..theory import CouplingFamily
from ..analysis.xi_tradeoff import (
    XiTradeoffResultHRC2,
    find_critical_xi_hrc2,
)
from ..constraints.observational import estimate_delta_H0


def plot_xi_tradeoff_hrc2(
    result: XiTradeoffResultHRC2,
    ax=None,
    figsize: Tuple[float, float] = (12, 7),
    show_stable_only: bool = True,
    save_path: Optional[str] = None,
):
    """Plot max |Delta G/G| vs xi for a single coupling family.

    Args:
        result: XiTradeoffResultHRC2 from scan
        ax: Matplotlib axes (creates new figure if None)
        figsize: Figure size
        show_stable_only: Show stable-only curve in addition to constrained
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xi = result.xi_values

    # Stable-only curve (dashed)
    has_stable = result.stable_fraction > 0
    if show_stable_only and np.any(has_stable):
        ax.plot(xi[has_stable], result.max_delta_G_stable[has_stable],
                'b--o', lw=2, ms=6, markerfacecolor='lightblue',
                label='Dynamically stable only')

    # Constrained curve (solid)
    has_allowed = result.obs_allowed_fraction > 0
    if np.any(has_allowed):
        ax.plot(xi[has_allowed], result.max_delta_G_allowed[has_allowed],
                'g-s', lw=2.5, ms=8, markerfacecolor='lightgreen',
                label=f'Stable + constraints ({result.constraint_level})')

    # Mark points that are stable but excluded
    stable_excluded = has_stable & ~has_allowed
    if np.any(stable_excluded):
        ax.scatter(xi[stable_excluded], result.max_delta_G_stable[stable_excluded],
                   marker='x', s=100, c='orange', lw=2, zorder=5,
                   label='Stable but excluded by constraints')

    # Mark unstable points
    unstable = ~has_stable
    if np.any(unstable):
        ax.scatter(xi[unstable], np.zeros(np.sum(unstable)),
                   marker='X', s=80, c='red', lw=2, zorder=5,
                   label='Dynamically unstable')

    # Critical xi lines
    xi_crit_stable, _ = find_critical_xi_hrc2(result, use_constraints=False)
    xi_crit_allowed, max_dg_allowed = find_critical_xi_hrc2(result, use_constraints=True)

    if xi_crit_stable > 0:
        ax.axvline(xi_crit_stable, color='blue', ls='--', lw=1.5, alpha=0.5)

    if xi_crit_allowed > 0:
        ax.axvline(xi_crit_allowed, color='green', ls='--', lw=1.5, alpha=0.5)

    # Hubble tension reference line
    # Delta H0 ~ 5 km/s/Mpc corresponds to |Delta G/G| ~ 0.14
    ax.axhline(0.14, color='purple', ls=':', lw=2, alpha=0.7,
               label=r'$|\Delta G/G| = 0.14$ ($\Delta H_0 \approx 5$ km/s/Mpc)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (coupling strength)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$', fontsize=12)
    ax.set_title(f'HRC 2.0 Stability-Effect Tradeoff\n'
                 f'({result.coupling_family.value} coupling, {result.potential_type.value} potential)',
                 fontsize=13)

    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return ax


def plot_coupling_comparison(
    results: Dict[CouplingFamily, XiTradeoffResultHRC2],
    ax=None,
    figsize: Tuple[float, float] = (12, 7),
    use_constraints: bool = True,
    save_path: Optional[str] = None,
):
    """Plot comparison of max |Delta G/G| vs xi across coupling families.

    Args:
        results: Dictionary of CouplingFamily -> XiTradeoffResultHRC2
        ax: Matplotlib axes
        figsize: Figure size
        use_constraints: If True, plot constrained values; else stable-only
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = {
        CouplingFamily.LINEAR: 'blue',
        CouplingFamily.QUADRATIC: 'green',
        CouplingFamily.EXPONENTIAL: 'red',
    }

    markers = {
        CouplingFamily.LINEAR: 'o',
        CouplingFamily.QUADRATIC: 's',
        CouplingFamily.EXPONENTIAL: '^',
    }

    for family, result in results.items():
        xi = result.xi_values
        color = colors.get(family, 'gray')
        marker = markers.get(family, 'o')

        if use_constraints:
            max_dg = result.max_delta_G_allowed
            has_solutions = result.obs_allowed_fraction > 0
            label_suffix = " (constrained)"
        else:
            max_dg = result.max_delta_G_stable
            has_solutions = result.stable_fraction > 0
            label_suffix = " (stable only)"

        if np.any(has_solutions):
            ax.plot(xi[has_solutions], max_dg[has_solutions],
                    marker=marker, color=color, lw=2, ms=8,
                    label=f'{family.value}{label_suffix}')

    # Hubble tension reference
    ax.axhline(0.14, color='purple', ls=':', lw=2, alpha=0.7,
               label=r'$|\Delta G/G| = 0.14$ (Hubble tension)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi$ (coupling strength)', fontsize=12)
    ax.set_ylabel(r'Max $|\Delta G/G|$', fontsize=12)

    title = 'HRC 2.0: Coupling Family Comparison'
    if use_constraints:
        title += '\n(with observational constraints)'
    ax.set_title(title, fontsize=13)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return ax


def create_hrc2_summary_figure(
    results: Dict[CouplingFamily, XiTradeoffResultHRC2],
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
):
    """Create 4-panel summary figure for HRC 2.0 analysis.

    Panels:
    1. Max |Delta G/G| vs xi for all families (constrained)
    2. Max |Delta G/G| vs xi (stable only)
    3. Stable fraction vs xi
    4. Allowed fraction vs xi

    Args:
        results: Dictionary of results
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = {
        CouplingFamily.LINEAR: 'blue',
        CouplingFamily.QUADRATIC: 'green',
        CouplingFamily.EXPONENTIAL: 'red',
    }

    markers = {
        CouplingFamily.LINEAR: 'o',
        CouplingFamily.QUADRATIC: 's',
        CouplingFamily.EXPONENTIAL: '^',
    }

    # Panel 1: Constrained comparison
    ax1 = axes[0, 0]
    for family, result in results.items():
        xi = result.xi_values
        has = result.obs_allowed_fraction > 0
        if np.any(has):
            ax1.plot(xi[has], result.max_delta_G_allowed[has],
                     marker=markers.get(family, 'o'),
                     color=colors.get(family, 'gray'),
                     lw=2, ms=7, label=family.value)
    ax1.axhline(0.14, color='purple', ls=':', lw=2, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'Max $|\Delta G/G|$ (constrained)')
    ax1.set_title('With Observational Constraints')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Stable-only comparison
    ax2 = axes[0, 1]
    for family, result in results.items():
        xi = result.xi_values
        has = result.stable_fraction > 0
        if np.any(has):
            ax2.plot(xi[has], result.max_delta_G_stable[has],
                     marker=markers.get(family, 'o'),
                     color=colors.get(family, 'gray'),
                     lw=2, ms=7, label=family.value)
    ax2.axhline(0.14, color='purple', ls=':', lw=2, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'Max $|\Delta G/G|$ (stable only)')
    ax2.set_title('Dynamically Stable Only')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Stable fraction
    ax3 = axes[1, 0]
    for family, result in results.items():
        ax3.plot(result.xi_values, 100 * result.stable_fraction,
                 marker=markers.get(family, 'o'),
                 color=colors.get(family, 'gray'),
                 lw=2, ms=7, label=family.value)
    ax3.set_xscale('log')
    ax3.set_xlabel(r'$\xi$')
    ax3.set_ylabel(r'Stable fraction (%)')
    ax3.set_title('Fraction of phi0 Values that are Stable')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    # Panel 4: Allowed fraction
    ax4 = axes[1, 1]
    for family, result in results.items():
        ax4.plot(result.xi_values, 100 * result.obs_allowed_fraction,
                 marker=markers.get(family, 'o'),
                 color=colors.get(family, 'gray'),
                 lw=2, ms=7, label=family.value)
    ax4.set_xscale('log')
    ax4.set_xlabel(r'$\xi$')
    ax4.set_ylabel(r'Allowed fraction (%)')
    ax4.set_title('Fraction Passing All Constraints')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
