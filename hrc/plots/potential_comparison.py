"""Comparison plots for different scalar field potentials.

This module creates diagnostic plots comparing different potential forms
(quadratic, plateau, symmetron, exponential) and their effects on:
- Parameter space validity
- G_eff evolution
- Hubble tension resolution
"""

from typing import Optional, Tuple, List, Dict, Union
import numpy as np
from numpy.typing import NDArray

from ..analysis import ParameterScanResult, PointClassification, scan_parameter_space
from ..potentials import (
    Potential,
    QuadraticPotential,
    PlateauPotential,
    SymmetronPotential,
    ExponentialPotential,
    POTENTIAL_REGISTRY,
)
from ..utils.config import HRCParameters
from ..background import BackgroundCosmology


# Default potentials for comparison with parameters tuned for cosmology
DEFAULT_POTENTIALS = {
    "quadratic": QuadraticPotential(V0=0.7, m=1.0),
    "plateau": PlateauPotential(V0=0.7, M=0.5, n=2.0),
    "symmetron": SymmetronPotential(V0=0.7, mu2=1.0, lambda_=2.0),
    "exponential": ExponentialPotential(V0=0.7, lambda_=0.5, M=1.0),
}


def plot_potential_shapes(
    potentials: Optional[Dict[str, Potential]] = None,
    phi_range: Tuple[float, float] = (-1.0, 1.0),
    n_points: int = 200,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    show_derivative: bool = False,
):
    """Plot V(phi) curves for multiple potentials.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        phi_range: Range of phi values to plot
        n_points: Number of points
        ax: Matplotlib axes
        figsize: Figure size
        show_derivative: If True, plot dV/dphi instead of V

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if potentials is None:
        potentials = DEFAULT_POTENTIALS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    phi = np.linspace(phi_range[0], phi_range[1], n_points)
    colors = plt.cm.tab10(np.linspace(0, 1, len(potentials)))

    for (name, pot), color in zip(potentials.items(), colors):
        if show_derivative:
            y = np.array([pot.dV_dphi(p) for p in phi])
            ylabel = r"$V'(\phi)$"
            title = "Potential Derivatives"
        else:
            y = np.array([pot.V(p) for p in phi])
            ylabel = r"$V(\phi)$"
            title = "Scalar Field Potentials"

        ax.plot(phi, y, label=name, color=color, lw=2)

    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)

    ax.set_xlabel(r"$\phi$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_validity_comparison(
    potentials: Optional[Dict[str, Potential]] = None,
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 20,
    n_phi_0: int = 20,
    z_max: float = 1100.0,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
):
    """Create comparison plot of validity regions for different potentials.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi: Number of xi grid points
        n_phi_0: Number of phi_0 grid points
        z_max: Maximum redshift for integration
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure and scan results dictionary
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if potentials is None:
        potentials = DEFAULT_POTENTIALS

    n_pots = len(potentials)
    ncols = 2
    nrows = (n_pots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_pots > 1 else [axes]

    # Colors for classification
    colors = ['#ff6b6b', '#74c0fc', '#69db7c']  # Red, Blue, Green
    cmap = ListedColormap(colors)

    results = {}

    for idx, (name, pot) in enumerate(potentials.items()):
        print(f"\nScanning with {name} potential...")

        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
            potential=pot,
        )
        results[name] = result

        # Create classification array
        class_values = np.zeros_like(result.geff_valid, dtype=int)
        for i in range(len(result.xi_grid)):
            for j in range(len(result.phi_0_grid)):
                if result.classification[i, j] == PointClassification.INVALID:
                    class_values[i, j] = 0
                elif result.classification[i, j] == PointClassification.VALID_NO_TENSION:
                    class_values[i, j] = 1
                else:
                    class_values[i, j] = 2

        ax = axes[idx]
        XI, PHI = np.meshgrid(result.xi_grid, result.phi_0_grid, indexing='ij')

        im = ax.pcolormesh(XI, PHI, class_values, cmap=cmap, vmin=-0.5, vmax=2.5,
                          shading='auto', alpha=0.8)

        # Statistics
        total = n_xi * n_phi_0
        valid_pct = 100 * result.geff_valid.sum() / total
        resolves_pct = 100 * result.n_valid_resolves / total

        ax.set_xlabel(r'$\xi$', fontsize=11)
        ax.set_ylabel(r'$\phi_0$', fontsize=11)
        ax.set_title(f'{name.capitalize()}\n(Valid: {valid_pct:.0f}%, Resolves: {resolves_pct:.0f}%)',
                    fontsize=12)
        ax.grid(True, alpha=0.3, ls=':')

    # Hide extra subplots
    for idx in range(n_pots, len(axes)):
        axes[idx].set_visible(False)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.set_yticks([0, 1, 2])
    cbar.ax.set_yticklabels(['Invalid', 'Valid\n(no tension)', 'Valid\n(resolves)'])

    fig.suptitle(f'Parameter Space Validity by Potential ($z_{{\\rm max}} = {z_max:.0f}$)',
                fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")

    return fig, results


def plot_geff_evolution_comparison(
    potentials: Optional[Dict[str, Potential]] = None,
    xi: float = 0.03,
    phi_0: float = 0.15,
    z_max: float = 100.0,
    z_points: int = 500,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
):
    """Compare G_eff(z) evolution for different potentials.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        xi: Non-minimal coupling constant
        phi_0: Initial scalar field value
        z_max: Maximum redshift
        z_points: Number of redshift points
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes and solutions dictionary
    """
    import matplotlib.pyplot as plt

    if potentials is None:
        potentials = DEFAULT_POTENTIALS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    params = HRCParameters(xi=xi, phi_0=phi_0)
    colors = plt.cm.tab10(np.linspace(0, 1, len(potentials)))
    solutions = {}

    for (name, pot), color in zip(potentials.items(), colors):
        try:
            cosmo = BackgroundCosmology(params, potential=pot)
            sol = cosmo.solve(z_max=z_max, z_points=z_points)
            solutions[name] = sol

            if sol.geff_valid:
                ax.plot(sol.z, sol.G_eff_ratio, label=name, color=color, lw=2)
            else:
                # Plot until divergence
                valid_idx = ~np.isnan(sol.G_eff_ratio)
                ax.plot(sol.z[valid_idx], sol.G_eff_ratio[valid_idx],
                       label=f'{name} (diverges)', color=color, lw=2, ls='--')

        except Exception as e:
            print(f"  {name}: Error - {e}")

    ax.axhline(1.0, color='gray', ls='--', lw=1, label='GR')
    ax.set_xlabel(r'Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$G_{\rm eff}/G$', fontsize=12)
    ax.set_title(f'$G_{{\\rm eff}}$ Evolution ($\\xi={xi}$, $\\phi_0={phi_0}$)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, z_max)

    return ax, solutions


def plot_phi_evolution_comparison(
    potentials: Optional[Dict[str, Potential]] = None,
    xi: float = 0.03,
    phi_0: float = 0.15,
    z_max: float = 100.0,
    z_points: int = 500,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
):
    """Compare scalar field phi(z) evolution for different potentials.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        xi: Non-minimal coupling constant
        phi_0: Initial scalar field value
        z_max: Maximum redshift
        z_points: Number of redshift points
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes and solutions dictionary
    """
    import matplotlib.pyplot as plt
    from ..utils.numerics import compute_critical_phi

    if potentials is None:
        potentials = DEFAULT_POTENTIALS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    params = HRCParameters(xi=xi, phi_0=phi_0)
    phi_crit = compute_critical_phi(xi)
    colors = plt.cm.tab10(np.linspace(0, 1, len(potentials)))
    solutions = {}

    for (name, pot), color in zip(potentials.items(), colors):
        try:
            cosmo = BackgroundCosmology(params, potential=pot)
            sol = cosmo.solve(z_max=z_max, z_points=z_points)
            solutions[name] = sol

            ax.plot(sol.z, sol.phi, label=name, color=color, lw=2)

        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Add critical value line
    ax.axhline(phi_crit, color='red', ls='--', lw=2, label=r'$\phi_c$')
    ax.axhline(-phi_crit, color='red', ls='--', lw=2)

    ax.set_xlabel(r'Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$\phi$', fontsize=12)
    ax.set_title(f'Scalar Field Evolution ($\\xi={xi}$, $\\phi_0={phi_0}$)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, z_max)

    return ax, solutions


def create_potential_comparison_figure(
    potentials: Optional[Dict[str, Potential]] = None,
    xi: float = 0.03,
    phi_0: float = 0.15,
    z_max: float = 100.0,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
):
    """Create a comprehensive comparison figure.

    Args:
        potentials: Dictionary of potential name -> Potential instance
        xi: Non-minimal coupling for evolution plots
        phi_0: Initial scalar field for evolution plots
        z_max: Maximum redshift for evolution plots
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if potentials is None:
        potentials = DEFAULT_POTENTIALS

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Potential shapes
    plot_potential_shapes(potentials, ax=axes[0, 0])

    # Panel 2: Potential derivatives
    plot_potential_shapes(potentials, ax=axes[0, 1], show_derivative=True)

    # Panel 3: phi(z) evolution
    plot_phi_evolution_comparison(potentials, xi=xi, phi_0=phi_0, z_max=z_max, ax=axes[1, 0])

    # Panel 4: G_eff(z) evolution
    plot_geff_evolution_comparison(potentials, xi=xi, phi_0=phi_0, z_max=z_max, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def summarize_potential_results(
    results: Dict[str, ParameterScanResult],
) -> Dict[str, Dict]:
    """Summarize scan results for each potential.

    Args:
        results: Dictionary of potential name -> ParameterScanResult

    Returns:
        Dictionary with summary statistics for each potential
    """
    summary = {}

    for name, result in results.items():
        total = len(result.points)
        n_valid = result.geff_valid.sum()
        n_resolves = result.n_valid_resolves

        summary[name] = {
            "total_points": total,
            "valid_count": int(n_valid),
            "valid_fraction": n_valid / total,
            "resolves_count": n_resolves,
            "resolves_fraction": n_resolves / total,
            "invalid_count": result.n_invalid,
            "invalid_fraction": result.n_invalid / total,
        }

    return summary


def print_potential_comparison_report(
    results: Dict[str, ParameterScanResult],
    z_max: float = 1100.0,
):
    """Print a text report comparing potential performance.

    Args:
        results: Dictionary of potential name -> ParameterScanResult
        z_max: Maximum redshift used in scan
    """
    summary = summarize_potential_results(results)

    print("\n" + "=" * 70)
    print(f"POTENTIAL COMPARISON REPORT (z_max = {z_max})")
    print("=" * 70)

    # Header
    print(f"\n{'Potential':<15} {'Valid %':>10} {'Resolves %':>12} {'Invalid %':>10}")
    print("-" * 50)

    # Sort by valid fraction (best first)
    sorted_names = sorted(summary.keys(), key=lambda k: summary[k]['valid_fraction'], reverse=True)

    for name in sorted_names:
        s = summary[name]
        print(f"{name:<15} {100*s['valid_fraction']:>9.1f}% {100*s['resolves_fraction']:>11.1f}% {100*s['invalid_fraction']:>9.1f}%")

    print("-" * 50)

    # Find best potential
    best_name = sorted_names[0]
    best = summary[best_name]
    print(f"\nBest potential for stability: {best_name}")
    print(f"  - {100*best['valid_fraction']:.1f}% of parameter space remains valid to z={z_max}")
    print(f"  - {100*best['resolves_fraction']:.1f}% can resolve Hubble tension")

    print("\n" + "=" * 70)
