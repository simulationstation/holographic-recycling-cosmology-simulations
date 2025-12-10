"""Plots for parameter space validity regions.

This module creates diagnostic plots showing the valid vs invalid
regions in (xi, phi_0) parameter space, including:
- 2D validity maps
- Hubble tension resolution regions
- Parameter space boundaries
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..analysis import ParameterScanResult, PointClassification, scan_parameter_space
from ..analysis.parameter_scan import compute_validity_boundary
from ..utils.numerics import compute_critical_phi


def plot_validity_map(
    result: Optional[ParameterScanResult] = None,
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 30,
    n_phi_0: int = 30,
    z_max: float = 1100.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap_invalid: str = 'Reds',
    cmap_valid: str = 'Greens',
    show_colorbar: bool = True,
    title: Optional[str] = None,
):
    """Plot 2D map of validity regions in parameter space.

    Args:
        result: Pre-computed scan result (will compute if None)
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi: Number of xi grid points
        n_phi_0: Number of phi_0 grid points
        z_max: Maximum redshift for integration
        ax: Matplotlib axes
        figsize: Figure size
        cmap_invalid: Colormap for invalid regions
        cmap_valid: Colormap for valid regions
        show_colorbar: Whether to show colorbar
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Run scan if result not provided
    if result is None:
        print("Running parameter space scan...")
        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
        )

    # Create classification array for coloring
    # 0 = invalid, 1 = valid (no tension), 2 = valid (resolves)
    class_values = np.zeros_like(result.geff_valid, dtype=int)
    for i in range(len(result.xi_grid)):
        for j in range(len(result.phi_0_grid)):
            if result.classification[i, j] == PointClassification.INVALID:
                class_values[i, j] = 0
            elif result.classification[i, j] == PointClassification.VALID_NO_TENSION:
                class_values[i, j] = 1
            else:  # VALID_RESOLVES
                class_values[i, j] = 2

    # Create custom colormap
    colors = ['#ff6b6b', '#74c0fc', '#69db7c']  # Red, Blue, Green
    cmap = ListedColormap(colors)

    # Plot
    XI, PHI = np.meshgrid(result.xi_grid, result.phi_0_grid, indexing='ij')

    im = ax.pcolormesh(XI, PHI, class_values, cmap=cmap, vmin=-0.5, vmax=2.5,
                       shading='auto', alpha=0.8)

    # Add critical curve
    xi_fine = np.linspace(result.xi_grid.min(), result.xi_grid.max(), 100)
    phi_crit = np.array([compute_critical_phi(xi) for xi in xi_fine])
    valid_mask = phi_crit < phi_0_range[1] * 2
    if np.any(valid_mask):
        ax.plot(xi_fine[valid_mask], phi_crit[valid_mask], 'k--', lw=2,
                label=r'$\phi_c = 1/(8\pi\xi)$')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Invalid', 'Valid\n(no tension)', 'Valid\n(resolves)'])

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_xlim(xi_range)
    ax.set_ylim(phi_0_range)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, ls=':')

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Parameter Space Validity ($z_{{\\rm max}} = {z_max:.0f}$)', fontsize=14)

    return ax


def plot_divergence_redshift(
    result: Optional[ParameterScanResult] = None,
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 30,
    n_phi_0: int = 30,
    z_max: float = 1100.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'RdYlGn_r',
):
    """Plot the redshift at which G_eff diverges for each parameter point.

    Args:
        result: Pre-computed scan result
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi, n_phi_0: Grid resolution
        z_max: Maximum redshift
        ax: Matplotlib axes
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if result is None:
        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
        )

    XI, PHI = np.meshgrid(result.xi_grid, result.phi_0_grid, indexing='ij')

    # Mask NaN values and create divergence z array
    div_z = result.geff_divergence_z.copy()
    div_z[result.geff_valid] = z_max  # Valid points "diverge" at z_max

    im = ax.pcolormesh(XI, PHI, div_z, cmap=cmap, shading='auto',
                       vmin=0, vmax=z_max)

    plt.colorbar(im, ax=ax, label=r'Divergence redshift $z_{\rm div}$')

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_title('Redshift of $G_{\\rm eff}$ Divergence', fontsize=14)
    ax.grid(True, alpha=0.3, ls=':')

    return ax


def plot_delta_h0(
    result: Optional[ParameterScanResult] = None,
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 30,
    n_phi_0: int = 30,
    z_max: float = 1100.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'RdBu',
):
    """Plot the Hubble tension resolution (Delta H0) across parameter space.

    Args:
        result: Pre-computed scan result
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi, n_phi_0: Grid resolution
        z_max: Maximum redshift
        ax: Matplotlib axes
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if result is None:
        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
        )

    XI, PHI = np.meshgrid(result.xi_grid, result.phi_0_grid, indexing='ij')

    # Mask invalid points
    delta_h0 = result.Delta_H0.copy()
    delta_h0[~result.geff_valid] = np.nan

    # Determine symmetric colorbar limits
    vmax = np.nanmax(np.abs(delta_h0))
    if np.isnan(vmax):
        vmax = 10.0

    im = ax.pcolormesh(XI, PHI, delta_h0, cmap=cmap, shading='auto',
                       vmin=-vmax, vmax=vmax)

    plt.colorbar(im, ax=ax, label=r'$\Delta H_0$ (km/s/Mpc)')

    # Contour for target tension
    if not np.all(np.isnan(delta_h0)):
        try:
            cs = ax.contour(XI, PHI, delta_h0, levels=[3, 5, 7],
                           colors=['green', 'darkgreen', 'black'],
                           linewidths=2)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f km/s/Mpc')
        except ValueError:
            pass  # Skip contours if data doesn't support them

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_title('Hubble Tension Resolution $\\Delta H_0$', fontsize=14)
    ax.grid(True, alpha=0.3, ls=':')

    return ax


def plot_geff_at_z(
    result: Optional[ParameterScanResult] = None,
    z_value: float = 0.0,
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 30,
    n_phi_0: int = 30,
    z_max: float = 1100.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'coolwarm',
):
    """Plot G_eff/G at a specific redshift across parameter space.

    Args:
        result: Pre-computed scan result
        z_value: Redshift at which to evaluate G_eff (0 or 1089)
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi, n_phi_0: Grid resolution
        z_max: Maximum redshift
        ax: Matplotlib axes
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if result is None:
        result = scan_parameter_space(
            xi_range=xi_range,
            phi_0_range=phi_0_range,
            n_xi=n_xi,
            n_phi_0=n_phi_0,
            z_max=z_max,
            verbose=False,
        )

    XI, PHI = np.meshgrid(result.xi_grid, result.phi_0_grid, indexing='ij')

    # Select appropriate G_eff array
    if z_value < 100:
        geff = result.G_eff_0.copy()
        z_label = '0'
    else:
        geff = result.G_eff_cmb.copy()
        z_label = '1089'

    # Mask invalid points
    geff[~result.geff_valid] = np.nan

    im = ax.pcolormesh(XI, PHI, geff, cmap=cmap, shading='auto',
                       vmin=0.8, vmax=1.5)

    plt.colorbar(im, ax=ax, label=r'$G_{\rm eff}/G$')

    # Contours
    if not np.all(np.isnan(geff)):
        try:
            cs = ax.contour(XI, PHI, geff, levels=[1.0, 1.1, 1.2, 1.3],
                           colors='black', linewidths=1)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        except ValueError:
            pass

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_title(f'$G_{{\\rm eff}}/G$ at $z = {z_label}$', fontsize=14)
    ax.grid(True, alpha=0.3, ls=':')

    return ax


def plot_validity_boundary(
    xi_range: Tuple[float, float] = (0.01, 0.1),
    n_xi: int = 50,
    z_max: float = 1100.0,
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
):
    """Plot the boundary curve between valid and invalid regions.

    Args:
        xi_range: Range of xi values
        n_xi: Number of points for boundary
        z_max: Maximum redshift
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    print("Computing validity boundary...")
    xi_vals, max_phi_0 = compute_validity_boundary(
        xi_range=xi_range,
        n_xi=n_xi,
        z_max=z_max,
        verbose=False,
    )

    # Also compute critical curve
    phi_crit = np.array([compute_critical_phi(xi) for xi in xi_vals])

    # Plot regions
    ax.fill_between(xi_vals, 0, max_phi_0, alpha=0.3, color='green',
                    label='Valid region')
    ax.fill_between(xi_vals, max_phi_0, phi_crit, alpha=0.3, color='orange',
                    label='Evolves to divergence')
    ax.fill_between(xi_vals, phi_crit, phi_crit.max() * 1.2, alpha=0.3, color='red',
                    label='Initially invalid')

    # Boundary curves
    ax.plot(xi_vals, max_phi_0, 'b-', lw=2, label='Validity boundary')
    ax.plot(xi_vals, phi_crit, 'r--', lw=2, label=r'$\phi_c = 1/(8\pi\xi)$')

    ax.set_xlabel(r'$\xi$ (non-minimal coupling)', fontsize=12)
    ax.set_ylabel(r'$\phi_0$ (initial scalar field)', fontsize=12)
    ax.set_xlim(xi_range)
    ax.set_ylim(0, phi_crit.min() * 1.5)
    ax.set_title(f'Parameter Space Boundaries ($z_{{\\rm max}} = {z_max:.0f}$)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def create_summary_figure(
    xi_range: Tuple[float, float] = (0.01, 0.1),
    phi_0_range: Tuple[float, float] = (0.05, 0.5),
    n_xi: int = 25,
    n_phi_0: int = 25,
    z_max: float = 1100.0,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
):
    """Create a 2x2 summary figure of parameter space diagnostics.

    Args:
        xi_range: Range of xi values
        phi_0_range: Range of phi_0 values
        n_xi, n_phi_0: Grid resolution
        z_max: Maximum redshift
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt

    # Run scan once
    print("Running parameter space scan for summary figure...")
    result = scan_parameter_space(
        xi_range=xi_range,
        phi_0_range=phi_0_range,
        n_xi=n_xi,
        n_phi_0=n_phi_0,
        z_max=z_max,
        verbose=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Validity map
    plot_validity_map(result, ax=axes[0, 0], show_colorbar=True)

    # Panel 2: Divergence redshift
    plot_divergence_redshift(result, ax=axes[0, 1])

    # Panel 3: G_eff at z=0
    plot_geff_at_z(result, z_value=0, ax=axes[1, 0])

    # Panel 4: G_eff at CMB
    plot_geff_at_z(result, z_value=1089, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
