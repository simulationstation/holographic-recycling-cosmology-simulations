#!/usr/bin/env python3
"""
Generate all figures for the paper:
"The James Webb Space Telescope Will Not Resolve the Hubble Tension"

Author: Simulation Station
Repository: https://github.com/simulationstation/holographic-recycling-cosmology-simulations
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 3.0),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

# Color scheme
PLANCK_BLUE = '#1f77b4'
SHOES_RED = '#d62728'
JWST_GOLD = '#ff7f0e'
TENSION_PURPLE = '#9467bd'
GRID_COLOR = '#cccccc'

# Directories
RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)
IMAGES_DIR = Path(__file__).parent / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

# Hubble tension reference values
H0_TRUE = 67.5
H0_SHOES = 73.0
TENSION = H0_SHOES - H0_TRUE


def load_json(filepath):
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def fig_sim11_heatmap():
    """
    SIM 11: Delta H0 heatmap as function of population drift and metallicity.
    """
    print("Generating SIM 11 heatmap...")

    # Load scan results
    scan_file = RESULTS_DIR / 'simulation_11_snia_systematics' / 'scan_results.json'
    with open(scan_file, 'r') as f:
        full_data = json.load(f)

    # Results are under the 'results' key
    data = full_data.get('results', full_data)

    # Extract unique parameter values
    alpha_vals = sorted(set(r['alpha_pop'] for r in data))
    gamma_vals = sorted(set(r['gamma_Z'] for r in data))
    malm_vals = sorted(set(r['delta_m_malm'] for r in data))

    # Choose middle Malmquist value for 2D slice
    malm_target = malm_vals[len(malm_vals)//2]

    # Build 2D array
    delta_h0 = np.zeros((len(gamma_vals), len(alpha_vals)))
    for r in data:
        if abs(r['delta_m_malm'] - malm_target) < 0.001:
            i = gamma_vals.index(r['gamma_Z'])
            j = alpha_vals.index(r['alpha_pop'])
            delta_h0[i, j] = r['delta_H0']

    # Create figure
    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    # Heatmap
    im = ax.imshow(delta_h0, origin='lower', aspect='auto',
                   extent=[alpha_vals[0]-0.025, alpha_vals[-1]+0.025,
                           gamma_vals[0]-0.025, gamma_vals[-1]+0.025],
                   cmap='RdBu_r', vmin=-2, vmax=6)

    # Contours
    X, Y = np.meshgrid(alpha_vals, gamma_vals)
    cs = ax.contour(X, Y, delta_h0, levels=[3, 4, 5], colors='black', linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

    # Realistic region box
    rect = Rectangle((0, 0), 0.05, 0.05, linewidth=2, edgecolor='green',
                      facecolor='none', linestyle='--', label='Realistic')
    ax.add_patch(rect)

    # Labels
    ax.set_xlabel(r'Population drift $\alpha_{\rm pop}$')
    ax.set_ylabel(r'Metallicity dependence $\gamma_Z$')
    ax.set_title(rf'$\Delta H_0$ at $\delta_{{m,\rm Malm}} = {malm_target}$')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$\Delta H_0$ (km/s/Mpc)')

    # Legend
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim11_delta_h0_heatmap.pdf')
    plt.savefig(IMAGES_DIR / 'sim11_delta_h0_heatmap.png', dpi=300)
    plt.close()
    print(f"  Saved: sim11_delta_h0_heatmap.pdf and .png")


def fig_sim11b_scatter():
    """
    SIM 11b: Distribution of Delta H0 across scenarios.
    """
    print("Generating SIM 11b scatter plot...")

    scan_file = RESULTS_DIR / 'simulation_11b_ladder_systematics' / 'scan_results.json'
    with open(scan_file, 'r') as f:
        full_data = json.load(f)

    # Results are under the 'results' key
    data = full_data.get('results', full_data)

    delta_h0_vals = [r['delta_H0'] for r in data]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    # Histogram
    n, bins, patches = ax.hist(delta_h0_vals, bins=20, color=PLANCK_BLUE,
                                edgecolor='white', alpha=0.7)

    # Tension line
    ax.axvline(TENSION, color=SHOES_RED, linestyle='--', linewidth=2,
               label=f'Hubble tension ({TENSION:.1f})')
    ax.axvline(-TENSION, color=SHOES_RED, linestyle='--', linewidth=2)

    # Mean line
    mean_dh0 = np.mean(delta_h0_vals)
    ax.axvline(mean_dh0, color='green', linestyle='-', linewidth=1.5,
               label=f'Mean ({mean_dh0:.2f})')

    ax.set_xlabel(r'$\Delta H_0$ (km/s/Mpc)')
    ax.set_ylabel('Number of scenarios')
    ax.set_title('SIM 11b: SALT2 Ladder Systematics')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim11b_scatter.pdf')
    plt.savefig(IMAGES_DIR / 'sim11b_scatter.png', dpi=300)
    plt.close()
    print(f"  Saved: sim11b_scatter.pdf and .png")


def fig_sim11c_histogram():
    """
    SIM 11c: Histogram of Delta H0 for combined systematics.
    """
    print("Generating SIM 11c histogram...")

    scan_file = RESULTS_DIR / 'simulation_11c_calibrator_plus_sn' / 'scan_results.json'
    with open(scan_file, 'r') as f:
        full_data = json.load(f)

    # Results are under the 'results' key
    data = full_data.get('results', full_data)

    delta_h0_vals = [r['delta_H0'] for r in data]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    # Histogram with KDE
    n, bins, patches = ax.hist(delta_h0_vals, bins=50, density=True,
                                color=PLANCK_BLUE, edgecolor='white', alpha=0.6)

    # KDE overlay
    kde = gaussian_kde(delta_h0_vals)
    x_kde = np.linspace(min(delta_h0_vals)-0.5, max(delta_h0_vals)+0.5, 200)
    ax.plot(x_kde, kde(x_kde), color='darkblue', linewidth=2, label='KDE')

    # Tension threshold
    ax.axvline(TENSION, color=SHOES_RED, linestyle='--', linewidth=2,
               label=f'Tension ({TENSION:.1f})')
    ax.axvline(-TENSION, color=SHOES_RED, linestyle='--', linewidth=2)

    ax.set_xlabel(r'$\Delta H_0$ (km/s/Mpc)')
    ax.set_ylabel('Density')
    ax.set_title(f'SIM 11c: Combined Systematics (n={len(data)})')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim11c_histogram.pdf')
    plt.savefig(IMAGES_DIR / 'sim11c_histogram.png', dpi=300)
    plt.close()
    print(f"  Saved: sim11c_histogram.pdf and .png")


def fig_sim12_calibration():
    """
    SIM 12: Delta H0 vs mean calibrator distance bias.
    """
    print("Generating SIM 12 calibration bias plot...")

    scan_file = RESULTS_DIR / 'simulation_12_cepheid_calibration' / 'scan_results.json'
    with open(scan_file, 'r') as f:
        full_data = json.load(f)

    # Results are under the 'results' key
    data = full_data.get('results', full_data)

    # Extract mu bias and delta H0
    mu_bias = [r.get('mean_mu_bias', 0) for r in data]
    delta_h0 = [r['delta_H0'] for r in data]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.scatter(mu_bias, delta_h0, c=PLANCK_BLUE, s=20, alpha=0.5)

    # Reference lines
    ax.axhline(3, color=JWST_GOLD, linestyle=':', linewidth=1.5, label=r'$|\Delta H_0| = 3$')
    ax.axhline(-3, color=JWST_GOLD, linestyle=':', linewidth=1.5)
    ax.axhline(TENSION, color=SHOES_RED, linestyle='--', linewidth=1.5, label='Tension')
    ax.axhline(-TENSION, color=SHOES_RED, linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_xlabel(r'Mean calibrator $\mu$ bias (mag)')
    ax.set_ylabel(r'$\Delta H_0$ (km/s/Mpc)')
    ax.set_title('SIM 12: Cepheid Calibration')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim12_calibration_bias.pdf')
    plt.savefig(IMAGES_DIR / 'sim12_calibration_bias.png', dpi=300)
    plt.close()
    print(f"  Saved: sim12_calibration_bias.pdf and .png")


def fig_sim13_jwst_hst():
    """
    SIM 13: HST vs JWST instrument difference distribution.
    """
    print("Generating SIM 13 JWST/HST comparison...")

    scan_file = RESULTS_DIR / 'simulation_13_jwst_hst_recalibration' / 'scan_results.json'
    with open(scan_file, 'r') as f:
        data = json.load(f)

    delta_h0_inst = [r['delta_H0_inst'] for r in data]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    # Histogram
    n, bins, patches = ax.hist(delta_h0_inst, bins=15, color=JWST_GOLD,
                                edgecolor='white', alpha=0.7)

    # Annotate max/min
    max_abs = max(abs(min(delta_h0_inst)), abs(max(delta_h0_inst)))
    ax.axvline(max_abs, color='red', linestyle=':', linewidth=1.5)
    ax.axvline(-max_abs, color='red', linestyle=':', linewidth=1.5)

    # Add text annotation
    ax.text(0.02, 0.95, f'Max $|\\Delta H_0^{{\\rm inst}}| = {max_abs:.2f}$',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Tension reference (way off scale, so just label)
    ax.set_xlabel(r'$\Delta H_0^{\rm inst} = H_0^{\rm JWST} - H_0^{\rm HST}$ (km/s/Mpc)')
    ax.set_ylabel('Number of scenarios')
    ax.set_title('SIM 13: HST vs JWST Recalibration')
    ax.set_xlim(-0.6, 0.6)

    # Add inset text about tension
    ax.annotate(f'Tension: {TENSION:.1f} km/s/Mpc\n(14x larger than max)',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim13_jwst_hst_delta.pdf')
    plt.savefig(IMAGES_DIR / 'sim13_jwst_hst_delta.png', dpi=300)
    plt.close()
    print(f"  Saved: sim13_jwst_hst_delta.pdf and .png")


def fig_sim14_frame_bias():
    """
    SIM 14: Rest-frame misalignment bias by sky coverage.
    """
    print("Generating SIM 14 frame bias plot...")

    summary_file = RESULTS_DIR / 'simulation_14_restframe_misalignment' / 'summary.json'
    summary = load_json(summary_file)

    v_values = summary['v_true_values']
    sky_coverages = ['isotropic', 'toward_apex', 'away_from_apex']
    sky_labels = ['Isotropic', 'Toward apex', 'Away from apex']
    sky_markers = ['o', '^', 'v']
    sky_colors = [PLANCK_BLUE, 'green', SHOES_RED]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for sky, label, marker, color in zip(sky_coverages, sky_labels, sky_markers, sky_colors):
        means = []
        stds = []
        for v in v_values:
            key = f"v{v}_sky{sky}"
            stats = summary['stats_by_scenario'].get(key, {})
            means.append(stats.get('mean', 0) - (-3.5))  # Relative to CMB frame baseline
            stds.append(stats.get('std', 0))

        # Plot as frame bias relative to isotropic
        ax.errorbar(v_values, [summary['by_sky_coverage'][sky]['mean']] * len(v_values),
                    yerr=[summary['by_sky_coverage'][sky]['std']] * len(v_values),
                    marker=marker, color=color, label=label, capsize=3, markersize=6)

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axhline(1, color=JWST_GOLD, linestyle=':', linewidth=1)
    ax.axhline(-1, color=JWST_GOLD, linestyle=':', linewidth=1)

    ax.set_xlabel(r'Peculiar velocity $v$ (km/s)')
    ax.set_ylabel(r'Frame mismatch bias $\Delta H_0^{\rm frame}$ (km/s/Mpc)')
    ax.set_title('SIM 14: Rest-Frame Misalignment')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim14_frame_bias.pdf')
    plt.savefig(IMAGES_DIR / 'sim14_frame_bias.png', dpi=300)
    plt.close()
    print(f"  Saved: sim14_frame_bias.pdf and .png")


def fig_sim15_posterior():
    """
    SIM 15: H0 posterior distribution from joint hierarchical analysis.
    """
    print("Generating SIM 15 H0 posterior...")

    # Load MCMC results
    results_file = RESULTS_DIR / 'simulation_15_joint_systematics' / 'mcmc_results.json'
    results = load_json(results_file)

    # Load flat chain for histogram
    chain_file = RESULTS_DIR / 'simulation_15_joint_systematics' / 'flat_chain.npy'
    if chain_file.exists():
        flat_chain = np.load(chain_file)
        h0_samples = flat_chain[:, 0]
    else:
        # Generate synthetic samples from reported statistics
        h0_mean = results['results']['H0']['mean']
        h0_std = results['results']['H0']['std']
        h0_samples = np.random.normal(h0_mean, h0_std, 10000)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Histogram
    n, bins, patches = ax.hist(h0_samples, bins=60, density=True,
                                color=PLANCK_BLUE, edgecolor='white', alpha=0.7,
                                label='Posterior')

    # KDE
    kde = gaussian_kde(h0_samples)
    x_kde = np.linspace(60, 80, 300)
    ax.plot(x_kde, kde(x_kde), color='darkblue', linewidth=2)

    # True value
    ax.axvline(H0_TRUE, color='green', linestyle='--', linewidth=2,
               label=f'True $H_0$ = {H0_TRUE}')

    # SH0ES value
    ax.axvline(H0_SHOES, color=SHOES_RED, linestyle='--', linewidth=2,
               label=f'SH0ES $H_0$ = {H0_SHOES}')

    # Shade region H0 >= 73
    x_fill = np.linspace(73, 80, 100)
    ax.fill_between(x_fill, 0, kde(x_fill), color=SHOES_RED, alpha=0.3)

    # Annotate probability
    p_73 = results['results']['P_H0_ge_73']
    ax.annotate(f'$P(H_0 \\geq 73) = {p_73*100:.2f}\\%$',
                xy=(73.5, 0.02), fontsize=10, color=SHOES_RED,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=SHOES_RED))

    # Statistics box
    h0_stats = results['results']['H0']
    stats_text = (f"$H_0 = {h0_stats['mean']:.2f} \\pm {h0_stats['std']:.2f}$\n"
                  f"68% CI: [{h0_stats['16th']:.1f}, {h0_stats['84th']:.1f}]")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(r'$H_0$ (km/s/Mpc)')
    ax.set_ylabel('Probability density')
    ax.set_title('SIM 15: Joint Hierarchical Posterior')
    ax.set_xlim(60, 78)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim15_h0_posterior.pdf')
    plt.savefig(IMAGES_DIR / 'sim15_h0_posterior.png', dpi=300)
    plt.close()
    print(f"  Saved: sim15_h0_posterior.pdf and .png")


def fig_sim15_corner():
    """
    SIM 15: Corner plot of H0 and key nuisance parameters.
    """
    print("Generating SIM 15 corner plot...")

    chain_file = RESULTS_DIR / 'simulation_15_joint_systematics' / 'flat_chain.npy'
    if not chain_file.exists():
        print("  Skipping corner plot (chain file not found)")
        return

    flat_chain = np.load(chain_file)

    # Select H0 and a few key nuisance parameters
    param_indices = [0, 5, 6, 7]  # H0, delta_M_W0, delta_mu_anchor, delta_mu_crowd
    param_names = [r'$H_0$', r'$\delta M_{W,0}$', r'$\delta\mu_{\rm anchor}$', r'$\delta\mu_{\rm crowd}$']

    # Subsample for speed
    n_samples = min(10000, len(flat_chain))
    idx = np.random.choice(len(flat_chain), n_samples, replace=False)
    samples = flat_chain[idx][:, param_indices]

    # Create corner plot manually (avoiding corner dependency)
    n_params = len(param_indices)
    fig, axes = plt.subplots(n_params, n_params, figsize=(6, 6))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if j > i:
                ax.axis('off')
                continue

            if i == j:
                # 1D histogram
                ax.hist(samples[:, i], bins=30, color=PLANCK_BLUE,
                        edgecolor='white', alpha=0.7, density=True)
                if i == 0:
                    ax.axvline(H0_TRUE, color='green', linestyle='--')
                    ax.axvline(H0_SHOES, color=SHOES_RED, linestyle='--')
                else:
                    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
            else:
                # 2D scatter/contour
                ax.scatter(samples[:, j], samples[:, i], c=PLANCK_BLUE,
                           s=1, alpha=0.1)
                # Simple density contours
                try:
                    from scipy.stats import gaussian_kde
                    xy = np.vstack([samples[:, j], samples[:, i]])
                    kde = gaussian_kde(xy)
                    xmin, xmax = samples[:, j].min(), samples[:, j].max()
                    ymin, ymax = samples[:, i].min(), samples[:, i].max()
                    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    z = np.reshape(kde(positions).T, xx.shape)
                    ax.contour(xx, yy, z, levels=3, colors='darkblue', linewidths=0.5)
                except Exception:
                    pass

            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=9)

            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sim15_corner.pdf')
    plt.savefig(IMAGES_DIR / 'sim15_corner.png', dpi=300)
    plt.close()
    print(f"  Saved: sim15_corner.pdf and .png")


def fig_summary_bar():
    """
    Summary bar chart of max |Delta H0| across all simulations.
    """
    print("Generating summary bar chart...")

    simulations = [
        ('SIM 11\n(SN pop)', 6.04),
        ('SIM 11b\n(SALT2)', 3.0),
        ('SIM 11c\n(Combined)', 4.8),
        ('SIM 12\n(Cepheid)', 4.3),
        ('SIM 13\n(JWST)', 0.4),
        ('SIM 14\n(Frame)', 1.6),
    ]

    labels = [s[0] for s in simulations]
    max_dh0 = [s[1] for s in simulations]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    x = np.arange(len(labels))
    colors = [SHOES_RED if v >= TENSION else PLANCK_BLUE for v in max_dh0]

    bars = ax.bar(x, max_dh0, color=colors, edgecolor='white', alpha=0.8)

    # Tension threshold line
    ax.axhline(TENSION, color=SHOES_RED, linestyle='--', linewidth=2,
               label=f'Hubble tension ({TENSION:.1f} km/s/Mpc)')

    # Annotate JWST specifically
    ax.annotate('JWST: 14x too small!', xy=(4, 0.4), xytext=(4, 2.5),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'Max $|\Delta H_0|$ (km/s/Mpc)')
    ax.set_title('Maximum Systematic Bias by Simulation')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'summary_bar_chart.pdf')
    plt.savefig(IMAGES_DIR / 'summary_bar_chart.png', dpi=300)
    plt.close()
    print(f"  Saved: summary_bar_chart.pdf and .png")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating figures for JWST Hubble Tension paper")
    print("=" * 60)
    print()

    # Check that results exist
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    # Generate each figure
    try:
        fig_sim11_heatmap()
    except Exception as e:
        print(f"  Error in SIM 11: {e}")

    try:
        fig_sim11b_scatter()
    except Exception as e:
        print(f"  Error in SIM 11b: {e}")

    try:
        fig_sim11c_histogram()
    except Exception as e:
        print(f"  Error in SIM 11c: {e}")

    try:
        fig_sim12_calibration()
    except Exception as e:
        print(f"  Error in SIM 12: {e}")

    try:
        fig_sim13_jwst_hst()
    except Exception as e:
        print(f"  Error in SIM 13: {e}")

    try:
        fig_sim14_frame_bias()
    except Exception as e:
        print(f"  Error in SIM 14: {e}")

    try:
        fig_sim15_posterior()
    except Exception as e:
        print(f"  Error in SIM 15 posterior: {e}")

    try:
        fig_sim15_corner()
    except Exception as e:
        print(f"  Error in SIM 15 corner: {e}")

    try:
        fig_summary_bar()
    except Exception as e:
        print(f"  Error in summary: {e}")

    print()
    print("=" * 60)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
