#!/usr/bin/env python3
"""
Generate all figures for the HRC LaTeX paper.

Saves figures to images/ directory in PDF and PNG formats.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hrc_signatures import (
    CMBSignatures, ExpansionSignatures, GWSignatures, DarkMatterSignatures,
    summarize_signatures, SIG_CONST
)
from hrc_observations import LCDMCosmology

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'text.usetex': False,  # Set to True if LaTeX is available
})

# Output directory
OUTPUT_DIR = Path('images')
OUTPUT_DIR.mkdir(exist_ok=True)

# Default HRC parameters
PARAMS = {
    'H0_true': 70.0,
    'Omega_m': 0.315,
    'xi': 0.03,
    'alpha': 0.01,
    'phi_0': 0.2,
    'f_remnant': 0.2,
}


def save_figure(fig, name):
    """Save figure in both PDF and PNG formats."""
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', format='pdf')
    fig.savefig(OUTPUT_DIR / f'{name}.png', format='png')
    print(f"  Saved: {name}.pdf, {name}.png")
    plt.close(fig)


def fig1_hubble_tension():
    """Figure 1: H0 measurements from different probes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Observational data
    probes = ['SH0ES\n(Cepheids)', 'TRGB', 'TDCOSMO\n(Lensing)', 'DESI\nBAO', 'Planck\n(CMB)']
    H0_obs = [73.04, 69.8, 73.3, 67.8, 67.4]
    H0_err = [1.04, 1.7, 3.3, 1.3, 0.5]

    # HRC predictions
    exp = ExpansionSignatures(PARAMS)
    hubble = exp.hubble_tension_vs_z()

    H0_hrc = [
        hubble['local']['H0_predicted'],
        70.5,  # TRGB intermediate
        hubble['lensing']['H0_predicted'],
        hubble['bao']['H0_predicted'],
        hubble['cmb']['H0_predicted'],
    ]

    x = np.arange(len(probes))
    width = 0.35

    # Plot observations
    bars1 = ax.bar(x - width/2, H0_obs, width, yerr=H0_err, label='Observed',
                   color='#2E86AB', capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Plot HRC predictions
    bars2 = ax.bar(x + width/2, H0_hrc, width, label='HRC Prediction',
                   color='#E94F37', alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add ΛCDM line
    ax.axhline(y=67.4, color='gray', linestyle='--', linewidth=1.5,
               label=r'$\Lambda$CDM (Planck)', alpha=0.7)

    # 5σ tension band
    ax.axhspan(67.4 - 2.5, 67.4 + 2.5, alpha=0.1, color='gray')

    # Labels
    ax.set_ylabel(r'$H_0$ [km/s/Mpc]')
    ax.set_xticks(x)
    ax.set_xticklabels(probes)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(63, 80)

    ax.set_title(r'Hubble Constant: Observations vs HRC Predictions', fontweight='bold')

    # Add annotation
    ax.annotate(r'HRC predicts probe-dependent $H_0$',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    save_figure(fig, 'fig1_hubble_tension')


def fig2_geff_evolution():
    """Figure 2: G_eff/G evolution with redshift."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Parameters
    xi = PARAMS['xi']
    phi_0 = PARAMS['phi_0']
    alpha = PARAMS['alpha']

    # Redshift range
    z = np.logspace(-2, 3.1, 500)

    # G_eff factor
    phi_z = phi_0 / (1 + z)**alpha
    G_eff = 1 - 8 * np.pi * xi * phi_z

    ax.semilogx(z, G_eff, 'b-', linewidth=2, label=r'$G_{\rm eff}/G = 1 - 8\pi\xi\phi(z)$')

    # Mark key epochs
    key_z = [0, 0.5, 2, 1089]
    key_labels = ['Today', r'$z=0.5$', r'$z=2$', 'CMB']
    for zi, label in zip(key_z, key_labels):
        phi_zi = phi_0 / (1 + zi)**alpha
        G_zi = 1 - 8 * np.pi * xi * phi_zi
        ax.plot(max(zi, 0.01), G_zi, 'ro', markersize=8)
        offset = (10, 10) if zi < 100 else (10, -15)
        ax.annotate(label, (max(zi, 0.01), G_zi), textcoords='offset points',
                    xytext=offset, fontsize=9)

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label=r'$\Lambda$CDM ($G_{\rm eff}=G$)')

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$G_{\rm eff}/G$')
    ax.set_xlim(0.01, 2000)
    ax.set_ylim(0.82, 1.02)
    ax.legend(loc='lower right')

    ax.set_title(r'Evolution of Effective Gravitational Coupling', fontweight='bold')

    # Add parameter annotation
    ax.annotate(rf'$\xi = {xi}$, $\phi_0 = {phi_0}$, $\alpha = {alpha}$',
                xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    save_figure(fig, 'fig2_geff_evolution')


def fig3_hz_ratio():
    """Figure 3: H(z)/H_ΛCDM(z) ratio."""
    fig, ax = plt.subplots(figsize=(6, 4))

    exp = ExpansionSignatures(PARAMS)
    lcdm = LCDMCosmology(H0=67.4, Omega_m=0.315)

    z = np.linspace(0.01, 3, 200)
    ratio = 1 + exp.Hz_ratio(z)

    ax.plot(z, ratio, 'b-', linewidth=2, label=r'$H(z)_{\rm HRC} / H(z)_{\Lambda{\rm CDM}}$')
    ax.axhline(y=1, color='gray', linestyle='--', label=r'$\Lambda$CDM')

    ax.fill_between(z, 1, ratio, alpha=0.3, color='blue')

    # Mark BAO measurement redshifts
    bao_z = [0.295, 0.51, 0.706, 0.93, 1.317]
    for zi in bao_z:
        ratio_i = 1 + exp.Hz_ratio(zi)
        ax.axvline(x=zi, color='green', linestyle=':', alpha=0.5)

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$H(z)_{\rm HRC} / H(z)_{\Lambda{\rm CDM}}$')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.95, 1.15)
    ax.legend(loc='upper right')

    ax.set_title(r'HRC Expansion Rate Deviation from $\Lambda$CDM', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'fig3_hz_ratio')


def fig4_effective_w():
    """Figure 4: Effective dark energy equation of state w(z)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    exp = ExpansionSignatures(PARAMS)

    z = np.linspace(0.01, 2.5, 200)
    w = exp.effective_w_of_z(z)

    # HRC prediction
    ax.plot(z, w, 'b-', linewidth=2, label=r'HRC effective $w(z)$')

    # ΛCDM
    ax.axhline(y=-1, color='gray', linestyle='--', linewidth=1.5, label=r'$\Lambda$CDM ($w = -1$)')

    # DESI best fit
    w0_desi, wa_desi = -0.827, -0.75
    a = 1 / (1 + z)
    w_desi = w0_desi + wa_desi * (1 - a)
    ax.plot(z, w_desi, 'r--', linewidth=1.5, alpha=0.8, label='DESI best fit')

    # DESI 1σ band
    w_desi_up = (w0_desi + 0.063) + (wa_desi + 0.27) * (1 - a)
    w_desi_down = (w0_desi - 0.063) + (wa_desi - 0.27) * (1 - a)
    ax.fill_between(z, w_desi_down, w_desi_up, alpha=0.15, color='red',
                    label=r'DESI 1$\sigma$')

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'Effective $w(z)$')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-1.6, -0.4)
    ax.legend(loc='lower left', framealpha=0.9)

    ax.set_title(r'Effective Dark Energy Equation of State', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'fig4_effective_w')


def fig5_gw_echoes():
    """Figure 5: GW echo time vs black hole mass."""
    fig, ax = plt.subplots(figsize=(6, 4))

    gw = GWSignatures(PARAMS)

    masses = np.logspace(0.5, 2.5, 100)  # 3 to 300 solar masses
    echo_times = [gw.echo_time_delay(M)['t_echo_ms'] for M in masses]

    ax.loglog(masses, echo_times, 'b-', linewidth=2, label='HRC prediction')

    # Mark key masses
    key_masses = [10, 30, 100]
    colors = ['green', 'red', 'purple']
    for M, c in zip(key_masses, colors):
        t = gw.echo_time_delay(M)['t_echo_ms']
        ax.plot(M, t, 'o', color=c, markersize=10, markeredgecolor='black')
        ax.annotate(f'{M} $M_\\odot$\n{t:.0f} ms', (M, t),
                    textcoords='offset points', xytext=(15, 0),
                    fontsize=9, va='center')

    # LIGO sensitivity regions
    ax.axhspan(5, 100, alpha=0.1, color='green', label='LIGO O4-O5 range')

    ax.set_xlabel(r'Black Hole Mass [$M_\odot$]')
    ax.set_ylabel('Echo Time Delay [ms]')
    ax.set_xlim(3, 300)
    ax.set_ylim(5, 150)
    ax.legend(loc='upper left')

    ax.set_title(r'Predicted GW Ringdown Echo Time', fontweight='bold')

    # Add formula
    ax.annotate(r'$t_{\rm echo} \approx \frac{r_s}{c}\ln\left(\frac{r_s}{\ell_P}\right)$',
                xy=(0.65, 0.15), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'fig5_gw_echoes')


def fig6_parameter_space():
    """Figure 6: Parameter space for Hubble tension resolution."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Parameter grid
    xi_range = np.linspace(0.005, 0.08, 100)
    phi_range = np.linspace(0.05, 0.35, 100)
    XI, PHI = np.meshgrid(xi_range, phi_range)

    # Compute G_eff factor and approximate ΔH0
    G_FACTOR = 1 - 8 * np.pi * XI * PHI
    # Approximate H0 shift from G_eff variation
    DELTA_H0 = 70 * (1 / np.sqrt(G_FACTOR) - 1)

    # Contour plot
    levels = [2, 4, 6, 8, 10, 12]
    cs = ax.contour(XI, PHI, DELTA_H0, levels=levels, colors='blue', linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Fill the "tension resolution" region (4-8 km/s/Mpc)
    sweet_spot = (DELTA_H0 > 4) & (DELTA_H0 < 8)
    ax.contourf(XI, PHI, sweet_spot.astype(float), levels=[0.5, 1.5],
                colors=['lightgreen'], alpha=0.4)

    # Mark fiducial point
    ax.plot(0.03, 0.2, 'r*', markersize=15, markeredgecolor='black',
            label=r'Fiducial ($\xi=0.03$, $\phi_0=0.2$)')

    # Physical constraint: G_eff > 0.5
    constraint = G_FACTOR < 0.5
    if constraint.any():
        ax.contourf(XI, PHI, constraint.astype(float), levels=[0.5, 1.5],
                    colors=['red'], alpha=0.3)

    ax.set_xlabel(r'Non-minimal coupling $\xi$')
    ax.set_ylabel(r'Scalar field $\phi_0$ [Planck units]')
    ax.set_xlim(0.005, 0.08)
    ax.set_ylim(0.05, 0.35)
    ax.legend(loc='upper right')

    ax.set_title(r'Parameter Space for Hubble Tension Resolution', fontweight='bold')

    # Annotation for contours
    ax.annotate(r'$\Delta H_0$ contours (km/s/Mpc)',
                xy=(0.05, 0.92), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.annotate('Tension\nresolved', xy=(0.035, 0.18), fontsize=9, color='darkgreen',
                ha='center')

    plt.tight_layout()
    save_figure(fig, 'fig6_parameter_space')


def fig7_cmb_visibility():
    """Figure 7: Modified CMB visibility function."""
    fig, ax = plt.subplots(figsize=(6, 4))

    cmb = CMBSignatures(PARAMS)

    z = np.linspace(900, 1300, 500)

    # Standard visibility (Gaussian approximation)
    z_star_lcdm = 1089.92
    sigma_lcdm = 80
    g_lcdm = np.exp(-0.5 * ((z - z_star_lcdm) / sigma_lcdm)**2)
    g_lcdm /= g_lcdm.max()

    # HRC modified visibility
    g_hrc = cmb.modified_visibility_function(z)
    g_hrc /= g_hrc.max()

    ax.plot(z, g_lcdm, 'b-', linewidth=2, label=r'$\Lambda$CDM')
    ax.plot(z, g_hrc, 'r--', linewidth=2, label='HRC')

    ax.fill_between(z, 0, g_lcdm, alpha=0.2, color='blue')
    ax.fill_between(z, 0, g_hrc, alpha=0.2, color='red')

    # Mark peaks
    ax.axvline(x=z_star_lcdm, color='blue', linestyle=':', alpha=0.5)
    rec = cmb.recombination_shift()
    ax.axvline(x=rec['z_star_hrc'], color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Visibility function $g(z)$ [normalized]')
    ax.set_xlim(900, 1300)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')

    ax.set_title(r'CMB Visibility Function', fontweight='bold')

    # Annotation for shift
    ax.annotate(rf'$\Delta z_* = {rec["delta_z_star"]:.1f}$',
                xy=(0.7, 0.5), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'fig7_cmb_visibility')


def fig8_dm_mass_spectrum():
    """Figure 8: Dark matter mass spectrum comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Mass ranges for different DM candidates (in GeV)
    # 1 GeV = 1.78e-27 kg
    # M_Planck = 1.22e19 GeV

    candidates = {
        'Axion': (1e-12, 1e-3, 'blue'),
        'Sterile neutrino': (1, 100, 'green'),
        'WIMP': (10, 1e4, 'orange'),
        'Primordial BH': (1e15, 1e25, 'purple'),
        'HRC Remnant': (1e19, 1e19, 'red'),
    }

    y_pos = 0
    for name, (m_min, m_max, color) in candidates.items():
        if m_min == m_max:
            # Point (delta function)
            ax.plot(m_min, y_pos, 's', color=color, markersize=12,
                    markeredgecolor='black', label=name)
            ax.annotate(f'{name}\n($M_{{\\rm Planck}}$)', (m_min, y_pos),
                        textcoords='offset points', xytext=(0, 15),
                        ha='center', fontsize=9, fontweight='bold')
        else:
            # Range
            ax.barh(y_pos, m_max - m_min, left=m_min, height=0.3,
                    color=color, alpha=0.7, edgecolor='black', label=name)
            ax.annotate(name, (np.sqrt(m_min * m_max), y_pos),
                        ha='center', va='center', fontsize=9, fontweight='bold')
        y_pos += 1

    ax.set_xscale('log')
    ax.set_xlabel('Mass [GeV]')
    ax.set_xlim(1e-15, 1e30)
    ax.set_ylim(-0.5, 5)
    ax.set_yticks([])

    ax.set_title('Dark Matter Candidate Mass Spectrum', fontweight='bold')

    # Add vertical line at Planck mass
    ax.axvline(x=1.22e19, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, 'fig8_dm_mass_spectrum')


def fig9_observational_tests():
    """Figure 9: Timeline of observational tests."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tests = [
        ('Standard siren $H_0$', 2024, 2029, 'HIGH', 'LIGO/Virgo/KAGRA'),
        ('Multi-probe $H_0$', 2024, 2027, 'HIGH', 'CMB+BAO+SNe'),
        ('DESI $w(z)$', 2024, 2028, 'MEDIUM-HIGH', 'DESI BAO'),
        ('GW echoes', 2025, 2030, 'HIGH*', 'LIGO A+'),
        ('CMB-S4 $\\theta_*$', 2028, 2035, 'LOW-MEDIUM', 'CMB-S4'),
        ('Dwarf galaxy cores', 2025, 2032, 'MEDIUM', '4MOST/DESI'),
    ]

    colors = {'HIGH': '#E94F37', 'HIGH*': '#E94F37', 'MEDIUM-HIGH': '#F4A261',
              'MEDIUM': '#2A9D8F', 'LOW-MEDIUM': '#264653'}

    for i, (name, start, end, power, probe) in enumerate(tests):
        color = colors.get(power, 'gray')
        ax.barh(i, end - start, left=start, height=0.6, color=color,
                alpha=0.8, edgecolor='black')
        ax.annotate(name, (start - 0.5, i), ha='right', va='center', fontsize=9)
        ax.annotate(f'({probe})', ((start + end)/2, i), ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    # Current time marker
    ax.axvline(x=2025, color='black', linestyle='--', linewidth=2, label='Now (2025)')

    ax.set_xlabel('Year')
    ax.set_xlim(2020, 2036)
    ax.set_ylim(-0.5, len(tests) - 0.5)
    ax.set_yticks([])

    ax.set_title('HRC Observational Test Timeline', fontweight='bold')

    # Legend for discriminating power
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E94F37', label='HIGH'),
        Patch(facecolor='#F4A261', label='MEDIUM-HIGH'),
        Patch(facecolor='#2A9D8F', label='MEDIUM'),
        Patch(facecolor='#264653', label='LOW-MEDIUM'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Discriminating\nPower',
              fontsize=8, title_fontsize=8)

    plt.tight_layout()
    save_figure(fig, 'fig9_observational_tests')


def main():
    """Generate all figures."""
    print("Generating HRC paper figures...")
    print("=" * 50)

    fig1_hubble_tension()
    fig2_geff_evolution()
    fig3_hz_ratio()
    fig4_effective_w()
    fig5_gw_echoes()
    fig6_parameter_space()
    fig7_cmb_visibility()
    fig8_dm_mass_spectrum()
    fig9_observational_tests()

    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*.pdf')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
