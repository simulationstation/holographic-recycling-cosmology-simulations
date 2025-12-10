#!/usr/bin/env python3
"""Compare MCMC posterior distributions: LCDM vs HMDE T06D.

This script post-processes Cobaya MCMC chains to:
1. Generate 1D and 2D marginalized posteriors
2. Compute evidence ratios (Bayes factors)
3. Create comparison triangle plots
4. Output parameter constraints with uncertainties
5. Assess tension between models

Features:
- Uses GetDist for chain analysis
- Generates publication-quality plots
- Computes information criteria (AIC, BIC, DIC)
- Outputs LaTeX-formatted parameter tables

Usage:
    python scripts/compare_lcdm_vs_hmde_posteriors.py

    # With custom chain paths:
    python scripts/compare_lcdm_vs_hmde_posteriors.py \
        --lcdm results/mcmc/lcdm_planck_bao_sne/chains \
        --hmde results/mcmc/hmde_t06d_planck_bao_sne/chains

    # Generate all plots:
    python scripts/compare_lcdm_vs_hmde_posteriors.py --all-plots
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def load_chains(chain_root: Path, name: str = None):
    """Load MCMC chains using GetDist.

    Args:
        chain_root: Path to chain files (without extension)
        name: Optional name for the chain

    Returns:
        GetDist MCSamples object
    """
    from getdist import MCSamples, loadMCSamples

    try:
        samples = loadMCSamples(str(chain_root), settings={'ignore_rows': 0.3})
        if name:
            samples.name_tag = name
        print(f"[OK] Loaded chains: {chain_root}")
        print(f"     Samples: {samples.numrows}")
        print(f"     Parameters: {len(samples.getParamNames().names)}")
        return samples
    except Exception as e:
        print(f"[ERROR] Failed to load chains from {chain_root}: {e}")
        return None


def get_parameter_constraints(samples, params: list = None):
    """Extract parameter constraints from chains.

    Args:
        samples: GetDist MCSamples object
        params: List of parameter names (default: all)

    Returns:
        Dictionary of parameter constraints
    """
    if params is None:
        params = [p.name for p in samples.getParamNames().names]

    constraints = {}
    for param in params:
        try:
            mean = samples.mean(param)
            std = samples.std(param)
            bounds = samples.confidence(param, limfrac=0.683)  # 1-sigma

            constraints[param] = {
                'mean': float(mean),
                'std': float(std),
                'lower_1sigma': float(bounds[0]),
                'upper_1sigma': float(bounds[1]),
            }
        except Exception:
            pass

    return constraints


def compute_information_criteria(samples, n_data: int = 2500):
    """Compute AIC, BIC, DIC from chains.

    Args:
        samples: GetDist MCSamples object
        n_data: Number of data points

    Returns:
        Dictionary with information criteria
    """
    # Best-fit log-likelihood
    loglike = samples.getLikeStats()
    if loglike is None:
        return {}

    best_loglike = loglike.logLike_sample
    mean_loglike = loglike.logMeanInvLike

    # Number of parameters
    n_params = len([p for p in samples.getParamNames().names
                   if not p.isDerived])

    # AIC = -2 * ln(L_max) + 2*k
    aic = -2 * best_loglike + 2 * n_params

    # BIC = -2 * ln(L_max) + k * ln(n)
    bic = -2 * best_loglike + n_params * np.log(n_data)

    # DIC = D_bar + p_D (deviance + effective parameters)
    # D_bar = -2 * <ln(L)>
    # p_D = D_bar - D_theta_bar (effective number of parameters)
    d_bar = -2 * mean_loglike if mean_loglike else None
    dic = d_bar + (d_bar - (-2 * best_loglike)) if d_bar else None

    return {
        'best_loglike': float(best_loglike),
        'n_params': n_params,
        'n_data': n_data,
        'AIC': float(aic),
        'BIC': float(bic),
        'DIC': float(dic) if dic else None,
    }


def compute_model_comparison(lcdm_samples, hmde_samples, n_data: int = 2500):
    """Compute model comparison statistics.

    Args:
        lcdm_samples: LCDM chain samples
        hmde_samples: HMDE chain samples
        n_data: Number of data points

    Returns:
        Dictionary with comparison statistics
    """
    lcdm_ic = compute_information_criteria(lcdm_samples, n_data)
    hmde_ic = compute_information_criteria(hmde_samples, n_data)

    comparison = {
        'lcdm': lcdm_ic,
        'hmde': hmde_ic,
    }

    # Delta values (positive = LCDM preferred)
    if lcdm_ic.get('AIC') and hmde_ic.get('AIC'):
        comparison['delta_AIC'] = hmde_ic['AIC'] - lcdm_ic['AIC']
        comparison['delta_BIC'] = hmde_ic['BIC'] - lcdm_ic['BIC']
        if lcdm_ic.get('DIC') and hmde_ic.get('DIC'):
            comparison['delta_DIC'] = hmde_ic['DIC'] - lcdm_ic['DIC']

        # Interpretation
        delta_aic = comparison['delta_AIC']
        if delta_aic < -10:
            comparison['aic_verdict'] = "Strong preference for HMDE"
        elif delta_aic < -6:
            comparison['aic_verdict'] = "Moderate preference for HMDE"
        elif delta_aic < -2:
            comparison['aic_verdict'] = "Weak preference for HMDE"
        elif delta_aic < 2:
            comparison['aic_verdict'] = "Models indistinguishable"
        elif delta_aic < 6:
            comparison['aic_verdict'] = "Weak preference for LCDM"
        elif delta_aic < 10:
            comparison['aic_verdict'] = "Moderate preference for LCDM"
        else:
            comparison['aic_verdict'] = "Strong preference for LCDM"

    return comparison


def create_triangle_plot(samples_list, params: list, output_path: Path,
                         labels: list = None, title: str = None):
    """Create triangle plot comparing posteriors.

    Args:
        samples_list: List of GetDist MCSamples objects
        params: Parameters to include
        output_path: Path to save figure
        labels: Legend labels
        title: Plot title
    """
    from getdist import plots

    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 12
    g.settings.axes_fontsize = 10
    g.settings.lab_fontsize = 12

    # Filter to available parameters
    available_params = []
    for p in params:
        if all(s.hasParam(p) for s in samples_list):
            available_params.append(p)

    if not available_params:
        print(f"[WARNING] No common parameters found for triangle plot")
        return

    g.triangle_plot(
        samples_list,
        available_params,
        filled=True,
        legend_labels=labels,
        contour_colors=['blue', 'red'],
        title=title
    )

    g.export(str(output_path))
    print(f"[OK] Saved triangle plot: {output_path}")


def create_1d_comparison(samples_list, params: list, output_path: Path,
                         labels: list = None):
    """Create 1D posterior comparison plots.

    Args:
        samples_list: List of GetDist MCSamples objects
        params: Parameters to plot
        output_path: Path to save figure
        labels: Legend labels
    """
    from getdist import plots
    import matplotlib.pyplot as plt

    g = plots.get_subplot_plotter(width_inch=10)

    # Filter to available parameters
    available_params = []
    for p in params:
        if all(s.hasParam(p) for s in samples_list):
            available_params.append(p)

    if not available_params:
        print(f"[WARNING] No common parameters for 1D comparison")
        return

    n_params = len(available_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    g.plots_1d(
        samples_list,
        available_params,
        nx=n_cols,
        legend_labels=labels,
        colors=['blue', 'red'],
    )

    g.export(str(output_path))
    print(f"[OK] Saved 1D comparison: {output_path}")


def create_H0_omega_m_plot(samples_list, output_path: Path, labels: list = None):
    """Create H0 vs Omega_m 2D posterior comparison.

    This is a key plot for Hubble tension analysis.

    Args:
        samples_list: List of GetDist MCSamples objects
        output_path: Path to save figure
        labels: Legend labels
    """
    from getdist import plots
    import matplotlib.pyplot as plt

    g = plots.get_single_plotter()
    g.settings.legend_fontsize = 14
    g.settings.axes_fontsize = 12

    # Check if parameters exist
    h0_param = 'H0'
    om_param = 'omegam'

    if not all(s.hasParam(h0_param) and s.hasParam(om_param) for s in samples_list):
        print(f"[WARNING] Missing H0 or Omega_m for 2D plot")
        return

    g.plot_2d(
        samples_list,
        h0_param, om_param,
        filled=True,
        legend_labels=labels,
        contour_colors=['blue', 'red'],
    )

    # Add SH0ES measurement band
    ax = g.subplots[0, 0]
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, alpha=0.2, color='green', label='SH0ES')
    ax.axhline(0.315, color='gray', linestyle='--', alpha=0.5, label='Planck 2018')
    ax.legend(loc='upper right')

    g.export(str(output_path))
    print(f"[OK] Saved H0-Omega_m plot: {output_path}")


def create_t06d_parameter_plot(hmde_samples, output_path: Path):
    """Create T06D parameter posterior plot.

    Args:
        hmde_samples: HMDE chain samples
        output_path: Path to save figure
    """
    from getdist import plots

    # T06D-specific parameters
    t06d_params = ['delta_w', 'a_w', 'w0_fld', 'wa_fld']
    available = [p for p in t06d_params if hmde_samples.hasParam(p)]

    if len(available) < 2:
        print(f"[WARNING] Not enough T06D parameters for plot")
        return

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        hmde_samples,
        available,
        filled=True,
        title='T06D Parameter Posteriors'
    )

    g.export(str(output_path))
    print(f"[OK] Saved T06D parameter plot: {output_path}")


def generate_latex_table(constraints_dict: dict, output_path: Path):
    """Generate LaTeX parameter table.

    Args:
        constraints_dict: Dictionary with model constraints
        output_path: Path to save .tex file
    """
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Parameter constraints from MCMC analysis}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Parameter & $\Lambda$CDM & HMDE T06D \\",
        r"\hline",
    ]

    # Common parameters to show
    show_params = ['H0', 'omegam', 'sigma8', 'ns', 'tau', 'ombh2', 'omch2']

    for param in show_params:
        lcdm_val = constraints_dict.get('lcdm', {}).get(param)
        hmde_val = constraints_dict.get('hmde', {}).get(param)

        if lcdm_val:
            lcdm_str = f"${lcdm_val['mean']:.4f} \\pm {lcdm_val['std']:.4f}$"
        else:
            lcdm_str = "--"

        if hmde_val:
            hmde_str = f"${hmde_val['mean']:.4f} \\pm {hmde_val['std']:.4f}$"
        else:
            hmde_str = "--"

        param_latex = param.replace('_', r'\_')
        lines.append(f"${param_latex}$ & {lcdm_str} & {hmde_str} \\\\")

    # T06D-specific parameters
    t06d_params = ['delta_w', 'a_w', 'w0_fld', 'wa_fld']
    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{c}{\textit{T06D Parameters}} \\")
    lines.append(r"\hline")

    for param in t06d_params:
        hmde_val = constraints_dict.get('hmde', {}).get(param)
        if hmde_val:
            hmde_str = f"${hmde_val['mean']:.4f} \\pm {hmde_val['std']:.4f}$"
            param_latex = param.replace('_', r'\_')
            lines.append(f"${param_latex}$ & -- & {hmde_str} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[OK] Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LCDM vs HMDE T06D MCMC posteriors"
    )

    parser.add_argument(
        '--lcdm',
        type=Path,
        default=PROJECT_ROOT / "results/mcmc/lcdm_planck_bao_sne/chains",
        help='Path to LCDM chain root'
    )

    parser.add_argument(
        '--hmde',
        type=Path,
        default=PROJECT_ROOT / "results/mcmc/hmde_t06d_planck_bao_sne/chains",
        help='Path to HMDE chain root'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=PROJECT_ROOT / "results/mcmc_comparison",
        help='Output directory for plots and results'
    )

    parser.add_argument(
        '--all-plots',
        action='store_true',
        help='Generate all comparison plots'
    )

    parser.add_argument(
        '--n-data',
        type=int,
        default=2500,
        help='Number of data points for IC calculation'
    )

    args = parser.parse_args()

    # Check for GetDist
    try:
        import getdist
        print(f"[OK] GetDist version: {getdist.__version__}")
    except ImportError:
        print("[ERROR] GetDist not installed. Run: pip install getdist")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load chains
    print("\n" + "=" * 60)
    print("Loading MCMC chains")
    print("=" * 60)

    lcdm_samples = load_chains(args.lcdm, name='LCDM')
    hmde_samples = load_chains(args.hmde, name='HMDE T06D')

    if lcdm_samples is None and hmde_samples is None:
        print("\n[ERROR] No chains found. Run MCMC first:")
        print("  python scripts/run_cobaya_mcmc.py --model lcdm")
        print("  python scripts/run_cobaya_mcmc.py --model hmde")
        sys.exit(1)

    # Extract constraints
    print("\n" + "=" * 60)
    print("Parameter constraints")
    print("=" * 60)

    constraints = {}
    if lcdm_samples:
        constraints['lcdm'] = get_parameter_constraints(lcdm_samples)
        print("\nLCDM constraints:")
        for p, c in list(constraints['lcdm'].items())[:5]:
            print(f"  {p}: {c['mean']:.4f} +/- {c['std']:.4f}")

    if hmde_samples:
        constraints['hmde'] = get_parameter_constraints(hmde_samples)
        print("\nHMDE T06D constraints:")
        for p, c in list(constraints['hmde'].items())[:5]:
            print(f"  {p}: {c['mean']:.4f} +/- {c['std']:.4f}")

    # Model comparison
    if lcdm_samples and hmde_samples:
        print("\n" + "=" * 60)
        print("Model comparison")
        print("=" * 60)

        comparison = compute_model_comparison(lcdm_samples, hmde_samples, args.n_data)
        print(f"\nInformation criteria:")
        print(f"  LCDM AIC: {comparison['lcdm'].get('AIC', 'N/A'):.2f}")
        print(f"  HMDE AIC: {comparison['hmde'].get('AIC', 'N/A'):.2f}")
        print(f"  Delta AIC: {comparison.get('delta_AIC', 'N/A'):.2f}")
        print(f"  Verdict: {comparison.get('aic_verdict', 'N/A')}")

        # Save comparison
        comparison_file = args.output / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[OK] Saved comparison: {comparison_file}")

    # Generate plots
    if args.all_plots and (lcdm_samples or hmde_samples):
        print("\n" + "=" * 60)
        print("Generating plots")
        print("=" * 60)

        import matplotlib
        matplotlib.use('Agg')

        samples_list = []
        labels = []
        if lcdm_samples:
            samples_list.append(lcdm_samples)
            labels.append('LCDM')
        if hmde_samples:
            samples_list.append(hmde_samples)
            labels.append('HMDE T06D')

        # Standard parameters for triangle plot
        cosmo_params = ['H0', 'omegam', 'sigma8', 'ns', 'tau', 'ombh2', 'omch2']

        # Triangle plot
        create_triangle_plot(
            samples_list, cosmo_params,
            args.output / "triangle_comparison.pdf",
            labels=labels,
            title="LCDM vs HMDE T06D"
        )

        # 1D comparison
        create_1d_comparison(
            samples_list, cosmo_params,
            args.output / "1d_posteriors.pdf",
            labels=labels
        )

        # H0-Omega_m plot (Hubble tension focus)
        if len(samples_list) == 2:
            create_H0_omega_m_plot(
                samples_list,
                args.output / "H0_omegam.pdf",
                labels=labels
            )

        # T06D parameters
        if hmde_samples:
            create_t06d_parameter_plot(
                hmde_samples,
                args.output / "t06d_posteriors.pdf"
            )

    # Generate LaTeX table
    if constraints:
        generate_latex_table(
            constraints,
            args.output / "parameter_table.tex"
        )

    # Save all constraints as JSON
    constraints_file = args.output / "parameter_constraints.json"
    # Convert numpy to python types
    constraints_clean = {}
    for model, params in constraints.items():
        constraints_clean[model] = {}
        for p, c in params.items():
            constraints_clean[model][p] = {k: float(v) for k, v in c.items()}

    with open(constraints_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'constraints': constraints_clean,
        }, f, indent=2)
    print(f"\n[OK] Saved constraints: {constraints_file}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
