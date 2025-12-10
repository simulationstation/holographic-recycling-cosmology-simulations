"""Plotting modules for HRC cosmology."""

from .phase_diagram import plot_phase_diagram, plot_stability_region
from .w_of_z import plot_effective_w, plot_w_comparison
from .geff_evolution import plot_geff_evolution, plot_hubble_tension_resolution
from .validity_regions import (
    plot_validity_map,
    plot_divergence_redshift,
    plot_delta_h0,
    plot_geff_at_z,
    plot_validity_boundary,
    create_summary_figure,
)
from .potential_comparison import (
    plot_potential_shapes,
    plot_validity_comparison,
    plot_geff_evolution_comparison,
    plot_phi_evolution_comparison,
    create_potential_comparison_figure,
    summarize_potential_results,
    print_potential_comparison_report,
    DEFAULT_POTENTIALS,
)
from .xi_tradeoff import (
    plot_max_delta_G_vs_xi,
    plot_stability_map,
    plot_delta_G_vs_phi0,
    plot_stable_fraction_vs_xi,
    create_xi_tradeoff_summary_figure,
    plot_potential_comparison,
)

__all__ = [
    "plot_phase_diagram",
    "plot_stability_region",
    "plot_effective_w",
    "plot_w_comparison",
    "plot_geff_evolution",
    "plot_hubble_tension_resolution",
    # Validity region plots
    "plot_validity_map",
    "plot_divergence_redshift",
    "plot_delta_h0",
    "plot_geff_at_z",
    "plot_validity_boundary",
    "create_summary_figure",
    # Potential comparison plots
    "plot_potential_shapes",
    "plot_validity_comparison",
    "plot_geff_evolution_comparison",
    "plot_phi_evolution_comparison",
    "create_potential_comparison_figure",
    "summarize_potential_results",
    "print_potential_comparison_report",
    "DEFAULT_POTENTIALS",
    # Xi tradeoff plots
    "plot_max_delta_G_vs_xi",
    "plot_stability_map",
    "plot_delta_G_vs_phi0",
    "plot_stable_fraction_vs_xi",
    "create_xi_tradeoff_summary_figure",
    "plot_potential_comparison",
]
