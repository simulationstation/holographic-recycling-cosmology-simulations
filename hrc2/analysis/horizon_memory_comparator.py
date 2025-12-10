"""Horizon-Memory Refinement Comparator.

This module provides tools for comparing all T06A-D refinement pathways:
- Loads and aggregates results from all refinements
- Computes merit scores based on multiple criteria
- Identifies the global best model
- Produces comparison plots (radar charts, parameter maps)
- Generates unified summary reports

Merit score calculation:
    score = w_SN * score_SN + w_BAO * score_BAO + w_growth * score_growth
          + w_CMB * score_CMB + w_stability * score_stability + w_effect * score_effect

where:
    - score_SN: SN Ia distance deviation penalty (lower is better)
    - score_BAO: BAO scale deviation penalty
    - score_growth: f*sigma_8 deviation penalty
    - score_CMB: CMB distance D_A deviation penalty
    - score_stability: Perturbation stability score
    - score_effect: H0 tension relief effectiveness (higher is better)
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ModelViability:
    """Viability assessment for a single model."""
    model_id: str
    refinement_type: str

    # Primary metrics
    delta_H0_percent: float  # Percent H0 shift at z=0
    cmb_deviation_percent: float  # Percent D_A deviation at z*
    sn_deviation_percent: float = 0.0  # SN Ia distance deviation
    bao_deviation_percent: float = 0.0  # BAO scale deviation
    growth_deviation_percent: float = 0.0  # f*sigma8 deviation

    # Stability
    perturbation_stable: bool = True

    # Classification
    viability_class: str = "unknown"  # "gold", "silver", "bronze", "ruled_out"
    merit_score: float = 0.0

    # Parameters
    parameters: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "refinement_type": self.refinement_type,
            "delta_H0_percent": float(self.delta_H0_percent),
            "cmb_deviation_percent": float(self.cmb_deviation_percent),
            "sn_deviation_percent": float(self.sn_deviation_percent),
            "bao_deviation_percent": float(self.bao_deviation_percent),
            "growth_deviation_percent": float(self.growth_deviation_percent),
            "perturbation_stable": self.perturbation_stable,
            "viability_class": self.viability_class,
            "merit_score": float(self.merit_score),
            "parameters": self.parameters,
        }


@dataclass
class ComparisonResult:
    """Result of comparing all refinement models."""
    refinement_results: Dict[str, List[ModelViability]]
    best_per_refinement: Dict[str, ModelViability]
    global_best: Optional[ModelViability]

    # Summary statistics
    n_gold: int = 0
    n_silver: int = 0
    n_bronze: int = 0
    n_ruled_out: int = 0

    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "n_gold": self.n_gold,
                "n_silver": self.n_silver,
                "n_bronze": self.n_bronze,
                "n_ruled_out": self.n_ruled_out,
                "total_models": sum([
                    len(v) for v in self.refinement_results.values()
                ]),
            },
            "thresholds": self.thresholds,
            "global_best": self.global_best.to_dict() if self.global_best else None,
            "best_per_refinement": {
                k: v.to_dict() for k, v in self.best_per_refinement.items()
            },
            "refinement_summaries": {
                ref: {
                    "n_models": len(models),
                    "n_viable": sum(1 for m in models if m.viability_class != "ruled_out"),
                    "best_cmb_deviation": min(m.cmb_deviation_percent for m in models) if models else None,
                    "best_H0_effect": max(m.delta_H0_percent for m in models) if models else None,
                }
                for ref, models in self.refinement_results.items()
            }
        }


class HorizonMemoryComparator:
    """Comparator for horizon-memory refinement models.

    Compares T06A-D refinement pathways and identifies the best models
    based on multiple viability criteria.
    """

    # Default weights for merit score
    DEFAULT_WEIGHTS = {
        "sn": 0.15,      # SN Ia constraints
        "bao": 0.15,     # BAO constraints
        "growth": 0.15,  # Growth of structure
        "cmb": 0.25,     # CMB distance (our focus!)
        "stability": 0.10,  # Perturbation stability
        "effect": 0.20,  # H0 tension relief effectiveness
    }

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        # Gold class: excellent on all metrics
        "gold_cmb_max": 0.3,  # < 0.3% CMB deviation
        "gold_sn_max": 2.0,   # < 2% SN deviation
        "gold_growth_max": 3.0,  # < 3% growth deviation
        "gold_H0_min": 5.0,   # > 5% H0 effect

        # Silver class: good overall
        "silver_cmb_max": 0.5,
        "silver_sn_max": 3.0,
        "silver_growth_max": 5.0,
        "silver_H0_min": 3.0,

        # Bronze class: acceptable
        "bronze_cmb_max": 1.0,
        "bronze_sn_max": 5.0,
        "bronze_growth_max": 7.0,
        "bronze_H0_min": 2.0,
    }

    def __init__(
        self,
        results_dir: str = "results/tests",
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize comparator.

        Args:
            results_dir: Directory containing test results
            weights: Custom merit score weights
            thresholds: Custom viability thresholds
        """
        self.results_dir = Path(results_dir)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def load_refinement_results(self, refinement: str) -> List[ModelViability]:
        """Load results for a specific refinement.

        Args:
            refinement: Refinement name (e.g., "T06A", "T06B", etc.)

        Returns:
            List of ModelViability objects
        """
        models = []

        # Look for result directories
        pattern = f"{refinement}_*"
        refinement_dirs = list(self.results_dir.glob(pattern))

        for result_dir in refinement_dirs:
            # Try to load viability.json or scan results
            viability_path = result_dir / "viability.json"
            scan_path = result_dir / "scan.npz"

            if viability_path.exists():
                models.extend(self._load_viability_json(viability_path, refinement))
            elif scan_path.exists():
                models.extend(self._load_scan_npz(scan_path, refinement))

        return models

    def _load_viability_json(
        self,
        path: Path,
        refinement: str
    ) -> List[ModelViability]:
        """Load models from a viability.json file."""
        models = []
        try:
            with open(path) as f:
                data = json.load(f)

            # Single model or list
            if isinstance(data, list):
                for item in data:
                    models.append(self._parse_model_data(item, refinement))
            else:
                models.append(self._parse_model_data(data, refinement))

        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

        return models

    def _load_scan_npz(
        self,
        path: Path,
        refinement: str
    ) -> List[ModelViability]:
        """Load models from a parameter scan .npz file."""
        models = []
        try:
            data = np.load(path, allow_pickle=True)

            # Extract arrays
            delta_H0 = data.get("delta_H0_percent", None)
            cmb_dev = data.get("cmb_deviation_percent", None)

            if delta_H0 is None or cmb_dev is None:
                return models

            # Flatten and iterate
            for i in range(delta_H0.size):
                idx = np.unravel_index(i, delta_H0.shape)
                h0 = delta_H0[idx]
                cmb = cmb_dev[idx]

                if np.isnan(h0) or np.isnan(cmb):
                    continue

                model = ModelViability(
                    model_id=f"{refinement}_{i}",
                    refinement_type=refinement,
                    delta_H0_percent=float(h0),
                    cmb_deviation_percent=float(cmb),
                )
                models.append(model)

        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

        return models

    def _parse_model_data(
        self,
        data: Dict[str, Any],
        refinement: str
    ) -> ModelViability:
        """Parse model data from dictionary."""
        return ModelViability(
            model_id=data.get("model_id", "unknown"),
            refinement_type=refinement,
            delta_H0_percent=float(data.get("delta_H0_frac", 0)) * 100,
            cmb_deviation_percent=float(data.get("cmb_distance_deviation", 0)),
            sn_deviation_percent=float(data.get("sn_deviation", 0)),
            bao_deviation_percent=float(data.get("bao_deviation", 0)),
            growth_deviation_percent=float(data.get("growth_deviation", 0)),
            perturbation_stable=data.get("perturbation_stable", True),
            parameters=data.get("parameters", {}),
        )

    def compute_merit_score(self, model: ModelViability) -> float:
        """Compute merit score for a model.

        Higher scores are better.

        Score components:
        - SN, BAO, growth, CMB: Penalty for deviation (inverted)
        - stability: Binary bonus
        - effect: Bonus for H0 tension relief

        Args:
            model: ModelViability to score

        Returns:
            Merit score (0-100 scale)
        """
        # Deviation penalties (lower deviation = higher score)
        # Use exponential decay: score = exp(-deviation/scale)
        scale_cmb = 0.5  # CMB is critical
        scale_sn = 3.0
        scale_bao = 3.0
        scale_growth = 5.0

        score_cmb = np.exp(-model.cmb_deviation_percent / scale_cmb) * 100
        score_sn = np.exp(-model.sn_deviation_percent / scale_sn) * 100
        score_bao = np.exp(-model.bao_deviation_percent / scale_bao) * 100
        score_growth = np.exp(-model.growth_deviation_percent / scale_growth) * 100

        # Stability bonus
        score_stability = 100.0 if model.perturbation_stable else 0.0

        # Effect bonus (want larger H0 effect)
        # Saturates around 10%
        score_effect = min(model.delta_H0_percent / 10.0, 1.0) * 100

        # Weighted sum
        w = self.weights
        merit = (
            w["sn"] * score_sn +
            w["bao"] * score_bao +
            w["growth"] * score_growth +
            w["cmb"] * score_cmb +
            w["stability"] * score_stability +
            w["effect"] * score_effect
        )

        return merit

    def classify_model(self, model: ModelViability) -> str:
        """Classify model into viability tier.

        Args:
            model: ModelViability to classify

        Returns:
            Classification: "gold", "silver", "bronze", or "ruled_out"
        """
        t = self.thresholds

        # Check stability first
        if not model.perturbation_stable:
            return "ruled_out"

        # Check gold criteria
        if (model.cmb_deviation_percent <= t["gold_cmb_max"] and
            model.sn_deviation_percent <= t["gold_sn_max"] and
            model.growth_deviation_percent <= t["gold_growth_max"] and
            model.delta_H0_percent >= t["gold_H0_min"]):
            return "gold"

        # Check silver criteria
        if (model.cmb_deviation_percent <= t["silver_cmb_max"] and
            model.sn_deviation_percent <= t["silver_sn_max"] and
            model.growth_deviation_percent <= t["silver_growth_max"] and
            model.delta_H0_percent >= t["silver_H0_min"]):
            return "silver"

        # Check bronze criteria
        if (model.cmb_deviation_percent <= t["bronze_cmb_max"] and
            model.sn_deviation_percent <= t["bronze_sn_max"] and
            model.growth_deviation_percent <= t["bronze_growth_max"] and
            model.delta_H0_percent >= t["bronze_H0_min"]):
            return "bronze"

        return "ruled_out"

    def compare_all_refinements(
        self,
        refinements: List[str] = None
    ) -> ComparisonResult:
        """Compare all refinement pathways.

        Args:
            refinements: List of refinement names to compare
                        (default: ["T06A", "T06B", "T06C", "T06D"])

        Returns:
            ComparisonResult with full analysis
        """
        if refinements is None:
            refinements = ["T06A", "T06B", "T06C", "T06D"]

        all_results: Dict[str, List[ModelViability]] = {}
        best_per_ref: Dict[str, ModelViability] = {}

        for ref in refinements:
            # Load results
            models = self.load_refinement_results(ref)

            # Score and classify each model
            for model in models:
                model.merit_score = self.compute_merit_score(model)
                model.viability_class = self.classify_model(model)

            all_results[ref] = models

            # Find best in this refinement
            if models:
                best = max(models, key=lambda m: m.merit_score)
                best_per_ref[ref] = best

        # Count by class
        all_models = [m for models in all_results.values() for m in models]
        n_gold = sum(1 for m in all_models if m.viability_class == "gold")
        n_silver = sum(1 for m in all_models if m.viability_class == "silver")
        n_bronze = sum(1 for m in all_models if m.viability_class == "bronze")
        n_ruled_out = sum(1 for m in all_models if m.viability_class == "ruled_out")

        # Find global best
        global_best = None
        if all_models:
            global_best = max(all_models, key=lambda m: m.merit_score)

        return ComparisonResult(
            refinement_results=all_results,
            best_per_refinement=best_per_ref,
            global_best=global_best,
            n_gold=n_gold,
            n_silver=n_silver,
            n_bronze=n_bronze,
            n_ruled_out=n_ruled_out,
            thresholds=self.thresholds,
        )

    def generate_radar_plot(
        self,
        comparison: ComparisonResult,
        output_path: str = None,
    ) -> None:
        """Generate radar plot comparing best model from each refinement.

        Args:
            comparison: ComparisonResult from compare_all_refinements
            output_path: Where to save the plot (optional)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        # Categories for radar
        categories = [
            'CMB Distance\n(lower=better)',
            'SN/BAO\n(lower=better)',
            'Growth\n(lower=better)',
            'H0 Effect\n(higher=better)',
            'Stability',
            'Merit Score',
        ]
        n_cats = len(categories)

        # Prepare data for each refinement
        refinements = list(comparison.best_per_refinement.keys())
        n_ref = len(refinements)

        if n_ref == 0:
            print("No refinement results to plot")
            return

        # Normalize scores to [0, 1] for radar
        def normalize_scores(model: ModelViability) -> List[float]:
            # CMB: lower is better, invert and normalize (assume max ~2%)
            cmb_score = max(0, 1 - model.cmb_deviation_percent / 2.0)
            # SN/BAO: lower is better
            sn_score = max(0, 1 - (model.sn_deviation_percent + model.bao_deviation_percent) / 10.0)
            # Growth: lower is better
            growth_score = max(0, 1 - model.growth_deviation_percent / 10.0)
            # H0 effect: higher is better (normalize to ~10%)
            h0_score = min(1, model.delta_H0_percent / 10.0)
            # Stability: binary
            stab_score = 1.0 if model.perturbation_stable else 0.0
            # Merit: already 0-100
            merit_score = model.merit_score / 100.0

            return [cmb_score, sn_score, growth_score, h0_score, stab_score, merit_score]

        # Setup radar chart
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (ref, color) in enumerate(zip(refinements, colors[:n_ref])):
            if ref not in comparison.best_per_refinement:
                continue

            model = comparison.best_per_refinement[ref]
            scores = normalize_scores(model)
            scores += scores[:1]  # Close the polygon

            ax.plot(angles, scores, 'o-', linewidth=2, label=ref, color=color)
            ax.fill(angles, scores, alpha=0.15, color=color)

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.title('Horizon-Memory Refinement Comparison\n(Best Model per Pathway)', fontsize=14, pad=20)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Radar plot saved to {output_path}")

        plt.close()

    def generate_summary_table(
        self,
        comparison: ComparisonResult,
    ) -> str:
        """Generate publication-style summary table.

        Args:
            comparison: ComparisonResult

        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 100)
        lines.append("HORIZON-MEMORY REFINEMENT COMPARISON - SUMMARY TABLE")
        lines.append("=" * 100)
        lines.append("")

        # Header
        header = f"{'Refinement':<12} {'Best Model ID':<25} {'ΔH0 (%)':<10} {'D_A err (%)':<12} {'Merit':<8} {'Class':<10}"
        lines.append(header)
        lines.append("-" * 100)

        # Results per refinement
        for ref in sorted(comparison.best_per_refinement.keys()):
            model = comparison.best_per_refinement[ref]
            row = (
                f"{ref:<12} "
                f"{model.model_id[:24]:<25} "
                f"{model.delta_H0_percent:>8.2f}  "
                f"{model.cmb_deviation_percent:>10.4f}  "
                f"{model.merit_score:>6.1f}  "
                f"{model.viability_class:<10}"
            )
            lines.append(row)

        lines.append("-" * 100)

        # Global best
        if comparison.global_best:
            lines.append("")
            lines.append(f"GLOBAL BEST: {comparison.global_best.model_id}")
            lines.append(f"  Refinement: {comparison.global_best.refinement_type}")
            lines.append(f"  ΔH0 effect: {comparison.global_best.delta_H0_percent:.2f}%")
            lines.append(f"  CMB D_A error: {comparison.global_best.cmb_deviation_percent:.4f}%")
            lines.append(f"  Merit score: {comparison.global_best.merit_score:.1f}")
            lines.append(f"  Class: {comparison.global_best.viability_class.upper()}")

        # Summary counts
        lines.append("")
        lines.append("-" * 100)
        lines.append("VIABILITY SUMMARY:")
        lines.append(f"  Gold:      {comparison.n_gold}")
        lines.append(f"  Silver:    {comparison.n_silver}")
        lines.append(f"  Bronze:    {comparison.n_bronze}")
        lines.append(f"  Ruled out: {comparison.n_ruled_out}")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_comparison(
        self,
        comparison: ComparisonResult,
        output_dir: str = "results/tests/T06_comparison",
    ) -> None:
        """Save full comparison results.

        Args:
            comparison: ComparisonResult to save
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"Summary saved to {summary_path}")

        # Save table
        table_path = os.path.join(output_dir, "comparison_table.txt")
        table = self.generate_summary_table(comparison)
        with open(table_path, 'w') as f:
            f.write(table)
        print(f"Table saved to {table_path}")

        # Generate radar plot
        radar_path = os.path.join(output_dir, "refinement_radar.png")
        self.generate_radar_plot(comparison, radar_path)


def compare_horizon_memory_refinements(
    results_dir: str = "results/tests",
    output_dir: str = "results/tests/T06_comparison",
) -> ComparisonResult:
    """Convenience function to run full comparison.

    Args:
        results_dir: Directory containing refinement results
        output_dir: Directory for comparison outputs

    Returns:
        ComparisonResult
    """
    comparator = HorizonMemoryComparator(results_dir=results_dir)
    comparison = comparator.compare_all_refinements()
    comparator.save_comparison(comparison, output_dir)

    # Print summary
    print("\n" + comparator.generate_summary_table(comparison))

    return comparison
