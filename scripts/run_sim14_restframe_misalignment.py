#!/usr/bin/env python3
"""
SIMULATION 14: Rest-Frame Misalignment Bias on the Hubble Constant (RFMB-H0)

This simulation tests how using the wrong cosmic rest frame can bias H0 inference.

The experiment:
1. Generate SN Ia samples with realistic sky coverage and redshift distribution
2. Apply kinematic effects from a "true" rest frame (radio dipole ~1000 km/s)
3. "Reduce" the data using a "wrong" rest frame (CMB dipole ~369 km/s)
4. Fit H0 and compute ΔH0 = H0_fit - H0_true

Grid of scenarios:
- v_true: True frame velocity [600, 800, 1000, 1200] km/s
- sky_coverage: "isotropic", "toward_apex", "away_from_apex"
- Monte Carlo realizations per scenario

Key output: Distribution of ΔH0 for each (v_true, sky_coverage) combination
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.restframe import (
    RestFrameDefinition,
    get_heliocentric_velocity,
    generate_sn_catalog,
    fit_H0_with_frame_correction,
    fit_H0_true_frame,
    C_LIGHT,
)


# =============================================================================
# Configuration
# =============================================================================

# True cosmology
H0_TRUE = 67.5
OMEGA_M = 0.315

# CMB dipole (what we assume)
V_CMB = 369.82  # km/s
L_CMB = 264.021  # degrees
B_CMB = 48.253   # degrees

# True rest frame velocity grid (radio dipole hypothesis)
V_TRUE_VALUES = [600.0, 800.0, 1000.0, 1200.0]

# Sky coverage scenarios
SKY_COVERAGE_VALUES = ["isotropic", "toward_apex", "away_from_apex"]

# Sample parameters
N_SN = 300           # Number of SNe per realization
Z_MIN = 0.01         # Minimum redshift
Z_MAX = 0.10         # Maximum redshift
SIGMA_MU = 0.15      # Distance modulus uncertainty [mag]

# Monte Carlo parameters
N_REALIZATIONS = 50  # Number of realizations per scenario
SEED = 20241214


def create_true_frame(v_true: float) -> RestFrameDefinition:
    """
    Create the "true" cosmic rest frame (radio dipole hypothesis).

    The direction is approximately aligned with CMB dipole but with
    larger amplitude.
    """
    return RestFrameDefinition(
        name=f"True_v{v_true:.0f}",
        v_mag=v_true,
        l_apex=270.0,  # Approximate radio dipole direction
        b_apex=45.0,
    )


def create_cmb_correction() -> RestFrameDefinition:
    """
    Create the CMB velocity correction (what we use).
    """
    return RestFrameDefinition(
        name="CMB",
        v_mag=V_CMB,
        l_apex=L_CMB,
        b_apex=B_CMB,
    )


def run_single_realization(
    v_true: float,
    sky_coverage: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single Monte Carlo realization.

    Returns dict with H0_fit, delta_H0, and parameters.
    """
    # Create frames
    true_frame = create_true_frame(v_true)
    cmb_frame = create_cmb_correction()

    # True heliocentric velocity (relative to true frame)
    helio_to_true = RestFrameDefinition(
        name="Helio->True",
        v_mag=v_true,
        l_apex=270.0,
        b_apex=45.0,
    )

    # Generate SN catalog
    sample = generate_sn_catalog(
        n_sn=N_SN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        H0_true=H0_TRUE,
        true_frame=true_frame,
        helio_velocity=helio_to_true,
        sigma_mu=SIGMA_MU,
        rng=rng,
        Omega_m=OMEGA_M,
        sky_coverage=sky_coverage,
    )

    # Fit H0 using TRUE frame (control)
    result_true = fit_H0_true_frame(sample)

    # Fit H0 using CMB frame correction (what we actually do)
    result_cmb = fit_H0_with_frame_correction(sample, cmb_frame)

    # Compute biases
    delta_H0_true = result_true.H0 - H0_TRUE
    delta_H0_cmb = result_cmb.H0 - H0_TRUE

    return {
        "v_true": v_true,
        "sky_coverage": sky_coverage,
        "H0_true_fit": result_true.H0,
        "H0_true_err": result_true.H0_err,
        "H0_cmb_fit": result_cmb.H0,
        "H0_cmb_err": result_cmb.H0_err,
        "delta_H0_true": delta_H0_true,
        "delta_H0_cmb": delta_H0_cmb,
        "bias_from_frame": result_cmb.H0 - result_true.H0,
        "n_sn": sample.n_sn,
        "z_mean": float(np.mean(sample.z_cosmo)),
        "chi2_true": result_true.chi2,
        "chi2_cmb": result_cmb.chi2,
    }


def main():
    """Run full SIM 14 parameter scan with Monte Carlo."""
    print("=" * 70)
    print("SIMULATION 14: Rest-Frame Misalignment Bias (RFMB-H0)")
    print("=" * 70)

    # Setup
    rng = np.random.default_rng(SEED)

    # Create output directory
    output_dir = Path("results/simulation_14_restframe_misalignment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter combinations
    param_combos = list(itertools.product(V_TRUE_VALUES, SKY_COVERAGE_VALUES))

    n_scenarios = len(param_combos)
    n_total = n_scenarios * N_REALIZATIONS

    print(f"True H0: {H0_TRUE} km/s/Mpc")
    print(f"CMB velocity (assumed): {V_CMB:.1f} km/s")
    print(f"True velocity grid: {V_TRUE_VALUES} km/s")
    print(f"Sky coverage scenarios: {SKY_COVERAGE_VALUES}")
    print(f"N_SN per realization: {N_SN}")
    print(f"Redshift range: [{Z_MIN}, {Z_MAX}]")
    print(f"Total scenarios: {n_scenarios}")
    print(f"Realizations per scenario: {N_REALIZATIONS}")
    print(f"Total realizations: {n_total}")
    print()

    results = []

    for i, (v_true, sky_coverage) in enumerate(param_combos):
        print(f"[{i+1}/{n_scenarios}] v_true={v_true:.0f} km/s, sky={sky_coverage}")

        for r in range(N_REALIZATIONS):
            if (r + 1) % 10 == 0:
                print(f"  Realization {r+1}/{N_REALIZATIONS}")

            result = run_single_realization(v_true, sky_coverage, rng)
            result["realization"] = r
            results.append(result)

    # Save all results
    output_file = output_dir / "scan_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved {len(results)} results to {output_file}")

    # Quick summary by scenario
    print()
    print("-" * 70)
    print("QUICK SUMMARY")
    print("-" * 70)
    print(f"{'v_true':<10} {'Sky':<15} {'Mean ΔH0':>12} {'Std ΔH0':>10} {'Max |ΔH0|':>12}")
    print("-" * 70)

    for v_true, sky_coverage in param_combos:
        subset = [r for r in results
                  if r["v_true"] == v_true and r["sky_coverage"] == sky_coverage]
        delta_H0_values = [r["delta_H0_cmb"] for r in subset]

        mean_dH0 = np.mean(delta_H0_values)
        std_dH0 = np.std(delta_H0_values)
        max_abs_dH0 = max(abs(d) for d in delta_H0_values)

        print(f"{v_true:<10.0f} {sky_coverage:<15} {mean_dH0:>12.2f} {std_dH0:>10.2f} {max_abs_dH0:>12.2f}")

    print()

    # Overall statistics
    all_delta_H0 = [r["delta_H0_cmb"] for r in results]
    all_bias_from_frame = [r["bias_from_frame"] for r in results]

    print(f"Overall Max |ΔH0_cmb|: {max(abs(d) for d in all_delta_H0):.2f} km/s/Mpc")
    print(f"Overall Mean |ΔH0_cmb|: {np.mean(np.abs(all_delta_H0)):.2f} km/s/Mpc")
    print(f"Overall Max |bias_from_frame|: {max(abs(d) for d in all_bias_from_frame):.2f} km/s/Mpc")


if __name__ == "__main__":
    main()
