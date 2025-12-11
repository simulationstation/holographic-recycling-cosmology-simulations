#!/usr/bin/env python3
"""
SIMULATION 11C: Combined Calibrator + SN Ia Systematics Scan

Extends SIM 11B by adding explicit biases in calibrator distances
(Cepheid/TRGB-based μ_calib).

Scans over:
- SN Ia standardization systematics (from 11B):
  - Population drift (alpha_pop)
  - Metallicity dependence (gamma_Z)
  - Host mass step (delta_M_step)
  - Color law mismatch (delta_beta)

- Calibrator distance biases (new in 11C):
  - Global zero-point (delta_mu_global)
  - Metallicity-dependent (k_mu_Z)
  - Crowding/blending (delta_mu_crowd)

Goal: Quantify whether combined systematics can produce ~5-6 km/s/Mpc H0 bias.
"""

import sys
import os
import json
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.ladder.cosmology_baseline import TrueCosmology
from hrc2.ladder.host_population import HostPopulationParams, sample_hosts
from hrc2.ladder.snia_salt2 import (
    SNSystematicParameters11B,
    simulate_snia_with_hosts,
    apply_magnitude_limit,
)
from hrc2.ladder.ladder_pipeline import calibrate_M_B_from_mu, fit_H0_from_flow
from hrc2.ladder.calibrator_bias import CalibratorBiasParameters, apply_calibrator_biases


# =============================================================================
# Configuration
# =============================================================================

# True cosmology
TRUE_H0 = 67.5
TRUE_OMEGA_M = 0.315

# Sample sizes
N_CALIB = 40
N_FLOW = 200

# Redshift ranges
Z_CALIB_MIN, Z_CALIB_MAX = 0.005, 0.03
Z_FLOW_MIN, Z_FLOW_MAX = 0.01, 0.15

# SN systematic parameter grids (from 11B)
ALPHA_POP_VALUES = [0.0, 0.05, 0.10]
GAMMA_Z_VALUES = [0.0, 0.05, 0.10]
DELTA_STEP_VALUES = [0.0, 0.05, 0.10]
DELTA_BETA_VALUES = [0.0, 0.3, 0.5]

# Calibrator bias parameter grids (new in 11C)
DELTA_MU_GLOBAL_VALUES = [0.0, 0.02, 0.04]   # Global zero-point (0%, 1%, 2% distance)
K_MU_Z_VALUES = [0.0, 0.03, 0.06]            # Metallicity dependence (mag/dex)
DELTA_MU_CROWD_VALUES = [0.0, 0.03, 0.06]    # Crowding/blending bias (mag)

# Output directories
RESULTS_DIR = Path("results/simulation_11c_calibrator_plus_sn")
FIGURES_DIR = Path("figures/simulation_11c_calibrator_plus_sn")

# Random seed
RNG_SEED = 12345


def run_scan():
    """Run the combined SN + calibrator systematics scan."""

    # Setup
    rng = np.random.default_rng(RNG_SEED)
    cosmo_true = TrueCosmology(H0=TRUE_H0, Omega_m=TRUE_OMEGA_M, Omega_L=1.0 - TRUE_OMEGA_M)
    host_params = HostPopulationParams()

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate redshift arrays
    z_calib = rng.uniform(Z_CALIB_MIN, Z_CALIB_MAX, size=N_CALIB)
    z_flow = rng.uniform(Z_FLOW_MIN, Z_FLOW_MAX, size=N_FLOW)

    # Sample host galaxies
    hosts_calib = sample_hosts(N_CALIB, "calib", host_params, rng)
    hosts_flow = sample_hosts(N_FLOW, "flow", host_params, rng)

    # Compute total scenarios
    n_sn_scenarios = (len(ALPHA_POP_VALUES) * len(GAMMA_Z_VALUES) *
                     len(DELTA_STEP_VALUES) * len(DELTA_BETA_VALUES))
    n_calib_scenarios = (len(DELTA_MU_GLOBAL_VALUES) * len(K_MU_Z_VALUES) *
                        len(DELTA_MU_CROWD_VALUES))
    n_total = n_sn_scenarios * n_calib_scenarios

    print("=" * 65)
    print("SIMULATION 11C: Combined Calibrator + SN Systematics Scan")
    print("=" * 65)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"True H0: {TRUE_H0} km/s/Mpc")
    print(f"N_calib: {N_CALIB}, N_flow: {N_FLOW}")
    print(f"\nCalibrators: z in [{Z_CALIB_MIN}, {Z_CALIB_MAX}], mean z = {np.mean(z_calib):.3f}")
    print(f"Hubble flow: z in [{Z_FLOW_MIN}, {Z_FLOW_MAX}], mean z = {np.mean(z_flow):.3f}")
    print(f"\nSN systematic scenarios: {n_sn_scenarios}")
    print(f"Calibrator bias scenarios: {n_calib_scenarios}")
    print(f"Total scenarios: {n_total}")
    print()

    all_results = []
    scenario_idx = 0

    # Outer loop: SN systematics (fixed per SN sample generation)
    for alpha_pop, gamma_Z, delta_step, delta_beta in itertools.product(
            ALPHA_POP_VALUES, GAMMA_Z_VALUES, DELTA_STEP_VALUES, DELTA_BETA_VALUES):

        # Create SN systematic parameters
        sn_params = SNSystematicParameters11B(
            M_B_0=-19.3,
            alpha_true=0.14,
            beta_true=3.1,
            alpha_fit=0.14,
            beta_fit=3.1 - delta_beta,  # Mismatch: fit assumes beta - delta_beta
            alpha_pop=alpha_pop,
            gamma_Z=gamma_Z,
            delta_M_step_true=delta_step,
            delta_M_step_fit=0.0,  # Fitter ignores host mass step
            R_V_true=3.1,
            R_V_fit=3.1,
            m_lim_flow=19.5,
            m_lim_calib=18.5,
            sigma_int=0.10,
            sigma_meas=0.08,
        )

        # Simulate SN samples (once per SN param combo)
        calib_raw = simulate_snia_with_hosts(z_calib, hosts_calib, sn_params, cosmo_true, rng)
        flow_raw = simulate_snia_with_hosts(z_flow, hosts_flow, sn_params, cosmo_true, rng)

        # Apply magnitude limits
        calib = apply_magnitude_limit(calib_raw, sn_params.m_lim_calib)
        flow = apply_magnitude_limit(flow_raw, sn_params.m_lim_flow)

        # Get calibrator host properties for bias application
        n_calib_surviving = len(calib["z"])
        hosts_calib_surviving = hosts_calib[:n_calib_surviving]

        # Inner loop: calibrator biases
        for delta_mu_global, k_mu_Z, delta_mu_crowd in itertools.product(
                DELTA_MU_GLOBAL_VALUES, K_MU_Z_VALUES, DELTA_MU_CROWD_VALUES):

            scenario_idx += 1

            # Create calibrator bias parameters
            bias_params = CalibratorBiasParameters(
                delta_mu_global=delta_mu_global,
                k_mu_Z=k_mu_Z,
                delta_mu_crowd=delta_mu_crowd,
                logM_crowd_threshold=10.5,
                sigma_field=0.0,  # No random field scatter for reproducibility
            )

            # Compute biased calibrator distances
            mu_true_calib = calib["mu_true"]
            mu_biased_calib = apply_calibrator_biases(
                mu_true_calib, hosts_calib_surviving, bias_params, rng
            )

            # Step 1: Calibrate M_B using biased distances
            M_B_fit = calibrate_M_B_from_mu(calib, mu_biased_calib, sn_params)

            # Step 2: Fit H0 from Hubble flow
            H0_fit, chi2_flow, dof_flow = fit_H0_from_flow(flow, M_B_fit, sn_params)
            delta_H0 = H0_fit - TRUE_H0

            # Store result
            result = {
                # SN systematics
                "alpha_pop": alpha_pop,
                "gamma_Z": gamma_Z,
                "delta_step_true": delta_step,
                "delta_beta": delta_beta,
                # Calibrator biases
                "delta_mu_global": delta_mu_global,
                "k_mu_Z": k_mu_Z,
                "delta_mu_crowd": delta_mu_crowd,
                # Results
                "H0_true": TRUE_H0,
                "H0_fit": H0_fit,
                "delta_H0": delta_H0,
                "M_B_fit": M_B_fit,
                "N_calib": n_calib_surviving,
                "N_flow": len(flow["z"]),
                "chi2_flow": chi2_flow,
                "dof_flow": dof_flow,
            }
            all_results.append(result)

            # Progress output
            if scenario_idx % 200 == 1 or scenario_idx == n_total:
                print(f"  [{scenario_idx}/{n_total}] "
                      f"α_pop={alpha_pop:.2f}, γ_Z={gamma_Z:.2f}, "
                      f"δμ_glob={delta_mu_global:.2f}, k_μZ={k_mu_Z:.2f} "
                      f"=> ΔH0={delta_H0:+.2f}")

    # Save results
    output = {
        "simulation": "SIM11C_CALIBRATOR_PLUS_SN",
        "date": datetime.now().isoformat(),
        "true_cosmology": {"H0": TRUE_H0, "Omega_m": TRUE_OMEGA_M},
        "sample_sizes": {
            "N_calib": N_CALIB,
            "N_flow": N_FLOW,
            "z_calib_range": [Z_CALIB_MIN, Z_CALIB_MAX],
            "z_flow_range": [Z_FLOW_MIN, Z_FLOW_MAX],
        },
        "parameter_grids": {
            "sn_systematics": {
                "alpha_pop": ALPHA_POP_VALUES,
                "gamma_Z": GAMMA_Z_VALUES,
                "delta_step": DELTA_STEP_VALUES,
                "delta_beta": DELTA_BETA_VALUES,
            },
            "calibrator_biases": {
                "delta_mu_global": DELTA_MU_GLOBAL_VALUES,
                "k_mu_Z": K_MU_Z_VALUES,
                "delta_mu_crowd": DELTA_MU_CROWD_VALUES,
            },
        },
        "n_scenarios": n_total,
        "results": all_results,
    }

    results_file = RESULTS_DIR / "scan_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print quick summary
    delta_H0_all = np.array([r["delta_H0"] for r in all_results])

    print("\n" + "=" * 65)
    print("SIMULATION 11C SCAN SUMMARY")
    print("=" * 65)
    print(f"True H0: {TRUE_H0} km/s/Mpc")
    print(f"N scenarios: {n_total}")
    print(f"\nMax |ΔH0|: {np.max(np.abs(delta_H0_all)):.2f} km/s/Mpc")
    print(f"Mean |ΔH0|: {np.mean(np.abs(delta_H0_all)):.2f} km/s/Mpc")
    print(f"Median |ΔH0|: {np.median(np.abs(delta_H0_all)):.2f} km/s/Mpc")

    # Count thresholds
    for thresh in [3, 4, 5, 6]:
        n_above = np.sum(np.abs(delta_H0_all) >= thresh)
        print(f"Scenarios with |ΔH0| >= {thresh}: {n_above}")

    # Find extremes
    idx_max_pos = np.argmax(delta_H0_all)
    idx_max_neg = np.argmin(delta_H0_all)

    r_pos = all_results[idx_max_pos]
    r_neg = all_results[idx_max_neg]

    print(f"\nLargest positive bias (H0 too HIGH):")
    print(f"  SN: α_pop={r_pos['alpha_pop']:.2f}, γ_Z={r_pos['gamma_Z']:.2f}, "
          f"ΔM_step={r_pos['delta_step_true']:.2f}, Δβ={r_pos['delta_beta']:.1f}")
    print(f"  Calib: δμ_glob={r_pos['delta_mu_global']:.2f}, k_μZ={r_pos['k_mu_Z']:.2f}, "
          f"δμ_crowd={r_pos['delta_mu_crowd']:.2f}")
    print(f"  H0_fit={r_pos['H0_fit']:.2f} km/s/Mpc, ΔH0={r_pos['delta_H0']:+.2f}")

    print(f"\nLargest negative bias (H0 too LOW):")
    print(f"  SN: α_pop={r_neg['alpha_pop']:.2f}, γ_Z={r_neg['gamma_Z']:.2f}, "
          f"ΔM_step={r_neg['delta_step_true']:.2f}, Δβ={r_neg['delta_beta']:.1f}")
    print(f"  Calib: δμ_glob={r_neg['delta_mu_global']:.2f}, k_μZ={r_neg['k_mu_Z']:.2f}, "
          f"δμ_crowd={r_neg['delta_mu_crowd']:.2f}")
    print(f"  H0_fit={r_neg['H0_fit']:.2f} km/s/Mpc, ΔH0={r_neg['delta_H0']:+.2f}")

    print("=" * 65)


if __name__ == "__main__":
    run_scan()
