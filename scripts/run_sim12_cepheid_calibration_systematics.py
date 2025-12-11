#!/usr/bin/env python3
"""
SIMULATION 12: Full Cepheid/TRGB Calibration Error Propagation

Extends SIM 11B/11C by modeling the complete SH0ES-like calibration chain:
  Anchors (NGC 4258, LMC, MW) → Cepheid PL relation → SN host distances → M_B → H0

Scans over:
1. Anchor systematics:
   - Global zero-point (delta_mu_anchor_global)
   - Individual anchor offsets (delta_mu_N4258, delta_mu_LMC, delta_mu_MW)

2. Cepheid PL relation biases:
   - Zero-point (delta_M_W0)
   - Slope (delta_b_W)
   - Metallicity term (delta_gamma_W)

3. Crowding/blending:
   - Anchor crowding (delta_mu_crowd_anchor)
   - Host crowding (delta_mu_crowd_hosts)

4. TRGB calibration:
   - Global TRGB zero-point (delta_mu_TRGB_global)
   - Use Cepheids only vs mixed Cepheid+TRGB

5. SN Ia systematics (subset from 11B):
   - Population drift (alpha_pop)
   - Metallicity dependence (gamma_Z)
   - Host mass step (delta_step)
   - Color law mismatch (delta_beta)

Goal: Quantify whether the combined calibration chain can produce ~5-6 km/s/Mpc H0 bias.
"""

import sys
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
from hrc2.ladder.ladder_pipeline import (
    calibrate_M_B_from_mu,
    fit_H0_from_flow,
)
from hrc2.ladder.cepheid_calibration import (
    get_default_anchors,
    get_default_cepheid_hosts,
    compute_calibrator_mu_from_chain,
)
from hrc2.ladder.cepheid_systematics import (
    CepheidSystematicParameters,
    build_cepheid_pl_params,
    build_trgb_params,
    get_anchor_biases,
)


# =============================================================================
# Configuration
# =============================================================================

# True cosmology
TRUE_H0 = 67.5
TRUE_OMEGA_M = 0.315

# Sample sizes
N_CALIB = 40   # Use 40 SN in calibrator hosts (tied to 5 Cepheid hosts, 8 per host)
N_FLOW = 200

# Redshift ranges
Z_CALIB_MIN, Z_CALIB_MAX = 0.003, 0.015   # Nearby for calibrators
Z_FLOW_MIN, Z_FLOW_MAX = 0.01, 0.15

# =============================================================================
# Parameter Grids
# =============================================================================

# SN systematic grids (minimal for speed)
ALPHA_POP_VALUES = [0.0, 0.05]
GAMMA_Z_VALUES = [0.0]              # Fix to reduce grid
DELTA_STEP_VALUES = [0.0, 0.05]
DELTA_BETA_VALUES = [0.0, 0.3]

# Anchor systematic grids (key parameters only)
DELTA_MU_ANCHOR_GLOBAL_VALUES = [0.0, 0.02, 0.04]  # 0-2% distance scale
DELTA_MU_N4258_VALUES = [0.0]                       # Fix individual anchors
DELTA_MU_LMC_VALUES = [0.0]
DELTA_MU_MW_VALUES = [0.0]

# PL relation bias grids (most important)
DELTA_M_W0_VALUES = [0.0, 0.03, 0.05]  # Zero-point bias - key parameter
DELTA_B_W_VALUES = [0.0]                # Fix slope
DELTA_GAMMA_W_VALUES = [0.0, 0.05]      # Metallicity term error

# Crowding grids
DELTA_MU_CROWD_ANCHOR_VALUES = [0.0, 0.03]
DELTA_MU_CROWD_HOSTS_VALUES = [0.0, 0.03]

# TRGB grid
DELTA_MU_TRGB_GLOBAL_VALUES = [0.0]     # Fix for speed

# Calibration mode
USE_TRGB_OPTIONS = [False]              # Cepheid-only for speed

# Output directories
RESULTS_DIR = Path("results/simulation_12_cepheid_calibration")
FIGURES_DIR = Path("figures/simulation_12_cepheid_calibration")

# Random seed
RNG_SEED = 54321


def run_scan():
    """Run the full Cepheid + SN systematics scan."""

    # Setup
    rng = np.random.default_rng(RNG_SEED)
    cosmo_true = TrueCosmology(H0=TRUE_H0, Omega_m=TRUE_OMEGA_M, Omega_L=1.0 - TRUE_OMEGA_M)
    host_params = HostPopulationParams()

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Get default anchors and Cepheid hosts
    anchors = get_default_anchors()
    cepheid_hosts = get_default_cepheid_hosts(cosmo_H0=TRUE_H0)

    # For SN simulation, we'll use the Cepheid host distances as calibrators
    # Generate redshifts that match the Cepheid host distances
    n_ceph_hosts = len(cepheid_hosts)
    n_sn_per_host = N_CALIB // n_ceph_hosts

    # Build calibrator redshifts from Cepheid hosts
    z_calib_list = []
    host_assignment = []  # Which Cepheid host each SN belongs to
    for i, ch in enumerate(cepheid_hosts):
        # Convert mu to z (approximate)
        d_Mpc = 10**((ch.mu_true - 25) / 5)
        z_host = d_Mpc * TRUE_H0 / 299792.458
        # Add small scatter around host redshift
        z_sn = z_host * np.ones(n_sn_per_host) + rng.normal(0, 0.0005, n_sn_per_host)
        z_sn = np.clip(z_sn, 0.001, 0.1)
        z_calib_list.extend(z_sn)
        host_assignment.extend([i] * n_sn_per_host)

    z_calib = np.array(z_calib_list)[:N_CALIB]
    host_assignment = np.array(host_assignment)[:N_CALIB]

    # Generate Hubble flow redshifts
    z_flow = rng.uniform(Z_FLOW_MIN, Z_FLOW_MAX, size=N_FLOW)

    # Sample host galaxies for SNe
    hosts_calib = sample_hosts(N_CALIB, "calib", host_params, rng)
    hosts_flow = sample_hosts(N_FLOW, "flow", host_params, rng)

    # Compute total scenarios
    n_sn_scenarios = (
        len(ALPHA_POP_VALUES) * len(GAMMA_Z_VALUES) *
        len(DELTA_STEP_VALUES) * len(DELTA_BETA_VALUES)
    )
    n_cepheid_scenarios = (
        len(DELTA_MU_ANCHOR_GLOBAL_VALUES) *
        len(DELTA_MU_N4258_VALUES) * len(DELTA_MU_LMC_VALUES) * len(DELTA_MU_MW_VALUES) *
        len(DELTA_M_W0_VALUES) * len(DELTA_B_W_VALUES) * len(DELTA_GAMMA_W_VALUES) *
        len(DELTA_MU_CROWD_ANCHOR_VALUES) * len(DELTA_MU_CROWD_HOSTS_VALUES) *
        len(DELTA_MU_TRGB_GLOBAL_VALUES) * len(USE_TRGB_OPTIONS)
    )
    n_total = n_sn_scenarios * n_cepheid_scenarios

    print("=" * 70)
    print("SIMULATION 12: Full Cepheid/TRGB Calibration Chain")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"True H0: {TRUE_H0} km/s/Mpc")
    print(f"N_calib: {N_CALIB} (from {n_ceph_hosts} Cepheid hosts)")
    print(f"N_flow: {N_FLOW}")
    print(f"\nAnchors: {[a.name for a in anchors]}")
    print(f"Cepheid hosts: {[ch.name for ch in cepheid_hosts]}")
    print(f"\nSN systematic scenarios: {n_sn_scenarios}")
    print(f"Cepheid/TRGB systematic scenarios: {n_cepheid_scenarios}")
    print(f"Total scenarios: {n_total}")
    print()

    all_results = []
    scenario_idx = 0

    # Outer loop: SN systematics
    for alpha_pop, gamma_Z, delta_step, delta_beta in itertools.product(
            ALPHA_POP_VALUES, GAMMA_Z_VALUES, DELTA_STEP_VALUES, DELTA_BETA_VALUES):

        # Create SN systematic parameters
        sn_params = SNSystematicParameters11B(
            M_B_0=-19.3,
            alpha_true=0.14,
            beta_true=3.1,
            alpha_fit=0.14,
            beta_fit=3.1 - delta_beta,
            alpha_pop=alpha_pop,
            gamma_Z=gamma_Z,
            delta_M_step_true=delta_step,
            delta_M_step_fit=0.0,
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

        n_calib_surviving = len(calib["z"])

        # Inner loop: Cepheid/TRGB systematics
        for (delta_mu_anchor_global, delta_mu_N4258, delta_mu_LMC, delta_mu_MW,
             delta_M_W0, delta_b_W, delta_gamma_W,
             delta_mu_crowd_anchor, delta_mu_crowd_hosts,
             delta_mu_TRGB_global, use_trgb) in itertools.product(
                DELTA_MU_ANCHOR_GLOBAL_VALUES,
                DELTA_MU_N4258_VALUES, DELTA_MU_LMC_VALUES, DELTA_MU_MW_VALUES,
                DELTA_M_W0_VALUES, DELTA_B_W_VALUES, DELTA_GAMMA_W_VALUES,
                DELTA_MU_CROWD_ANCHOR_VALUES, DELTA_MU_CROWD_HOSTS_VALUES,
                DELTA_MU_TRGB_GLOBAL_VALUES, USE_TRGB_OPTIONS):

            scenario_idx += 1

            # Create Cepheid systematic parameters
            ceph_sys_params = CepheidSystematicParameters(
                delta_mu_anchor_global=delta_mu_anchor_global,
                delta_mu_N4258=delta_mu_N4258,
                delta_mu_LMC=delta_mu_LMC,
                delta_mu_MW=delta_mu_MW,
                delta_M_W0=delta_M_W0,
                delta_b_W=delta_b_W,
                delta_gamma_W=delta_gamma_W,
                delta_mu_crowd_anchor=delta_mu_crowd_anchor,
                delta_mu_crowd_hosts=delta_mu_crowd_hosts,
                delta_mu_TRGB_global=delta_mu_TRGB_global,
                use_cepheids=True,
                use_trgb=use_trgb,
            )

            # Build PL and TRGB parameters
            pl_params = build_cepheid_pl_params(ceph_sys_params)
            trgb_params = build_trgb_params(ceph_sys_params)
            anchor_biases = get_anchor_biases(ceph_sys_params)

            # Compute biased calibrator distances through the chain
            mu_cepheid_hosts, _ = compute_calibrator_mu_from_chain(
                anchors=anchors,
                hosts=cepheid_hosts,
                pl_params=pl_params,
                trgb_params=trgb_params,
                anchor_biases=anchor_biases,
                delta_mu_anchor_global=delta_mu_anchor_global,
                delta_mu_crowd_anchor=delta_mu_crowd_anchor,
                delta_mu_trgb_global=delta_mu_TRGB_global,
                use_cepheids=True,
                use_trgb=use_trgb,
                rng=None,  # No random noise for reproducibility
            )

            # Map Cepheid host distances to SN calibrators
            # Each SN gets the distance of its host galaxy
            mu_calib_biased = np.array([
                mu_cepheid_hosts[host_assignment[i] % len(mu_cepheid_hosts)]
                for i in range(n_calib_surviving)
            ])

            # Step 1: Calibrate M_B using biased distances
            M_B_fit = calibrate_M_B_from_mu(calib, mu_calib_biased, sn_params)

            # Step 2: Fit H0 from Hubble flow
            H0_fit, chi2_flow, dof_flow = fit_H0_from_flow(flow, M_B_fit, sn_params)
            delta_H0 = H0_fit - TRUE_H0

            # Compute mean calibrator distance bias
            mu_true_hosts = np.array([ch.mu_true for ch in cepheid_hosts])
            mean_mu_bias = float(np.mean(mu_cepheid_hosts - mu_true_hosts))

            # Store result
            result = {
                # SN systematics
                "alpha_pop": alpha_pop,
                "gamma_Z": gamma_Z,
                "delta_step_true": delta_step,
                "delta_beta": delta_beta,
                # Anchor systematics
                "delta_mu_anchor_global": delta_mu_anchor_global,
                "delta_mu_N4258": delta_mu_N4258,
                "delta_mu_LMC": delta_mu_LMC,
                "delta_mu_MW": delta_mu_MW,
                # PL relation biases
                "delta_M_W0": delta_M_W0,
                "delta_b_W": delta_b_W,
                "delta_gamma_W": delta_gamma_W,
                # Crowding
                "delta_mu_crowd_anchor": delta_mu_crowd_anchor,
                "delta_mu_crowd_hosts": delta_mu_crowd_hosts,
                # TRGB
                "delta_mu_TRGB_global": delta_mu_TRGB_global,
                "use_trgb": use_trgb,
                # Results
                "H0_true": TRUE_H0,
                "H0_fit": H0_fit,
                "delta_H0": delta_H0,
                "M_B_fit": M_B_fit,
                "mean_mu_bias": mean_mu_bias,
                "N_calib": n_calib_surviving,
                "N_flow": len(flow["z"]),
                "chi2_flow": chi2_flow,
                "dof_flow": dof_flow,
            }
            all_results.append(result)

            # Progress output
            if scenario_idx % 500 == 1 or scenario_idx == n_total:
                print(f"  [{scenario_idx}/{n_total}] "
                      f"α_pop={alpha_pop:.2f}, δμ_anch={delta_mu_anchor_global:.2f}, "
                      f"δM_W0={delta_M_W0:.2f}, TRGB={use_trgb} "
                      f"=> ΔH0={delta_H0:+.2f}")

    # Save results
    output = {
        "simulation": "SIM12_CEPHEID_CALIBRATION",
        "date": datetime.now().isoformat(),
        "true_cosmology": {"H0": TRUE_H0, "Omega_m": TRUE_OMEGA_M},
        "sample_sizes": {
            "N_calib": N_CALIB,
            "N_flow": N_FLOW,
            "n_cepheid_hosts": len(cepheid_hosts),
            "z_calib_range": [float(z_calib.min()), float(z_calib.max())],
            "z_flow_range": [Z_FLOW_MIN, Z_FLOW_MAX],
        },
        "anchors": [{"name": a.name, "mu_true": a.mu_true, "Z": a.Z} for a in anchors],
        "cepheid_hosts": [
            {"name": ch.name, "mu_true": ch.mu_true, "Z": ch.Z, "anchor": ch.anchor_name}
            for ch in cepheid_hosts
        ],
        "parameter_grids": {
            "sn_systematics": {
                "alpha_pop": ALPHA_POP_VALUES,
                "gamma_Z": GAMMA_Z_VALUES,
                "delta_step": DELTA_STEP_VALUES,
                "delta_beta": DELTA_BETA_VALUES,
            },
            "anchor_biases": {
                "delta_mu_anchor_global": DELTA_MU_ANCHOR_GLOBAL_VALUES,
                "delta_mu_N4258": DELTA_MU_N4258_VALUES,
                "delta_mu_LMC": DELTA_MU_LMC_VALUES,
                "delta_mu_MW": DELTA_MU_MW_VALUES,
            },
            "pl_biases": {
                "delta_M_W0": DELTA_M_W0_VALUES,
                "delta_b_W": DELTA_B_W_VALUES,
                "delta_gamma_W": DELTA_GAMMA_W_VALUES,
            },
            "crowding": {
                "delta_mu_crowd_anchor": DELTA_MU_CROWD_ANCHOR_VALUES,
                "delta_mu_crowd_hosts": DELTA_MU_CROWD_HOSTS_VALUES,
            },
            "trgb": {
                "delta_mu_TRGB_global": DELTA_MU_TRGB_GLOBAL_VALUES,
                "use_trgb": USE_TRGB_OPTIONS,
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

    print("\n" + "=" * 70)
    print("SIMULATION 12 SCAN SUMMARY")
    print("=" * 70)
    print(f"True H0: {TRUE_H0} km/s/Mpc")
    print(f"N scenarios: {n_total}")
    print(f"\nMax |ΔH0|: {np.max(np.abs(delta_H0_all)):.2f} km/s/Mpc")
    print(f"Mean |ΔH0|: {np.mean(np.abs(delta_H0_all)):.2f} km/s/Mpc")
    print(f"Median |ΔH0|: {np.median(np.abs(delta_H0_all)):.2f} km/s/Mpc")

    # Count thresholds
    for thresh in [2, 3, 4, 5, 6]:
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
    print(f"  Anchor: δμ_glob={r_pos['delta_mu_anchor_global']:.2f}, "
          f"δμ_N4258={r_pos['delta_mu_N4258']:.2f}")
    print(f"  PL: δM_W0={r_pos['delta_M_W0']:.2f}, δb_W={r_pos['delta_b_W']:.2f}, "
          f"δγ_W={r_pos['delta_gamma_W']:.2f}")
    print(f"  Crowd: anchor={r_pos['delta_mu_crowd_anchor']:.2f}, "
          f"hosts={r_pos['delta_mu_crowd_hosts']:.2f}")
    print(f"  TRGB: δμ_TRGB={r_pos['delta_mu_TRGB_global']:.2f}, use={r_pos['use_trgb']}")
    print(f"  H0_fit={r_pos['H0_fit']:.2f} km/s/Mpc, ΔH0={r_pos['delta_H0']:+.2f}")
    print(f"  Mean μ bias: {r_pos['mean_mu_bias']:+.3f} mag")

    print(f"\nLargest negative bias (H0 too LOW):")
    print(f"  SN: α_pop={r_neg['alpha_pop']:.2f}, γ_Z={r_neg['gamma_Z']:.2f}, "
          f"ΔM_step={r_neg['delta_step_true']:.2f}, Δβ={r_neg['delta_beta']:.1f}")
    print(f"  Anchor: δμ_glob={r_neg['delta_mu_anchor_global']:.2f}, "
          f"δμ_N4258={r_neg['delta_mu_N4258']:.2f}")
    print(f"  PL: δM_W0={r_neg['delta_M_W0']:.2f}, δb_W={r_neg['delta_b_W']:.2f}, "
          f"δγ_W={r_neg['delta_gamma_W']:.2f}")
    print(f"  Crowd: anchor={r_neg['delta_mu_crowd_anchor']:.2f}, "
          f"hosts={r_neg['delta_mu_crowd_hosts']:.2f}")
    print(f"  TRGB: δμ_TRGB={r_neg['delta_mu_TRGB_global']:.2f}, use={r_neg['use_trgb']}")
    print(f"  H0_fit={r_neg['H0_fit']:.2f} km/s/Mpc, ΔH0={r_neg['delta_H0']:+.2f}")
    print(f"  Mean μ bias: {r_neg['mean_mu_bias']:+.3f} mag")

    print("=" * 70)


if __name__ == "__main__":
    run_scan()
