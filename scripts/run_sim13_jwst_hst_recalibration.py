#!/usr/bin/env python3
"""
SIMULATION 13: HST vs JWST Cepheid Recalibration Test

This simulation tests how instrument-dependent Cepheid photometry systematics
(HST vs JWST) can shift the inferred H0 in a SH0ES-like ladder.

The experiment:
1. Simulate Cepheid observations in two instruments (HST-like and JWST-like)
2. Include: zero-point offsets, color terms, non-linearity, crowding
3. Calibrate PL zero-point using each instrument
4. Propagate through the SN Ia ladder to H0
5. Compute ΔH0 = H0_JWST - H0_HST

Grid of instrument systematics explored:
- Zero-point differences (HST vs JWST)
- Color term differences
- Non-linearity differences
- Crowding differences (JWST has better resolution)
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrc2.ladder import (
    TrueCosmology,
    CepheidPLParameters,
    CepheidHost,
    get_default_cepheid_hosts,
    get_default_anchors,
    SNSystematicParameters11B,
    simulate_snia_with_hosts,
    apply_magnitude_limit,
    calibrate_M_B_from_mu,
    InstrumentPhotometrySystematics,
    create_hst_baseline,
    generate_cepheid_data_for_host,
    fit_PL_zero_point_from_instrument,
    compute_host_mu_from_instrument_cepheids,
)
from hrc2.ladder.ladder_pipeline import fit_H0_from_flow


# =============================================================================
# Configuration
# =============================================================================

# True cosmology (Planck-like)
H0_TRUE = 67.5
OMEGA_M = 0.315
OMEGA_L = 0.685

# Instrument systematic grids (keep small for speed)
# Zero-point differences: JWST - HST (mag)
ZP_DIFF_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04]

# Color term differences: JWST - HST (mag per mag)
C_COLOR_DIFF_VALUES = [0.0, 0.02, 0.04, 0.06]

# Non-linearity differences: JWST - HST
C_NL_DIFF_VALUES = [0.0, 0.01, 0.02]

# JWST crowding bias (HST has 0.03 baseline)
JWST_CROWD_VALUES = [0.0]  # Better resolution = less crowding

# Sample sizes
N_CALIB_SN = 40      # Calibrator SNe (per host)
N_FLOW_SN = 200      # Hubble flow SNe
Z_MIN_FLOW = 0.023
Z_MAX_FLOW = 0.15

# Random seed for reproducibility
SEED = 20241213


def create_anchor_hosts_for_sim13(H0: float = 67.5) -> List[CepheidHost]:
    """
    Create anchor-like hosts for PL zero-point calibration.

    These are analogous to anchors (NGC4258, LMC, MW) but represented
    as CepheidHost objects for the instrument simulation.
    """
    # Simplified anchor proxies with known distances
    anchors = [
        CepheidHost(
            name="NGC4258_anchor",
            mu_true=29.40,  # ~7.6 Mpc (maser distance)
            logM_star=10.8,  # High mass = some crowding
            Z=0.0,
            anchor_name="self",
            n_cepheids=50,  # Many Cepheids observed
        ),
        CepheidHost(
            name="LMC_anchor",
            mu_true=18.49,  # ~50 kpc
            logM_star=9.5,  # Lower mass
            Z=-0.3,
            anchor_name="self",
            n_cepheids=100,
        ),
    ]
    return anchors


def run_single_instrument_scenario(
    zp_diff: float,
    c_color_diff: float,
    c_nl_diff: float,
    jwst_crowd: float,
    cosmo_true: TrueCosmology,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single instrument systematic scenario.

    Returns dict with H0_HST, H0_JWST, delta_H0_inst, and all parameters.
    """
    # Create instruments
    inst_hst = create_hst_baseline()  # HST with ~0.03 crowding bias

    inst_jwst = InstrumentPhotometrySystematics(
        name="JWST",
        zp_offset=zp_diff,
        c_color=c_color_diff,
        c_nonlinearity=c_nl_diff,
        delta_mu_crowd=jwst_crowd,
    )

    # PL parameters (true and fit are the same - no PL systematics in SIM 13)
    pl_params = CepheidPLParameters()

    # Get anchor hosts for PL calibration
    anchor_hosts = create_anchor_hosts_for_sim13(cosmo_true.H0)

    # Get SN calibrator hosts
    sn_hosts = get_default_cepheid_hosts(cosmo_true.H0)

    # Generate Cepheid photometry for all hosts
    all_hosts = anchor_hosts + sn_hosts
    cepheid_data_by_host = {}

    for host in all_hosts:
        cepheid_data_by_host[host.name] = generate_cepheid_data_for_host(
            host, pl_params, inst_hst, inst_jwst, rng
        )

    # --- HST-only ladder ---
    # Fit PL zero-point from anchor hosts using HST photometry
    M_W0_fit_hst, rms_hst = fit_PL_zero_point_from_instrument(
        cepheid_data_by_host, anchor_hosts, pl_params, "m_hst"
    )

    # Compute SN host distances using HST-derived PL
    mu_calib_hst = np.array([
        compute_host_mu_from_instrument_cepheids(
            cepheid_data_by_host[host.name], host, M_W0_fit_hst, pl_params, "m_hst"
        )
        for host in sn_hosts
    ])

    # --- JWST-recalibrated ladder ---
    # Fit PL zero-point from anchor hosts using JWST photometry
    M_W0_fit_jwst, rms_jwst = fit_PL_zero_point_from_instrument(
        cepheid_data_by_host, anchor_hosts, pl_params, "m_jwst"
    )

    # Compute SN host distances using JWST-derived PL
    mu_calib_jwst = np.array([
        compute_host_mu_from_instrument_cepheids(
            cepheid_data_by_host[host.name], host, M_W0_fit_jwst, pl_params, "m_jwst"
        )
        for host in sn_hosts
    ])

    # --- Generate SN samples ---
    # SN systematic parameters (no SN systematics for SIM 13)
    sn_params = SNSystematicParameters11B()

    # Import host population
    from hrc2.ladder.host_population import HostGalaxy, HostPopulationParams, sample_hosts

    # Calibrator SNe (one per host)
    # Simplified: assign each calibrator SN to a host
    n_calib_per_host = max(1, N_CALIB_SN // len(sn_hosts))

    calib_z = []
    calib_mu_true = []
    calib_hosts = []
    for host in sn_hosts:
        for _ in range(n_calib_per_host):
            # Low-z calibrator redshifts (approximate)
            z = 0.001 + rng.exponential(0.005)
            calib_z.append(z)
            calib_mu_true.append(host.mu_true)
            # Create a HostGalaxy for the SN simulation
            calib_hosts.append(HostGalaxy(
                logM_star=host.logM_star,
                Z=host.Z,
                E_BV=rng.exponential(0.03),
            ))

    calib_z = np.array(calib_z)
    calib_mu_true = np.array(calib_mu_true)

    # Simulate calibrator SN observables
    calib_sample = simulate_snia_with_hosts(
        calib_z, calib_hosts, sn_params, cosmo_true, rng,
    )
    calib_sample["mu_true"] = calib_mu_true

    # Hubble flow SNe - generate hosts and redshifts
    flow_z = rng.uniform(Z_MIN_FLOW, Z_MAX_FLOW, N_FLOW_SN)
    pop_params = HostPopulationParams()
    flow_hosts = sample_hosts(N_FLOW_SN, "flow", pop_params, rng)

    flow_sample = simulate_snia_with_hosts(
        flow_z, flow_hosts, sn_params, cosmo_true, rng,
    )
    flow_sample = apply_magnitude_limit(flow_sample, m_lim=sn_params.m_lim_flow)

    # --- Calibrate M_B and fit H0 for each instrument ---
    # Expand mu_calib to match calibrator sample size
    mu_calib_hst_expanded = np.tile(mu_calib_hst, n_calib_per_host)[:len(calib_z)]
    mu_calib_jwst_expanded = np.tile(mu_calib_jwst, n_calib_per_host)[:len(calib_z)]

    # HST ladder
    M_B_fit_hst = calibrate_M_B_from_mu(calib_sample, mu_calib_hst_expanded, sn_params)
    H0_fit_hst, chi2_hst, dof_hst = fit_H0_from_flow(flow_sample, M_B_fit_hst, sn_params)

    # JWST ladder
    M_B_fit_jwst = calibrate_M_B_from_mu(calib_sample, mu_calib_jwst_expanded, sn_params)
    H0_fit_jwst, chi2_jwst, dof_jwst = fit_H0_from_flow(flow_sample, M_B_fit_jwst, sn_params)

    # Compute bias
    delta_H0_hst = H0_fit_hst - H0_TRUE
    delta_H0_jwst = H0_fit_jwst - H0_TRUE
    delta_H0_inst = H0_fit_jwst - H0_fit_hst

    # Mean mu bias
    mu_true_hosts = np.array([h.mu_true for h in sn_hosts])
    mean_mu_bias_hst = float(np.mean(mu_calib_hst - mu_true_hosts))
    mean_mu_bias_jwst = float(np.mean(mu_calib_jwst - mu_true_hosts))

    return {
        "zp_diff": zp_diff,
        "c_color_diff": c_color_diff,
        "c_nl_diff": c_nl_diff,
        "jwst_crowd": jwst_crowd,
        "M_W0_fit_hst": M_W0_fit_hst,
        "M_W0_fit_jwst": M_W0_fit_jwst,
        "M_B_fit_hst": M_B_fit_hst,
        "M_B_fit_jwst": M_B_fit_jwst,
        "H0_fit_hst": H0_fit_hst,
        "H0_fit_jwst": H0_fit_jwst,
        "delta_H0_hst": delta_H0_hst,
        "delta_H0_jwst": delta_H0_jwst,
        "delta_H0_inst": delta_H0_inst,
        "mean_mu_bias_hst": mean_mu_bias_hst,
        "mean_mu_bias_jwst": mean_mu_bias_jwst,
        "chi2_hst": chi2_hst,
        "chi2_jwst": chi2_jwst,
    }


def main():
    """Run full SIM 13 parameter scan."""
    print("=" * 70)
    print("SIMULATION 13: HST vs JWST Cepheid Recalibration Test")
    print("=" * 70)

    # Setup
    cosmo_true = TrueCosmology(H0=H0_TRUE, Omega_m=OMEGA_M, Omega_L=OMEGA_L)
    rng = np.random.default_rng(SEED)

    # Create output directory
    output_dir = Path("results/simulation_13_jwst_hst_recalibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter grid
    param_combos = list(itertools.product(
        ZP_DIFF_VALUES,
        C_COLOR_DIFF_VALUES,
        C_NL_DIFF_VALUES,
        JWST_CROWD_VALUES,
    ))

    n_total = len(param_combos)
    print(f"True H0: {H0_TRUE} km/s/Mpc")
    print(f"Total instrument scenarios: {n_total}")
    print(f"  Zero-point diffs: {ZP_DIFF_VALUES}")
    print(f"  Color term diffs: {C_COLOR_DIFF_VALUES}")
    print(f"  Non-linearity diffs: {C_NL_DIFF_VALUES}")
    print()

    results = []

    for i, (zp_diff, c_color, c_nl, jwst_crowd) in enumerate(param_combos):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] zp={zp_diff:.3f}, color={c_color:.3f}, nl={c_nl:.3f}")

        result = run_single_instrument_scenario(
            zp_diff, c_color, c_nl, jwst_crowd, cosmo_true, rng
        )
        results.append(result)

    # Save results
    output_file = output_dir / "scan_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved {len(results)} results to {output_file}")

    # Quick summary
    delta_H0_inst_values = [r["delta_H0_inst"] for r in results]
    print()
    print("Quick Summary:")
    print(f"  Max |ΔH0_inst| (JWST - HST): {max(abs(d) for d in delta_H0_inst_values):.2f} km/s/Mpc")
    print(f"  Mean |ΔH0_inst|: {np.mean(np.abs(delta_H0_inst_values)):.2f} km/s/Mpc")
    print(f"  Median |ΔH0_inst|: {np.median(np.abs(delta_H0_inst_values)):.2f} km/s/Mpc")


if __name__ == "__main__":
    main()
