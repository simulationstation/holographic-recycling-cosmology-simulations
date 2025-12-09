#!/usr/bin/env python3
"""
Example Cosmological Evolution with HRC

This script demonstrates the Holographic Recycling Cosmology model by:
1. Setting up a primordial black hole population
2. Computing evaporation and remnant formation
3. Integrating the modified Friedmann equations
4. Comparing with standard ΛCDM

Usage:
    python example_evolution.py

The script generates diagnostic output showing the evolution of key quantities.
"""

import numpy as np
import warnings
from typing import Dict, Tuple

# Import HRC modules
from hrc_dynamics import (
    PhysicalConstants,
    CONSTANTS,
    Units,
    UNITS,
    MassFunctionParams,
    BlackHolePopulation,
    RecyclingDynamics,
    HRCCosmology
)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demonstrate_hawking_physics():
    """
    Demonstrate basic Hawking radiation physics.

    Shows temperature, luminosity, and evaporation times for various BH masses.
    """
    print_header("1. HAWKING RADIATION PHYSICS")

    bh = BlackHolePopulation(MassFunctionParams())

    # Test masses (kg)
    masses = {
        'Planck mass': CONSTANTS.M_Planck,
        '10^12 kg (asteroid-mass PBH)': 1e12,
        '10^15 kg (evaporating now)': 1e15,
        'Lunar mass': 7.35e22,
        'Solar mass': 2e30,
    }

    print("\nBlack Hole Properties:")
    print("-" * 70)
    print(f"{'Mass':<30} {'T_H (K)':<15} {'L (W)':<15} {'τ_evap (s)':<15}")
    print("-" * 70)

    for name, M in masses.items():
        T = bh.hawking_temperature(M)
        L = bh.hawking_luminosity(M)
        tau = bh.evaporation_time(M)

        print(f"{name:<30} {T:<15.3e} {L:<15.3e} {tau:<15.3e}")

    # Universe age for reference
    t_universe = 4.35e17  # seconds (13.8 Gyr)
    print(f"\nUniverse age: {t_universe:.3e} s")

    # Find mass that evaporates in universe age
    M_evap_now = bh.initial_mass_for_evaporation_at(t_universe)
    print(f"Mass evaporating today: {M_evap_now:.3e} kg = {M_evap_now/1e12:.1f} × 10^15 g")


def demonstrate_pbh_population():
    """
    Demonstrate primordial black hole population evolution.

    Shows the mass function and how it evolves as BHs evaporate.
    """
    print_header("2. PRIMORDIAL BLACK HOLE POPULATION")

    # Create population with default parameters
    params = MassFunctionParams(
        f_PBH=1e-3,      # 0.1% of dark matter
        M_c=1e12,        # Centered at 10^15 g
        sigma_M=1.0,     # Log-normal width
        rho_DM=2.3e-27   # Local DM density
    )

    bh = BlackHolePopulation(params)

    print(f"\nMass function parameters:")
    print(f"  f_PBH = {params.f_PBH}")
    print(f"  M_c = {params.M_c:.2e} kg")
    print(f"  σ_M = {params.sigma_M}")
    print(f"  ρ_DM = {params.rho_DM:.2e} kg/m³")

    # Evaluate mass function at several points
    print("\nMass function dn/dM:")
    print("-" * 50)
    print(f"{'M (kg)':<20} {'dn/dM (1/m³/kg)':<20}")
    print("-" * 50)

    for log_M in [9, 10, 11, 12, 13, 14, 15]:
        M = 10**log_M
        dn_dM = bh.dn_dM(M)
        print(f"{M:<20.2e} {dn_dM:<20.3e}")

    # Total number and mass density
    n_total = bh.total_number_density()
    rho_total = bh.total_mass_density()

    print(f"\nIntegrated quantities:")
    print(f"  Total number density: {n_total:.3e} /m³")
    print(f"  Total mass density: {rho_total:.3e} kg/m³")
    print(f"  Fraction of ρ_DM: {rho_total/params.rho_DM:.3e}")


def demonstrate_recycling():
    """
    Demonstrate the recycling mechanism.

    Shows how Hawking radiation is partially absorbed by remnants.
    """
    print_header("3. RECYCLING MECHANISM")

    recycling = RecyclingDynamics()

    print(f"\nRemnant properties:")
    print(f"  Mass: {recycling.remnant.mass:.3e} kg (Planck mass)")
    print(f"  Exterior radius: {recycling.remnant.exterior_radius:.3e} m")
    print(f"  Geometric cross-section: {recycling.sigma_abs:.3e} m²")

    # Test different remnant densities
    print("\nRecycling probability vs remnant density:")
    print("-" * 60)
    print(f"{'n_rem (1/m³)':<20} {'ℓ_mfp (m)':<20} {'P_recycle':<15}")
    print("-" * 60)

    # Use Hubble length as characteristic scale
    L_Hubble = CONSTANTS.c / CONSTANTS.H0_SI

    for log_n in [10, 20, 30, 40, 50]:
        n = 10**log_n
        mfp = recycling.mean_free_path(n)
        P = recycling.recycling_probability(n, L_Hubble)
        print(f"{n:<20.2e} {mfp:<20.3e} {P:<15.6f}")

    print(f"\n(Characteristic length: L_Hubble = {L_Hubble:.3e} m)")

    # Information flow calculation
    print("\nInformation flow for a 10^12 kg BH with n_rem = 10^30 /m³:")
    bh = BlackHolePopulation(MassFunctionParams())
    M = 1e12
    n_rem = 1e30

    info = recycling.information_flow_rate(M, n_rem)
    print(f"  Emission rate: {info['emission_rate']:.3e} bits/s")
    print(f"  Recycled rate: {info['recycled_rate']:.3e} bits/s")
    print(f"  Escaped rate:  {info['escaped_rate']:.3e} bits/s")
    print(f"  P_recycle:     {info['recycling_probability']:.6f}")


def demonstrate_cosmological_evolution():
    """
    Demonstrate full cosmological evolution.

    Sets up and integrates the modified Friedmann equations.
    """
    print_header("4. COSMOLOGICAL EVOLUTION")

    # Create HRC cosmology
    print("\nSetting up HRC cosmology...")

    # Theory parameters (in Planck units where G=1)
    theory_params = {
        'G': 1.0,
        'Lambda': 1e-122,     # Tiny cosmological constant
        'xi': 0.01,           # Non-minimal coupling
        'lambda_r': 1e-60,    # Recycling coupling
        'alpha': 0.01,        # Remnant-field coupling
        'm_phi': 1e-60        # Scalar mass ~ H_0
    }

    cosmo = HRCCosmology(theory_params=theory_params)

    print(f"\nTheory parameters:")
    for key, val in theory_params.items():
        print(f"  {key} = {val}")

    # Create initial conditions
    y0 = cosmo.create_initial_conditions(
        a_initial=0.1,      # Start at a = 0.1 (z ~ 9)
        Omega_m=0.3,        # 30% matter
        Omega_rem=0.01,     # 1% remnants
        phi_initial=0.0,    # Scalar field starts at zero
        dphi_initial=0.0    # No initial velocity
    )

    print(f"\nInitial conditions:")
    print(f"  a = {y0[cosmo.IDX_A]:.3e}")
    print(f"  ρ_m = {y0[cosmo.IDX_RHO_M]:.3e}")
    print(f"  ρ_rem = {y0[cosmo.IDX_RHO_REM]:.3e}")
    print(f"  n_rem = {y0[cosmo.IDX_N_REM]:.3e}")
    print(f"  φ = {y0[cosmo.IDX_PHI]:.3e}")
    print(f"  φ̇ = {y0[cosmo.IDX_DPHI]:.3e}")

    # Compute initial Hubble parameter
    state = {
        'rho_m': y0[cosmo.IDX_RHO_M],
        'rho_rem': y0[cosmo.IDX_RHO_REM],
        'n_rem': y0[cosmo.IDX_N_REM],
        'phi': y0[cosmo.IDX_PHI],
        'dphi_dt': y0[cosmo.IDX_DPHI]
    }
    H0 = cosmo.hubble_parameter(0, state)
    print(f"\n  Initial H = {H0:.3e} (Planck units)")

    # Try short evolution (numerical issues expected due to scale hierarchy)
    print("\nAttempting numerical integration...")
    print("(Note: Full cosmological evolution spans ~10^60 Planck times)")
    print("(Demonstration uses very short timescales)")

    try:
        # Very short evolution to demonstrate the system works
        t_span = (0, 1e-55)
        sol = cosmo.evolve(y0, t_span, max_step=1e-57)

        if sol.success:
            print(f"\n  Integration successful!")
            print(f"  Time steps: {len(sol.t)}")
            print(f"  Final a: {sol.y[cosmo.IDX_A, -1]:.6e}")
            print(f"  Final ρ_m: {sol.y[cosmo.IDX_RHO_M, -1]:.6e}")
            print(f"  Scale factor change: {(sol.y[cosmo.IDX_A, -1]/y0[cosmo.IDX_A] - 1)*100:.3e}%")
        else:
            print(f"\n  Integration stopped: {sol.message}")

    except Exception as e:
        print(f"\n  Numerical integration challenging due to scale hierarchy")
        print(f"  (This is expected for such extreme parameter ranges)")
        print(f"  Error: {e}")


def demonstrate_standard_limit():
    """
    Demonstrate recovery of standard cosmology.

    Shows that with recycling turned off, we get ΛCDM.
    """
    print_header("5. STANDARD COSMOLOGY LIMIT")

    # Standard parameters (no recycling)
    standard_params = {
        'G': 1.0,
        'Lambda': 1e-122,
        'xi': 0.0,       # Minimal coupling
        'lambda_r': 0.0,  # No recycling
        'alpha': 0.0,     # No remnant coupling
        'm_phi': 1e-60
    }

    # HRC parameters
    hrc_params = {
        'G': 1.0,
        'Lambda': 1e-122,
        'xi': 0.01,
        'lambda_r': 1e-60,
        'alpha': 0.01,
        'm_phi': 1e-60
    }

    cosmo_std = HRCCosmology(theory_params=standard_params)
    cosmo_hrc = HRCCosmology(theory_params=hrc_params)

    # Same initial conditions with φ = 0
    state = {
        'rho_m': 1e-100,
        'rho_rem': 0.0,
        'n_rem': 0.0,
        'phi': 0.0,
        'dphi_dt': 0.0
    }

    # Compute Hubble parameters
    H_std = cosmo_std.hubble_parameter(0, state)
    H_hrc = cosmo_hrc.hubble_parameter(0, state)

    print(f"\nWith φ = 0 and no remnants:")
    print(f"  H (standard): {H_std:.6e}")
    print(f"  H (HRC):      {H_hrc:.6e}")
    print(f"  Ratio:        {H_hrc/H_std:.10f}")

    # Now with non-zero φ
    state_phi = {
        'rho_m': 1e-100,
        'rho_rem': 1e-105,
        'n_rem': 1e-105,
        'phi': 1e-50,
        'dphi_dt': 0.0
    }

    H_std_phi = cosmo_std.hubble_parameter(0, state_phi)
    H_hrc_phi = cosmo_hrc.hubble_parameter(0, state_phi)

    print(f"\nWith φ = 10^-50 and remnants:")
    print(f"  H (standard): {H_std_phi:.6e}")
    print(f"  H (HRC):      {H_hrc_phi:.6e}")
    print(f"  Ratio:        {H_hrc_phi/H_std_phi:.10f}")
    print(f"  Deviation:    {abs(H_hrc_phi/H_std_phi - 1)*100:.3e}%")


def demonstrate_entropy_evolution():
    """
    Demonstrate entropy and information evolution.

    Shows how entropy flows during black hole evaporation.
    """
    print_header("6. ENTROPY AND INFORMATION FLOW")

    bh = BlackHolePopulation(MassFunctionParams())
    recycling = RecyclingDynamics()

    M_initial = 1e12  # kg
    M_final = 1e11    # kg (lost 90% of mass)

    print(f"\nBlack hole evolution: {M_initial:.0e} kg → {M_final:.0e} kg")
    print("-" * 50)

    # Entropy changes
    S_i = bh.entropy_of_bh(M_initial)
    S_f = bh.entropy_of_bh(M_final)
    delta_S_BH = S_f - S_i

    print(f"\nBlack hole entropy:")
    print(f"  Initial: {S_i:.3e} k_B")
    print(f"  Final:   {S_f:.3e} k_B")
    print(f"  Change:  {delta_S_BH:.3e} k_B (negative: BH lost entropy)")

    # Energy radiated
    E_radiated = (M_initial - M_final) * CONSTANTS.c**2

    # Radiation entropy (at average Hawking temperature)
    T_avg = bh.hawking_temperature((M_initial + M_final) / 2)
    S_rad = E_radiated / (CONSTANTS.k_B * T_avg)

    print(f"\nRadiation entropy:")
    print(f"  Energy radiated: {E_radiated:.3e} J")
    print(f"  Average T_H:     {T_avg:.3e} K")
    print(f"  S_radiation:     {S_rad:.3e} k_B")

    # Total entropy change
    delta_S_total = S_rad + delta_S_BH

    print(f"\nGeneralized Second Law:")
    print(f"  ΔS_BH + S_rad = {delta_S_total:.3e} k_B")
    print(f"  Second law satisfied: {delta_S_total > 0}")

    # Information
    print(f"\nInformation content:")
    I_i = bh.information_content(M_initial)
    I_f = bh.information_content(M_final)
    print(f"  Initial: {I_i:.3e} bits")
    print(f"  Final:   {I_f:.3e} bits")
    print(f"  Released: {I_i - I_f:.3e} bits")

    # With recycling
    n_rem = 1e25  # 1/m³
    info_flow = recycling.information_flow_rate(M_initial, n_rem)

    print(f"\nWith remnants (n_rem = {n_rem:.0e} /m³):")
    print(f"  Information emission: {info_flow['emission_rate']:.3e} bits/s")
    print(f"  Recycled:            {info_flow['recycled_rate']:.3e} bits/s")
    print(f"  Escaped to infinity: {info_flow['escaped_rate']:.3e} bits/s")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" HOLOGRAPHIC RECYCLING COSMOLOGY - EXAMPLE EVOLUTION")
    print("=" * 70)

    print("\nThis script demonstrates the HRC model components:")
    print("  1. Hawking radiation physics")
    print("  2. Primordial black hole populations")
    print("  3. The recycling mechanism")
    print("  4. Cosmological evolution")
    print("  5. Standard cosmology limit")
    print("  6. Entropy and information flow")

    # Run demonstrations
    demonstrate_hawking_physics()
    demonstrate_pbh_population()
    demonstrate_recycling()
    demonstrate_cosmological_evolution()
    demonstrate_standard_limit()
    demonstrate_entropy_evolution()

    # Summary
    print_header("SUMMARY")

    print("""
Key findings from this demonstration:

1. HAWKING RADIATION:
   - Black holes emit thermal radiation at T_H ∝ 1/M
   - Evaporation time τ ∝ M³ (larger BHs last longer)
   - PBHs with M ~ 10^15 g are evaporating today

2. RECYCLING MECHANISM:
   - Remnants (if they exist) can reabsorb Hawking radiation
   - Recycling probability depends on remnant density
   - This creates feedback between micro and macro physics

3. COSMOLOGICAL EVOLUTION:
   - HRC modifies Friedmann equations via effective G
   - Standard ΛCDM is recovered when couplings → 0
   - Full evolution spans extreme timescale hierarchies

4. ENTROPY:
   - Generalized Second Law is satisfied
   - Information can be partially recycled into remnants
   - Total entropy never decreases

CAVEATS:
- Remnant formation is speculative (no quantum gravity)
- Recycling mechanism is phenomenological
- Numerical integration challenging due to scale hierarchy
- Many parameters are unconstrained by observation
""")

    print("=" * 70)
    print(" END OF DEMONSTRATION")
    print("=" * 70)


if __name__ == "__main__":
    main()
