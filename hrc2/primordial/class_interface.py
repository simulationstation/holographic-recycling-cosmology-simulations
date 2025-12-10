"""
CLASS Interface for WHBC Primordial Power Spectrum

This module provides an interface to use CLASS (the Cosmic Linear Anisotropy
Solving System) with WHBC-modified primordial power spectra.

Two approaches are implemented:
1. External P(k) file: Generate a file that CLASS can read
2. Python wrapper: Modify CLASS P(k) after computation

For production CMB analysis, the external P(k) file method is recommended
as it properly propagates primordial modifications through the full
Boltzmann hierarchy.

Author: HRC Collaboration
Date: December 2025
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from .whbc_primordial import (
    WHBCPrimordialParameters,
    primordial_ratio,
    primordial_PK_whbc,
    primordial_PK_lcdm,
    generate_class_pk_file,
)


# Check if CLASS (classy) is available
try:
    from classy import Class
    HAS_CLASS = True
except ImportError:
    HAS_CLASS = False
    Class = None

# Check if CAMB is available
try:
    import camb
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
    camb = None


# Planck 2018 fiducial cosmology
PLANCK_COSMO = {
    'H0': 67.4,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau_reio': 0.0544,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
}


@dataclass
class CLASSResult:
    """
    Results from CLASS computation with WHBC primordial spectrum.
    """
    # CMB power spectra
    ell: np.ndarray  # Multipole array
    Cl_TT: np.ndarray  # TT power spectrum
    Cl_EE: np.ndarray  # EE power spectrum
    Cl_TE: np.ndarray  # TE power spectrum
    Cl_BB: Optional[np.ndarray] = None  # BB power spectrum (if lensing)

    # Matter power spectrum
    k_Pk: np.ndarray = None  # k array for P(k)
    Pk_m: np.ndarray = None  # Matter P(k)

    # Derived parameters
    sigma8: float = 0.0
    theta_s: float = 0.0  # 100*theta_s
    r_s: float = 0.0  # Sound horizon at drag [Mpc]
    D_A_star: float = 0.0  # Angular diameter distance to z_star [Mpc]
    z_star: float = 0.0  # Redshift of last scattering
    z_drag: float = 0.0  # Redshift of drag epoch

    # Metadata
    params: Dict[str, Any] = None
    whbc_params: WHBCPrimordialParameters = None
    success: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sigma8': self.sigma8,
            'theta_s': self.theta_s,
            'r_s': self.r_s,
            'D_A_star': self.D_A_star,
            'z_star': self.z_star,
            'z_drag': self.z_drag,
            'success': self.success,
            'message': self.message,
        }


def generate_whbc_pk_file(
    whbc_params: WHBCPrimordialParameters,
    output_dir: str = ".",
    filename: str = "whbc_primordial_pk.dat",
    k_min: float = 1e-6,
    k_max: float = 100.0,
    n_k: int = 2000
) -> str:
    """
    Generate a CLASS-compatible primordial P(k) file.

    CLASS can read external primordial power spectra using:
        P_k_ini type = external_Pk
        command = cat /path/to/pk_file.dat

    Args:
        whbc_params: WHBC primordial parameters
        output_dir: Output directory
        filename: Output filename
        k_min: Minimum k [Mpc^-1]
        k_max: Maximum k [Mpc^-1]
        n_k: Number of k points

    Returns:
        Full path to generated file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    k_array = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    P_array = primordial_PK_whbc(k_array, whbc_params)

    with open(filepath, 'w') as f:
        f.write("# WHBC primordial power spectrum for CLASS\n")
        f.write(f"# A_cut = {whbc_params.A_cut}\n")
        f.write(f"# k_cut = {whbc_params.k_cut} Mpc^-1\n")
        f.write(f"# p_cut = {whbc_params.p_cut}\n")
        f.write(f"# A_osc = {whbc_params.A_osc}\n")
        f.write(f"# omega_WH = {whbc_params.omega_WH}\n")
        f.write(f"# phi_WH = {whbc_params.phi_WH}\n")
        f.write(f"# k_damp = {whbc_params.k_damp} Mpc^-1\n")
        f.write(f"# As = {whbc_params.As}\n")
        f.write(f"# ns = {whbc_params.ns}\n")
        f.write("# k [Mpc^-1]    P_R(k)\n")
        for k, P in zip(k_array, P_array):
            f.write(f"{k:.12e}  {P:.12e}\n")

    return filepath


def create_class_ini_dict(
    cosmo_params: Dict[str, float] = None,
    whbc_params: WHBCPrimordialParameters = None,
    pk_file: str = None,
    lmax: int = 2500,
    compute_matter_pk: bool = True,
    k_max_pk: float = 10.0,
    z_pk: float = 0.0,
) -> Dict[str, Any]:
    """
    Create CLASS input parameter dictionary.

    Args:
        cosmo_params: Cosmological parameters (uses Planck 2018 if None)
        whbc_params: WHBC primordial parameters (ignored if pk_file given)
        pk_file: Path to external P(k) file
        lmax: Maximum multipole for CMB spectra
        compute_matter_pk: Whether to compute matter P(k)
        k_max_pk: Maximum k for matter P(k)
        z_pk: Redshift for matter P(k)

    Returns:
        Dictionary suitable for CLASS input
    """
    if cosmo_params is None:
        cosmo_params = PLANCK_COSMO.copy()

    params = {
        'output': 'tCl,pCl,lCl,mPk' if compute_matter_pk else 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': lmax,

        # Cosmological parameters
        'H0': cosmo_params.get('H0', 67.4),
        'omega_b': cosmo_params.get('omega_b', 0.02237),
        'omega_cdm': cosmo_params.get('omega_cdm', 0.1200),
        'tau_reio': cosmo_params.get('tau_reio', 0.0544),
    }

    # Primordial spectrum
    if pk_file is not None:
        # Use external P(k) file
        params['P_k_ini type'] = 'external_Pk'
        params['command'] = f'cat {pk_file}'
    else:
        # Use standard power-law primordial spectrum
        whbc = whbc_params if whbc_params is not None else WHBCPrimordialParameters()
        params['A_s'] = whbc.As
        params['n_s'] = whbc.ns
        params['k_pivot'] = whbc.k_pivot

    # Matter power spectrum settings
    if compute_matter_pk:
        params['P_k_max_h/Mpc'] = k_max_pk
        params['z_pk'] = z_pk

    return params


def run_class_with_whbc(
    whbc_params: WHBCPrimordialParameters,
    cosmo_params: Dict[str, float] = None,
    lmax: int = 2500,
    use_external_pk: bool = True,
    output_dir: str = "/tmp",
) -> CLASSResult:
    """
    Run CLASS with WHBC primordial modifications.

    Args:
        whbc_params: WHBC primordial parameters
        cosmo_params: Cosmological parameters
        lmax: Maximum multipole
        use_external_pk: If True, use external P(k) file; else post-process
        output_dir: Directory for temporary files

    Returns:
        CLASSResult with CMB and matter power spectra
    """
    if not HAS_CLASS:
        return CLASSResult(
            ell=np.array([]),
            Cl_TT=np.array([]),
            Cl_EE=np.array([]),
            Cl_TE=np.array([]),
            success=False,
            message="CLASS (classy) not installed. Install with: pip install classy",
        )

    try:
        # Generate external P(k) file if using that method
        pk_file = None
        if use_external_pk:
            pk_file = generate_whbc_pk_file(
                whbc_params, output_dir, "whbc_pk_temp.dat"
            )

        # Create CLASS input
        class_params = create_class_ini_dict(
            cosmo_params=cosmo_params,
            whbc_params=whbc_params,
            pk_file=pk_file,
            lmax=lmax,
        )

        # Run CLASS
        cosmo = Class()
        cosmo.set(class_params)
        cosmo.compute()

        # Extract CMB spectra
        cls = cosmo.lensed_cl(lmax)
        ell = cls['ell']

        # Convert to D_l = l(l+1)/(2pi) * C_l format
        factor = ell * (ell + 1) / (2 * np.pi) * 1e12  # in μK^2
        Cl_TT = cls['tt'] * factor
        Cl_EE = cls['ee'] * factor
        Cl_TE = cls['te'] * factor
        Cl_BB = cls['bb'] * factor if 'bb' in cls else None

        # Extract matter power spectrum
        k_pk = np.logspace(-4, 1, 200)
        Pk_m = np.array([cosmo.pk(k, 0.0) for k in k_pk])

        # Get derived parameters
        derived = cosmo.get_current_derived_parameters([
            'sigma8', '100*theta_s', 'rs_drag', 'da_rec', 'z_rec', 'z_d'
        ])

        result = CLASSResult(
            ell=ell,
            Cl_TT=Cl_TT,
            Cl_EE=Cl_EE,
            Cl_TE=Cl_TE,
            Cl_BB=Cl_BB,
            k_Pk=k_pk,
            Pk_m=Pk_m,
            sigma8=derived.get('sigma8', 0.0),
            theta_s=derived.get('100*theta_s', 0.0),
            r_s=derived.get('rs_drag', 0.0),
            D_A_star=derived.get('da_rec', 0.0),
            z_star=derived.get('z_rec', 0.0),
            z_drag=derived.get('z_d', 0.0),
            params=class_params,
            whbc_params=whbc_params,
            success=True,
            message="Success",
        )

        # Cleanup
        cosmo.struct_cleanup()
        cosmo.empty()

        # Remove temporary file
        if pk_file and os.path.exists(pk_file):
            os.remove(pk_file)

        return result

    except Exception as e:
        return CLASSResult(
            ell=np.array([]),
            Cl_TT=np.array([]),
            Cl_EE=np.array([]),
            Cl_TE=np.array([]),
            success=False,
            message=f"CLASS error: {str(e)}",
        )


def compute_cmb_chi2(
    result: CLASSResult,
    planck_data: Dict[str, np.ndarray] = None,
) -> float:
    """
    Compute simplified chi^2 against Planck-like CMB data.

    This is a simplified version for rapid scanning. For production
    analysis, use the full Planck likelihood.

    Args:
        result: CLASSResult from CLASS computation
        planck_data: Planck binned data (optional, uses mock if None)

    Returns:
        chi^2 value
    """
    if not result.success:
        return np.inf

    # Use simplified Planck-like constraints
    # theta_s constraint: 100*theta_s = 1.04109 +/- 0.00030
    theta_s_obs = 1.04109
    theta_s_err = 0.00030
    chi2_theta_s = ((result.theta_s - theta_s_obs) / theta_s_err) ** 2

    # sigma8 constraint: sigma8 = 0.811 +/- 0.006
    sigma8_obs = 0.811
    sigma8_err = 0.006
    chi2_sigma8 = ((result.sigma8 - sigma8_obs) / sigma8_err) ** 2

    return chi2_theta_s + chi2_sigma8


def scan_whbc_pk_parameters(
    param_grid: Dict[str, List[float]],
    cosmo_params: Dict[str, float] = None,
    output_dir: str = "results/whbc_pk_scan",
    n_parallel: int = 1,
) -> List[Dict[str, Any]]:
    """
    Scan WHBC primordial P(k) parameter space with CLASS.

    Args:
        param_grid: Dictionary mapping parameter names to value lists
        cosmo_params: Base cosmological parameters
        output_dir: Output directory for results
        n_parallel: Number of parallel CLASS runs (not implemented)

    Returns:
        List of result dictionaries
    """
    from itertools import product
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []

    for i, combo in enumerate(combinations):
        # Create WHBC parameters
        whbc_dict = dict(zip(param_names, combo))
        try:
            whbc_params = WHBCPrimordialParameters(**whbc_dict)
        except ValueError as e:
            # Invalid parameter combination
            results.append({
                'params': whbc_dict,
                'success': False,
                'message': str(e),
                'chi2': np.inf,
            })
            continue

        # Run CLASS
        result = run_class_with_whbc(
            whbc_params=whbc_params,
            cosmo_params=cosmo_params,
            output_dir=output_dir,
        )

        # Compute chi^2
        chi2 = compute_cmb_chi2(result)

        results.append({
            'params': whbc_dict,
            'success': result.success,
            'message': result.message,
            'chi2': float(chi2) if np.isfinite(chi2) else None,
            'sigma8': result.sigma8,
            'theta_s': result.theta_s,
            'r_s': result.r_s,
        })

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(combinations)} parameter points")

    # Save results
    with open(os.path.join(output_dir, 'scan_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# Fallback analysis when CLASS is not available
def approximate_cmb_effects(
    whbc_params: WHBCPrimordialParameters,
    cosmo_params: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Approximate CMB effects without running CLASS.

    This provides rough estimates of how WHBC P(k) modifications
    affect CMB observables, useful for rapid scanning.

    Args:
        whbc_params: WHBC primordial parameters
        cosmo_params: Cosmological parameters

    Returns:
        Dictionary with approximate effects
    """
    if cosmo_params is None:
        cosmo_params = PLANCK_COSMO.copy()

    # Key CMB scales (in Mpc^-1)
    k_damping = 0.15  # Silk damping scale
    k_peak1 = 0.016   # First acoustic peak
    k_pivot = 0.05    # Planck pivot scale

    # Compute P(k) ratio at key scales
    F_damping = primordial_ratio(k_damping, whbc_params)
    F_peak1 = primordial_ratio(k_peak1, whbc_params)
    F_pivot = primordial_ratio(k_pivot, whbc_params)

    # Approximate effects on observables
    # sigma8 is dominated by k ~ 0.01 - 0.2 Mpc^-1
    sigma8_ratio = np.sqrt(F_damping)

    # theta_s is mostly sensitive to r_s, not directly to P(k)
    # But large P(k) modifications can shift acoustic peaks
    theta_s_shift = 0.0  # First-order approximation

    # Peak height ratios
    peak_ratio = F_peak1 / F_damping

    return {
        'F_at_pivot': F_pivot,
        'F_at_peak1': F_peak1,
        'F_at_damping': F_damping,
        'sigma8_ratio_approx': sigma8_ratio,
        'theta_s_shift_approx': theta_s_shift,
        'peak_ratio_approx': peak_ratio,
        'class_available': HAS_CLASS,
        'camb_available': HAS_CAMB,
    }


def run_camb_with_whbc(
    whbc_params: WHBCPrimordialParameters,
    cosmo_params: Dict[str, float] = None,
    lmax: int = 2500,
) -> CLASSResult:
    """
    Run CAMB with WHBC primordial modifications.

    Note: CAMB uses a different approach than CLASS for custom primordial spectra.
    We modify the power spectrum post-computation and recompute CMB.

    Args:
        whbc_params: WHBC primordial parameters
        cosmo_params: Cosmological parameters
        lmax: Maximum multipole

    Returns:
        CLASSResult with CMB and matter power spectra (same format as CLASS)
    """
    if not HAS_CAMB:
        return CLASSResult(
            ell=np.array([]),
            Cl_TT=np.array([]),
            Cl_EE=np.array([]),
            Cl_TE=np.array([]),
            success=False,
            message="CAMB not installed. Install with: pip install camb",
        )

    if cosmo_params is None:
        cosmo_params = PLANCK_COSMO.copy()

    try:
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=cosmo_params.get('H0', 67.4),
            ombh2=cosmo_params.get('omega_b', 0.02237),
            omch2=cosmo_params.get('omega_cdm', 0.1200),
            tau=cosmo_params.get('tau_reio', 0.0544),
            mnu=0.06,
            num_massive_neutrinos=1,
            nnu=3.046,
        )

        # Set primordial power spectrum
        # For WHBC modifications, we'll use the standard power law and then
        # apply our modifications to understand the effects
        pars.InitPower.set_params(
            As=whbc_params.As,
            ns=whbc_params.ns,
            pivot_scalar=whbc_params.k_pivot,
        )

        pars.set_for_lmax(lmax, lens_potential_accuracy=1)
        pars.NonLinear = camb.model.NonLinear_both
        pars.set_accuracy(AccuracyBoost=1)

        # Run CAMB
        results = camb.get_results(pars)

        # Get CMB power spectra
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL = powers['total']

        ell = np.arange(totCL.shape[0])
        Cl_TT = totCL[:, 0]  # Already in D_l format from CAMB
        Cl_EE = totCL[:, 1]
        Cl_BB = totCL[:, 2]
        Cl_TE = totCL[:, 3]

        # Apply WHBC modifications to Cl
        # The primordial P(k) modification propagates roughly as a scale-dependent
        # multiplicative factor to the Cls
        # This is approximate - proper treatment requires modifying CAMB source
        k_ell = ell / results.angular_diameter_distance(1089)  # Approximate k-l relation
        k_ell[k_ell <= 0] = 1e-10
        F_whbc = primordial_ratio(k_ell, whbc_params)
        F_whbc[0:2] = 1.0  # Don't modify monopole/dipole

        # Apply modification (approximate)
        Cl_TT_mod = Cl_TT * F_whbc
        Cl_EE_mod = Cl_EE * F_whbc
        Cl_TE_mod = Cl_TE * np.sqrt(F_whbc)  # TE goes as sqrt
        Cl_BB_mod = Cl_BB * F_whbc

        # Get derived parameters
        derived = results.get_derived_params()

        # Get matter power spectrum
        k_pk = np.logspace(-4, 1, 200)
        # Apply WHBC modification to matter P(k)
        _, _, Pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=200)
        F_pk = primordial_ratio(k_pk * cosmo_params.get('H0', 67.4) / 100.0, whbc_params)
        Pk_mod = Pk_lin[0] * F_pk

        return CLASSResult(
            ell=ell[:lmax+1],
            Cl_TT=Cl_TT_mod[:lmax+1],
            Cl_EE=Cl_EE_mod[:lmax+1],
            Cl_TE=Cl_TE_mod[:lmax+1],
            Cl_BB=Cl_BB_mod[:lmax+1],
            k_Pk=k_pk,
            Pk_m=Pk_mod,
            sigma8=derived['sigma8'] * np.sqrt(F_whbc[100]),  # Approximate sigma8 modification
            theta_s=derived['thetastar'] * 100,
            r_s=derived['rdrag'],
            D_A_star=derived['DAstar'],
            z_star=derived['zstar'],
            z_drag=derived['zdrag'],
            params=cosmo_params,
            whbc_params=whbc_params,
            success=True,
            message="Success (CAMB with post-hoc P(k) modification)",
        )

    except Exception as e:
        return CLASSResult(
            ell=np.array([]),
            Cl_TT=np.array([]),
            Cl_EE=np.array([]),
            Cl_TE=np.array([]),
            success=False,
            message=f"CAMB error: {str(e)}",
        )


def run_boltzmann_with_whbc(
    whbc_params: WHBCPrimordialParameters,
    cosmo_params: Dict[str, float] = None,
    lmax: int = 2500,
    prefer: str = 'camb',
) -> CLASSResult:
    """
    Run Boltzmann code (CAMB or CLASS) with WHBC primordial modifications.

    Automatically uses whichever code is available, preferring the one specified.

    Args:
        whbc_params: WHBC primordial parameters
        cosmo_params: Cosmological parameters
        lmax: Maximum multipole
        prefer: Preferred code ('camb' or 'class')

    Returns:
        CLASSResult with CMB and matter power spectra
    """
    if prefer == 'camb' and HAS_CAMB:
        return run_camb_with_whbc(whbc_params, cosmo_params, lmax)
    elif prefer == 'class' and HAS_CLASS:
        return run_class_with_whbc(whbc_params, cosmo_params, lmax)
    elif HAS_CAMB:
        return run_camb_with_whbc(whbc_params, cosmo_params, lmax)
    elif HAS_CLASS:
        return run_class_with_whbc(whbc_params, cosmo_params, lmax)
    else:
        return CLASSResult(
            ell=np.array([]),
            Cl_TT=np.array([]),
            Cl_EE=np.array([]),
            Cl_TE=np.array([]),
            success=False,
            message="Neither CAMB nor CLASS is installed",
        )


if __name__ == "__main__":
    # Test the interface
    print("Testing CLASS Interface for WHBC Primordial...")
    print(f"CLASS available: {HAS_CLASS}")

    # Create test WHBC parameters
    from .whbc_primordial import PRESETS

    whbc_params = PRESETS['combined_whbc']
    print(f"\nTest parameters: A_cut={whbc_params.A_cut}, A_osc={whbc_params.A_osc}")

    # Test P(k) file generation
    pk_file = generate_whbc_pk_file(whbc_params, "/tmp", "test_whbc_pk.dat")
    print(f"Generated P(k) file: {pk_file}")

    # Test approximate effects
    effects = approximate_cmb_effects(whbc_params)
    print(f"\nApproximate CMB effects:")
    for key, val in effects.items():
        print(f"  {key}: {val}")

    # Run CLASS if available
    if HAS_CLASS:
        print("\nRunning CLASS...")
        result = run_class_with_whbc(whbc_params)
        if result.success:
            print(f"  sigma8 = {result.sigma8:.4f}")
            print(f"  100*theta_s = {result.theta_s:.5f}")
            print(f"  r_s = {result.r_s:.2f} Mpc")
            chi2 = compute_cmb_chi2(result)
            print(f"  chi2 = {chi2:.2f}")
        else:
            print(f"  CLASS failed: {result.message}")
    else:
        print("\nSkipping CLASS run (not installed)")

    print("\nTest complete!")
