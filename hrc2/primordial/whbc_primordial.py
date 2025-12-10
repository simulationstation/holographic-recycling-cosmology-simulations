"""
T08_WHBC_PK: WHBC-Motivated Primordial Power Spectrum Modifications

This module implements primordial curvature power spectrum modifications
motivated by White-Hole Boundary Cosmology (WHBC). The modifications act
ONLY through P(k), not the background H(z).

The modified primordial spectrum is:
    P_WHBC(k) = P_LCDM(k) * F_WHBC(k)

where:
    F_WHBC(k) = 1
                + A_cut * exp[-(k / k_cut)^p_cut]            # IR cutoff
                + A_osc * sin(omega * ln(k/k_pivot) + phi)   # Oscillations
                        * exp[-(k / k_damp)^2]               # UV damping

Physical motivation:
- IR cutoff (A_cut term): Pre-geometric phase suppresses large-scale modes
- Oscillations (A_osc term): Boundary condition echo creates log-periodic features
- UV damping: High-k modes damped by boundary smoothing

Author: HRC Collaboration
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from scipy.integrate import quad


# Physical constants
K_PIVOT = 0.05  # Mpc^-1, standard pivot scale
AS_FIDUCIAL = 2.1e-9  # Planck 2018 amplitude
NS_FIDUCIAL = 0.9649  # Planck 2018 spectral index


@dataclass
class WHBCPrimordialParameters:
    """
    Parameters for WHBC-motivated primordial power spectrum.

    The modified spectrum is:
        P_WHBC(k) = P_LCDM(k) * F_WHBC(k)

    where:
        F_WHBC(k) = 1 + A_cut * exp[-(k/k_cut)^p_cut]
                      + A_osc * sin(omega_WH * ln(k/k_pivot) + phi_WH)
                              * exp[-(k/k_damp)^2]

    Attributes:
        A_cut: Amplitude of IR cutoff suppression [-0.1, 0.1]
               Positive = enhancement at low-k
               Negative = suppression at low-k
        k_cut: Characteristic scale of IR cutoff [Mpc^-1], typically < 0.01
        p_cut: Power of cutoff exponential [0.5, 3.0]

        A_osc: Amplitude of log-periodic oscillations [-0.1, 0.1]
        omega_WH: Frequency of oscillations in log(k) space [0, 20]
        phi_WH: Phase of oscillations [0, 2*pi]
        k_damp: UV damping scale [Mpc^-1], typically > 0.1

        k_pivot: Pivot scale for oscillations [Mpc^-1]
        As: Standard LCDM amplitude at pivot
        ns: Standard LCDM spectral index
    """
    # IR cutoff parameters
    A_cut: float = 0.0
    k_cut: float = 0.001  # Mpc^-1
    p_cut: float = 2.0

    # Oscillation parameters
    A_osc: float = 0.0
    omega_WH: float = 0.0  # Frequency in ln(k) space
    phi_WH: float = 0.0    # Phase
    k_damp: float = 1.0    # Mpc^-1, UV damping

    # Standard LCDM parameters
    k_pivot: float = K_PIVOT
    As: float = AS_FIDUCIAL
    ns: float = NS_FIDUCIAL

    def __post_init__(self):
        """Validate parameters."""
        if not -0.5 <= self.A_cut <= 0.5:
            raise ValueError(f"A_cut must be in [-0.5, 0.5], got {self.A_cut}")
        if self.k_cut <= 0:
            raise ValueError(f"k_cut must be positive, got {self.k_cut}")
        if not 0.5 <= self.p_cut <= 4.0:
            raise ValueError(f"p_cut must be in [0.5, 4.0], got {self.p_cut}")
        if not -0.5 <= self.A_osc <= 0.5:
            raise ValueError(f"A_osc must be in [-0.5, 0.5], got {self.A_osc}")
        if self.omega_WH < 0:
            raise ValueError(f"omega_WH must be non-negative, got {self.omega_WH}")
        if self.k_damp <= 0:
            raise ValueError(f"k_damp must be positive, got {self.k_damp}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'A_cut': self.A_cut,
            'k_cut': self.k_cut,
            'p_cut': self.p_cut,
            'A_osc': self.A_osc,
            'omega_WH': self.omega_WH,
            'phi_WH': self.phi_WH,
            'k_damp': self.k_damp,
            'k_pivot': self.k_pivot,
            'As': self.As,
            'ns': self.ns,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WHBCPrimordialParameters':
        """Create from dictionary."""
        return cls(**d)

    def copy(self, **kwargs) -> 'WHBCPrimordialParameters':
        """Create a copy with optional parameter updates."""
        d = self.to_dict()
        d.update(kwargs)
        return WHBCPrimordialParameters(**d)


def primordial_PK_lcdm(
    k: Union[float, np.ndarray],
    As: float = AS_FIDUCIAL,
    ns: float = NS_FIDUCIAL,
    k_pivot: float = K_PIVOT
) -> Union[float, np.ndarray]:
    """
    Standard LCDM primordial curvature power spectrum.

    P_R(k) = A_s * (k / k_pivot)^(n_s - 1)

    Args:
        k: Wavenumber(s) in Mpc^-1
        As: Amplitude at pivot scale
        ns: Scalar spectral index
        k_pivot: Pivot scale in Mpc^-1

    Returns:
        P_R(k) - dimensionless primordial power spectrum
    """
    k = np.asarray(k)
    return As * (k / k_pivot) ** (ns - 1)


def primordial_ratio(
    k: Union[float, np.ndarray],
    params: WHBCPrimordialParameters
) -> Union[float, np.ndarray]:
    """
    WHBC modification factor F_WHBC(k) = P_WHBC(k) / P_LCDM(k).

    F_WHBC(k) = 1
                + A_cut * exp[-(k / k_cut)^p_cut]
                + A_osc * sin(omega_WH * ln(k/k_pivot) + phi_WH)
                        * exp[-(k / k_damp)^2]

    Args:
        k: Wavenumber(s) in Mpc^-1
        params: WHBC primordial parameters

    Returns:
        F_WHBC(k) - ratio of WHBC to LCDM primordial spectrum
    """
    k = np.asarray(k)

    # Base is unity (LCDM limit)
    F = np.ones_like(k, dtype=float)

    # IR cutoff term: suppression/enhancement at low k
    if params.A_cut != 0:
        ir_term = params.A_cut * np.exp(-(k / params.k_cut) ** params.p_cut)
        F = F + ir_term

    # Oscillation term: log-periodic features with UV damping
    if params.A_osc != 0 and params.omega_WH > 0:
        # Log-periodic oscillation
        phase = params.omega_WH * np.log(k / params.k_pivot) + params.phi_WH
        oscillation = np.sin(phase)

        # UV damping envelope
        uv_damping = np.exp(-(k / params.k_damp) ** 2)

        osc_term = params.A_osc * oscillation * uv_damping
        F = F + osc_term

    # Ensure F > 0 (physical constraint)
    F = np.maximum(F, 0.01)

    return F


def primordial_PK_whbc(
    k: Union[float, np.ndarray],
    params: WHBCPrimordialParameters
) -> Union[float, np.ndarray]:
    """
    WHBC-modified primordial curvature power spectrum.

    P_WHBC(k) = P_LCDM(k) * F_WHBC(k)

    Args:
        k: Wavenumber(s) in Mpc^-1
        params: WHBC primordial parameters

    Returns:
        P_WHBC(k) - modified primordial power spectrum
    """
    P_lcdm = primordial_PK_lcdm(k, params.As, params.ns, params.k_pivot)
    F_whbc = primordial_ratio(k, params)
    return P_lcdm * F_whbc


def effective_spectral_index(
    k: float,
    params: WHBCPrimordialParameters,
    dlnk: float = 0.01
) -> float:
    """
    Effective spectral index n_eff(k) = d ln P / d ln k + 1.

    Args:
        k: Wavenumber in Mpc^-1
        params: WHBC primordial parameters
        dlnk: Step size for numerical derivative

    Returns:
        n_eff(k) - effective spectral index at k
    """
    k_minus = k * np.exp(-dlnk / 2)
    k_plus = k * np.exp(dlnk / 2)

    P_minus = primordial_PK_whbc(k_minus, params)
    P_plus = primordial_PK_whbc(k_plus, params)

    dln_P = np.log(P_plus / P_minus)
    n_eff = dln_P / dlnk + 1

    return n_eff


def running_of_spectral_index(
    k: float,
    params: WHBCPrimordialParameters,
    dlnk: float = 0.01
) -> float:
    """
    Running of spectral index alpha = d n_s / d ln k.

    Args:
        k: Wavenumber in Mpc^-1
        params: WHBC primordial parameters
        dlnk: Step size for numerical derivative

    Returns:
        alpha(k) - running at k
    """
    k_minus = k * np.exp(-dlnk / 2)
    k_plus = k * np.exp(dlnk / 2)

    n_minus = effective_spectral_index(k_minus, params, dlnk)
    n_plus = effective_spectral_index(k_plus, params, dlnk)

    alpha = (n_plus - n_minus) / dlnk
    return alpha


def compute_sigma8_ratio(
    params: WHBCPrimordialParameters,
    k_min: float = 1e-4,
    k_max: float = 10.0
) -> float:
    """
    Estimate sigma_8 ratio (WHBC / LCDM) from primordial spectrum ratio.

    sigma_8^2 ~ integral k^2 P(k) W^2(kR) dk

    For R = 8 h^-1 Mpc, the main contribution is around k ~ 0.01 - 0.2 Mpc^-1.

    This is an approximation assuming the transfer function is the same.

    Args:
        params: WHBC primordial parameters
        k_min: Minimum k for integration
        k_max: Maximum k for integration

    Returns:
        sigma8_WHBC / sigma8_LCDM ratio
    """
    R = 8.0  # h^-1 Mpc (approximately 8/0.67 ~ 12 Mpc)

    def window_tophat(k, R):
        """Top-hat window function in Fourier space."""
        x = k * R
        if np.isscalar(x):
            if x < 1e-10:
                return 1.0
            return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        result = np.ones_like(x)
        mask = x > 1e-10
        result[mask] = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / x[mask]**3
        return result

    def integrand_lcdm(k):
        P = primordial_PK_lcdm(k, params.As, params.ns, params.k_pivot)
        W = window_tophat(k, R)
        return k**2 * P * W**2

    def integrand_whbc(k):
        P = primordial_PK_whbc(k, params)
        W = window_tophat(k, R)
        return k**2 * P * W**2

    # Use log-spaced integration for better numerical stability
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), 500)

    integral_lcdm = np.trapezoid(integrand_lcdm(k_array), k_array)
    integral_whbc = np.trapezoid(integrand_whbc(k_array), k_array)

    # sigma8 ratio
    if integral_lcdm > 0:
        return np.sqrt(integral_whbc / integral_lcdm)
    return 1.0


@dataclass
class WHBCPrimordialResult:
    """
    Results from WHBC primordial spectrum analysis.
    """
    params: WHBCPrimordialParameters
    k_array: np.ndarray = field(default_factory=lambda: np.array([]))
    P_lcdm: np.ndarray = field(default_factory=lambda: np.array([]))
    P_whbc: np.ndarray = field(default_factory=lambda: np.array([]))
    F_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    n_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma8_ratio: float = 1.0
    n_s_eff_pivot: float = 0.0
    alpha_pivot: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'params': self.params.to_dict(),
            'sigma8_ratio': self.sigma8_ratio,
            'n_s_eff_pivot': self.n_s_eff_pivot,
            'alpha_pivot': self.alpha_pivot,
        }


def analyze_whbc_primordial(
    params: WHBCPrimordialParameters,
    k_min: float = 1e-5,
    k_max: float = 10.0,
    n_k: int = 500
) -> WHBCPrimordialResult:
    """
    Analyze WHBC primordial spectrum modifications.

    Args:
        params: WHBC primordial parameters
        k_min: Minimum k in Mpc^-1
        k_max: Maximum k in Mpc^-1
        n_k: Number of k points

    Returns:
        WHBCPrimordialResult with computed quantities
    """
    result = WHBCPrimordialResult(params=params)

    # Generate k array
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    result.k_array = k_array

    # Compute spectra
    result.P_lcdm = primordial_PK_lcdm(k_array, params.As, params.ns, params.k_pivot)
    result.P_whbc = primordial_PK_whbc(k_array, params)
    result.F_ratio = primordial_ratio(k_array, params)

    # Compute effective spectral index
    result.n_eff = np.array([effective_spectral_index(k, params) for k in k_array])

    # Compute summary statistics
    result.sigma8_ratio = compute_sigma8_ratio(params, k_min, k_max)
    result.n_s_eff_pivot = effective_spectral_index(params.k_pivot, params)
    result.alpha_pivot = running_of_spectral_index(params.k_pivot, params)

    return result


def generate_class_pk_file(
    params: WHBCPrimordialParameters,
    filename: str,
    k_min: float = 1e-6,
    k_max: float = 100.0,
    n_k: int = 1000
) -> None:
    """
    Generate a CLASS-compatible primordial P(k) file.

    CLASS can read external P(k) files for the primordial spectrum.

    Args:
        params: WHBC primordial parameters
        filename: Output filename
        k_min: Minimum k in Mpc^-1
        k_max: Maximum k in Mpc^-1
        n_k: Number of k points
    """
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    P_array = primordial_PK_whbc(k_array, params)

    with open(filename, 'w') as f:
        f.write("# WHBC primordial power spectrum\n")
        f.write(f"# Parameters: A_cut={params.A_cut}, k_cut={params.k_cut}, p_cut={params.p_cut}\n")
        f.write(f"#             A_osc={params.A_osc}, omega_WH={params.omega_WH}, phi_WH={params.phi_WH}\n")
        f.write(f"#             k_damp={params.k_damp}, As={params.As}, ns={params.ns}\n")
        f.write("# k [Mpc^-1]    P_R(k)\n")
        for k, P in zip(k_array, P_array):
            f.write(f"{k:.10e}  {P:.10e}\n")


def create_cobaya_theory_provider(
    params: WHBCPrimordialParameters
) -> Dict[str, Any]:
    """
    Create a Cobaya-compatible theory provider dictionary for WHBC P(k).

    This can be used to define a custom primordial spectrum in Cobaya.

    Args:
        params: WHBC primordial parameters

    Returns:
        Dictionary suitable for Cobaya theory block
    """
    return {
        'whbc_primordial': {
            'A_cut': params.A_cut,
            'k_cut': params.k_cut,
            'p_cut': params.p_cut,
            'A_osc': params.A_osc,
            'omega_WH': params.omega_WH,
            'phi_WH': params.phi_WH,
            'k_damp': params.k_damp,
        }
    }


# Preset configurations for specific physical scenarios
PRESETS = {
    'lcdm': WHBCPrimordialParameters(),  # Pure LCDM

    'ir_suppression': WHBCPrimordialParameters(
        A_cut=-0.05,
        k_cut=0.001,
        p_cut=2.0,
    ),

    'ir_enhancement': WHBCPrimordialParameters(
        A_cut=0.05,
        k_cut=0.001,
        p_cut=2.0,
    ),

    'weak_oscillations': WHBCPrimordialParameters(
        A_osc=0.02,
        omega_WH=5.0,
        phi_WH=0.0,
        k_damp=0.5,
    ),

    'strong_oscillations': WHBCPrimordialParameters(
        A_osc=0.05,
        omega_WH=10.0,
        phi_WH=0.0,
        k_damp=0.3,
    ),

    'combined_whbc': WHBCPrimordialParameters(
        A_cut=-0.03,
        k_cut=0.0005,
        p_cut=2.0,
        A_osc=0.02,
        omega_WH=6.0,
        phi_WH=np.pi/4,
        k_damp=0.4,
    ),
}


if __name__ == "__main__":
    # Quick test
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Testing WHBC Primordial Module...")

    # Test LCDM
    params_lcdm = WHBCPrimordialParameters()
    k_test = np.logspace(-4, 1, 100)
    P_lcdm = primordial_PK_lcdm(k_test, params_lcdm.As, params_lcdm.ns)

    print(f"LCDM P(k=0.05) = {primordial_PK_lcdm(0.05, params_lcdm.As, params_lcdm.ns):.3e}")
    print(f"Expected As = {params_lcdm.As:.3e}")

    # Test IR suppression
    params_ir = PRESETS['ir_suppression']
    P_ir = primordial_PK_whbc(k_test, params_ir)
    F_ir = primordial_ratio(k_test, params_ir)

    print(f"\nIR suppression F(k=0.0001) = {primordial_ratio(0.0001, params_ir):.4f}")
    print(f"IR suppression F(k=0.01) = {primordial_ratio(0.01, params_ir):.4f}")

    # Test oscillations
    params_osc = PRESETS['weak_oscillations']
    F_osc = primordial_ratio(k_test, params_osc)

    print(f"\nOscillation F range: [{F_osc.min():.4f}, {F_osc.max():.4f}]")

    # Test combined
    params_comb = PRESETS['combined_whbc']
    result = analyze_whbc_primordial(params_comb)

    print(f"\nCombined WHBC analysis:")
    print(f"  sigma8 ratio = {result.sigma8_ratio:.4f}")
    print(f"  n_s_eff at pivot = {result.n_s_eff_pivot:.4f}")
    print(f"  running alpha at pivot = {result.alpha_pivot:.6f}")

    # Create test plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: P(k) comparison
    ax = axes[0, 0]
    ax.loglog(k_test, P_lcdm, 'k-', label='LCDM', lw=2)
    ax.loglog(k_test, primordial_PK_whbc(k_test, params_ir), 'b--', label='IR suppression')
    ax.loglog(k_test, primordial_PK_whbc(k_test, params_osc), 'r--', label='Oscillations')
    ax.loglog(k_test, primordial_PK_whbc(k_test, params_comb), 'g--', label='Combined')
    ax.set_xlabel('k [Mpc$^{-1}$]')
    ax.set_ylabel('$P_R(k)$')
    ax.set_title('Primordial Power Spectra')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: F(k) ratio
    ax = axes[0, 1]
    ax.semilogx(k_test, np.ones_like(k_test), 'k-', label='LCDM', lw=2)
    ax.semilogx(k_test, primordial_ratio(k_test, params_ir), 'b--', label='IR suppression')
    ax.semilogx(k_test, primordial_ratio(k_test, params_osc), 'r--', label='Oscillations')
    ax.semilogx(k_test, primordial_ratio(k_test, params_comb), 'g--', label='Combined')
    ax.set_xlabel('k [Mpc$^{-1}$]')
    ax.set_ylabel('$F_{WHBC}(k) = P_{WHBC}/P_{LCDM}$')
    ax.set_title('WHBC Modification Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.2)

    # Plot 3: Effective spectral index
    ax = axes[1, 0]
    n_eff_lcdm = np.array([effective_spectral_index(k, params_lcdm) for k in k_test])
    n_eff_comb = np.array([effective_spectral_index(k, params_comb) for k in k_test])
    ax.semilogx(k_test, n_eff_lcdm, 'k-', label='LCDM', lw=2)
    ax.semilogx(k_test, n_eff_comb, 'g--', label='Combined WHBC')
    ax.axhline(params_lcdm.ns, color='gray', linestyle=':', label=f'$n_s$ = {params_lcdm.ns}')
    ax.set_xlabel('k [Mpc$^{-1}$]')
    ax.set_ylabel('$n_{eff}(k)$')
    ax.set_title('Effective Spectral Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Parameter scan effect on sigma8
    ax = axes[1, 1]
    A_cut_values = np.linspace(-0.1, 0.1, 21)
    sigma8_ratios = []
    for A_cut in A_cut_values:
        p = WHBCPrimordialParameters(A_cut=A_cut, k_cut=0.001, p_cut=2.0)
        sigma8_ratios.append(compute_sigma8_ratio(p))

    ax.plot(A_cut_values, sigma8_ratios, 'b-', lw=2)
    ax.axhline(1.0, color='k', linestyle='--')
    ax.axvline(0.0, color='gray', linestyle=':')
    ax.set_xlabel('$A_{cut}$')
    ax.set_ylabel('$\\sigma_8^{WHBC} / \\sigma_8^{LCDM}$')
    ax.set_title('Effect of IR Cutoff on $\\sigma_8$')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/primary/hard-animals-to-break/figures/whbc_primordial_test.png', dpi=150)
    print(f"\nSaved test plot to figures/whbc_primordial_test.png")

    print("\nWHBC Primordial Module test complete!")
