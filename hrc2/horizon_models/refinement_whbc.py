"""
T07_WHBC: White-Hole Boundary Cosmology

This module implements the White-Hole Boundary Hypothesis where our universe
emerges from a single white-hole-like boundary connected to a pre-geometric
evaporated spacetime "foam" (Hawking evaporation endstate).

Key effects:
1. Initial expansion shift: H(a→0) = H_GR(a→0) * (1 + epsilon_WH)
2. Sound horizon shift: r_s_WH = r_s_GR * (1 - beta_WH)
3. Perturbation damping: D_WH(k,a) = exp[-gamma_WH * (k/k_pivot)^2 * a^2]
4. Optional inherited horizon memory (xi_WH): active at z > 50

Author: HRC Collaboration
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d


# Physical constants
C_LIGHT = 299792.458  # km/s
H0_FIDUCIAL = 67.4  # km/s/Mpc (Planck 2018)
OMEGA_M_FIDUCIAL = 0.315
OMEGA_B_FIDUCIAL = 0.0493
OMEGA_R_FIDUCIAL = 9.24e-5  # radiation density today
T_CMB = 2.7255  # K
Z_REC = 1089.0  # recombination redshift
Z_DRAG = 1059.94  # drag epoch
K_PIVOT = 0.05  # Mpc^-1, pivot scale for perturbations


@dataclass
class WHBCParameters:
    """
    Parameters for White-Hole Boundary Cosmology (T07_WHBC).

    Attributes:
        epsilon_WH: Dimensionless initial expansion shift [0, 0.2]
                   H(a→0) = H_GR(a→0) * (1 + epsilon_WH)
        beta_WH: Sound horizon modification factor [0, 0.1]
                r_s_WH = r_s_GR * (1 - beta_WH)
        gamma_WH: Perturbation damping coefficient [0, 2]
                 D_WH(k,a) = exp[-gamma_WH * (k/k_pivot)^2 * a^2]
        xi_WH: Inherited horizon memory coupling [0, 0.1]
               Active only at z > z_memory_threshold
        z_memory_threshold: Redshift above which xi_WH is active (default 50)
        H0: Hubble constant today [km/s/Mpc]
        Omega_m: Matter density parameter
        Omega_b: Baryon density parameter
        Omega_r: Radiation density parameter
        ns: Scalar spectral index
        As: Amplitude of primordial perturbations
    """
    epsilon_WH: float = 0.0
    beta_WH: float = 0.0
    gamma_WH: float = 0.0
    xi_WH: float = 0.0
    z_memory_threshold: float = 50.0

    # Standard cosmological parameters
    H0: float = H0_FIDUCIAL
    Omega_m: float = OMEGA_M_FIDUCIAL
    Omega_b: float = OMEGA_B_FIDUCIAL
    Omega_r: float = OMEGA_R_FIDUCIAL
    ns: float = 0.9649
    As: float = 2.1e-9

    def __post_init__(self):
        """Validate parameters."""
        if not 0 <= self.epsilon_WH <= 0.3:
            raise ValueError(f"epsilon_WH must be in [0, 0.3], got {self.epsilon_WH}")
        if not 0 <= self.beta_WH <= 0.15:
            raise ValueError(f"beta_WH must be in [0, 0.15], got {self.beta_WH}")
        if not 0 <= self.gamma_WH <= 3.0:
            raise ValueError(f"gamma_WH must be in [0, 3], got {self.gamma_WH}")
        if not 0 <= self.xi_WH <= 0.2:
            raise ValueError(f"xi_WH must be in [0, 0.2], got {self.xi_WH}")

    @property
    def Omega_Lambda(self) -> float:
        """Dark energy density (flat universe)."""
        return 1.0 - self.Omega_m - self.Omega_r

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epsilon_WH': self.epsilon_WH,
            'beta_WH': self.beta_WH,
            'gamma_WH': self.gamma_WH,
            'xi_WH': self.xi_WH,
            'z_memory_threshold': self.z_memory_threshold,
            'H0': self.H0,
            'Omega_m': self.Omega_m,
            'Omega_b': self.Omega_b,
            'Omega_r': self.Omega_r,
            'Omega_Lambda': self.Omega_Lambda,
            'ns': self.ns,
            'As': self.As
        }


@dataclass
class WHBCResult:
    """
    Results from WHBC model computation.

    Attributes:
        params: Input parameters
        H_ratio: H_WHBC(z) / H_LCDM(z) array
        r_s_WHBC: Modified sound horizon at drag epoch [Mpc]
        r_s_LCDM: Standard sound horizon [Mpc]
        theta_s_WHBC: Modified CMB acoustic scale [rad]
        theta_s_LCDM: Standard CMB acoustic scale [rad]
        z_array: Redshift array
        H_array: H(z) array [km/s/Mpc]
        D_A_array: Angular diameter distance array [Mpc]
        D_L_array: Luminosity distance array [Mpc]
        chi_array: Comoving distance array [Mpc]
        D_V_array: BAO volume distance array [Mpc]
        damping_function: D_WH(k) at various k values
        chi2_theta_s: chi^2 for theta_s deviation
        chi2_bao: chi^2 for BAO deviations
        chi2_sn: chi^2 for SNe Ia
        chi2_growth: chi^2 for growth rate
        passes_constraints: Whether model passes all constraints
    """
    params: WHBCParameters
    H_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    r_s_WHBC: float = 0.0
    r_s_LCDM: float = 0.0
    theta_s_WHBC: float = 0.0
    theta_s_LCDM: float = 0.0
    z_array: np.ndarray = field(default_factory=lambda: np.array([]))
    H_array: np.ndarray = field(default_factory=lambda: np.array([]))
    D_A_array: np.ndarray = field(default_factory=lambda: np.array([]))
    D_L_array: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_array: np.ndarray = field(default_factory=lambda: np.array([]))
    D_V_array: np.ndarray = field(default_factory=lambda: np.array([]))
    k_array: np.ndarray = field(default_factory=lambda: np.array([]))
    damping_function: np.ndarray = field(default_factory=lambda: np.array([]))
    chi2_theta_s: float = 0.0
    chi2_bao: float = 0.0
    chi2_sn: float = 0.0
    chi2_growth: float = 0.0
    chi2_total: float = 0.0
    passes_constraints: bool = False
    constraint_details: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'params': self.params.to_dict(),
            'r_s_WHBC': self.r_s_WHBC,
            'r_s_LCDM': self.r_s_LCDM,
            'r_s_ratio': self.r_s_WHBC / self.r_s_LCDM if self.r_s_LCDM > 0 else 0,
            'theta_s_WHBC': self.theta_s_WHBC,
            'theta_s_LCDM': self.theta_s_LCDM,
            'theta_s_deviation_percent': 100 * abs(self.theta_s_WHBC - self.theta_s_LCDM) / self.theta_s_LCDM if self.theta_s_LCDM > 0 else 0,
            'chi2_theta_s': self.chi2_theta_s,
            'chi2_bao': self.chi2_bao,
            'chi2_sn': self.chi2_sn,
            'chi2_growth': self.chi2_growth,
            'chi2_total': self.chi2_total,
            'passes_constraints': self.passes_constraints,
            'constraint_details': self.constraint_details
        }


class WHBCModel:
    """
    White-Hole Boundary Cosmology Model.

    Implements the T07_WHBC model with:
    1. Modified initial expansion rate
    2. Modified sound horizon
    3. Perturbation damping window
    4. Optional inherited horizon memory
    """

    def __init__(self, params: WHBCParameters):
        """
        Initialize WHBC model.

        Args:
            params: WHBC parameters
        """
        self.params = params
        self._setup_cosmology()

    def _setup_cosmology(self):
        """Set up cosmological quantities."""
        p = self.params
        self.h = p.H0 / 100.0
        self.Omega_m = p.Omega_m
        self.Omega_r = p.Omega_r
        self.Omega_Lambda = p.Omega_Lambda
        self.Omega_b = p.Omega_b

        # Precompute LCDM reference
        self._compute_lcdm_reference()

    def _compute_lcdm_reference(self):
        """Compute LCDM reference values."""
        # LCDM Hubble parameter
        def H_LCDM(z):
            a = 1.0 / (1.0 + z)
            return self.params.H0 * np.sqrt(
                self.Omega_m * (1 + z)**3 +
                self.Omega_r * (1 + z)**4 +
                self.Omega_Lambda
            )

        self._H_LCDM = H_LCDM

        # Compute LCDM sound horizon
        self.r_s_LCDM = self._compute_sound_horizon_lcdm()

        # Compute LCDM comoving distance to last scattering
        self.chi_star_LCDM = self._compute_comoving_distance(Z_REC, use_whbc=False)

        # LCDM theta_s
        self.theta_s_LCDM = self.r_s_LCDM / self.chi_star_LCDM

    def H_LCDM(self, z: float) -> float:
        """
        Standard LCDM Hubble parameter.

        Args:
            z: Redshift

        Returns:
            H(z) in km/s/Mpc
        """
        return self._H_LCDM(z)

    def H_WHBC(self, z: float) -> float:
        """
        WHBC-modified Hubble parameter.

        Implements:
        - epsilon_WH expansion shift (strongest at high z)
        - xi_WH horizon memory (active at z > z_threshold)

        Args:
            z: Redshift

        Returns:
            H(z) in km/s/Mpc
        """
        H_base = self.H_LCDM(z)

        # Initial expansion shift (fades at low z)
        # Use a smooth transition that's maximal at high z
        a = 1.0 / (1.0 + z)

        # Expansion shift: maximum effect at a→0, fades by a~0.1 (z~9)
        epsilon_effect = self.params.epsilon_WH * np.exp(-10 * a)

        # Horizon memory effect (active at z > z_threshold)
        xi_effect = 0.0
        if z > self.params.z_memory_threshold and self.params.xi_WH > 0:
            # Smooth activation above threshold
            z_rel = (z - self.params.z_memory_threshold) / self.params.z_memory_threshold
            activation = 1.0 - np.exp(-z_rel)
            xi_effect = self.params.xi_WH * activation * (1 + z) / (1 + Z_REC)

        return H_base * (1.0 + epsilon_effect + xi_effect)

    def H_ratio(self, z: float) -> float:
        """
        Ratio H_WHBC / H_LCDM.

        Args:
            z: Redshift

        Returns:
            Ratio of Hubble parameters
        """
        return self.H_WHBC(z) / self.H_LCDM(z)

    def _sound_speed(self, z: float) -> float:
        """
        Baryon sound speed c_s(z).

        c_s = c / sqrt(3(1 + R_b))
        where R_b = 3 * rho_b / (4 * rho_gamma)

        Args:
            z: Redshift

        Returns:
            Sound speed in km/s
        """
        # Baryon-to-photon ratio
        # R_b = 3 * Omega_b / (4 * Omega_gamma) * (1+z)^{-1}
        # Omega_gamma = Omega_r * (1 - 0.405)  (neutrinos contribute ~40.5%)
        Omega_gamma = self.Omega_r * 0.595
        R_b = 3.0 * self.Omega_b / (4.0 * Omega_gamma) / (1.0 + z)

        c_s = C_LIGHT / np.sqrt(3.0 * (1.0 + R_b))
        return c_s

    def _compute_sound_horizon_lcdm(self) -> float:
        """
        Compute LCDM sound horizon at drag epoch.

        r_s = integral_0^{a_drag} c_s / (a^2 H) da

        Returns:
            Sound horizon in Mpc
        """
        a_drag = 1.0 / (1.0 + Z_DRAG)

        def integrand(a):
            if a < 1e-10:
                return 0.0
            z = 1.0 / a - 1.0
            c_s = self._sound_speed(z)
            H = self.H_LCDM(z)
            return c_s / (a**2 * H)

        result, _ = quad(integrand, 1e-8, a_drag, limit=200)
        return result  # in Mpc (c is in km/s, H is in km/s/Mpc)

    def _compute_sound_horizon_whbc(self) -> float:
        """
        Compute WHBC-modified sound horizon.

        Two effects:
        1. beta_WH direct modification: r_s_WH = r_s_GR * (1 - beta_WH)
        2. Modified H(z) affects the integral

        Returns:
            Modified sound horizon in Mpc
        """
        a_drag = 1.0 / (1.0 + Z_DRAG)

        def integrand(a):
            if a < 1e-10:
                return 0.0
            z = 1.0 / a - 1.0
            c_s = self._sound_speed(z)
            H = self.H_WHBC(z)  # Use WHBC Hubble
            return c_s / (a**2 * H)

        result, _ = quad(integrand, 1e-8, a_drag, limit=200)

        # Apply beta_WH modification
        r_s_whbc = result * (1.0 - self.params.beta_WH)

        return r_s_whbc

    def _compute_comoving_distance(self, z: float, use_whbc: bool = True) -> float:
        """
        Compute comoving distance to redshift z.

        chi(z) = c * integral_0^z dz' / H(z')

        Args:
            z: Target redshift
            use_whbc: If True, use WHBC H(z), else use LCDM

        Returns:
            Comoving distance in Mpc
        """
        def integrand(z_prime):
            H = self.H_WHBC(z_prime) if use_whbc else self.H_LCDM(z_prime)
            return C_LIGHT / H

        result, _ = quad(integrand, 0, z, limit=200)
        return result

    def angular_diameter_distance(self, z: float) -> float:
        """
        Angular diameter distance D_A(z).

        D_A = chi(z) / (1 + z)

        Args:
            z: Redshift

        Returns:
            D_A in Mpc
        """
        chi = self._compute_comoving_distance(z)
        return chi / (1.0 + z)

    def luminosity_distance(self, z: float) -> float:
        """
        Luminosity distance D_L(z).

        D_L = chi(z) * (1 + z)

        Args:
            z: Redshift

        Returns:
            D_L in Mpc
        """
        chi = self._compute_comoving_distance(z)
        return chi * (1.0 + z)

    def D_V(self, z: float) -> float:
        """
        BAO volume distance D_V(z).

        D_V = [z * D_H(z) * D_M(z)^2]^(1/3)

        where D_H = c/H and D_M = chi (comoving distance)

        Args:
            z: Redshift

        Returns:
            D_V in Mpc
        """
        chi = self._compute_comoving_distance(z)
        D_H = C_LIGHT / self.H_WHBC(z)
        D_V = (z * D_H * chi**2)**(1.0/3.0)
        return D_V

    def damping_window(self, k: float, a: float = 1.0) -> float:
        """
        Perturbation damping window D_WH(k, a).

        D_WH(k, a) = exp[-gamma_WH * (k / k_pivot)^2 * a^2]

        This multiplies the primordial power spectrum.

        Args:
            k: Wavenumber in Mpc^-1
            a: Scale factor (default 1.0 for today)

        Returns:
            Damping factor in [0, 1]
        """
        if self.params.gamma_WH == 0:
            return 1.0

        k_ratio = k / K_PIVOT
        return np.exp(-self.params.gamma_WH * k_ratio**2 * a**2)

    def primordial_power_spectrum(self, k: float) -> float:
        """
        WHBC-modified primordial scalar power spectrum.

        P_s(k) = A_s * (k / k_pivot)^(n_s - 1) * D_WH(k)

        Args:
            k: Wavenumber in Mpc^-1

        Returns:
            Power spectrum amplitude
        """
        As = self.params.As
        ns = self.params.ns

        P_s = As * (k / K_PIVOT)**(ns - 1)

        # Apply WHBC damping
        P_s *= self.damping_window(k)

        return P_s

    def solve(self, z_max: float = 2000.0, n_points: int = 500) -> WHBCResult:
        """
        Solve the WHBC model and compute all observables.

        Args:
            z_max: Maximum redshift for computation
            n_points: Number of redshift points

        Returns:
            WHBCResult with all computed quantities
        """
        result = WHBCResult(params=self.params)

        # Set up redshift array (log-spaced for better sampling at high z)
        z_array = np.logspace(-3, np.log10(z_max), n_points)
        z_array = np.sort(np.unique(np.concatenate([z_array, [0.01, 0.1, 0.5, 1.0, 2.0, Z_DRAG, Z_REC]])))
        result.z_array = z_array

        # Compute H(z) for both models
        H_whbc = np.array([self.H_WHBC(z) for z in z_array])
        H_lcdm = np.array([self.H_LCDM(z) for z in z_array])
        result.H_array = H_whbc
        result.H_ratio = H_whbc / H_lcdm

        # Compute sound horizons
        result.r_s_LCDM = self.r_s_LCDM
        result.r_s_WHBC = self._compute_sound_horizon_whbc()

        # Compute distances
        result.chi_array = np.array([self._compute_comoving_distance(z) for z in z_array])
        result.D_A_array = result.chi_array / (1.0 + z_array)
        result.D_L_array = result.chi_array * (1.0 + z_array)
        result.D_V_array = np.array([self.D_V(z) for z in z_array])

        # Compute theta_s
        chi_star = self._compute_comoving_distance(Z_REC)
        result.theta_s_WHBC = result.r_s_WHBC / chi_star
        result.theta_s_LCDM = self.theta_s_LCDM

        # Compute damping function
        k_array = np.logspace(-4, 1, 100)  # k from 0.0001 to 10 Mpc^-1
        result.k_array = k_array
        result.damping_function = np.array([self.damping_window(k) for k in k_array])

        # Compute chi^2 values
        result.chi2_theta_s = self._compute_chi2_theta_s(result)
        result.chi2_bao = self._compute_chi2_bao(result)
        result.chi2_sn = self._compute_chi2_sn(result)
        result.chi2_growth = self._compute_chi2_growth(result)
        result.chi2_total = (result.chi2_theta_s + result.chi2_bao +
                           result.chi2_sn + result.chi2_growth)

        # Check constraints
        result.constraint_details = self._check_constraints(result)
        result.passes_constraints = all(result.constraint_details.values())

        return result

    def _compute_chi2_theta_s(self, result: WHBCResult) -> float:
        """
        Compute chi^2 for CMB acoustic scale theta_s.

        Planck constraint: theta_s = (1.04109 ± 0.00030) * 10^-2 rad
        """
        theta_s_obs = 1.04109e-2  # Planck 2018
        sigma = 0.00030e-2

        chi2 = ((result.theta_s_WHBC - theta_s_obs) / sigma)**2
        return chi2

    def _compute_chi2_bao(self, result: WHBCResult) -> float:
        """
        Compute chi^2 for BAO measurements.

        Uses simplified BAO constraints at z = 0.38, 0.51, 0.61 (BOSS)
        """
        # BOSS BAO measurements (D_V / r_d)
        bao_data = [
            (0.38, 10.27, 0.15),  # z, D_V/r_d, sigma
            (0.51, 13.38, 0.18),
            (0.61, 15.45, 0.20),
        ]

        chi2 = 0.0
        r_d = result.r_s_WHBC  # Use WHBC sound horizon

        for z_bao, dv_rd_obs, sigma in bao_data:
            D_V = self.D_V(z_bao)
            dv_rd_model = D_V / r_d
            chi2 += ((dv_rd_model - dv_rd_obs) / sigma)**2

        return chi2

    def _compute_chi2_sn(self, result: WHBCResult) -> float:
        """
        Compute chi^2 for SNe Ia distance moduli.

        Uses simplified Pantheon+ constraints.
        """
        # Simplified Pantheon+ data points
        sn_data = [
            (0.01, 32.95, 0.10),  # z, mu, sigma
            (0.03, 35.15, 0.08),
            (0.1, 38.30, 0.06),
            (0.3, 40.85, 0.05),
            (0.5, 42.20, 0.05),
            (0.8, 43.35, 0.06),
            (1.0, 43.95, 0.08),
        ]

        chi2 = 0.0
        for z, mu_obs, sigma in sn_data:
            D_L = self.luminosity_distance(z)
            mu_model = 5.0 * np.log10(D_L) + 25.0  # Distance modulus
            chi2 += ((mu_model - mu_obs) / sigma)**2

        return chi2

    def _compute_chi2_growth(self, result: WHBCResult) -> float:
        """
        Compute chi^2 for growth rate f*sigma8.

        Simplified constraint assuming WHBC affects growth minimally.
        """
        # For this simplified implementation, assume growth is LCDM-like
        # unless gamma_WH significantly damps perturbations

        if self.params.gamma_WH > 1.0:
            # Significant damping would affect growth
            chi2 = 10.0 * self.params.gamma_WH
        else:
            chi2 = 0.0

        return chi2

    def _check_constraints(self, result: WHBCResult) -> Dict[str, bool]:
        """
        Check if model passes observational constraints.
        """
        constraints = {}

        # theta_s deviation < 0.05%
        theta_s_dev = abs(result.theta_s_WHBC - result.theta_s_LCDM) / result.theta_s_LCDM
        constraints['theta_s_pass'] = bool(theta_s_dev < 0.0005)

        # BAO D_V deviation < 1% (check at z=0.5)
        D_V_whbc = self.D_V(0.5)
        D_V_lcdm = self._compute_D_V_lcdm(0.5)
        bao_dev = abs(D_V_whbc - D_V_lcdm) / D_V_lcdm
        constraints['bao_pass'] = bool(bao_dev < 0.01)

        # SNe D_L deviation < 1.5% (check at z=0.5)
        D_L_whbc = self.luminosity_distance(0.5)
        D_L_lcdm = self._compute_D_L_lcdm(0.5)
        sn_dev = abs(D_L_whbc - D_L_lcdm) / D_L_lcdm
        constraints['sn_pass'] = bool(sn_dev < 0.015)

        # Growth constraint (simplified)
        constraints['growth_pass'] = bool(self.params.gamma_WH < 1.5)

        return constraints

    def _compute_D_V_lcdm(self, z: float) -> float:
        """LCDM D_V for comparison."""
        chi = self._compute_comoving_distance(z, use_whbc=False)
        D_H = C_LIGHT / self.H_LCDM(z)
        return (z * D_H * chi**2)**(1.0/3.0)

    def _compute_D_L_lcdm(self, z: float) -> float:
        """LCDM D_L for comparison."""
        chi = self._compute_comoving_distance(z, use_whbc=False)
        return chi * (1.0 + z)


def create_whbc_model(
    epsilon_WH: float = 0.0,
    beta_WH: float = 0.0,
    gamma_WH: float = 0.0,
    xi_WH: float = 0.0,
    H0: float = H0_FIDUCIAL,
    Omega_m: float = OMEGA_M_FIDUCIAL
) -> WHBCModel:
    """
    Factory function to create a WHBC model.

    Args:
        epsilon_WH: Initial expansion shift
        beta_WH: Sound horizon modification
        gamma_WH: Perturbation damping
        xi_WH: Inherited horizon memory coupling
        H0: Hubble constant
        Omega_m: Matter density

    Returns:
        WHBCModel instance
    """
    params = WHBCParameters(
        epsilon_WH=epsilon_WH,
        beta_WH=beta_WH,
        gamma_WH=gamma_WH,
        xi_WH=xi_WH,
        H0=H0,
        Omega_m=Omega_m
    )
    return WHBCModel(params)


def scan_whbc_parameter_space(
    epsilon_values: list = None,
    beta_values: list = None,
    gamma_values: list = None,
    xi_values: list = None,
    parallel: bool = True,
    n_jobs: int = -1
) -> list:
    """
    Scan WHBC parameter space.

    Args:
        epsilon_values: List of epsilon_WH values to scan
        beta_values: List of beta_WH values to scan
        gamma_values: List of gamma_WH values to scan
        xi_values: List of xi_WH values to scan
        parallel: Whether to parallelize
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        List of WHBCResult objects
    """
    if epsilon_values is None:
        epsilon_values = [0.00, 0.05, 0.10, 0.15]
    if beta_values is None:
        beta_values = [0.00, 0.02, 0.04, 0.06]
    if gamma_values is None:
        gamma_values = [0.0, 0.5, 1.0]
    if xi_values is None:
        xi_values = [0.0, 0.05]

    # Generate parameter combinations
    param_combos = []
    for eps in epsilon_values:
        for beta in beta_values:
            for gamma in gamma_values:
                for xi in xi_values:
                    param_combos.append((eps, beta, gamma, xi))

    def evaluate_point(params):
        eps, beta, gamma, xi = params
        try:
            model = create_whbc_model(
                epsilon_WH=eps,
                beta_WH=beta,
                gamma_WH=gamma,
                xi_WH=xi
            )
            return model.solve()
        except Exception as e:
            print(f"Error at eps={eps}, beta={beta}, gamma={gamma}, xi={xi}: {e}")
            return None

    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(evaluate_point, param_combos))
    else:
        results = [evaluate_point(p) for p in param_combos]

    # Filter out None results
    results = [r for r in results if r is not None]

    return results


if __name__ == "__main__":
    # Quick test
    print("Testing WHBC Model...")

    # Test with default parameters (LCDM-like)
    model = create_whbc_model()
    result = model.solve()

    print(f"\nLCDM Reference:")
    print(f"  r_s_LCDM = {result.r_s_LCDM:.4f} Mpc")
    print(f"  theta_s_LCDM = {result.theta_s_LCDM:.6e} rad")

    print(f"\nWHBC (eps=0, beta=0):")
    print(f"  r_s_WHBC = {result.r_s_WHBC:.4f} Mpc")
    print(f"  theta_s_WHBC = {result.theta_s_WHBC:.6e} rad")
    print(f"  Passes constraints: {result.passes_constraints}")

    # Test with non-trivial parameters
    model2 = create_whbc_model(epsilon_WH=0.10, beta_WH=0.04, gamma_WH=0.5, xi_WH=0.05)
    result2 = model2.solve()

    print(f"\nWHBC (eps=0.10, beta=0.04, gamma=0.5, xi=0.05):")
    print(f"  r_s_WHBC = {result2.r_s_WHBC:.4f} Mpc")
    print(f"  theta_s_WHBC = {result2.theta_s_WHBC:.6e} rad")
    print(f"  theta_s deviation: {100*abs(result2.theta_s_WHBC - result2.theta_s_LCDM)/result2.theta_s_LCDM:.4f}%")
    print(f"  H ratio at z=1000: {result2.H_ratio[result2.z_array > 900][0]:.4f}")
    print(f"  chi2_total = {result2.chi2_total:.2f}")
    print(f"  Passes constraints: {result2.passes_constraints}")
    print(f"  Constraint details: {result2.constraint_details}")
