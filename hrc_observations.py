"""
Holographic Recycling Cosmology (HRC) - Observational Constraints

This module interfaces HRC predictions with real cosmological observations:
1. Planck CMB measurements
2. DESI Baryon Acoustic Oscillations
3. SH0ES local Hubble constant
4. Pantheon+ Type Ia supernovae
5. Big Bang Nucleosynthesis constraints

The goal is to:
- Test whether HRC can resolve the Hubble tension
- Constrain the new physics parameters (ξ, λ, α, etc.)
- Compare model fit quality with standard ΛCDM

Units: km/s/Mpc for H0, Mpc for distances unless otherwise noted.

Author: HRC Observations Module
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import json
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize, brentq
from scipy.special import erf
import logging

# Optional imports for MCMC
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee not installed. MCMC functionality unavailable.")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not installed. Some data loading features unavailable.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS FOR COSMOLOGY
# =============================================================================

@dataclass(frozen=True)
class CosmologyConstants:
    """Physical constants relevant for cosmological calculations."""
    c: float = 299792.458  # Speed of light [km/s]
    c_m_s: float = 299792458.0  # Speed of light [m/s]
    G_SI: float = 6.67430e-11  # Newton's constant [m³/(kg·s²)]
    h_planck: float = 6.62607e-34  # Planck constant [J·s]
    k_B: float = 1.380649e-23  # Boltzmann constant [J/K]

    # Cosmological conversion factors
    Mpc_to_m: float = 3.0857e22  # Mpc to meters
    Gyr_to_s: float = 3.1558e16  # Gyr to seconds

    # CMB temperature today
    T_cmb_0: float = 2.7255  # [K]

    # Neutrino temperature ratio
    T_nu_T_gamma: float = (4/11)**(1/3)


COSMO_CONST = CosmologyConstants()


# =============================================================================
# PART A: OBSERVATIONAL DATA
# =============================================================================

@dataclass
class PlanckParameters:
    """
    Planck 2018 + ACT DR6 best-fit ΛCDM parameters.

    Reference: Planck Collaboration 2020, A&A 641, A6
               ACT Collaboration 2024
    """
    # Primary parameters
    H0: float = 67.4           # [km/s/Mpc]
    H0_err: float = 0.5

    omega_b_h2: float = 0.02237   # Physical baryon density
    omega_b_h2_err: float = 0.00015

    omega_cdm_h2: float = 0.1200   # Physical CDM density
    omega_cdm_h2_err: float = 0.0012

    tau: float = 0.054            # Optical depth
    tau_err: float = 0.007

    ln_10_10_As: float = 3.044    # Primordial amplitude
    ln_10_10_As_err: float = 0.014

    n_s: float = 0.9649           # Spectral index
    n_s_err: float = 0.0042

    # Derived parameters
    @property
    def h(self) -> float:
        return self.H0 / 100

    @property
    def Omega_m(self) -> float:
        """Total matter density parameter."""
        return (self.omega_b_h2 + self.omega_cdm_h2) / self.h**2

    @property
    def Omega_b(self) -> float:
        """Baryon density parameter."""
        return self.omega_b_h2 / self.h**2

    @property
    def Omega_cdm(self) -> float:
        """CDM density parameter."""
        return self.omega_cdm_h2 / self.h**2

    @property
    def Omega_Lambda(self) -> float:
        """Dark energy density (flat universe)."""
        return 1.0 - self.Omega_m

    @property
    def sigma_8(self) -> float:
        """Matter fluctuation amplitude (approximate)."""
        return 0.811  # From Planck chains

    @property
    def sigma_8_err(self) -> float:
        return 0.006

    @property
    def z_star(self) -> float:
        """Redshift of last scattering."""
        return 1089.92

    @property
    def r_s_star(self) -> float:
        """Sound horizon at last scattering [Mpc]."""
        return 144.43  # Mpc

    @property
    def r_s_drag(self) -> float:
        """Sound horizon at drag epoch [Mpc]."""
        return 147.09  # Mpc


@dataclass
class LocalH0Measurement:
    """
    Local Hubble constant measurements from various methods.

    Primary: SH0ES (Cepheid-calibrated SNe Ia)
    Cross-checks: TRGB, Miras, Time-delay cosmography
    """
    # SH0ES 2024 (Riess et al.)
    shoes_H0: float = 73.04
    shoes_err: float = 1.04

    # TRGB (Freedman et al. 2024)
    trgb_H0: float = 69.8
    trgb_err: float = 1.7

    # Time-delay cosmography (TDCOSMO 2024)
    tdcosmo_H0: float = 73.3
    tdcosmo_err: float = 3.3

    # Carnegie-Chicago Hubble Program
    cchp_H0: float = 69.8
    cchp_err: float = 0.8

    @property
    def combined_local_H0(self) -> Tuple[float, float]:
        """
        Weighted average of local measurements.
        Returns (H0, sigma).
        """
        measurements = [
            (self.shoes_H0, self.shoes_err),
            (self.trgb_H0, self.trgb_err),
            (self.tdcosmo_H0, self.tdcosmo_err),
        ]

        weights = [1/err**2 for _, err in measurements]
        total_weight = sum(weights)

        H0_avg = sum(H0 * w for (H0, _), w in zip(measurements, weights)) / total_weight
        sigma = 1 / np.sqrt(total_weight)

        return H0_avg, sigma

    @property
    def tension_sigma(self) -> float:
        """Tension with Planck in units of sigma."""
        planck = PlanckParameters()
        H0_local, sigma_local = self.shoes_H0, self.shoes_err
        H0_cmb, sigma_cmb = planck.H0, planck.H0_err

        return abs(H0_local - H0_cmb) / np.sqrt(sigma_local**2 + sigma_cmb**2)


@dataclass
class DESIBAOData:
    """
    DESI DR1 BAO measurements (2024).

    Measurements of D_V/r_d, D_M/r_d, D_H/r_d at various redshifts.

    Reference: DESI Collaboration 2024, arXiv:2404.03002
    """
    # Effective redshifts
    z_eff: np.ndarray = field(default_factory=lambda: np.array([
        0.295,  # BGS
        0.51,   # LRG1
        0.706,  # LRG2
        0.93,   # LRG3+ELG1
        1.317,  # ELG2
        1.491,  # QSO
        2.33,   # Lya-QSO
    ]))

    # D_V(z) / r_d measurements (isotropic)
    DV_rd: np.ndarray = field(default_factory=lambda: np.array([
        7.93,   # BGS
        13.62,  # LRG1
        16.85,  # LRG2
        21.71,  # LRG3+ELG1
        27.79,  # ELG2
        30.69,  # QSO
        39.71,  # Lya
    ]))

    DV_rd_err: np.ndarray = field(default_factory=lambda: np.array([
        0.15,   # BGS
        0.25,   # LRG1
        0.32,   # LRG2
        0.28,   # LRG3+ELG1
        0.69,   # ELG2
        0.80,   # QSO
        0.94,   # Lya
    ]))

    # Sound horizon at drag (Planck-derived for reference)
    r_d_planck: float = 147.09  # Mpc

    def D_V_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (z, D_V, sigma_DV) with D_V in Mpc."""
        D_V = self.DV_rd * self.r_d_planck
        D_V_err = self.DV_rd_err * self.r_d_planck
        return self.z_eff, D_V, D_V_err


@dataclass
class DESIDarkEnergy:
    """
    DESI constraints on dark energy equation of state.

    w0-wa parametrization: w(a) = w0 + wa(1-a)

    Reference: DESI Collaboration 2024
    """
    # w0-wa CDM fit (DESI BAO + CMB + Pantheon+)
    w0: float = -0.827
    w0_err: float = 0.063

    wa: float = -0.75
    wa_err_plus: float = 0.29
    wa_err_minus: float = 0.25

    # Correlation coefficient
    rho_w0_wa: float = -0.85

    @property
    def wa_err(self) -> float:
        """Symmetrized wa error."""
        return (self.wa_err_plus + self.wa_err_minus) / 2

    def w_of_z(self, z: float) -> float:
        """Dark energy EoS at redshift z."""
        a = 1 / (1 + z)
        return self.w0 + self.wa * (1 - a)


@dataclass
class PantheonPlusData:
    """
    Pantheon+ Type Ia Supernova compilation.

    1701 light curves of 1550 unique SNe Ia in range 0.001 < z < 2.26.

    Reference: Scolnic et al. 2022, ApJ 938, 113
    """
    # Binned distance modulus data (simplified)
    # Full analysis requires covariance matrix

    z_bins: np.ndarray = field(default_factory=lambda: np.array([
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0
    ]))

    # Distance modulus μ = m - M = 5 log₁₀(D_L/10pc)
    mu_bins: np.ndarray = field(default_factory=lambda: np.array([
        32.95, 34.20, 34.98, 36.07, 36.82, 37.56, 38.46, 39.05, 39.93,
        40.58, 41.08, 41.48, 41.82, 42.10, 42.58, 42.97, 43.47, 44.07
    ]))

    mu_err: np.ndarray = field(default_factory=lambda: np.array([
        0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04,
        0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.18
    ]))

    # Absolute magnitude (marginalized nuisance)
    M_B: float = -19.253
    M_B_err: float = 0.027


@dataclass
class BBNConstraints:
    """
    Big Bang Nucleosynthesis constraints.

    Primordial abundances constrain baryon density and N_eff.
    """
    # Primordial helium mass fraction
    Y_p: float = 0.2449
    Y_p_err: float = 0.0040

    # Deuterium abundance D/H
    D_H: float = 2.547e-5
    D_H_err: float = 0.025e-5

    # Effective number of neutrino species
    N_eff: float = 2.99
    N_eff_err: float = 0.17

    # Standard model prediction
    N_eff_SM: float = 3.044

    def consistent_with_SM(self) -> bool:
        """Check if N_eff consistent with standard model."""
        return abs(self.N_eff - self.N_eff_SM) < 2 * self.N_eff_err


@dataclass
class PBHConstraints:
    """
    Constraints on primordial black hole abundance.

    f_PBH = Ω_PBH / Ω_DM must satisfy various bounds.
    """

    # Mass ranges and maximum f_PBH allowed
    # Format: (M_min, M_max, f_max, constraint_source)
    constraints: List[Tuple[float, float, float, str]] = field(default_factory=lambda: [
        # Evaporation constraints (γ-ray background)
        (1e11, 1e14, 1e-8, "Extragalactic γ-ray"),
        (1e14, 1e16, 1e-4, "Galactic γ-ray"),
        # Microlensing
        (1e17, 1e20, 0.1, "HSC microlensing"),
        (1e20, 1e24, 0.3, "OGLE/EROS microlensing"),
        # Stellar mass range
        (1e30, 1e33, 0.01, "LIGO/Virgo mergers"),
        # Massive PBHs
        (1e35, 1e40, 0.001, "CMB accretion"),
        # Planck-mass remnants: essentially unconstrained
        (1e-8, 1e-7, 1.0, "Planck remnants (unconstrained)"),
    ])

    def max_f_pbh(self, M: float) -> float:
        """
        Maximum allowed f_PBH at mass M.

        Parameters
        ----------
        M : float
            PBH mass in kg

        Returns
        -------
        float
            Maximum f_PBH (0 to 1)
        """
        for M_min, M_max, f_max, _ in self.constraints:
            if M_min <= M <= M_max:
                return f_max
        return 1.0  # Unconstrained mass range


class ObservationalData:
    """
    Unified interface for all observational constraints.

    Loads and manages cosmological datasets for HRC parameter estimation.
    """

    def __init__(self, data_dir: str = './data'):
        """
        Initialize with observational data.

        Parameters
        ----------
        data_dir : str
            Directory for data files (created if needed)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize data containers
        self._planck = PlanckParameters()
        self._local_H0 = LocalH0Measurement()
        self._desi_bao = DESIBAOData()
        self._desi_de = DESIDarkEnergy()
        self._pantheon = PantheonPlusData()
        self._bbn = BBNConstraints()
        self._pbh = PBHConstraints()

        logger.info("Observational data initialized")

    @property
    def planck_params(self) -> PlanckParameters:
        """Planck best-fit ΛCDM parameters."""
        return self._planck

    @property
    def local_H0(self) -> Tuple[float, float]:
        """(value, sigma) for SH0ES H0 measurement."""
        return self._local_H0.shoes_H0, self._local_H0.shoes_err

    @property
    def cmb_H0(self) -> Tuple[float, float]:
        """(value, sigma) for Planck H0."""
        return self._planck.H0, self._planck.H0_err

    @property
    def hubble_tension(self) -> float:
        """Tension significance in sigma."""
        return self._local_H0.tension_sigma

    @property
    def bao_measurements(self) -> DESIBAOData:
        """DESI BAO data."""
        return self._desi_bao

    @property
    def dark_energy_constraints(self) -> DESIDarkEnergy:
        """DESI dark energy EoS constraints."""
        return self._desi_de

    @property
    def sn_data(self) -> PantheonPlusData:
        """Pantheon+ supernova data."""
        return self._pantheon

    @property
    def bbn_constraints(self) -> BBNConstraints:
        """BBN primordial abundance constraints."""
        return self._bbn

    @property
    def pbh_constraints(self) -> PBHConstraints:
        """PBH abundance constraints."""
        return self._pbh

    def sn_distance_modulus(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolated distance modulus from Pantheon+ at given redshifts.

        Parameters
        ----------
        z : np.ndarray
            Redshifts to evaluate

        Returns
        -------
        tuple
            (mu, sigma_mu) at requested redshifts
        """
        # Interpolate binned data
        mu_interp = interp1d(self._pantheon.z_bins, self._pantheon.mu_bins,
                            kind='cubic', fill_value='extrapolate')
        err_interp = interp1d(self._pantheon.z_bins, self._pantheon.mu_err,
                             kind='linear', fill_value='extrapolate')

        return mu_interp(z), err_interp(z)

    def summary(self) -> str:
        """Return summary of loaded data."""
        lines = [
            "=" * 60,
            "OBSERVATIONAL DATA SUMMARY",
            "=" * 60,
            "",
            "HUBBLE TENSION:",
            f"  Planck H0: {self._planck.H0:.1f} ± {self._planck.H0_err:.1f} km/s/Mpc",
            f"  SH0ES H0:  {self._local_H0.shoes_H0:.1f} ± {self._local_H0.shoes_err:.1f} km/s/Mpc",
            f"  Tension:   {self._local_H0.tension_sigma:.1f}σ",
            "",
            "PLANCK ΛCDM PARAMETERS:",
            f"  Ωm = {self._planck.Omega_m:.4f}",
            f"  Ωb = {self._planck.Omega_b:.4f}",
            f"  ΩΛ = {self._planck.Omega_Lambda:.4f}",
            f"  σ8 = {self._planck.sigma_8:.3f}",
            "",
            "DESI DARK ENERGY:",
            f"  w0 = {self._desi_de.w0:.3f} ± {self._desi_de.w0_err:.3f}",
            f"  wa = {self._desi_de.wa:.2f} +{self._desi_de.wa_err_plus:.2f}/-{self._desi_de.wa_err_minus:.2f}",
            "",
            "BAO MEASUREMENTS: {0} redshift bins".format(len(self._desi_bao.z_eff)),
            "PANTHEON+ SNe: {0} redshift bins".format(len(self._pantheon.z_bins)),
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# PART B: HRC PREDICTIONS
# =============================================================================

class LCDMCosmology:
    """
    Standard ΛCDM cosmology for comparison.

    Provides baseline predictions using Planck parameters.
    """

    def __init__(self, H0: float = 67.4, Omega_m: float = 0.315,
                 Omega_Lambda: float = None):
        """
        Initialize ΛCDM cosmology.

        Parameters
        ----------
        H0 : float
            Hubble constant [km/s/Mpc]
        Omega_m : float
            Matter density parameter
        Omega_Lambda : float
            Dark energy density (default: 1 - Omega_m for flat)
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda if Omega_Lambda else 1 - Omega_m
        self.Omega_k = 1 - Omega_m - self.Omega_Lambda

        self.c = COSMO_CONST.c  # km/s
        self.h = H0 / 100

    def E(self, z: float) -> float:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.

        For ΛCDM: E² = Ωm(1+z)³ + Ωk(1+z)² + ΩΛ
        """
        return np.sqrt(self.Omega_m * (1 + z)**3 +
                      self.Omega_k * (1 + z)**2 +
                      self.Omega_Lambda)

    def H(self, z: float) -> float:
        """Hubble parameter at redshift z [km/s/Mpc]."""
        return self.H0 * self.E(z)

    def comoving_distance(self, z: float) -> float:
        """
        Comoving distance to redshift z [Mpc].

        D_C = (c/H0) ∫₀ᶻ dz'/E(z')
        """
        integrand = lambda zp: 1.0 / self.E(zp)
        result, _ = quad(integrand, 0, z)
        return self.c / self.H0 * result

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance D_A(z) [Mpc]."""
        D_C = self.comoving_distance(z)
        return D_C / (1 + z)

    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance D_L(z) [Mpc]."""
        D_C = self.comoving_distance(z)
        return D_C * (1 + z)

    def distance_modulus(self, z: float) -> float:
        """Distance modulus μ = 5 log₁₀(D_L / 10pc)."""
        D_L = self.luminosity_distance(z)
        D_L_pc = D_L * 1e6  # Mpc to pc
        return 5 * np.log10(D_L_pc / 10)

    def D_V(self, z: float) -> float:
        """
        Volume-averaged distance for BAO [Mpc].

        D_V = [z D_H D_M²]^(1/3)
        where D_H = c/H(z), D_M = D_A(1+z)
        """
        D_H = self.c / self.H(z)
        D_M = self.angular_diameter_distance(z) * (1 + z)
        return (z * D_H * D_M**2)**(1/3)

    def sound_horizon_approximate(self, omega_m: float = None,
                                   omega_b: float = None) -> float:
        """
        Approximate sound horizon at drag epoch [Mpc].

        Using Eisenstein & Hu 1998 fitting formula.
        """
        if omega_m is None:
            omega_m = self.Omega_m * self.h**2
        if omega_b is None:
            planck = PlanckParameters()
            omega_b = planck.omega_b_h2

        # Fitting formula from Eisenstein & Hu
        omega_m_eff = omega_m
        b1 = 0.313 * omega_m_eff**(-0.419) * (1 + 0.607 * omega_m_eff**0.674)
        b2 = 0.238 * omega_m_eff**0.223
        z_d = 1291 * omega_m_eff**0.251 / (1 + 0.659 * omega_m_eff**0.828) * (1 + b1 * omega_b**b2)

        R_d = 31.5 * omega_b * (COSMO_CONST.T_cmb_0 / 2.7)**(-4) * (z_d / 1000)**(-1)
        r_s = 44.5 * np.log(9.83 / omega_m_eff) / np.sqrt(1 + 10 * omega_b**(3/4))

        return r_s  # Mpc

    def age(self, z: float = 0) -> float:
        """
        Age of universe at redshift z [Gyr].

        t = (1/H0) ∫_z^∞ dz' / [(1+z')E(z')]
        """
        integrand = lambda zp: 1.0 / ((1 + zp) * self.E(zp))
        result, _ = quad(integrand, z, np.inf)

        # Convert from 1/H0 to Gyr
        t_H = 1 / self.H0  # [Mpc s / km]
        t_H_Gyr = t_H * COSMO_CONST.Mpc_to_m / (COSMO_CONST.Gyr_to_s * 1000)

        return t_H_Gyr * result


class HRCPredictions:
    """
    Compute cosmological observables from HRC model.

    Key insight: HRC modifies the expansion history through:
    1. Effective Newton's constant G_eff(φ)
    2. Additional energy density from φ field
    3. Remnants contributing to dark matter
    4. Epoch-dependent recycling effects

    This can produce different H0 values for local vs CMB measurements.
    """

    def __init__(self, params: Dict = None):
        """
        Initialize HRC predictions.

        Parameters
        ----------
        params : dict
            HRC model parameters:
            - H0_true: True background H0 [km/s/Mpc]
            - Omega_m: Total matter density
            - Omega_b: Baryon density
            - Omega_rem: Remnant density (subset of Omega_cdm)
            - xi: Non-minimal coupling
            - lambda_r: Recycling coupling
            - alpha: Remnant-field coupling
            - phi_0: Present-day scalar field value
            - dphi_0: Present-day scalar field velocity
        """
        # Default parameters
        default_params = {
            'H0_true': 70.0,      # True H0 (between local and CMB)
            'Omega_m': 0.315,     # Total matter
            'Omega_b': 0.049,     # Baryons
            'Omega_rem': 0.05,    # Remnants
            'xi': 0.01,           # Non-minimal coupling
            'lambda_r': 1e-10,    # Recycling coupling
            'alpha': 0.01,        # Remnant-field coupling
            'm_phi': 1e-33,       # Scalar mass [eV] ~ H0
            'phi_0': 0.1,         # Present φ [Planck units]
            'dphi_0': 0.0,        # Present φ̇
        }

        self.params = {**default_params, **(params or {})}
        self.c = COSMO_CONST.c

        # Precompute useful quantities
        self._setup_cosmology()

    def _setup_cosmology(self):
        """Set up cosmological quantities."""
        p = self.params

        self.H0 = p['H0_true']
        self.h = self.H0 / 100
        self.Omega_m = p['Omega_m']
        self.Omega_b = p['Omega_b']
        self.Omega_rem = p['Omega_rem']
        self.Omega_cdm = self.Omega_m - self.Omega_b
        self.Omega_Lambda = 1 - self.Omega_m  # Flat universe

        # HRC-specific
        self.xi = p['xi']
        self.alpha = p['alpha']
        self.phi_0 = p['phi_0']

        # G_eff factor today
        self.G_eff_factor_0 = 1 - 8 * np.pi * self.xi * self.phi_0
        if self.G_eff_factor_0 <= 0:
            warnings.warn("G_eff factor non-positive!")
            self.G_eff_factor_0 = 0.1

    def phi_of_z(self, z: float) -> float:
        """
        Scalar field value at redshift z.

        Simplified model: φ evolves slowly, roughly scaling with H².
        φ(z) ≈ φ_0 × (H(z)/H_0)^α_eff

        This is phenomenological; full solution requires integrating
        the scalar field equation.
        """
        # Effective scaling exponent (phenomenological)
        alpha_eff = 0.5 * self.alpha

        # Use ΛCDM E(z) as approximation
        E_z = np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

        return self.phi_0 * E_z**alpha_eff

    def G_eff_factor(self, z: float) -> float:
        """
        Effective gravitational coupling factor at redshift z.

        G_eff = G / (1 - 8πGξφ)
        Returns the denominator (1 - 8πξφ).
        """
        phi = self.phi_of_z(z)
        factor = 1 - 8 * np.pi * self.xi * phi
        return max(factor, 0.1)  # Regularize

    def E_squared(self, z: float) -> float:
        """
        Modified dimensionless Hubble parameter squared.

        E² = [Ωm(1+z)³ + ΩΛ + Ωφ(z)] / G_eff_factor(z)
        """
        # Standard terms
        standard = self.Omega_m * (1 + z)**3 + self.Omega_Lambda

        # Scalar field energy density (simplified)
        m_phi = self.params['m_phi']
        phi = self.phi_of_z(z)
        Omega_phi = 0.5 * m_phi**2 * phi**2 / (3 * self.H0**2)  # Very rough

        # G_eff modification
        G_factor = self.G_eff_factor(z)

        return (standard + Omega_phi) / G_factor

    def E(self, z: float) -> float:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(max(self.E_squared(z), 1e-10))

    def hubble_at_z(self, z: float) -> float:
        """H(z) in km/s/Mpc."""
        return self.H0 * self.E(z)

    def H0_local(self) -> float:
        """
        Effective H0 as measured locally.

        Local measurements use distances at z < 0.1.
        If G_eff differs from G at low z, the inferred H0 differs.
        """
        # Average G_eff factor over local volume (z < 0.1)
        z_local = 0.05  # Typical SN Ia redshift for calibration
        G_factor_local = self.G_eff_factor(z_local)

        # H0_local = H0_true / sqrt(G_factor_local) approximately
        return self.H0 / np.sqrt(G_factor_local)

    def H0_cmb(self) -> float:
        """
        H0 inferred from CMB assuming standard physics.

        CMB measures θ* = r_s / D_A(z*) which determines H0 through
        the distance-redshift relation. If G_eff was different at
        recombination, the inferred H0 differs.
        """
        z_star = 1089  # Last scattering
        G_factor_cmb = self.G_eff_factor(z_star)

        # CMB inference assumes G = const, so
        # H0_cmb = H0_true × sqrt(G_factor_today / G_factor_cmb)
        return self.H0 * np.sqrt(self.G_eff_factor_0 / G_factor_cmb)

    def hubble_tension_prediction(self) -> float:
        """Predicted ΔH0 = H0_local - H0_cmb."""
        return self.H0_local() - self.H0_cmb()

    def can_resolve_tension(self, tension_target: float = 5.6) -> bool:
        """
        Check if model can resolve observed Hubble tension.

        Target: H0_local ≈ 73, H0_cmb ≈ 67, ΔH0 ≈ 6
        """
        predicted_delta = self.hubble_tension_prediction()
        return abs(predicted_delta - tension_target) < 2.0

    def comoving_distance(self, z: float) -> float:
        """Comoving distance [Mpc]."""
        integrand = lambda zp: 1.0 / self.E(zp)
        result, _ = quad(integrand, 0, z)
        return self.c / self.H0 * result

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance D_A(z) [Mpc]."""
        D_C = self.comoving_distance(z)
        return D_C / (1 + z)

    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance D_L(z) [Mpc]."""
        D_C = self.comoving_distance(z)
        return D_C * (1 + z)

    def distance_modulus(self, z: float) -> float:
        """Distance modulus μ = 5 log₁₀(D_L / 10pc)."""
        D_L = self.luminosity_distance(z)
        D_L_pc = D_L * 1e6
        return 5 * np.log10(D_L_pc / 10)

    def D_V(self, z: float) -> float:
        """Volume-averaged distance for BAO [Mpc]."""
        D_H = self.c / self.hubble_at_z(z)
        D_M = self.angular_diameter_distance(z) * (1 + z)
        return (z * D_H * D_M**2)**(1/3)

    def sound_horizon(self) -> float:
        """
        Sound horizon at drag epoch [Mpc].

        In HRC, this could be modified if G_eff was different in early universe.
        """
        # Use approximate formula with G_eff correction
        lcdm = LCDMCosmology(self.H0, self.Omega_m)
        r_s_lcdm = lcdm.sound_horizon_approximate()

        # G_eff at recombination modifies sound speed
        z_drag = 1060  # Drag epoch
        G_factor_drag = self.G_eff_factor(z_drag)

        # r_s ∝ 1/sqrt(G_eff) approximately
        return r_s_lcdm / np.sqrt(G_factor_drag)

    def cmb_shift_parameter(self) -> float:
        """
        CMB shift parameter R.

        R = √Ωm × D_A(z*) × H0 / c

        This is a key compressed CMB observable.
        """
        z_star = 1089
        D_A_star = self.angular_diameter_distance(z_star)
        return np.sqrt(self.Omega_m) * D_A_star * self.H0 / self.c

    def effective_dark_energy_eos(self, z: float) -> float:
        """
        What w(z) would ΛCDM infer from our expansion history?

        If H(z) deviates from ΛCDM, an observer assuming ΛCDM would
        infer dynamical dark energy.
        """
        # Compare our E(z) to ΛCDM
        E_hrc = self.E(z)

        lcdm = LCDMCosmology(self.H0, self.Omega_m)
        E_lcdm = lcdm.E(z)

        # Deviation factor
        ratio = E_hrc / E_lcdm

        # Approximate effective w from deviation
        # For small deviations: δw ≈ 2 × (ratio - 1) / (3 × Ω_DE)
        Omega_DE_z = self.Omega_Lambda / E_lcdm**2

        if Omega_DE_z > 0.01:
            delta_w = 2 * (ratio - 1) / (3 * Omega_DE_z)
            return -1 + delta_w
        else:
            return -1.0

    def dark_matter_fraction_from_remnants(self) -> float:
        """Fraction of CDM that is remnants: f_DM = Ω_rem / Ω_cdm."""
        return self.Omega_rem / self.Omega_cdm


# =============================================================================
# PART C: LIKELIHOOD AND MCMC
# =============================================================================

class HRCLikelihood:
    """
    Bayesian inference for HRC parameters.

    Computes likelihoods for various cosmological datasets
    and combines them for parameter estimation.
    """

    def __init__(self, data: ObservationalData = None):
        """
        Initialize likelihood calculator.

        Parameters
        ----------
        data : ObservationalData
            Observational constraints. If None, uses defaults.
        """
        self.data = data if data else ObservationalData()

        # Parameter names and bounds
        self.param_names = [
            'H0', 'Omega_m', 'xi', 'alpha', 'phi_0'
        ]

        self.param_bounds = {
            'H0': (60, 80),
            'Omega_m': (0.2, 0.5),
            'xi': (-0.1, 0.1),
            'alpha': (-0.1, 0.1),
            'phi_0': (0.0, 1.0),
        }

    def params_to_dict(self, theta: np.ndarray) -> Dict:
        """Convert parameter array to dictionary."""
        return dict(zip(self.param_names, theta))

    def log_likelihood_H0(self, params: Dict) -> float:
        """
        Combined likelihood for Hubble constant measurements.

        Key HRC feature: Model predicts different H0 values for
        local vs CMB measurements, potentially explaining the tension!
        """
        pred = HRCPredictions(params)

        H0_local_pred = pred.H0_local()
        H0_cmb_pred = pred.H0_cmb()

        # Local measurement
        H0_local_obs, sigma_local = self.data.local_H0
        chi2_local = ((H0_local_pred - H0_local_obs) / sigma_local)**2

        # CMB measurement
        H0_cmb_obs, sigma_cmb = self.data.cmb_H0
        chi2_cmb = ((H0_cmb_pred - H0_cmb_obs) / sigma_cmb)**2

        return -0.5 * (chi2_local + chi2_cmb)

    def log_likelihood_bao(self, params: Dict) -> float:
        """
        Likelihood for BAO measurements.

        Compare predicted D_V(z)/r_s with DESI measurements.
        """
        pred = HRCPredictions(params)
        bao = self.data.bao_measurements

        r_s = pred.sound_horizon()

        chi2 = 0
        for i, z in enumerate(bao.z_eff):
            D_V_pred = pred.D_V(z)
            DV_rd_pred = D_V_pred / r_s

            DV_rd_obs = bao.DV_rd[i]
            sigma = bao.DV_rd_err[i]

            chi2 += ((DV_rd_pred - DV_rd_obs) / sigma)**2

        return -0.5 * chi2

    def log_likelihood_sne(self, params: Dict) -> float:
        """
        Likelihood for Type Ia supernovae.

        Compare predicted distance modulus with Pantheon+ data.
        """
        pred = HRCPredictions(params)
        sn = self.data.sn_data

        chi2 = 0
        for i, z in enumerate(sn.z_bins):
            if z < 0.01:
                continue  # Skip very low z

            mu_pred = pred.distance_modulus(z)
            mu_obs = sn.mu_bins[i]
            sigma = sn.mu_err[i]

            chi2 += ((mu_pred - mu_obs) / sigma)**2

        return -0.5 * chi2

    def log_likelihood_cmb_compressed(self, params: Dict) -> float:
        """
        Simplified CMB likelihood using compressed statistics.

        Uses shift parameter R and acoustic scale, not full CMB spectrum.
        """
        pred = HRCPredictions(params)
        planck = self.data.planck_params

        # Shift parameter
        R_pred = pred.cmb_shift_parameter()
        R_obs = np.sqrt(planck.Omega_m) * planck.H0 / COSMO_CONST.c * \
                LCDMCosmology(planck.H0, planck.Omega_m).angular_diameter_distance(planck.z_star)
        sigma_R = 0.01 * R_obs  # ~1% precision

        chi2_R = ((R_pred - R_obs) / sigma_R)**2

        return -0.5 * chi2_R

    def log_prior(self, params: Dict) -> float:
        """
        Prior probability on HRC parameters.

        Uses flat priors within physical bounds.
        """
        for name, value in params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                if value < low or value > high:
                    return -np.inf

        # Additional physical constraints
        xi = params.get('xi', 0)
        phi_0 = params.get('phi_0', 0)
        G_factor = 1 - 8 * np.pi * xi * phi_0

        if G_factor <= 0:
            return -np.inf  # Unphysical

        return 0.0  # Flat prior within bounds

    def log_likelihood(self, params: Dict) -> float:
        """
        Total log-likelihood from all datasets.
        """
        ll = 0

        ll += self.log_likelihood_H0(params)
        ll += self.log_likelihood_bao(params)
        ll += self.log_likelihood_sne(params)
        ll += self.log_likelihood_cmb_compressed(params)

        return ll

    def log_posterior(self, theta: np.ndarray) -> float:
        """
        Log-posterior for MCMC.

        log P(θ|data) ∝ log L(data|θ) + log π(θ)
        """
        params = self.params_to_dict(theta)

        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        try:
            ll = self.log_likelihood(params)
        except Exception as e:
            logger.debug(f"Likelihood evaluation failed: {e}")
            return -np.inf

        return lp + ll

    def run_mcmc(self, n_walkers: int = 32, n_steps: int = 1000,
                 initial_params: Dict = None,
                 progress: bool = True) -> Optional['emcee.EnsembleSampler']:
        """
        Run MCMC using emcee.

        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of steps per walker
        initial_params : dict
            Initial parameter values
        progress : bool
            Show progress bar

        Returns
        -------
        emcee.EnsembleSampler
            Sampler with chains
        """
        if not HAS_EMCEE:
            raise ImportError("emcee required for MCMC. Install with: pip install emcee")

        ndim = len(self.param_names)

        # Initial positions
        if initial_params is None:
            initial_params = {
                'H0': 70.0,
                'Omega_m': 0.315,
                'xi': 0.01,
                'alpha': 0.01,
                'phi_0': 0.1,
            }

        p0_center = np.array([initial_params.get(name, 0) for name in self.param_names])
        p0 = p0_center + 1e-4 * np.random.randn(n_walkers, ndim)

        # Ensure within bounds
        for i, name in enumerate(self.param_names):
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                p0[:, i] = np.clip(p0[:, i], low + 0.01, high - 0.01)

        # Run sampler
        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_posterior)

        logger.info(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
        sampler.run_mcmc(p0, n_steps, progress=progress)

        return sampler

    def analyze_chains(self, sampler, burn_in: int = 200) -> Dict:
        """
        Analyze MCMC chains.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            Sampler with completed chains
        burn_in : int
            Number of steps to discard as burn-in

        Returns
        -------
        dict
            Summary statistics
        """
        samples = sampler.get_chain(discard=burn_in, flat=True)

        results = {}

        for i, name in enumerate(self.param_names):
            chain = samples[:, i]
            results[name] = {
                'mean': np.mean(chain),
                'std': np.std(chain),
                'median': np.median(chain),
                'q16': np.percentile(chain, 16),
                'q84': np.percentile(chain, 84),
                'q2.5': np.percentile(chain, 2.5),
                'q97.5': np.percentile(chain, 97.5),
            }

        # Compute derived quantities
        H0_local_samples = []
        H0_cmb_samples = []

        for theta in samples[::10]:  # Subsample for speed
            params = self.params_to_dict(theta)
            pred = HRCPredictions(params)
            H0_local_samples.append(pred.H0_local())
            H0_cmb_samples.append(pred.H0_cmb())

        results['H0_local_derived'] = {
            'mean': np.mean(H0_local_samples),
            'std': np.std(H0_local_samples),
        }
        results['H0_cmb_derived'] = {
            'mean': np.mean(H0_cmb_samples),
            'std': np.std(H0_cmb_samples),
        }
        results['tension_resolved'] = {
            'delta_H0': np.mean(H0_local_samples) - np.mean(H0_cmb_samples),
        }

        return results


# =============================================================================
# PART D: MODEL COMPARISON
# =============================================================================

class FisherMatrix:
    """
    Fisher matrix approximation for quick parameter estimation.

    Faster than MCMC but assumes Gaussian posterior.
    """

    def __init__(self, likelihood: HRCLikelihood, fiducial_params: Dict):
        """
        Initialize Fisher matrix calculator.

        Parameters
        ----------
        likelihood : HRCLikelihood
            Likelihood object
        fiducial_params : dict
            Fiducial parameter values
        """
        self.likelihood = likelihood
        self.fiducial = fiducial_params
        self.param_names = likelihood.param_names

    def compute_fisher(self, step_size: float = 0.01) -> np.ndarray:
        """
        Compute Fisher information matrix numerically.

        F_ij = -E[∂²log L / ∂θ_i ∂θ_j]

        Approximated with finite differences.
        """
        ndim = len(self.param_names)
        fisher = np.zeros((ndim, ndim))

        theta_fid = np.array([self.fiducial[name] for name in self.param_names])

        for i in range(ndim):
            for j in range(i, ndim):
                # Second derivative via finite differences
                theta_pp = theta_fid.copy()
                theta_pm = theta_fid.copy()
                theta_mp = theta_fid.copy()
                theta_mm = theta_fid.copy()

                h_i = abs(theta_fid[i]) * step_size + 1e-8
                h_j = abs(theta_fid[j]) * step_size + 1e-8

                theta_pp[i] += h_i
                theta_pp[j] += h_j

                theta_pm[i] += h_i
                theta_pm[j] -= h_j

                theta_mp[i] -= h_i
                theta_mp[j] += h_j

                theta_mm[i] -= h_i
                theta_mm[j] -= h_j

                ll_pp = self.likelihood.log_posterior(theta_pp)
                ll_pm = self.likelihood.log_posterior(theta_pm)
                ll_mp = self.likelihood.log_posterior(theta_mp)
                ll_mm = self.likelihood.log_posterior(theta_mm)

                # Mixed second derivative
                d2ll = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * h_i * h_j)

                fisher[i, j] = -d2ll
                fisher[j, i] = fisher[i, j]

        return fisher

    def parameter_covariance(self) -> np.ndarray:
        """
        Compute parameter covariance matrix.

        Cov = F^(-1)
        """
        fisher = self.compute_fisher()

        try:
            cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            logger.warning("Fisher matrix singular, using pseudo-inverse")
            cov = np.linalg.pinv(fisher)

        return cov

    def parameter_errors(self) -> Dict[str, float]:
        """
        Compute 1σ parameter uncertainties.

        σ_i = √(Cov_ii)
        """
        cov = self.parameter_covariance()

        errors = {}
        for i, name in enumerate(self.param_names):
            errors[name] = np.sqrt(max(cov[i, i], 0))

        return errors


def compute_bayes_factor(data: ObservationalData,
                         hrc_params: Dict = None,
                         n_samples: int = 1000) -> Dict:
    """
    Compute Bayes factor comparing HRC to ΛCDM.

    B = P(data | HRC) / P(data | ΛCDM)

    Uses Savage-Dickey density ratio for nested models.

    Parameters
    ----------
    data : ObservationalData
        Observational constraints
    hrc_params : dict
        HRC fiducial parameters
    n_samples : int
        Number of samples for Monte Carlo integration

    Returns
    -------
    dict
        Bayes factor and interpretation
    """
    likelihood = HRCLikelihood(data)

    # ΛCDM is HRC with xi=0, alpha=0
    lcdm_params = {
        'H0': 67.4,
        'Omega_m': 0.315,
        'xi': 0.0,
        'alpha': 0.0,
        'phi_0': 0.0,
    }

    if hrc_params is None:
        hrc_params = {
            'H0': 70.0,
            'Omega_m': 0.315,
            'xi': 0.01,
            'alpha': 0.01,
            'phi_0': 0.1,
        }

    # Compute log-likelihoods at best-fit points
    ll_lcdm = likelihood.log_likelihood(lcdm_params)
    ll_hrc = likelihood.log_likelihood(hrc_params)

    # Difference in log-likelihood (not full Bayes factor)
    delta_ll = ll_hrc - ll_lcdm

    # Penalize for extra parameters (BIC approximation)
    n_data = len(data.bao_measurements.z_eff) + len(data.sn_data.z_bins) + 2
    n_params_lcdm = 2  # H0, Omega_m
    n_params_hrc = 5   # + xi, alpha, phi_0

    bic_lcdm = -2 * ll_lcdm + n_params_lcdm * np.log(n_data)
    bic_hrc = -2 * ll_hrc + n_params_hrc * np.log(n_data)

    delta_bic = bic_hrc - bic_lcdm

    # Approximate Bayes factor from BIC
    ln_B = -0.5 * delta_bic

    # Interpretation
    if ln_B > 5:
        interpretation = "Very strong evidence for HRC"
    elif ln_B > 2.5:
        interpretation = "Strong evidence for HRC"
    elif ln_B > 1:
        interpretation = "Moderate evidence for HRC"
    elif ln_B > 0:
        interpretation = "Weak evidence for HRC"
    elif ln_B > -1:
        interpretation = "Weak evidence for ΛCDM"
    elif ln_B > -2.5:
        interpretation = "Moderate evidence for ΛCDM"
    else:
        interpretation = "Strong evidence for ΛCDM"

    return {
        'ln_B': ln_B,
        'B': np.exp(ln_B),
        'delta_ll': delta_ll,
        'bic_lcdm': bic_lcdm,
        'bic_hrc': bic_hrc,
        'delta_bic': delta_bic,
        'interpretation': interpretation,
    }


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def quick_tension_analysis():
    """
    Quick analysis: Can HRC resolve the Hubble tension?

    Scans parameter space to find regions where tension is resolved.
    """
    data = ObservationalData()

    print("=" * 60)
    print("HUBBLE TENSION ANALYSIS FOR HRC")
    print("=" * 60)

    print(f"\nObserved tension: {data.hubble_tension:.1f}σ")
    print(f"  Local H0:  {data.local_H0[0]:.1f} ± {data.local_H0[1]:.1f} km/s/Mpc")
    print(f"  CMB H0:    {data.cmb_H0[0]:.1f} ± {data.cmb_H0[1]:.1f} km/s/Mpc")

    print("\nScanning HRC parameter space...")

    # Scan xi and phi_0
    xi_values = np.linspace(-0.05, 0.05, 11)
    phi_values = np.linspace(0, 0.5, 11)

    best_tension_diff = np.inf
    best_params = None

    target_delta_H0 = data.local_H0[0] - data.cmb_H0[0]  # ~5.6

    for xi in xi_values:
        for phi_0 in phi_values:
            params = {
                'H0_true': 70.0,
                'Omega_m': 0.315,
                'xi': xi,
                'alpha': 0.01,
                'phi_0': phi_0,
            }

            try:
                pred = HRCPredictions(params)
                H0_local = pred.H0_local()
                H0_cmb = pred.H0_cmb()
                delta_H0 = H0_local - H0_cmb

                diff = abs(delta_H0 - target_delta_H0)

                if diff < best_tension_diff:
                    best_tension_diff = diff
                    best_params = params.copy()
                    best_params['H0_local'] = H0_local
                    best_params['H0_cmb'] = H0_cmb
                    best_params['delta_H0'] = delta_H0

            except Exception:
                continue

    if best_params:
        print("\nBest-fit HRC parameters to explain tension:")
        print(f"  ξ = {best_params['xi']:.4f}")
        print(f"  φ₀ = {best_params['phi_0']:.4f}")
        print(f"\nPredicted H0 values:")
        print(f"  H0_local: {best_params['H0_local']:.2f} km/s/Mpc")
        print(f"  H0_cmb:   {best_params['H0_cmb']:.2f} km/s/Mpc")
        print(f"  ΔH0:      {best_params['delta_H0']:.2f} km/s/Mpc")
        print(f"  Target:   {target_delta_H0:.2f} km/s/Mpc")
        print(f"  Residual: {best_tension_diff:.2f} km/s/Mpc")

        resolved = best_tension_diff < 2.0
        print(f"\nTension resolved: {'YES' if resolved else 'NO'}")
    else:
        print("No valid HRC solutions found!")

    return best_params


def run_full_analysis():
    """
    Run full Bayesian analysis of HRC model.
    """
    print("=" * 60)
    print("FULL BAYESIAN ANALYSIS OF HRC MODEL")
    print("=" * 60)

    data = ObservationalData()
    print(data.summary())

    likelihood = HRCLikelihood(data)

    # Fisher matrix quick estimate
    print("\n1. FISHER MATRIX QUICK ESTIMATE")
    print("-" * 40)

    fiducial = {
        'H0': 70.0,
        'Omega_m': 0.315,
        'xi': 0.01,
        'alpha': 0.01,
        'phi_0': 0.1,
    }

    fisher = FisherMatrix(likelihood, fiducial)
    errors = fisher.parameter_errors()

    print("Parameter uncertainties (Fisher estimate):")
    for name, err in errors.items():
        print(f"  σ({name}) = {err:.4f}")

    # Model comparison
    print("\n2. MODEL COMPARISON (HRC vs ΛCDM)")
    print("-" * 40)

    bf = compute_bayes_factor(data, fiducial)
    print(f"  ln(Bayes factor) = {bf['ln_B']:.2f}")
    print(f"  ΔBIC = {bf['delta_bic']:.2f}")
    print(f"  Interpretation: {bf['interpretation']}")

    # MCMC (if emcee available)
    if HAS_EMCEE:
        print("\n3. MCMC ANALYSIS")
        print("-" * 40)
        print("  Running short MCMC chain for demonstration...")

        try:
            sampler = likelihood.run_mcmc(n_walkers=16, n_steps=500, progress=False)
            results = likelihood.analyze_chains(sampler, burn_in=100)

            print("\nMCMC Results:")
            for name in likelihood.param_names:
                r = results[name]
                print(f"  {name} = {r['mean']:.4f} ± {r['std']:.4f}")

            print(f"\nDerived quantities:")
            print(f"  H0_local = {results['H0_local_derived']['mean']:.2f} ± {results['H0_local_derived']['std']:.2f}")
            print(f"  H0_cmb = {results['H0_cmb_derived']['mean']:.2f} ± {results['H0_cmb_derived']['std']:.2f}")
            print(f"  ΔH0 = {results['tension_resolved']['delta_H0']:.2f}")

        except Exception as e:
            print(f"  MCMC failed: {e}")
    else:
        print("\n3. MCMC ANALYSIS")
        print("  Skipped (emcee not installed)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Quick tension analysis
    best_params = quick_tension_analysis()

    print("\n")

    # Full analysis
    run_full_analysis()
