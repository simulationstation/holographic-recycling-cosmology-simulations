"""
Mapping from No-Boundary Primordial Parameters to Cosmological Parameters

This module implements the physical mapping from Hawking-Hartle primordial
parameters (N_e, V_scale, phi_init, epsilon_corr) to observable cosmological
parameters (H0, Omega_m, Omega_k, A_s, n_s, etc.).

The mapping encodes:
1. Inflation determines primordial perturbations (A_s, n_s, r)
2. Number of e-folds determines spatial curvature (Omega_k)
3. epsilon_corr modifies early-time H(z) affecting sound horizon

Theory References:
    - Slow-roll inflation: A_s ~ V/epsilon, n_s ~ 1 - 6*epsilon + 2*eta
    - Curvature from e-folds: Omega_k ~ exp(-2*N_e) for sufficient N
    - Sound horizon: r_s = integral of c_s/H from z_star to infinity
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import numpy as np
from scipy import integrate

from .no_boundary_prior import NoBoundaryParams


# Physical constants (Planck 2018)
M_PL = 1.0  # Planck mass in natural units
RHO_CRIT_H100 = 2.775e11  # h^2 M_sun / Mpc^3
C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255  # K
Z_STAR = 1089.80  # Recombination redshift (Planck 2018)
Z_DRAG = 1059.94  # Baryon drag epoch


@dataclass
class CosmoParams:
    """
    Standard cosmological parameters derived from primordial params.

    These are the parameters that enter into observables like CMB spectra,
    BAO scales, and distance-redshift relations.
    """
    # Background geometry
    H0: float          # Hubble constant [km/s/Mpc]
    Omega_m: float     # Total matter density
    Omega_b: float     # Baryon density
    Omega_k: float     # Curvature parameter
    Omega_L: float     # Dark energy density (cosmological constant)

    # Primordial perturbations
    A_s: float         # Scalar amplitude (at k_pivot = 0.05/Mpc)
    n_s: float         # Scalar spectral index
    r: float           # Tensor-to-scalar ratio

    # Derived quantities
    h: float           # H0 / 100
    theta_s: float     # CMB acoustic scale (in radians * 100)
    r_s: float         # Sound horizon at drag epoch [Mpc]

    # Early-time modification
    epsilon_corr: float  # H(z) correction factor at high-z

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_b": self.Omega_b,
            "Omega_k": self.Omega_k,
            "Omega_L": self.Omega_L,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "r": self.r,
            "h": self.h,
            "theta_s": self.theta_s,
            "r_s": self.r_s,
            "epsilon_corr": self.epsilon_corr
        }


@dataclass
class InflationModel:
    """
    Single-field slow-roll inflation model parameters.

    Encodes the mapping from V_scale, phi_init to slow-roll parameters.
    """
    # Potential type: "quadratic", "starobinsky", "natural"
    potential_type: str = "quadratic"

    # Model-specific parameters
    m_inflaton: float = 1e-6   # Inflaton mass (for quadratic) [M_pl]
    f_axion: float = 1.0       # Axion decay constant (for natural) [M_pl]
    alpha_R2: float = 1e9      # R^2 coefficient (for Starobinsky)


def compute_slow_roll_params(
    V_scale: float,
    phi_init: float,
    model: InflationModel
) -> Tuple[float, float, float]:
    """
    Compute slow-roll parameters (epsilon, eta) from potential.

    For quadratic potential V = (1/2) m^2 phi^2:
        epsilon = (1/2) * (V'/V)^2 = 2 / phi^2
        eta = V''/V = 2 / phi^2

    For Starobinsky (approximate):
        epsilon ~ 3 / (4 * N_remaining^2)
        eta ~ -2 / N_remaining

    Args:
        V_scale: log10 of potential scale
        phi_init: Initial field value [M_pl]
        model: Inflation model parameters

    Returns:
        (epsilon, eta, V) - slow-roll params and potential value
    """
    if model.potential_type == "quadratic":
        # V = (1/2) * m^2 * phi^2
        V = 0.5 * model.m_inflaton**2 * phi_init**2

        # Slow-roll parameters
        if abs(phi_init) > 1e-10:
            epsilon = 2.0 / phi_init**2
            eta = 2.0 / phi_init**2
        else:
            epsilon = 1e10
            eta = 1e10

    elif model.potential_type == "starobinsky":
        # Starobinsky-like: V ~ V0 * (1 - exp(-sqrt(2/3) * phi))^2
        x = np.sqrt(2.0/3.0) * phi_init
        exp_x = np.exp(-x)

        V = 10**V_scale * (1 - exp_x)**2

        if abs(1 - exp_x) > 1e-10:
            epsilon = (4.0/3.0) * exp_x**2 / (1 - exp_x)**2
            eta = (4.0/3.0) * exp_x * (2*exp_x - 1) / (1 - exp_x)**2
        else:
            epsilon = 1e10
            eta = 1e10

    elif model.potential_type == "natural":
        # Natural inflation: V = V0 * (1 + cos(phi/f))
        V = 10**V_scale * (1 + np.cos(phi_init / model.f_axion))

        denom = 1 + np.cos(phi_init / model.f_axion)
        if abs(denom) > 1e-10:
            epsilon = 0.5 * (np.sin(phi_init / model.f_axion) / model.f_axion / denom)**2
            eta = -np.cos(phi_init / model.f_axion) / model.f_axion**2 / denom
        else:
            epsilon = 1e10
            eta = 1e10
    else:
        # Default to quadratic
        V = 0.5 * model.m_inflaton**2 * phi_init**2
        epsilon = 2.0 / phi_init**2 if abs(phi_init) > 1e-10 else 1e10
        eta = epsilon

    return epsilon, eta, V


def compute_primordial_spectrum(
    Ne: float,
    V_scale: float,
    phi_init: float,
    model: Optional[InflationModel] = None
) -> Tuple[float, float, float]:
    """
    Compute primordial perturbation parameters from inflation.

    The scalar amplitude A_s and spectral index n_s are determined by:
        A_s = V / (24 * pi^2 * epsilon * M_pl^4)
        n_s = 1 - 6*epsilon + 2*eta
        r = 16 * epsilon

    Args:
        Ne: Number of e-folds
        V_scale: log10 of potential scale
        phi_init: Initial field value
        model: Inflation model (default: quadratic)

    Returns:
        (A_s, n_s, r) - primordial spectrum parameters
    """
    if model is None:
        model = InflationModel()

    epsilon, eta, V = compute_slow_roll_params(V_scale, phi_init, model)

    # Amplitude - normalized to observed value
    # A_s ~ 2.1e-9 from Planck
    # This comes from V / (24 * pi^2 * epsilon)
    if epsilon > 0 and epsilon < 1:
        A_s_raw = V / (24 * np.pi**2 * epsilon)
        # Normalize by scaling V_scale
        # A_s = A_s_raw scaled appropriately
        # Use V_scale to set the overall amplitude
        A_s = 10**(V_scale + 9) * 2.1e-9  # Simplified normalization
    else:
        A_s = 2.1e-9  # Planck fiducial

    # Spectral index
    n_s = 1 - 6*epsilon + 2*eta
    # Keep n_s in physical range
    n_s = np.clip(n_s, 0.8, 1.1)

    # Tensor-to-scalar ratio
    r = 16 * epsilon
    r = np.clip(r, 0, 1)  # Physical bound

    return A_s, n_s, r


def compute_curvature_from_efolds(
    Ne: float,
    Ne_total: float = 60.0
) -> float:
    """
    Compute spatial curvature from number of e-folds.

    More e-folds -> universe gets stretched flatter
    Omega_k ~ exp(-2 * N) for large N

    In practice, for N > 60, Omega_k is unmeasurably small.
    We parameterize as: Omega_k = Omega_k_max * exp(-alpha * (N - N_min))

    Args:
        Ne: Number of e-folds of inflation
        Ne_total: Reference total e-folds (where Omega_k ~ 0)

    Returns:
        Omega_k (can be positive or negative)
    """
    # Maximum |Omega_k| for N = 50 (minimal inflation)
    Omega_k_max = 0.01  # |Omega_k| < 0.01 from Planck

    # Decay constant - 10 e-folds reduces by factor of ~20
    alpha = 0.3

    # Reference point
    N_ref = 50.0

    if Ne < N_ref:
        # Very few e-folds - significant curvature possible
        Omega_k = Omega_k_max * np.exp(alpha * (N_ref - Ne))
        Omega_k = np.clip(Omega_k, -0.1, 0.1)
    else:
        # Many e-folds - curvature exponentially suppressed
        Omega_k = Omega_k_max * np.exp(-alpha * (Ne - N_ref))

    # Sign is random (quantum fluctuation) - we'll use positive
    # In practice, |Omega_k| << 1 for reasonable Ne
    return Omega_k


def compute_sound_horizon(
    H0: float,
    Omega_m: float,
    Omega_b: float,
    epsilon_corr: float = 0.0,
    z_transition: float = 3000.0
) -> float:
    """
    Compute the comoving sound horizon at the drag epoch.

    r_s = integral from z_drag to infinity of c_s / H(z) dz

    where c_s = c / sqrt(3(1 + R)) and R = 3*rho_b / (4*rho_gamma)

    The epsilon_corr parameter modifies H(z) at z > z_transition:
        H(z) -> H(z) * (1 + epsilon_corr * f(z))

    where f(z) smoothly transitions from 0 at z < z_transition to 1 at z >> z_transition.

    Args:
        H0: Hubble constant [km/s/Mpc]
        Omega_m: Matter density
        Omega_b: Baryon density
        epsilon_corr: Early-time H(z) correction
        z_transition: Transition redshift for correction

    Returns:
        r_s: Sound horizon at drag epoch [Mpc]
    """
    h = H0 / 100.0
    omega_b = Omega_b * h**2
    omega_m = Omega_m * h**2

    # Photon density from CMB temperature
    Omega_gamma = 2.469e-5 / h**2 * (T_CMB / 2.7255)**4
    omega_gamma = Omega_gamma * h**2

    def integrand(z):
        # Baryon-to-photon ratio
        R = 3 * omega_b / (4 * omega_gamma) * 1/(1+z)

        # Sound speed
        c_s = C_LIGHT / np.sqrt(3 * (1 + R))

        # Hubble parameter (matter + radiation + Lambda)
        Omega_r = Omega_gamma * (1 + 0.2271 * 3.046)  # Include neutrinos
        E_z = np.sqrt(Omega_m * (1+z)**3 + Omega_r * (1+z)**4 +
                      (1 - Omega_m - Omega_r))
        H_z = H0 * E_z

        # Apply epsilon correction at high-z
        if z > z_transition:
            # Smooth transition function
            f_z = 1 - np.exp(-(z - z_transition) / z_transition)
            H_z *= (1 + epsilon_corr * f_z)

        return c_s / H_z

    # Integrate from drag epoch to high redshift
    z_max = 1e6  # High enough for convergence
    result, _ = integrate.quad(integrand, Z_DRAG, z_max, limit=100)

    return result


def compute_theta_star(
    H0: float,
    Omega_m: float,
    Omega_k: float,
    r_s: float
) -> float:
    """
    Compute the CMB acoustic angular scale theta_*.

    theta_* = r_s / D_A(z_*)

    where D_A is the angular diameter distance to recombination.

    Args:
        H0: Hubble constant [km/s/Mpc]
        Omega_m: Matter density
        Omega_k: Curvature
        r_s: Sound horizon [Mpc]

    Returns:
        theta_s: Acoustic scale in radians * 100
    """
    h = H0 / 100.0
    Omega_L = 1 - Omega_m - Omega_k

    def E_inv(z):
        return 1.0 / np.sqrt(Omega_m * (1+z)**3 + Omega_k * (1+z)**2 + Omega_L)

    # Comoving distance to z_star
    d_C, _ = integrate.quad(E_inv, 0, Z_STAR, limit=100)
    d_C *= C_LIGHT / H0  # Convert to Mpc

    # Angular diameter distance
    if abs(Omega_k) < 1e-6:
        D_A = d_C / (1 + Z_STAR)
    elif Omega_k > 0:
        # Open universe
        K = np.sqrt(Omega_k) * H0 / C_LIGHT
        D_A = np.sinh(K * d_C) / K / (1 + Z_STAR)
    else:
        # Closed universe
        K = np.sqrt(-Omega_k) * H0 / C_LIGHT
        D_A = np.sin(K * d_C) / K / (1 + Z_STAR)

    # Angular scale
    theta_s = r_s / D_A

    # Return in units of radians * 100 (standard convention)
    return theta_s * 100


def primordial_to_cosmo(
    params: NoBoundaryParams,
    inflation_model: Optional[InflationModel] = None,
    Omega_b: float = 0.0493,    # Planck 2018 fiducial
    Omega_m_base: float = 0.315, # Planck 2018 fiducial
    H0_base: float = 67.4,       # Planck 2018 fiducial
    z_transition: float = 3000.0
) -> CosmoParams:
    """
    Map no-boundary primordial parameters to cosmological parameters.

    This is the core mapping function that converts:
        (N_e, log10_V_scale, phi_init, epsilon_corr) -> (H0, Omega_m, n_s, ...)

    The mapping incorporates:
    1. Curvature from e-folds: Omega_k = f(N_e)
    2. Primordial spectrum from slow-roll: A_s, n_s, r = g(V_scale, phi_init)
    3. Sound horizon modification: r_s = r_s(epsilon_corr)
    4. H0 inferred from CMB: H0 = H0(r_s, theta_s_observed)

    Args:
        params: NoBoundaryParams (primordial parameters)
        inflation_model: Optional inflation model specification
        Omega_b: Baryon density (fixed or from prior)
        Omega_m_base: Base matter density
        H0_base: Base Hubble constant
        z_transition: Redshift where epsilon_corr turns on

    Returns:
        CosmoParams: Derived cosmological parameters
    """
    if inflation_model is None:
        inflation_model = InflationModel()

    # 1. Compute curvature from e-folds
    Omega_k = compute_curvature_from_efolds(params.Ne)

    # 2. Compute primordial spectrum
    A_s, n_s, r = compute_primordial_spectrum(
        params.Ne,
        params.log10_V_scale,
        params.phi_init,
        inflation_model
    )

    # 3. Adjust Omega_m to maintain closure
    Omega_m = Omega_m_base
    Omega_L = 1 - Omega_m - Omega_k

    # Ensure physical values
    if Omega_L < 0:
        Omega_m = 1 - Omega_k - 0.01
        Omega_L = 0.01

    # 4. Compute sound horizon with epsilon correction
    r_s = compute_sound_horizon(
        H0_base, Omega_m, Omega_b,
        params.epsilon_corr, z_transition
    )

    # 5. Infer H0 from CMB acoustic scale
    # The observed theta_s is fixed by Planck: theta_s_obs ~ 1.04110 * 10^-2 rad
    theta_s_obs = 1.04110  # in radians * 100

    # Compute what H0 would give this theta_s
    # theta_s = r_s / D_A(z_*) ~ r_s * H0 / c * integral
    # So H0 ~ theta_s_obs * c / (r_s * integral_factor)

    # Iteratively solve for H0 that gives observed theta_s
    H0 = H0_base
    for _ in range(5):  # Converges quickly
        r_s_current = compute_sound_horizon(H0, Omega_m, Omega_b,
                                            params.epsilon_corr, z_transition)
        theta_s_current = compute_theta_star(H0, Omega_m, Omega_k, r_s_current)

        # Adjust H0 to match observed theta_s
        # theta ~ r_s / D_A, D_A ~ c/H0, so theta ~ r_s * H0 / c
        # Therefore H0 should scale as theta_obs / theta_current
        H0 *= theta_s_obs / theta_s_current

        # Keep in physical range
        H0 = np.clip(H0, 50.0, 100.0)

    # Update sound horizon with final H0
    r_s = compute_sound_horizon(H0, Omega_m, Omega_b,
                                params.epsilon_corr, z_transition)
    theta_s = compute_theta_star(H0, Omega_m, Omega_k, r_s)

    return CosmoParams(
        H0=H0,
        Omega_m=Omega_m,
        Omega_b=Omega_b,
        Omega_k=Omega_k,
        Omega_L=Omega_L,
        A_s=A_s,
        n_s=n_s,
        r=r,
        h=H0/100.0,
        theta_s=theta_s,
        r_s=r_s,
        epsilon_corr=params.epsilon_corr
    )


def compute_early_H0(cosmo: CosmoParams) -> float:
    """
    Compute the "early" H0 inferred from CMB/BAO data.

    This is the H0 you would infer by fitting LCDM to CMB+BAO,
    assuming the sound horizon r_s is correctly computed.

    In the presence of epsilon_corr > 0, the true H(z) at high-z
    is larger than LCDM predicts, making r_s smaller, and leading
    to a higher inferred H0 when assuming standard LCDM.

    Args:
        cosmo: CosmoParams with epsilon_corr

    Returns:
        H0_early: CMB-inferred H0 [km/s/Mpc]
    """
    # If epsilon_corr = 0, early and late H0 are the same
    if abs(cosmo.epsilon_corr) < 1e-10:
        return cosmo.H0

    # Compute sound horizon without correction (what CMB analysis assumes)
    r_s_standard = compute_sound_horizon(
        cosmo.H0, cosmo.Omega_m, cosmo.Omega_b,
        epsilon_corr=0.0
    )

    # The ratio of sound horizons tells us how H0 would be biased
    # Larger H(z) at high-z -> smaller r_s -> higher inferred H0
    r_s_ratio = cosmo.r_s / r_s_standard

    # To first order: H0_inferred ~ H0_true / r_s_ratio
    # (because D_A ~ c/H0 and theta = r_s/D_A is fixed)
    H0_early = cosmo.H0 / r_s_ratio

    return H0_early


def compute_late_H0(cosmo: CosmoParams) -> float:
    """
    Compute the "late" H0 from local distance ladder.

    In the no-boundary framework, the late-time universe is
    standard LCDM, so the local H0 equals the true H0.

    Args:
        cosmo: CosmoParams

    Returns:
        H0_late: Local/ladder H0 [km/s/Mpc]
    """
    # Late-time H0 is unaffected by early-time modifications
    return cosmo.H0
