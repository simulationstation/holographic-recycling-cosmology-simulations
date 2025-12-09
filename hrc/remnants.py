"""Black hole remnant physics for HRC.

This module implements the physics of Planck-mass remnants from black hole
evaporation, including:
- Remnant formation rate from primordial black holes
- Mass evolution during evaporation
- Number density evolution
- Contribution to dark matter density

The key assumption is that Hawking evaporation halts at the Planck mass,
leaving stable remnants.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

from .utils.config import HRCParameters
from .utils.constants import SI_UNITS


@dataclass
class RemnantProperties:
    """Properties of a single remnant."""

    M_rem: float = SI_UNITS.M_Planck  # Remnant mass [kg]
    M_rem_Mpl: float = 1.0  # Remnant mass in Planck units

    # Quantum properties
    spin: float = 0.0  # Remnant spin
    charge: float = 0.0  # Remnant charge

    # Information storage
    n_qubits: float = 1e40  # Information capacity (rough estimate)


@dataclass
class RemnantPopulation:
    """Statistical properties of remnant population."""

    z: NDArray[np.floating]  # Redshift array
    n_rem: NDArray[np.floating]  # Number density [m⁻³]
    rho_rem: NDArray[np.floating]  # Mass density [kg/m³]
    Omega_rem: NDArray[np.floating]  # Density parameter

    # Formation history
    formation_rate: NDArray[np.floating]  # dn/dt [m⁻³ s⁻¹]

    # Derived quantities
    f_dm: float = 0.0  # Fraction of DM in remnants today


class HawkingEvaporation:
    """Model for black hole evaporation to remnants.

    A black hole of initial mass M evaporates via Hawking radiation:
        dM/dt = -α/M² (for M >> M_Pl)

    where α = ℏc⁴/(15360πG²) ≈ 3.5 × 10⁻⁸ kg³/s.

    Evaporation halts at M = M_Pl, forming a stable remnant.
    """

    def __init__(
        self,
        M_rem: float = SI_UNITS.M_Planck,
    ):
        """Initialize evaporation model.

        Args:
            M_rem: Final remnant mass [kg]
        """
        self.M_rem = M_rem

        # Hawking evaporation coefficient
        # α = ℏc⁴/(15360πG²)
        hbar = SI_UNITS.hbar
        c = SI_UNITS.c
        G = SI_UNITS.G
        self.alpha = hbar * c**4 / (15360 * np.pi * G**2)

    def evaporation_time(self, M_initial: float) -> float:
        """Compute time to evaporate from M_initial to M_rem.

        t_evap = (M_initial³ - M_rem³) / (3α)

        Args:
            M_initial: Initial BH mass [kg]

        Returns:
            Evaporation time [s]
        """
        return (M_initial**3 - self.M_rem**3) / (3 * self.alpha)

    def mass_at_time(
        self,
        M_initial: float,
        t: float,
    ) -> float:
        """Compute BH mass at time t after formation.

        M(t) = (M_initial³ - 3αt)^(1/3)

        Args:
            M_initial: Initial mass [kg]
            t: Time since formation [s]

        Returns:
            Mass at time t [kg], clamped at M_rem
        """
        M_cubed = M_initial**3 - 3 * self.alpha * t

        if M_cubed <= self.M_rem**3:
            return self.M_rem

        return M_cubed ** (1.0 / 3.0)

    def hawking_temperature(self, M: float) -> float:
        """Compute Hawking temperature.

        T_H = ℏc³/(8πGMk_B)

        Args:
            M: BH mass [kg]

        Returns:
            Temperature [K]
        """
        hbar = SI_UNITS.hbar
        c = SI_UNITS.c
        G = SI_UNITS.G
        k_B = SI_UNITS.k_B

        return hbar * c**3 / (8 * np.pi * G * M * k_B)

    def luminosity(self, M: float) -> float:
        """Compute Hawking luminosity.

        L = α/M² = ℏc⁴/(15360πG²M²)

        Args:
            M: BH mass [kg]

        Returns:
            Luminosity [W]
        """
        return self.alpha / M**2


class PrimordialBlackHoles:
    """Model for primordial black hole formation and evolution.

    PBHs form during radiation domination when density perturbations
    collapse. Their mass spectrum and abundance are determined by
    the primordial power spectrum.
    """

    def __init__(
        self,
        M_min: float = 1e-5 * SI_UNITS.M_Planck,  # Minimum PBH mass
        M_max: float = 1e20 * SI_UNITS.M_Planck,  # Maximum PBH mass
        f_pbh: float = 1e-10,  # Fraction of DM in PBHs at formation
        spectral_index: float = 0.0,  # Mass spectrum power law
    ):
        """Initialize PBH model.

        Args:
            M_min: Minimum PBH mass at formation
            M_max: Maximum PBH mass at formation
            f_pbh: Initial PBH fraction of DM
            spectral_index: Power law index (dn/dM ∝ M^spectral_index)
        """
        self.M_min = M_min
        self.M_max = M_max
        self.f_pbh = f_pbh
        self.spectral_index = spectral_index

    def mass_spectrum(
        self,
        M: float,
        normalization: float = 1.0,
    ) -> float:
        """Compute PBH mass spectrum dn/dM.

        dn/dM ∝ M^spectral_index

        Args:
            M: Mass [kg]
            normalization: Overall normalization

        Returns:
            dn/dM [kg⁻¹ m⁻³]
        """
        if M < self.M_min or M > self.M_max:
            return 0.0

        return normalization * M**self.spectral_index

    def formation_redshift(self, M: float) -> float:
        """Estimate formation redshift for PBH of mass M.

        PBHs form when the horizon mass equals M:
        M ~ M_H(z) ~ M_Pl² / m_H(z)

        For radiation domination:
        M ≈ (t/t_Pl)^(1/2) M_Pl ≈ (T_Pl/T)² M_Pl

        Args:
            M: PBH mass [kg]

        Returns:
            Formation redshift
        """
        # Very rough approximation
        # M ~ 10^5 (t/1s) M_sun ~ 10^5 (10^19 s / T[GeV])^2 M_sun
        M_solar = 2e30  # kg
        M_ratio = M / M_solar

        if M_ratio > 1e10:
            return 1e6  # Late formation
        elif M_ratio < 1e-10:
            return 1e20  # Very early formation

        # z ~ 10^9 (M/M_solar)^(-1/2) roughly
        return 1e9 * M_ratio ** (-0.5)


class RemnantFormation:
    """Compute remnant formation from PBH evaporation."""

    def __init__(
        self,
        evaporation: Optional[HawkingEvaporation] = None,
        pbh_model: Optional[PrimordialBlackHoles] = None,
    ):
        """Initialize remnant formation model.

        Args:
            evaporation: Hawking evaporation model
            pbh_model: PBH model
        """
        self.evaporation = evaporation or HawkingEvaporation()
        self.pbh_model = pbh_model or PrimordialBlackHoles()

    def remnants_formed_by_z(
        self,
        z: float,
        z_formation: float = 1e10,
    ) -> float:
        """Compute remnant number density at redshift z.

        Remnants form when PBHs complete evaporation. The number of
        remnants equals the number of PBHs that have evaporated.

        Args:
            z: Current redshift
            z_formation: Redshift of PBH formation

        Returns:
            Remnant number density [m⁻³]
        """
        # This is a placeholder for a more detailed calculation
        # Real calculation would integrate over PBH mass spectrum

        M_rem = self.evaporation.M_rem

        # Critical mass: PBHs lighter than this have evaporated by z
        # M_crit³ ≈ 3α × t(z)
        t_z = self._cosmic_time(z)
        M_crit = (3 * self.evaporation.alpha * t_z) ** (1.0 / 3.0)

        # Number of remnants ∝ integral of mass spectrum up to M_crit
        if M_crit < self.pbh_model.M_min:
            return 0.0

        M_upper = min(M_crit, self.pbh_model.M_max)

        # Rough estimate: n_rem ~ f_pbh × ρ_DM / M_rem
        rho_c = 1e-26  # Critical density kg/m³ (rough)
        Omega_DM = 0.25
        rho_DM = Omega_DM * rho_c * (1 + z) ** 3

        n_rem = self.pbh_model.f_pbh * rho_DM / M_rem

        return n_rem

    def _cosmic_time(self, z: float) -> float:
        """Estimate cosmic time at redshift z.

        Rough approximation: t ≈ 2/(3H₀) × (1+z)^(-3/2) for matter domination.
        """
        H0_si = 70 * 1e3 / SI_UNITS.Mpc_in_m  # s⁻¹
        t_H = 1 / H0_si

        return 2.0 / 3.0 * t_H * (1 + z) ** (-1.5)


def compute_remnant_omega(
    params: HRCParameters,
    z_array: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute remnant contribution to Ω_DM.

    Ω_rem(z) = (1/ρ_c) ∫ n_rem(M,z) M dM

    For monochromatic spectrum with M = M_Pl:
    Ω_rem = n_rem × M_Pl / ρ_c

    Args:
        params: HRC parameters
        z_array: Redshift array

    Returns:
        Ω_rem(z) array
    """
    M_rem = SI_UNITS.M_Planck
    rho_c_0 = SI_UNITS.rho_crit_h2 * params.h**2

    # f_rem is the fraction of DM in remnants
    f_rem = params.f_rem
    Omega_c = params.Omega_c

    # Ω_rem = f_rem × Ω_c × (1+z)³
    # But we define f_rem as the fraction today
    Omega_rem = f_rem * Omega_c * np.ones_like(z_array)

    return Omega_rem


def compute_remnant_number_density(
    params: HRCParameters,
    z: float = 0.0,
) -> float:
    """Compute remnant number density today.

    n_rem = Ω_rem × ρ_c / M_rem = f_rem × Ω_c × ρ_c / M_Pl

    Args:
        params: HRC parameters
        z: Redshift

    Returns:
        Number density [m⁻³]
    """
    M_rem = SI_UNITS.M_Planck
    rho_c = SI_UNITS.rho_crit_h2 * params.h**2

    rho_rem = params.f_rem * params.Omega_c * rho_c * (1 + z) ** 3
    n_rem = rho_rem / M_rem

    return n_rem


def remnant_summary(params: HRCParameters) -> dict:
    """Compute summary of remnant properties.

    Args:
        params: HRC parameters

    Returns:
        Dictionary with remnant properties
    """
    M_rem = SI_UNITS.M_Planck
    n_rem = compute_remnant_number_density(params)

    # Mean separation
    d_mean = n_rem ** (-1.0 / 3.0) if n_rem > 0 else np.inf

    # Total remnant mass in observable universe
    # V ~ (c/H0)³ ~ (4000 Mpc)³ ~ 10^80 m³
    V_obs = (4000 * SI_UNITS.Mpc_in_m) ** 3
    N_remnants = n_rem * V_obs
    M_total = N_remnants * M_rem

    return {
        "M_rem_kg": M_rem,
        "M_rem_Mpl": 1.0,
        "n_rem_m3": n_rem,
        "n_rem_Mpc3": n_rem * SI_UNITS.Mpc_in_m**3,
        "mean_separation_m": d_mean,
        "mean_separation_pc": d_mean / 3.086e16,
        "N_observable_universe": N_remnants,
        "f_rem": params.f_rem,
        "Omega_rem": params.f_rem * params.Omega_c,
    }
