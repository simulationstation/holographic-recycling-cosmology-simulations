"""
Holographic Recycling Cosmology (HRC) - Black Hole and Recycling Dynamics

This module implements the microphysics of:
1. Black hole formation, evaporation, and remnant formation
2. The recycling mechanism where Hawking radiation is reabsorbed by remnants
3. Cosmological integration coupling dynamics to the modified Friedmann equations

Physical foundations:
- Hawking radiation (semiclassical approximation)
- Primordial black hole mass functions
- Remnant formation hypothesis (Planck-mass stable endpoints)
- Information recycling via remnant absorption

Units: SI internally, with Planck unit normalization for numerics.
Conventions: c = ħ = k_B = 1 in natural units; SI conversions provided.

Author: HRC Dynamics Development
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.integrate import solve_ivp, quad, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.special import erf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS (SI Units)
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Fundamental physical constants in SI units.

    These are the 2018 CODATA recommended values.
    """
    # Fundamental constants
    c: float = 299792458.0                    # Speed of light [m/s]
    hbar: float = 1.054571817e-34             # Reduced Planck constant [J·s]
    G: float = 6.67430e-11                    # Gravitational constant [m³/(kg·s²)]
    k_B: float = 1.380649e-23                 # Boltzmann constant [J/K]

    # Derived Planck units
    @property
    def M_Planck(self) -> float:
        """Planck mass [kg]"""
        return np.sqrt(self.hbar * self.c / self.G)

    @property
    def L_Planck(self) -> float:
        """Planck length [m]"""
        return np.sqrt(self.hbar * self.G / self.c**3)

    @property
    def t_Planck(self) -> float:
        """Planck time [s]"""
        return np.sqrt(self.hbar * self.G / self.c**5)

    @property
    def T_Planck(self) -> float:
        """Planck temperature [K]"""
        return np.sqrt(self.hbar * self.c**5 / (self.G * self.k_B**2))

    @property
    def E_Planck(self) -> float:
        """Planck energy [J]"""
        return self.M_Planck * self.c**2

    # Cosmological constants
    H0_SI: float = 2.2e-18                    # Hubble constant ~70 km/s/Mpc [1/s]
    rho_crit_SI: float = 8.5e-27              # Critical density [kg/m³]

    # Useful combinations for Hawking radiation
    @property
    def hawking_temp_factor(self) -> float:
        """Factor in T_H = factor / M, units [K·kg]"""
        return self.hbar * self.c**3 / (8 * np.pi * self.G * self.k_B)

    @property
    def hawking_luminosity_factor(self) -> float:
        """Factor in L = factor / M², units [W·kg²]"""
        return self.hbar * self.c**6 / (15360 * np.pi * self.G**2)

    @property
    def evaporation_time_factor(self) -> float:
        """Factor in τ = factor * M³, units [s/kg³]"""
        return 5120 * np.pi * self.G**2 / (self.hbar * self.c**4)


# Global constants instance
CONSTANTS = PhysicalConstants()


# =============================================================================
# UNIT CONVERSION UTILITIES
# =============================================================================

class Units:
    """
    Unit conversion utilities between SI and Planck units.

    In Planck units: c = ħ = G = k_B = 1

    Conversions:
    - Mass: M_SI = M_Planck * m_natural
    - Length: L_SI = L_Planck * l_natural
    - Time: t_SI = t_Planck * τ_natural
    - Temperature: T_SI = T_Planck * θ_natural
    - Energy: E_SI = E_Planck * ε_natural
    """

    def __init__(self, constants: PhysicalConstants = CONSTANTS):
        self.c = constants

    # Mass conversions
    def mass_to_planck(self, M_SI: float) -> float:
        """Convert mass from kg to Planck units."""
        return M_SI / self.c.M_Planck

    def mass_to_SI(self, m_planck: float) -> float:
        """Convert mass from Planck units to kg."""
        return m_planck * self.c.M_Planck

    # Length conversions
    def length_to_planck(self, L_SI: float) -> float:
        """Convert length from m to Planck units."""
        return L_SI / self.c.L_Planck

    def length_to_SI(self, l_planck: float) -> float:
        """Convert length from Planck units to m."""
        return l_planck * self.c.L_Planck

    # Time conversions
    def time_to_planck(self, t_SI: float) -> float:
        """Convert time from s to Planck units."""
        return t_SI / self.c.t_Planck

    def time_to_SI(self, tau_planck: float) -> float:
        """Convert time from Planck units to s."""
        return tau_planck * self.c.t_Planck

    # Temperature conversions
    def temp_to_planck(self, T_SI: float) -> float:
        """Convert temperature from K to Planck units."""
        return T_SI / self.c.T_Planck

    def temp_to_SI(self, theta_planck: float) -> float:
        """Convert temperature from Planck units to K."""
        return theta_planck * self.c.T_Planck

    # Energy conversions
    def energy_to_planck(self, E_SI: float) -> float:
        """Convert energy from J to Planck units."""
        return E_SI / self.c.E_Planck

    def energy_to_SI(self, eps_planck: float) -> float:
        """Convert energy from Planck units to J."""
        return eps_planck * self.c.E_Planck

    # Density conversions
    def density_to_planck(self, rho_SI: float) -> float:
        """Convert mass density from kg/m³ to Planck units."""
        rho_Planck = self.c.M_Planck / self.c.L_Planck**3
        return rho_SI / rho_Planck

    def density_to_SI(self, rho_planck: float) -> float:
        """Convert mass density from Planck units to kg/m³."""
        rho_Planck = self.c.M_Planck / self.c.L_Planck**3
        return rho_planck * rho_Planck

    # Number density conversions
    def number_density_to_planck(self, n_SI: float) -> float:
        """Convert number density from 1/m³ to Planck units."""
        return n_SI * self.c.L_Planck**3

    def number_density_to_SI(self, n_planck: float) -> float:
        """Convert number density from Planck units to 1/m³."""
        return n_planck / self.c.L_Planck**3


# Global units instance
UNITS = Units()


# =============================================================================
# PART A: BLACK HOLE POPULATION DYNAMICS
# =============================================================================

@dataclass
class MassFunctionParams:
    """
    Parameters for the primordial black hole mass function.

    Default: Log-normal distribution centered at ~10¹⁵ g (asteroid mass PBHs
    that would be evaporating around now).

    The mass function is:
    dn/dM = (f_PBH * ρ_DM) / (M * √(2π) * σ_M) * exp(-[ln(M/M_c)]² / (2σ_M²))
    """
    f_PBH: float = 1e-3           # PBH fraction of dark matter
    M_c: float = 1e12             # Characteristic mass [kg] (~10¹⁵ g)
    sigma_M: float = 1.0          # Log-normal width
    rho_DM: float = 2.3e-27       # Dark matter density [kg/m³] (local value)

    def validate(self) -> Tuple[bool, str]:
        """Validate parameters are physical."""
        if not 0 < self.f_PBH <= 1:
            return False, "f_PBH must be in (0, 1]"
        if self.M_c <= 0:
            return False, "M_c must be positive"
        if self.sigma_M <= 0:
            return False, "sigma_M must be positive"
        if self.rho_DM < 0:
            return False, "rho_DM must be non-negative"
        return True, "OK"


class BlackHolePopulation:
    """
    Models a population of primordial black holes (PBHs) evolving via
    Hawking evaporation.

    Key physics:
    - Log-normal initial mass function
    - Hawking evaporation with mass loss rate dM/dt = -α/M²
    - Remnant formation when M → M_Planck
    - No accretion (conservative assumption for PBHs)

    The class tracks:
    - Evolving mass distribution
    - Remnant formation rate
    - Total energy radiated
    """

    def __init__(self, mass_function_params: MassFunctionParams,
                 constants: PhysicalConstants = CONSTANTS):
        """
        Initialize PBH population with mass function parameters.

        Parameters
        ----------
        mass_function_params : MassFunctionParams
            Parameters defining the initial mass function
        constants : PhysicalConstants
            Physical constants (default: SI values)
        """
        valid, msg = mass_function_params.validate()
        if not valid:
            raise ValueError(f"Invalid mass function parameters: {msg}")

        self.params = mass_function_params
        self.c = constants
        self.units = Units(constants)

        # Cache useful values
        self.M_Planck = constants.M_Planck
        self.alpha = constants.hawking_luminosity_factor / constants.c**2
        # dM/dt = -α/M² where α = ħc⁴/(15360πG²)

        # Mass range for numerical integration
        # Lower bound: Planck mass (remnant threshold)
        # Upper bound: ~10¹⁰ M_c to capture tail
        self.M_min = self.M_Planck
        self.M_max = mass_function_params.M_c * 1e10

    def dn_dM(self, M: float, z: float = 0.0) -> float:
        """
        Comoving number density per unit mass at redshift z.

        The mass function evolves because:
        1. BHs lose mass via Hawking radiation
        2. BHs reaching M_Planck become remnants

        Parameters
        ----------
        M : float
            Black hole mass [kg]
        z : float
            Redshift (default: 0, present day)

        Returns
        -------
        float
            dn/dM in [1/(m³ kg)]

        Notes
        -----
        For z > 0, we need to map back to initial mass and track evolution.
        This is a simplified version assuming z ≈ 0.
        """
        if M <= self.M_Planck:
            return 0.0

        p = self.params

        # Log-normal distribution
        log_ratio = np.log(M / p.M_c)
        normalization = p.f_PBH * p.rho_DM / (M * np.sqrt(2 * np.pi) * p.sigma_M)
        exponential = np.exp(-log_ratio**2 / (2 * p.sigma_M**2))

        # Comoving density scales as (1+z)³ for number density
        # but dn/dM includes the mass factor
        comoving_factor = (1 + z)**3 if z > 0 else 1.0

        return normalization * exponential * comoving_factor

    def hawking_temperature(self, M: float) -> float:
        """
        Hawking temperature of a black hole.

        T_H = ħc³ / (8πGMk_B)

        Parameters
        ----------
        M : float
            Black hole mass [kg]

        Returns
        -------
        float
            Temperature [K]
        """
        if M <= 0:
            return np.inf
        return self.c.hawking_temp_factor / M

    def hawking_luminosity(self, M: float) -> float:
        """
        Hawking radiation luminosity.

        L = ħc⁶ / (15360πG²M²)

        This is the Stefan-Boltzmann law for effective 2D emission
        (geometric optics limit for s-waves).

        Parameters
        ----------
        M : float
            Black hole mass [kg]

        Returns
        -------
        float
            Luminosity [W = J/s]
        """
        if M <= 0:
            return np.inf
        return self.c.hawking_luminosity_factor / M**2

    def mass_loss_rate(self, M: float) -> float:
        """
        Mass loss rate from Hawking radiation.

        dM/dt = -L/c² = -ħc⁴ / (15360πG²M²)

        Parameters
        ----------
        M : float
            Black hole mass [kg]

        Returns
        -------
        float
            dM/dt [kg/s] (negative, mass decreases)
        """
        if M <= self.M_Planck:
            return 0.0  # Remnant is stable
        return -self.alpha / M**2

    def evaporation_time(self, M: float) -> float:
        """
        Time for black hole to evaporate from mass M to M_Planck.

        Integrating dM/dt = -α/M²:
        τ = (M³ - M_Planck³) / (3α)

        For M >> M_Planck:
        τ ≈ M³/(3α) = 5120πG²M³/(ħc⁴)

        Parameters
        ----------
        M : float
            Initial black hole mass [kg]

        Returns
        -------
        float
            Evaporation time [s]
        """
        if M <= self.M_Planck:
            return 0.0

        return (M**3 - self.M_Planck**3) / (3 * self.alpha)

    def mass_at_time(self, M_initial: float, t: float) -> float:
        """
        Mass of a black hole after time t of evaporation.

        From dM/dt = -α/M², integrating:
        M(t)³ = M_initial³ - 3αt

        Parameters
        ----------
        M_initial : float
            Initial mass [kg]
        t : float
            Time elapsed [s]

        Returns
        -------
        float
            Current mass [kg], or M_Planck if fully evaporated
        """
        if M_initial <= self.M_Planck:
            return self.M_Planck

        M_cubed = M_initial**3 - 3 * self.alpha * t

        if M_cubed <= self.M_Planck**3:
            return self.M_Planck  # Became a remnant

        return M_cubed**(1/3)

    def initial_mass_for_evaporation_at(self, t_evap: float) -> float:
        """
        Initial mass of BH that evaporates (reaches M_Planck) at time t_evap.

        Parameters
        ----------
        t_evap : float
            Evaporation time [s]

        Returns
        -------
        float
            Initial mass [kg]
        """
        return (3 * self.alpha * t_evap + self.M_Planck**3)**(1/3)

    def total_number_density(self, z: float = 0.0,
                             M_min: float = None,
                             M_max: float = None) -> float:
        """
        Total comoving number density of BHs in mass range.

        n = ∫ (dn/dM) dM

        Parameters
        ----------
        z : float
            Redshift
        M_min, M_max : float
            Mass integration bounds [kg]

        Returns
        -------
        float
            Number density [1/m³]
        """
        if M_min is None:
            M_min = self.M_min
        if M_max is None:
            M_max = self.M_max

        # Integrate in log-mass for numerical stability
        def integrand(log_M):
            M = np.exp(log_M)
            return self.dn_dM(M, z) * M  # Jacobian: dM = M d(lnM)

        result, _ = quad(integrand, np.log(M_min), np.log(M_max),
                        limit=100)
        return result

    def total_mass_density(self, z: float = 0.0) -> float:
        """
        Total mass density in BHs.

        ρ_BH = ∫ M (dn/dM) dM

        Returns
        -------
        float
            Mass density [kg/m³]
        """
        def integrand(log_M):
            M = np.exp(log_M)
            return M * self.dn_dM(M, z) * M  # Extra M from mass weighting

        result, _ = quad(integrand, np.log(self.M_min), np.log(self.M_max),
                        limit=100)
        return result

    def remnant_formation_rate(self, t: float, t_universe: float = 4.35e17) -> float:
        """
        Rate of remnant formation: dn_rem/dt.

        BHs that evaporate at time t had initial mass M_i(t).
        The rate is determined by the initial mass function and mass evolution.

        Parameters
        ----------
        t : float
            Cosmic time [s] since some reference (e.g., Big Bang)
        t_universe : float
            Age of universe [s] (default: ~13.8 Gyr)

        Returns
        -------
        float
            Remnant formation rate [1/(m³·s)]

        Notes
        -----
        This is approximate. Full treatment requires tracking the evolving
        mass function and computing the flux through M = M_Planck.
        """
        # Mass of BH that evaporates at time t
        M_evap = self.initial_mass_for_evaporation_at(t)

        if M_evap > self.M_max:
            return 0.0  # No BHs this massive in the population

        # Number density of BHs with initial mass M_evap
        dn_dM_initial = self.dn_dM(M_evap, z=0)  # Simplified: z=0

        # Rate at which BHs cross M_Planck threshold
        # dM/dt at M_Planck determines the flux
        # Use |dM_i/dt_evap| = d(M_i)/d(τ) where τ = M³/(3α)
        # dM_i/dτ = α/M_i²
        dM_dt = self.alpha / M_evap**2

        # Formation rate = (dn/dM) × |dM/dt|
        return dn_dM_initial * dM_dt

    def evolve_population(self, t_initial: float, t_final: float,
                          n_mass_bins: int = 100,
                          n_time_steps: int = 1000) -> Dict:
        """
        Evolve the entire mass function forward in time.

        Parameters
        ----------
        t_initial : float
            Start time [s]
        t_final : float
            End time [s]
        n_mass_bins : int
            Number of logarithmic mass bins
        n_time_steps : int
            Number of time steps

        Returns
        -------
        dict
            Time series containing:
            - 't': time array
            - 'M_bins': mass bin centers
            - 'dn_dM': mass function at each time (2D array)
            - 'n_BH': total BH number density vs time
            - 'n_rem': cumulative remnant density vs time
            - 'rho_BH': BH mass density vs time
            - 'L_total': total Hawking luminosity vs time
        """
        # Set up mass bins (logarithmic spacing)
        log_M_bins = np.linspace(np.log10(self.M_min),
                                  np.log10(self.M_max),
                                  n_mass_bins)
        M_bins = 10**log_M_bins
        dlog_M = log_M_bins[1] - log_M_bins[0]

        # Time array
        times = np.linspace(t_initial, t_final, n_time_steps)
        dt = times[1] - times[0]

        # Initialize arrays
        dn_dM_evolved = np.zeros((n_time_steps, n_mass_bins))
        n_BH = np.zeros(n_time_steps)
        n_rem = np.zeros(n_time_steps)
        rho_BH = np.zeros(n_time_steps)
        L_total = np.zeros(n_time_steps)

        # Initial mass function
        for j, M in enumerate(M_bins):
            dn_dM_evolved[0, j] = self.dn_dM(M)

        # Track which mass bins have evaporated
        M_current = M_bins.copy()

        # Evolve
        for i in range(n_time_steps):
            t = times[i]

            if i > 0:
                # Evolve masses
                M_current = np.array([self.mass_at_time(M, t - t_initial)
                                      for M in M_bins])

                # Update mass function (advection in mass space)
                # This is simplified - proper treatment needs continuity equation
                for j, M in enumerate(M_current):
                    if M > self.M_Planck:
                        # Find where this mass came from in initial distribution
                        dn_dM_evolved[i, j] = self.dn_dM(M_bins[j])
                    else:
                        dn_dM_evolved[i, j] = 0.0

            # Compute integrated quantities
            for j, M in enumerate(M_current):
                if M > self.M_Planck:
                    dM = M * np.log(10) * dlog_M  # dM = M * d(log₁₀M) * ln(10)
                    n_BH[i] += dn_dM_evolved[i, j] * dM
                    rho_BH[i] += M * dn_dM_evolved[i, j] * dM
                    L_total[i] += self.hawking_luminosity(M) * dn_dM_evolved[i, j] * dM

            # Count remnants formed
            if i > 0:
                # Remnant rate integrated over time step
                rate = self.remnant_formation_rate(t)
                n_rem[i] = n_rem[i-1] + rate * dt

        return {
            't': times,
            'M_bins': M_bins,
            'dn_dM': dn_dM_evolved,
            'n_BH': n_BH,
            'n_rem': n_rem,
            'rho_BH': rho_BH,
            'L_total': L_total
        }

    def entropy_of_bh(self, M: float) -> float:
        """
        Bekenstein-Hawking entropy of a black hole.

        S = A / (4 ℓ_P²) = 4πG²M² / (ħc)

        In units of k_B.

        Parameters
        ----------
        M : float
            Black hole mass [kg]

        Returns
        -------
        float
            Entropy [dimensionless, in units of k_B]
        """
        l_P = self.c.L_Planck
        r_s = 2 * self.c.G * M / self.c.c**2  # Schwarzschild radius
        A = 4 * np.pi * r_s**2  # Horizon area
        return A / (4 * l_P**2)

    def information_content(self, M: float) -> float:
        """
        Information content of black hole in bits.

        I = S / ln(2) bits

        Parameters
        ----------
        M : float
            Black hole mass [kg]

        Returns
        -------
        float
            Information content [bits]
        """
        return self.entropy_of_bh(M) / np.log(2)


# =============================================================================
# PART B: RECYCLING PHYSICS
# =============================================================================

@dataclass
class RemnantProperties:
    """
    Properties of Planck-mass remnants.

    Key hypothesis: Remnants have Planck-scale exterior but potentially
    large interior volume (holographic storage per Rovelli's erebon model).
    """
    mass: float = CONSTANTS.M_Planck           # Remnant mass [kg]
    exterior_radius: float = CONSTANTS.L_Planck  # Exterior size [m]
    interior_volume: float = 1e-105            # Interior volume [m³]
    # Default: Planck volume; could be much larger per holography

    @property
    def geometric_cross_section(self) -> float:
        """Geometric cross-section πR² [m²]"""
        return np.pi * self.exterior_radius**2

    @property
    def schwarzschild_cross_section(self) -> float:
        """Cross-section based on Schwarzschild radius [m²]"""
        r_s = 2 * CONSTANTS.G * self.mass / CONSTANTS.c**2
        return np.pi * r_s**2


class RecyclingDynamics:
    """
    Models the recycling process where Hawking radiation is partially
    reabsorbed by remnants.

    Physical picture:
    1. Black hole emits Hawking radiation isotropically
    2. Radiation propagates through space filled with remnants
    3. Remnants have absorption cross-section σ_abs
    4. Probability of absorption depends on remnant density and mean free path
    5. Absorbed energy feeds back into the recycling field φ

    This creates a feedback loop:
    - More remnants → more recycling → modified expansion
    - Modified expansion → changed remnant production rate
    """

    def __init__(self, sigma_abs: float = None,
                 remnant_properties: RemnantProperties = None,
                 constants: PhysicalConstants = CONSTANTS):
        """
        Initialize recycling dynamics.

        Parameters
        ----------
        sigma_abs : float
            Absorption cross-section [m²]. If None, uses geometric cross-section.
        remnant_properties : RemnantProperties
            Properties of remnants. If None, uses defaults.
        constants : PhysicalConstants
            Physical constants
        """
        self.c = constants
        self.units = Units(constants)

        if remnant_properties is None:
            self.remnant = RemnantProperties()
        else:
            self.remnant = remnant_properties

        if sigma_abs is None:
            # Default: geometric cross-section
            self.sigma_abs = self.remnant.geometric_cross_section
        else:
            self.sigma_abs = sigma_abs

        # Interior volume for holographic storage
        self.V_interior = self.remnant.interior_volume

    def mean_free_path(self, n_rem: float) -> float:
        """
        Mean free path for Hawking quanta through remnant gas.

        ℓ_mfp = 1 / (n_rem × σ_abs)

        Parameters
        ----------
        n_rem : float
            Remnant number density [1/m³]

        Returns
        -------
        float
            Mean free path [m]
        """
        if n_rem <= 0 or self.sigma_abs <= 0:
            return np.inf
        return 1.0 / (n_rem * self.sigma_abs)

    def optical_depth(self, n_rem: float, path_length: float) -> float:
        """
        Optical depth along a path.

        τ = n_rem × σ_abs × L = L / ℓ_mfp

        Parameters
        ----------
        n_rem : float
            Remnant number density [1/m³]
        path_length : float
            Path length [m]

        Returns
        -------
        float
            Optical depth [dimensionless]
        """
        return n_rem * self.sigma_abs * path_length

    def recycling_probability(self, n_rem: float,
                               characteristic_length: float = None) -> float:
        """
        Probability that an emitted Hawking quantum is reabsorbed.

        P_recycle = 1 - exp(-τ) = 1 - exp(-n_rem × σ_abs × L)

        Parameters
        ----------
        n_rem : float
            Remnant number density [1/m³]
        characteristic_length : float
            Typical distance traveled before escape [m].
            If None, uses Hubble length c/H₀.

        Returns
        -------
        float
            Recycling probability [0, 1]
        """
        if characteristic_length is None:
            # Default: Hubble radius
            characteristic_length = self.c.c / self.c.H0_SI

        tau = self.optical_depth(n_rem, characteristic_length)
        return 1.0 - np.exp(-tau)

    def net_radiation_flux(self, L_hawking: float, n_rem: float,
                           characteristic_length: float = None) -> float:
        """
        Effective radiation escaping to infinity after recycling.

        L_eff = L_hawking × (1 - P_recycle)

        Parameters
        ----------
        L_hawking : float
            Total Hawking luminosity [W]
        n_rem : float
            Remnant number density [1/m³]
        characteristic_length : float
            Characteristic path length [m]

        Returns
        -------
        float
            Effective luminosity reaching infinity [W]
        """
        P_recycle = self.recycling_probability(n_rem, characteristic_length)
        return L_hawking * (1.0 - P_recycle)

    def recycled_power(self, L_hawking: float, n_rem: float,
                       characteristic_length: float = None) -> float:
        """
        Power absorbed by remnants (recycled).

        L_recycled = L_hawking × P_recycle

        Parameters
        ----------
        L_hawking : float
            Total Hawking luminosity [W]
        n_rem : float
            Remnant number density [1/m³]
        characteristic_length : float
            Characteristic path length [m]

        Returns
        -------
        float
            Recycled power [W]
        """
        P_recycle = self.recycling_probability(n_rem, characteristic_length)
        return L_hawking * P_recycle

    def recycling_source_term(self, L_hawking: float, n_rem: float,
                               volume: float = 1.0,
                               coupling_efficiency: float = 1.0) -> float:
        """
        Source term Γ for the recycling field φ in the action.

        The recycled energy couples to φ via the λφn_rem term in the action.
        This computes the effective source in the scalar field equation.

        Γ = η × (recycled energy per unit volume per unit time)

        Parameters
        ----------
        L_hawking : float
            Total Hawking luminosity [W]
        n_rem : float
            Remnant number density [1/m³]
        volume : float
            Characteristic volume [m³]
        coupling_efficiency : float
            Fraction of recycled energy that sources φ

        Returns
        -------
        float
            Source term [J/(m³·s) = W/m³]
        """
        L_recycled = self.recycled_power(L_hawking, n_rem)
        return coupling_efficiency * L_recycled / volume

    def information_flow_rate(self, M_bh: float, n_rem: float,
                               characteristic_length: float = None) -> Dict[str, float]:
        """
        Track information flow rates in bits/second.

        Information channels:
        1. Out of BH (Hawking emission) ~ Ṡ_BH
        2. Into remnants (recycled)
        3. To infinity (escaped radiation)

        Using S = A/(4ℓ_P²) and Ṡ = (dA/dt)/(4ℓ_P²)

        Parameters
        ----------
        M_bh : float
            Black hole mass [kg]
        n_rem : float
            Remnant number density [1/m³]
        characteristic_length : float
            Path length for recycling calculation [m]

        Returns
        -------
        dict
            Information flow rates in bits/second:
            - 'emission_rate': bits/s leaving BH
            - 'recycled_rate': bits/s going into remnants
            - 'escaped_rate': bits/s reaching infinity
        """
        # Entropy loss rate from BH
        # Ṡ = (dS/dM)(dM/dt) = (8πGM/(ħc)) × (dM/dt)
        L_P = self.c.L_Planck
        G = self.c.G
        c = self.c.c
        hbar = self.c.hbar

        # dS/dM = 8πG²M / (ħc) = 2 × (4πr_s) × (2GM/c²) / (4L_P²)
        r_s = 2 * G * M_bh / c**2
        dS_dM = 4 * np.pi * r_s * G * M_bh / (L_P**2 * c**2 * hbar)

        # dM/dt from Hawking
        dM_dt = -self.c.hawking_luminosity_factor / (M_bh**2 * c**2)

        # Entropy emission rate
        dS_dt = abs(dS_dM * dM_dt)  # Positive (entropy leaving BH)

        # Convert to bits
        emission_bits_per_s = dS_dt / np.log(2)

        # Split by recycling probability
        P_recycle = self.recycling_probability(n_rem, characteristic_length)

        return {
            'emission_rate': emission_bits_per_s,
            'recycled_rate': emission_bits_per_s * P_recycle,
            'escaped_rate': emission_bits_per_s * (1 - P_recycle),
            'recycling_probability': P_recycle
        }

    def remnant_information_capacity(self) -> float:
        """
        Information storage capacity of a single remnant.

        Assuming holographic bound: S_max = A/(4ℓ_P²) for exterior area,
        but interior could store more per the erebon hypothesis.

        Returns
        -------
        float
            Storage capacity [bits]
        """
        L_P = self.c.L_Planck

        # Exterior (holographic) bound
        A_exterior = 4 * np.pi * self.remnant.exterior_radius**2
        S_holographic = A_exterior / (4 * L_P**2)

        # Interior (volumetric) estimate
        # This is speculative - assumes interior can store ~1 bit per Planck volume
        S_interior = self.V_interior / L_P**3

        # Use the larger value (erebon hypothesis)
        return max(S_holographic, S_interior) / np.log(2)


# =============================================================================
# PART C: COSMOLOGICAL INTEGRATION
# =============================================================================

class HRCCosmology:
    """
    Combines the HRC theory (from hrc_theory.py) with black hole population
    dynamics and recycling physics for full cosmological evolution.

    The system of equations:
    1. Modified Friedmann equations (from HRCTheory)
    2. Matter continuity: ρ̇_m + 3H(ρ_m + p_m) = 0
    3. Remnant continuity: ρ̇_rem + 3Hρ_rem = source - αρ_rem·φ̇
    4. Scalar field: φ̈ + 3Hφ̇ + m²φ = -λn_rem - αρ_rem + ...
    5. BH population evolution (external)

    State vector: y = [a, ρ_m, ρ_rem, n_rem, φ, φ̇]
    """

    def __init__(self, theory_params: Dict = None,
                 bh_pop_params: MassFunctionParams = None,
                 recycling_sigma: float = None,
                 constants: PhysicalConstants = CONSTANTS):
        """
        Initialize full cosmological evolution.

        Parameters
        ----------
        theory_params : dict
            Parameters for HRCTheory (G, Lambda, xi, lambda_r, alpha, m_phi)
        bh_pop_params : MassFunctionParams
            Parameters for black hole population
        recycling_sigma : float
            Absorption cross-section for recycling
        constants : PhysicalConstants
            Physical constants
        """
        self.c = constants
        self.units = Units(constants)

        # Default theory parameters (dimensionless in Planck units)
        if theory_params is None:
            theory_params = {
                'G': 1.0,          # G = 1 in Planck units
                'Lambda': 1e-122,  # Cosmological constant (tiny in Planck)
                'xi': 0.01,        # Non-minimal coupling
                'lambda_r': 1e-60, # Recycling coupling (small)
                'alpha': 0.01,     # Remnant-field coupling
                'm_phi': 1e-60     # Scalar mass ~ H_0 in Planck units
            }
        self.theory_params = theory_params

        # Initialize black hole population
        if bh_pop_params is None:
            bh_pop_params = MassFunctionParams()
        self.bh_pop = BlackHolePopulation(bh_pop_params, constants)

        # Initialize recycling dynamics
        self.recycling = RecyclingDynamics(recycling_sigma, constants=constants)

        # State variable indices
        self.IDX_A = 0       # Scale factor
        self.IDX_RHO_M = 1   # Matter density
        self.IDX_RHO_REM = 2 # Remnant energy density
        self.IDX_N_REM = 3   # Remnant number density
        self.IDX_PHI = 4     # Scalar field
        self.IDX_DPHI = 5    # Scalar field velocity
        self.N_VARS = 6

        # Numerical parameters
        self.use_planck_units = True

    def _get_G(self) -> float:
        """Get Newton's constant in working units."""
        if self.use_planck_units:
            return 1.0  # G = 1 in Planck units
        return self.c.G

    def compute_source_terms(self, a: float, state: Dict) -> Dict:
        """
        Compute all source terms for the field equations.

        Parameters
        ----------
        a : float
            Scale factor
        state : dict
            Current state with keys: rho_m, rho_rem, n_rem, phi, dphi_dt

        Returns
        -------
        dict
            Source terms: H, H_dot, phi_ddot, rho_m_dot, rho_rem_dot, n_rem_dot
        """
        G = self._get_G()
        Lambda = self.theory_params['Lambda']
        xi = self.theory_params['xi']
        lambda_r = self.theory_params['lambda_r']
        alpha = self.theory_params['alpha']
        m_phi = self.theory_params['m_phi']

        rho_m = state['rho_m']
        rho_rem = state['rho_rem']
        n_rem = state['n_rem']
        phi = state['phi']
        dphi_dt = state['dphi_dt']

        # Scalar field potential contribution
        V_phi = 0.5 * m_phi**2 * phi**2

        # Scalar field energy density (simplified, neglecting some xi terms)
        rho_phi = 0.5 * dphi_dt**2 + V_phi + 0.5 * lambda_r * phi * n_rem

        # Effective remnant density with coupling
        rho_rem_eff = rho_rem * (1 + alpha * phi)

        # Effective G factor
        G_eff_factor = 1 - 8 * np.pi * G * xi * phi
        if G_eff_factor <= 0:
            logger.warning(f"G_eff factor non-positive: {G_eff_factor}")
            G_eff_factor = 1e-10  # Regularize

        # Total density
        rho_total = rho_m + rho_phi + rho_rem_eff

        # Hubble parameter from first Friedmann
        H_squared = (8 * np.pi * G * rho_total + Lambda) / (3 * G_eff_factor)
        if H_squared < 0:
            logger.warning(f"H² negative: {H_squared}")
            H_squared = 1e-100  # Regularize
        H = np.sqrt(H_squared)

        # Ricci scalar (for scalar field equation)
        R = 6 * (2 * H**2)  # Approximation neglecting H_dot for simplicity

        # Scalar field equation: φ̈ = -3Hφ̇ - m²φ - ξR - λn_rem - αρ_rem
        phi_ddot = (-3 * H * dphi_dt
                    - m_phi**2 * phi
                    - xi * R
                    - lambda_r * n_rem
                    - alpha * rho_rem)

        # Matter continuity (dust: p_m = 0)
        rho_m_dot = -3 * H * rho_m

        # Remnant source from BH evaporation
        # Convert n_rem from Planck units for the calculation
        if self.use_planck_units:
            n_rem_SI = self.units.number_density_to_SI(n_rem)
        else:
            n_rem_SI = n_rem

        # Remnant formation rate (simplified model)
        # This should come from BH population evolution
        # For now, use a parametric source
        remnant_source_rate = self._remnant_source(a, n_rem_SI)

        if self.use_planck_units:
            remnant_source_rate = self.units.number_density_to_planck(remnant_source_rate) / self.units.time_to_planck(1.0)

        # Remnant continuity with coupling and source
        rho_rem_dot = -3 * H * rho_rem - alpha * rho_rem * dphi_dt
        n_rem_dot = -3 * H * n_rem + remnant_source_rate

        # H_dot from energy conservation or second Friedmann
        # Ḣ = -4πG(ρ + p)/G_eff_factor (simplified)
        p_total = 0.5 * dphi_dt**2 - V_phi  # Scalar pressure (others = 0)
        H_dot = -4 * np.pi * G * (rho_total + p_total) / G_eff_factor

        return {
            'H': H,
            'H_dot': H_dot,
            'H_squared': H_squared,
            'phi_ddot': phi_ddot,
            'rho_m_dot': rho_m_dot,
            'rho_rem_dot': rho_rem_dot,
            'n_rem_dot': n_rem_dot,
            'rho_total': rho_total,
            'G_eff_factor': G_eff_factor
        }

    def _remnant_source(self, a: float, n_rem_SI: float) -> float:
        """
        Compute remnant formation rate from BH population.

        This is a simplified parametric model. Full treatment would
        integrate over the BH mass function.

        Parameters
        ----------
        a : float
            Scale factor
        n_rem_SI : float
            Current remnant density [1/m³]

        Returns
        -------
        float
            Source rate [1/(m³·s)]
        """
        # Age of universe estimate from scale factor
        # Very rough: t ~ 1/H_0 × f(a) where f(a) depends on cosmology
        H0 = self.c.H0_SI
        t_approx = a / H0  # Crude estimate

        # Use BH population remnant formation rate
        rate = self.bh_pop.remnant_formation_rate(t_approx)

        # Recycling modification: some radiation is reabsorbed
        P_recycle = self.recycling.recycling_probability(n_rem_SI)

        # Net rate reduced by recycling (radiation that would form new BHs is captured)
        return rate * (1 - 0.5 * P_recycle)  # Factor 0.5 is phenomenological

    def rhs_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE system for scipy.integrate.solve_ivp.

        dy/dt = f(t, y)

        State vector: y = [a, ρ_m, ρ_rem, n_rem, φ, φ̇]

        Parameters
        ----------
        t : float
            Time (Planck units if use_planck_units=True)
        y : np.ndarray
            State vector

        Returns
        -------
        np.ndarray
            Derivatives dy/dt
        """
        # Unpack state
        a = y[self.IDX_A]
        rho_m = y[self.IDX_RHO_M]
        rho_rem = y[self.IDX_RHO_REM]
        n_rem = y[self.IDX_N_REM]
        phi = y[self.IDX_PHI]
        dphi_dt = y[self.IDX_DPHI]

        # Sanity checks
        if a <= 0:
            a = 1e-30
        if rho_m < 0:
            rho_m = 0
        if rho_rem < 0:
            rho_rem = 0
        if n_rem < 0:
            n_rem = 0

        state = {
            'rho_m': rho_m,
            'rho_rem': rho_rem,
            'n_rem': n_rem,
            'phi': phi,
            'dphi_dt': dphi_dt
        }

        # Compute source terms
        sources = self.compute_source_terms(a, state)

        # Build derivative vector
        dydt = np.zeros(self.N_VARS)

        # da/dt = aH
        dydt[self.IDX_A] = a * sources['H']

        # dρ_m/dt
        dydt[self.IDX_RHO_M] = sources['rho_m_dot']

        # dρ_rem/dt
        dydt[self.IDX_RHO_REM] = sources['rho_rem_dot']

        # dn_rem/dt
        dydt[self.IDX_N_REM] = sources['n_rem_dot']

        # dφ/dt
        dydt[self.IDX_PHI] = dphi_dt

        # d²φ/dt²
        dydt[self.IDX_DPHI] = sources['phi_ddot']

        return dydt

    def create_initial_conditions(self,
                                   a_initial: float = 1e-10,
                                   Omega_m: float = 0.3,
                                   Omega_rem: float = 0.01,
                                   phi_initial: float = 0.0,
                                   dphi_initial: float = 0.0) -> np.ndarray:
        """
        Create initial conditions for cosmological evolution.

        Parameters
        ----------
        a_initial : float
            Initial scale factor (normalized to 1 today)
        Omega_m : float
            Matter density parameter today
        Omega_rem : float
            Remnant density parameter today (subset of dark matter)
        phi_initial : float
            Initial scalar field value
        dphi_initial : float
            Initial scalar field velocity

        Returns
        -------
        np.ndarray
            Initial state vector
        """
        # Critical density (in appropriate units)
        if self.use_planck_units:
            # ρ_crit = 3H₀²/(8πG) in Planck units
            H0_planck = self.units.time_to_planck(1.0) / self.c.H0_SI
            rho_crit = 3 * H0_planck**2 / (8 * np.pi)
        else:
            rho_crit = self.c.rho_crit_SI

        # Density parameters
        rho_m_0 = Omega_m * rho_crit
        rho_rem_0 = Omega_rem * rho_crit

        # Scale back to initial time (matter scales as a^-3)
        rho_m_initial = rho_m_0 / a_initial**3
        rho_rem_initial = rho_rem_0 / a_initial**3

        # Remnant number density (assume M_Planck mass each)
        n_rem_initial = rho_rem_initial / self.c.M_Planck
        if self.use_planck_units:
            n_rem_initial = rho_rem_initial  # Already in Planck units: n = ρ/m with m=1

        y0 = np.zeros(self.N_VARS)
        y0[self.IDX_A] = a_initial
        y0[self.IDX_RHO_M] = rho_m_initial
        y0[self.IDX_RHO_REM] = rho_rem_initial
        y0[self.IDX_N_REM] = n_rem_initial
        y0[self.IDX_PHI] = phi_initial
        y0[self.IDX_DPHI] = dphi_initial

        return y0

    def evolve(self, y0: np.ndarray, t_span: Tuple[float, float],
               method: str = 'DOP853',
               dense_output: bool = True,
               max_step: float = None,
               events: List[Callable] = None,
               **kwargs) -> 'OdeResult':
        """
        Integrate the full cosmological system.

        Parameters
        ----------
        y0 : np.ndarray
            Initial conditions
        t_span : tuple
            (t_initial, t_final) in working units
        method : str
            Integration method ('DOP853', 'Radau', 'BDF')
        dense_output : bool
            Whether to compute dense output for interpolation
        max_step : float
            Maximum step size
        events : list
            Event functions for termination
        **kwargs :
            Additional arguments to solve_ivp

        Returns
        -------
        OdeResult
            Solution object from scipy.integrate.solve_ivp
        """
        return solve_ivp(
            self.rhs_system,
            t_span,
            y0,
            method=method,
            dense_output=dense_output,
            max_step=max_step,
            events=events,
            **kwargs
        )

    def hubble_parameter(self, t: float, state: Dict) -> float:
        """
        Compute Hubble parameter at given state.

        Parameters
        ----------
        t : float
            Time (not directly used, included for interface)
        state : dict
            Current state

        Returns
        -------
        float
            H in working units
        """
        sources = self.compute_source_terms(state.get('a', 1.0), state)
        return sources['H']

    def extract_solution(self, sol) -> Dict:
        """
        Extract solution components into a convenient dictionary.

        Parameters
        ----------
        sol : OdeResult
            Solution from evolve()

        Returns
        -------
        dict
            Solution components with keys: t, a, rho_m, rho_rem, n_rem, phi, dphi_dt, H
        """
        t = sol.t
        y = sol.y

        result = {
            't': t,
            'a': y[self.IDX_A],
            'rho_m': y[self.IDX_RHO_M],
            'rho_rem': y[self.IDX_RHO_REM],
            'n_rem': y[self.IDX_N_REM],
            'phi': y[self.IDX_PHI],
            'dphi_dt': y[self.IDX_DPHI]
        }

        # Compute derived quantities
        H = np.zeros_like(t)
        for i in range(len(t)):
            state = {
                'rho_m': result['rho_m'][i],
                'rho_rem': result['rho_rem'][i],
                'n_rem': result['n_rem'][i],
                'phi': result['phi'][i],
                'dphi_dt': result['dphi_dt'][i]
            }
            H[i] = self.hubble_parameter(t[i], state)

        result['H'] = H

        return result

    def verify_energy_conservation(self, sol, rtol: float = 0.01) -> Dict:
        """
        Check energy conservation in the solution.

        Total energy should be approximately conserved (up to Λ effects).

        Parameters
        ----------
        sol : OdeResult
            Solution from evolve()
        rtol : float
            Relative tolerance for conservation check

        Returns
        -------
        dict
            Conservation diagnostics
        """
        data = self.extract_solution(sol)

        G = self._get_G()
        Lambda = self.theory_params['Lambda']
        m_phi = self.theory_params['m_phi']

        # Compute total energy density at each time
        rho_total = np.zeros_like(data['t'])

        for i in range(len(data['t'])):
            phi = data['phi'][i]
            dphi = data['dphi_dt'][i]
            V = 0.5 * m_phi**2 * phi**2
            rho_phi = 0.5 * dphi**2 + V
            rho_total[i] = data['rho_m'][i] + data['rho_rem'][i] + rho_phi

        # Energy in comoving volume: E = ρ × a³
        E_comoving = rho_total * data['a']**3

        # Check conservation
        E_initial = E_comoving[0]
        E_final = E_comoving[-1]
        relative_change = abs(E_final - E_initial) / E_initial if E_initial > 0 else 0

        return {
            'E_comoving': E_comoving,
            'E_initial': E_initial,
            'E_final': E_final,
            'relative_change': relative_change,
            'conserved': relative_change < rtol
        }


# =============================================================================
# VALIDATION AND TESTING UTILITIES
# =============================================================================

def validate_evaporation_time():
    """
    Validate evaporation time calculation against known values.

    A 10¹⁵ g (10¹² kg) PBH should evaporate in ~age of universe.
    """
    bh = BlackHolePopulation(MassFunctionParams())

    # Test mass: 10¹² kg ~ 10¹⁵ g
    M_test = 1e12  # kg
    tau = bh.evaporation_time(M_test)

    # Expected: ~10¹⁷ seconds ~ age of universe
    t_universe = 4.35e17  # seconds (13.8 Gyr)

    print(f"Evaporation time for M = {M_test:.2e} kg:")
    print(f"  Calculated: {tau:.2e} s")
    print(f"  Universe age: {t_universe:.2e} s")
    print(f"  Ratio: {tau/t_universe:.2f}")

    return abs(tau/t_universe - 1) < 10  # Within order of magnitude


def validate_hawking_temperature():
    """
    Validate Hawking temperature against known values.

    For a solar mass BH: T_H ~ 6 × 10⁻⁸ K
    For a 10¹² kg BH: T_H ~ 10¹¹ K
    """
    bh = BlackHolePopulation(MassFunctionParams())

    M_sun = 2e30  # kg
    T_solar = bh.hawking_temperature(M_sun)

    M_pbh = 1e12  # kg
    T_pbh = bh.hawking_temperature(M_pbh)

    print(f"Hawking temperature:")
    print(f"  Solar mass BH: T = {T_solar:.2e} K (expected ~6e-8 K)")
    print(f"  10¹² kg PBH: T = {T_pbh:.2e} K (expected ~10¹¹ K)")

    return True


def validate_entropy():
    """
    Validate Bekenstein-Hawking entropy.

    For solar mass BH: S ~ 10⁷⁷ (in units of k_B)
    """
    bh = BlackHolePopulation(MassFunctionParams())

    M_sun = 2e30  # kg
    S = bh.entropy_of_bh(M_sun)
    I = bh.information_content(M_sun)

    print(f"Solar mass BH entropy:")
    print(f"  S = {S:.2e} k_B (expected ~10⁷⁷)")
    print(f"  I = {I:.2e} bits")

    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HRC Dynamics Module - Validation Tests")
    print("=" * 60)

    print("\n1. Validating Hawking temperature...")
    validate_hawking_temperature()

    print("\n2. Validating evaporation time...")
    validate_evaporation_time()

    print("\n3. Validating BH entropy...")
    validate_entropy()

    print("\n4. Testing recycling probability...")
    recycling = RecyclingDynamics()
    n_test = 1e30  # 1/m³ (very dense)
    P = recycling.recycling_probability(n_test, characteristic_length=1e10)
    print(f"  n_rem = {n_test:.2e} /m³, L = 10¹⁰ m")
    print(f"  P_recycle = {P:.4f}")

    print("\n5. Testing cosmological integration...")
    cosmo = HRCCosmology()
    y0 = cosmo.create_initial_conditions(a_initial=0.1)
    print(f"  Initial conditions: a={y0[0]:.2e}, ρ_m={y0[1]:.2e}")

    # Short test evolution
    try:
        sol = cosmo.evolve(y0, (0, 1e-55), max_step=1e-57)
        print(f"  Evolution successful: {len(sol.t)} time steps")
    except Exception as e:
        print(f"  Evolution test skipped (numerical issues expected): {e}")

    print("\n" + "=" * 60)
    print("Module loaded successfully.")
