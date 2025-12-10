"""
Early Black-Hole Dominated Fertility Cosmology (BHFC) - GR-Consistent Implementation.

This module implements a cosmological model where:
- A fraction of energy density in the early universe collapses into primordial black holes
- The BH component (rho_BH) behaves like cold matter (w ~ 0)
- Optionally, a fraction of BH mass evaporates into radiation/dark sector
- A "fertility" parameter A_eff controls collapse efficiency and timing

Everything remains GR-consistent: all energy that gravitates lives in T_mu_nu.

Physical interpretation:
- At z > z_form: standard radiation + matter + Lambda
- Around z ~ z_form: fraction f_BH_init of matter collapses into BHs
- rho_BH(a) redshifts like matter (a^-3)
- Optionally at z < z_evap: fraction f_evap of BH energy converts to radiation
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d


@dataclass
class BHFCRealParameters:
    """Parameters for Early Black-Hole Dominated Fertility Cosmology.

    Attributes:
        f_BH_init: Fraction of total matter-like energy in BHs at formation (0-1)
        z_form: Redshift at which BHs effectively form (10^3 - 10^9)
        z_evap: Redshift at which fraction evaporates (None = no evaporation)
        A_eff: Fertility parameter controlling collapse sharpness (0.1-10)
        f_evap: Fraction of BH mass that evaporates into radiation (0-1)

        # Standard cosmological parameters
        H0: Hubble constant in km/s/Mpc
        Omega_m0: Total matter density parameter today
        Omega_b0: Baryon density parameter today
        Omega_r0: Radiation density parameter today
        Omega_Lambda0: Dark energy density parameter today (computed)
    """
    # BHFC-specific parameters
    f_BH_init: float = 0.1      # 10% of matter into BHs
    z_form: float = 1e5         # Formation at z = 10^5
    z_evap: Optional[float] = None  # No evaporation by default
    A_eff: float = 1.0          # Formation sharpness
    f_evap: float = 0.0         # No evaporation

    # Standard cosmology (Planck 2018 baseline)
    H0: float = 67.4            # km/s/Mpc
    Omega_m0: float = 0.315     # Total matter
    Omega_b0: float = 0.0493    # Baryons
    Omega_r0: float = 9.24e-5   # Radiation (photons + neutrinos)

    @property
    def Omega_cdm0(self) -> float:
        """CDM density parameter."""
        return self.Omega_m0 - self.Omega_b0

    @property
    def Omega_Lambda0(self) -> float:
        """Dark energy density (assuming flatness)."""
        return 1.0 - self.Omega_m0 - self.Omega_r0

    @property
    def a_form(self) -> float:
        """Scale factor at BH formation."""
        return 1.0 / (1.0 + self.z_form)

    @property
    def a_evap(self) -> Optional[float]:
        """Scale factor at BH evaporation (if any)."""
        if self.z_evap is None:
            return None
        return 1.0 / (1.0 + self.z_evap)


def bh_formation_window(a: float, params: BHFCRealParameters) -> float:
    """Compute the BH formation window function.

    This is a smooth transition function that goes from 0 to 1 around a_form.
    Higher A_eff means sharper (more sudden) formation.

    The function is:
        W(a) = 1 / (1 + exp(-A_eff * (ln(a) - ln(a_form))))
             = 1 / (1 + (a_form/a)^A_eff)

    At a << a_form: W ~ 0 (no BHs yet)
    At a = a_form:  W = 0.5
    At a >> a_form: W ~ 1 (full BH formation)

    Args:
        a: Scale factor
        params: BHFC parameters

    Returns:
        Formation window value in [0, 1]
    """
    if a <= 0:
        return 0.0

    a_form = params.a_form
    A_eff = params.A_eff

    # Sigmoid in ln(a)
    x = A_eff * (np.log(a) - np.log(a_form))

    # Numerically stable sigmoid
    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-x))


def bh_evaporation_window(a: float, params: BHFCRealParameters) -> float:
    """Compute the BH evaporation window function.

    Similar to formation window but for evaporation at later times.

    At a << a_evap: W_evap ~ 0 (no evaporation yet)
    At a = a_evap:  W_evap = 0.5
    At a >> a_evap: W_evap ~ 1 (full evaporation)

    Args:
        a: Scale factor
        params: BHFC parameters

    Returns:
        Evaporation window value in [0, 1], or 0 if no evaporation
    """
    if params.z_evap is None or params.f_evap <= 0:
        return 0.0

    if a <= 0:
        return 0.0

    a_evap = params.a_evap
    A_eff = params.A_eff  # Use same sharpness

    x = A_eff * (np.log(a) - np.log(a_evap))

    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-x))


def rho_BH(a: float, params: BHFCRealParameters) -> float:
    """Compute BH energy density at scale factor a.

    The BH component:
    1. Forms from a fraction f_BH_init of the CDM at z_form
    2. Redshifts like matter (a^-3)
    3. Optionally loses fraction f_evap to radiation at z_evap

    Energy conservation:
    - At formation: rho_BH(a_form) = f_BH_init * rho_cdm(a_form)
    - After formation: rho_BH(a) = rho_BH(a_form) * (a_form/a)^3
    - After evaporation: rho_BH reduced by (1 - f_evap)

    Args:
        a: Scale factor
        params: BHFC parameters

    Returns:
        BH energy density in units of 3*H0^2 (same as Omega_i)
    """
    if a <= 0:
        return 0.0

    # Formation window
    W_form = bh_formation_window(a, params)

    # CDM density at current a (what would exist without BH formation)
    rho_cdm_a = params.Omega_cdm0 * a**(-3)

    # BH density from converted CDM
    # W_form smoothly transitions: before formation, rho_BH ~ 0
    rho_BH_base = params.f_BH_init * rho_cdm_a * W_form

    # Evaporation reduction
    if params.z_evap is not None and params.f_evap > 0:
        W_evap = bh_evaporation_window(a, params)
        # After evaporation, BH mass is reduced by f_evap
        evap_factor = 1.0 - params.f_evap * W_evap
        rho_BH_base *= evap_factor

    return rho_BH_base


def rho_extra_rad(a: float, params: BHFCRealParameters) -> float:
    """Compute extra radiation density from BH evaporation.

    When BHs evaporate, the energy goes into radiation (or dark radiation).
    This component redshifts as a^-4 after being injected.

    The calculation tracks the cumulative energy injected at each scale factor
    and evolves it as radiation.

    For simplicity, we model this as:
    - Energy injected ~ f_evap * f_BH_init * rho_cdm(a_evap) at a ~ a_evap
    - This then redshifts as (a_evap/a)^4

    Args:
        a: Scale factor
        params: BHFC parameters

    Returns:
        Extra radiation density in units of 3*H0^2
    """
    if params.z_evap is None or params.f_evap <= 0:
        return 0.0

    if a <= 0:
        return 0.0

    a_evap = params.a_evap

    # Evaporation window (how much has evaporated)
    W_evap = bh_evaporation_window(a, params)

    # Energy that was in BHs at evaporation epoch
    # At a_evap: rho_BH would be f_BH_init * rho_cdm(a_evap) (before evap)
    rho_cdm_evap = params.Omega_cdm0 * a_evap**(-3)
    rho_BH_evap = params.f_BH_init * rho_cdm_evap

    # Energy released into radiation
    energy_released = params.f_evap * rho_BH_evap

    # This energy redshifts as radiation from a_evap onwards
    # But we need to account for when the evaporation actually happens
    # Using a smooth model: radiation density builds up as evaporation proceeds

    if a > a_evap:
        # After evaporation: radiation redshifts as (a_evap/a)^4
        rho_rad = energy_released * (a_evap / a)**4 * W_evap
    else:
        # Before/during evaporation: radiation just starting to appear
        # Scale by W_evap and account for some redshift
        rho_rad = energy_released * W_evap * (a / a)**(-4) * (a_evap / max(a, 1e-10))**(-4)
        # Actually simpler: radiation appears as evaporation happens, then redshifts
        rho_rad = energy_released * W_evap

    return rho_rad


def rho_cdm_residual(a: float, params: BHFCRealParameters) -> float:
    """Compute residual CDM density after BH formation.

    The CDM that remains after a fraction is converted to BHs:
        rho_cdm_res = rho_cdm_standard * (1 - f_BH_init * W_form)

    This ensures energy conservation: at any a,
        rho_cdm_res + rho_BH = rho_cdm_standard (ignoring evaporation)

    Args:
        a: Scale factor
        params: BHFC parameters

    Returns:
        Residual CDM density in units of 3*H0^2
    """
    if a <= 0:
        return 0.0

    # Standard CDM density
    rho_cdm_std = params.Omega_cdm0 * a**(-3)

    # Formation window
    W_form = bh_formation_window(a, params)

    # Residual = standard - converted to BH
    return rho_cdm_std * (1.0 - params.f_BH_init * W_form)


class BHFCBackgroundCosmology:
    """Background cosmology solver for BHFC model.

    Integrates the Friedmann equation including:
    - Standard radiation (photons + neutrinos)
    - Baryons
    - Residual CDM (after BH formation)
    - Primordial black holes (BH component)
    - Extra radiation from BH evaporation (optional)
    - Cosmological constant

    All energy is conserved and GR-consistent.
    """

    def __init__(self, params: BHFCRealParameters):
        """Initialize BHFC background solver.

        Args:
            params: BHFC parameters
        """
        self.params = params
        self.H0 = params.H0

        # Speed of light
        self.c = 299792.458  # km/s

        # H0 in inverse Mpc
        self.H0_invMpc = self.H0 / self.c  # 1/Mpc

    def rho_total(self, a: float) -> float:
        """Compute total energy density at scale factor a.

        Total density = radiation + baryons + residual_CDM + BH + extra_rad + Lambda

        Args:
            a: Scale factor

        Returns:
            Total density in units of 3*H0^2
        """
        p = self.params

        # Standard radiation
        rho_r = p.Omega_r0 * a**(-4)

        # Baryons (always standard)
        rho_b = p.Omega_b0 * a**(-3)

        # Residual CDM
        rho_cdm_res = rho_cdm_residual(a, p)

        # Black hole component
        rho_bh = rho_BH(a, p)

        # Extra radiation from evaporation
        rho_extra = rho_extra_rad(a, p)

        # Cosmological constant
        rho_L = p.Omega_Lambda0

        return rho_r + rho_b + rho_cdm_res + rho_bh + rho_extra + rho_L

    def H(self, a: float) -> float:
        """Compute Hubble parameter at scale factor a.

        H(a) = H0 * sqrt(rho_total / rho_crit0)
             = H0 * sqrt(rho_total)   [since rho_crit0 = 1 in our units]

        Args:
            a: Scale factor

        Returns:
            H in km/s/Mpc
        """
        rho = self.rho_total(a)
        if rho <= 0:
            return 0.0
        return self.H0 * np.sqrt(rho)

    def H_of_z(self, z: float) -> float:
        """Compute Hubble parameter at redshift z.

        Args:
            z: Redshift

        Returns:
            H(z) in km/s/Mpc
        """
        a = 1.0 / (1.0 + z)
        return self.H(a)

    def E(self, z: float) -> float:
        """Compute E(z) = H(z)/H0.

        Args:
            z: Redshift

        Returns:
            E(z) = H(z)/H0
        """
        return self.H_of_z(z) / self.H0

    def comoving_distance(self, z: float, z_ref: float = 0.0) -> float:
        """Compute comoving distance from z_ref to z.

        chi(z) = c * integral_0^z dz' / H(z')

        Args:
            z: Target redshift
            z_ref: Reference redshift (default 0)

        Returns:
            Comoving distance in Mpc
        """
        if z <= z_ref:
            return 0.0

        def integrand(zp):
            return 1.0 / self.H_of_z(zp)

        result, _ = quad(integrand, z_ref, z, limit=200)
        return self.c * result

    def angular_diameter_distance(self, z: float) -> float:
        """Compute angular diameter distance D_A(z).

        D_A(z) = chi(z) / (1 + z)    [flat universe]

        Args:
            z: Redshift

        Returns:
            D_A in Mpc
        """
        chi = self.comoving_distance(z)
        return chi / (1.0 + z)

    def luminosity_distance(self, z: float) -> float:
        """Compute luminosity distance D_L(z).

        D_L(z) = (1 + z) * chi(z)    [flat universe]

        Args:
            z: Redshift

        Returns:
            D_L in Mpc
        """
        chi = self.comoving_distance(z)
        return (1.0 + z) * chi

    def D_V(self, z: float) -> float:
        """Compute BAO volume-averaged distance D_V(z).

        D_V(z) = [z * D_A(z)^2 * c / H(z)]^(1/3)

        Args:
            z: Redshift

        Returns:
            D_V in Mpc
        """
        D_A = self.angular_diameter_distance(z)
        Hz = self.H_of_z(z)
        return (z * D_A**2 * self.c / Hz)**(1.0/3.0)

    def sound_horizon(self, z_drag: float = 1059.94) -> float:
        """Compute the sound horizon at drag epoch.

        r_s = integral_z_drag^inf c_s(z) / H(z) dz

        Using the approximation c_s ~ c/sqrt(3(1 + R_b))
        where R_b = 3*rho_b / (4*rho_gamma)

        Args:
            z_drag: Redshift at baryon drag epoch

        Returns:
            Sound horizon r_s in Mpc
        """
        p = self.params

        # Photon-to-baryon ratio factor
        # R_b = 3*Omega_b / (4*Omega_gamma) * (1+z)^{-1}
        # Omega_gamma ~ 2.47e-5 * h^{-2}
        h = self.H0 / 100.0
        Omega_gamma = 2.47e-5 / h**2

        def integrand(z):
            a = 1.0 / (1.0 + z)
            # Baryon-to-photon density ratio
            R_b = 3.0 * p.Omega_b0 / (4.0 * Omega_gamma) * a
            # Sound speed
            c_s = self.c / np.sqrt(3.0 * (1.0 + R_b))
            return c_s / self.H_of_z(z)

        # Integrate from z_drag to very high z (approximating infinity)
        z_max = min(self.params.z_form * 10, 1e9)  # Don't go beyond reason
        result, _ = quad(integrand, z_drag, z_max, limit=500)
        return result

    def theta_s(self, z_star: float = 1089.92, z_drag: float = 1059.94) -> float:
        """Compute the CMB acoustic scale theta_s.

        theta_s = r_s(z_drag) / D_M(z_star)

        where D_M is the comoving (transverse) distance, equal to chi
        for a flat universe.

        Note: This is the standard definition used by Planck and other
        CMB experiments. The comoving distance D_M appears because both
        r_s and D_M are comoving quantities.

        Args:
            z_star: Redshift of last scattering
            z_drag: Redshift at baryon drag epoch

        Returns:
            Acoustic scale theta_s (radians)
        """
        r_s = self.sound_horizon(z_drag)
        D_M = self.comoving_distance(z_star)  # D_M = chi for flat universe
        return r_s / D_M

    def compute_distances(self) -> Dict[str, float]:
        """Compute key cosmological distances.

        Returns:
            Dictionary with r_s, D_M(z*), theta_s, D_V at BAO redshifts
        """
        z_star = 1089.92  # Last scattering
        z_drag = 1059.94  # Baryon drag

        r_s = self.sound_horizon(z_drag)
        D_M_star = self.comoving_distance(z_star)  # Comoving distance
        theta_s = r_s / D_M_star  # Use comoving distance for theta_s

        # BAO measurements at standard redshifts
        z_bao = [0.38, 0.51, 0.61, 2.33]
        D_V_bao = {f"D_V_z{z:.2f}": self.D_V(z) for z in z_bao}

        # D_L for SN at sample redshifts
        z_sn = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
        D_L_sn = {f"D_L_z{z:.2f}": self.luminosity_distance(z) for z in z_sn}

        return {
            "r_s": r_s,
            "D_M_star": D_M_star,
            "theta_s": theta_s,
            **D_V_bao,
            **D_L_sn,
        }

    def H_ratio_vs_LCDM(self, z_array: NDArray) -> NDArray:
        """Compute H_BHFC(z) / H_LCDM(z) ratio.

        Args:
            z_array: Array of redshifts

        Returns:
            Array of H ratios
        """
        p = self.params

        def H_LCDM(z):
            a = 1.0 / (1.0 + z)
            rho = (p.Omega_r0 * a**(-4) +
                   p.Omega_m0 * a**(-3) +
                   p.Omega_Lambda0)
            return p.H0 * np.sqrt(rho)

        return np.array([self.H_of_z(z) / H_LCDM(z) for z in z_array])

    def BH_fraction(self, z_array: NDArray) -> NDArray:
        """Compute rho_BH / rho_total as function of redshift.

        Args:
            z_array: Array of redshifts

        Returns:
            Array of BH fraction values
        """
        fractions = []
        for z in z_array:
            a = 1.0 / (1.0 + z)
            rho_bh = rho_BH(a, self.params)
            rho_tot = self.rho_total(a)
            f = rho_bh / rho_tot if rho_tot > 0 else 0.0
            fractions.append(f)
        return np.array(fractions)


def compute_H0_early_late(cosmo: BHFCBackgroundCosmology) -> Dict[str, float]:
    """Compute early-universe vs late-universe H0 inference.

    The idea: if the expansion history differs from LCDM, then:
    - "Early" H0 inferred from CMB distance measures
    - "Late" H0 inferred from local distances (D_L at low z)

    may differ.

    Args:
        cosmo: BHFC cosmology object

    Returns:
        Dictionary with H0_Early, H0_Late, Delta_H0
    """
    # Reference LCDM values (Planck 2018)
    theta_s_Planck = 1.04109e-2  # radians
    r_s_Planck = 147.09  # Mpc (sound horizon at drag)
    D_M_star_Planck = 14134.0  # Mpc (comoving distance to z*)
    H0_Planck = 67.4  # km/s/Mpc

    # Compute BHFC distances
    theta_s_BHFC = cosmo.theta_s()
    r_s_BHFC = cosmo.sound_horizon()
    D_M_star_BHFC = cosmo.comoving_distance(1089.92)  # Comoving distance

    # Distance ratios
    r_s_ratio = r_s_BHFC / r_s_Planck
    D_M_ratio = D_M_star_BHFC / D_M_star_Planck

    # The CMB measures theta_s = r_s / D_M precisely.
    # If our model has different theta_s at the same input H0,
    # Planck would infer a different H0 when fitting LCDM.
    #
    # Key insight: Both r_s and D_M scale inversely with H0.
    # So theta_s = r_s/D_M is H0-independent to first order.
    # Any change in theta_s comes from MODIFIED EXPANSION HISTORY,
    # not from H0 rescaling.
    #
    # For H0_Early: We use the D_M comparison.
    # If BHFC has smaller D_M at same H0, Planck (assuming LCDM) would infer larger H0.
    H0_Early = H0_Planck / D_M_ratio

    # H0_Late: From local distance ladder at low z
    # At low z, D_L ~ c*z/H0 to first order
    z_local = 0.1
    D_L_BHFC = cosmo.luminosity_distance(z_local)

    # LCDM reference D_L
    p = cosmo.params
    def H_LCDM(z):
        a = 1.0 / (1.0 + z)
        rho = p.Omega_r0 * a**(-4) + p.Omega_m0 * a**(-3) + p.Omega_Lambda0
        return p.H0 * np.sqrt(rho)

    def integrand_LCDM(z):
        return 1.0 / H_LCDM(z)

    chi_LCDM, _ = quad(integrand_LCDM, 0, z_local, limit=100)
    chi_LCDM *= cosmo.c
    D_L_LCDM = (1 + z_local) * chi_LCDM

    # If BHFC gives larger D_L at same H0, the inferred H0 is smaller
    D_L_ratio = D_L_BHFC / D_L_LCDM
    H0_Late = H0_Planck / D_L_ratio

    # The "tension" would be the difference
    Delta_H0 = H0_Late - H0_Early

    return {
        "H0_Early": H0_Early,
        "H0_Late": H0_Late,
        "Delta_H0": Delta_H0,
        "theta_s_BHFC": theta_s_BHFC,
        "theta_s_Planck": theta_s_Planck,
        "theta_s_ratio": theta_s_BHFC / theta_s_Planck,
        "r_s_ratio": r_s_ratio,
        "D_M_ratio": D_M_ratio,
        "D_L_ratio": D_L_ratio,
    }


def check_constraints(cosmo: BHFCBackgroundCosmology) -> Dict[str, Any]:
    """Check observational constraints on BHFC model.

    Constraints:
    - |Delta theta_s| < 0.3% (CMB acoustic scale)
    - |Delta D_M(z*)| < 0.5% (CMB comoving distance)
    - |Delta D_V| < 2% at BAO redshifts
    - |Delta D_L| < 2% over SN redshift range

    Args:
        cosmo: BHFC cosmology object

    Returns:
        Dictionary with constraint checks and pass/fail status
    """
    p = cosmo.params

    # Reference LCDM cosmology
    def compute_LCDM_distances():
        """Compute LCDM reference distances."""
        def H_LCDM(z):
            a = 1.0 / (1.0 + z)
            rho = p.Omega_r0 * a**(-4) + p.Omega_m0 * a**(-3) + p.Omega_Lambda0
            return p.H0 * np.sqrt(rho)

        def chi_LCDM(z):
            result, _ = quad(lambda zp: 1.0/H_LCDM(zp), 0, z, limit=200)
            return cosmo.c * result

        def D_A_LCDM(z):
            return chi_LCDM(z) / (1 + z)

        def D_L_LCDM(z):
            return (1 + z) * chi_LCDM(z)

        def D_V_LCDM(z):
            D_A = D_A_LCDM(z)
            Hz = H_LCDM(z)
            return (z * D_A**2 * cosmo.c / Hz)**(1./3.)

        # Sound horizon
        h = p.H0 / 100.0
        Omega_gamma = 2.47e-5 / h**2
        z_drag = 1059.94
        z_max = 1e7

        def rs_integrand(z):
            a = 1.0 / (1.0 + z)
            R_b = 3.0 * p.Omega_b0 / (4.0 * Omega_gamma) * a
            c_s = cosmo.c / np.sqrt(3.0 * (1.0 + R_b))
            return c_s / H_LCDM(z)

        r_s_LCDM, _ = quad(rs_integrand, z_drag, z_max, limit=500)

        z_star = 1089.92
        D_M_star = chi_LCDM(z_star)  # Comoving distance
        theta_s_LCDM = r_s_LCDM / D_M_star  # Use comoving distance

        return {
            "r_s": r_s_LCDM,
            "D_M_star": D_M_star,
            "theta_s": theta_s_LCDM,
            "D_V": {z: D_V_LCDM(z) for z in [0.38, 0.51, 0.61, 2.33]},
            "D_L": {z: D_L_LCDM(z) for z in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]},
        }

    # Get LCDM reference
    lcdm = compute_LCDM_distances()

    # Get BHFC distances
    bhfc_dist = cosmo.compute_distances()

    # Constraint checks
    theta_s_BHFC = bhfc_dist["theta_s"]
    theta_s_LCDM = lcdm["theta_s"]
    delta_theta_s = abs(theta_s_BHFC - theta_s_LCDM) / theta_s_LCDM * 100
    theta_s_pass = delta_theta_s < 0.3

    D_M_star_BHFC = bhfc_dist["D_M_star"]
    D_M_star_LCDM = lcdm["D_M_star"]
    delta_D_M = abs(D_M_star_BHFC - D_M_star_LCDM) / D_M_star_LCDM * 100
    D_M_pass = delta_D_M < 0.5

    # BAO constraints
    bao_pass = True
    bao_deltas = {}
    for z in [0.38, 0.51, 0.61, 2.33]:
        D_V_BHFC = bhfc_dist[f"D_V_z{z:.2f}"]
        D_V_LCDM = lcdm["D_V"][z]
        delta = abs(D_V_BHFC - D_V_LCDM) / D_V_LCDM * 100
        bao_deltas[z] = delta
        if delta >= 2.0:
            bao_pass = False

    # SN constraints
    sn_pass = True
    sn_deltas = {}
    for z in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]:
        D_L_BHFC = bhfc_dist[f"D_L_z{z:.2f}"]
        D_L_LCDM = lcdm["D_L"][z]
        delta = abs(D_L_BHFC - D_L_LCDM) / D_L_LCDM * 100
        sn_deltas[z] = delta
        if delta >= 2.0:
            sn_pass = False

    all_pass = theta_s_pass and D_M_pass and bao_pass and sn_pass

    return {
        "passes_all": all_pass,
        "theta_s_pass": theta_s_pass,
        "D_M_pass": D_M_pass,
        "bao_pass": bao_pass,
        "sn_pass": sn_pass,
        "delta_theta_s_percent": delta_theta_s,
        "delta_D_M_percent": delta_D_M,
        "bao_deltas_percent": bao_deltas,
        "sn_deltas_percent": sn_deltas,
        "theta_s_BHFC": theta_s_BHFC,
        "theta_s_LCDM": theta_s_LCDM,
        "D_M_star_BHFC": D_M_star_BHFC,
        "D_M_star_LCDM": D_M_star_LCDM,
    }
