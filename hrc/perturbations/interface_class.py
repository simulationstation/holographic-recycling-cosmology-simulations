"""CLASS Boltzmann code interface for HRC.

This module provides a wrapper for the CLASS cosmology code that allows
computing CMB and matter power spectra with HRC modifications.

If CLASS is not installed, a fallback stub provides approximate results
based on analytical scaling relations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import warnings

from ..utils.config import HRCParameters, HRCConfig
from ..utils.constants import PLANCK_2018
from ..background import BackgroundSolution


@dataclass
class CLASSOutput:
    """Output from CLASS computation."""

    # CMB power spectra
    ell: NDArray[np.floating]  # Multipoles
    Cl_TT: NDArray[np.floating]  # TT spectrum [μK²]
    Cl_EE: NDArray[np.floating]  # EE spectrum [μK²]
    Cl_TE: NDArray[np.floating]  # TE spectrum [μK²]

    # Matter power spectrum
    k: NDArray[np.floating]  # Wavenumber [h/Mpc]
    Pk: NDArray[np.floating]  # Power spectrum [Mpc/h]³

    # Derived quantities
    theta_star: float  # Angular sound horizon [rad]
    r_s: float  # Sound horizon at last scattering [Mpc]
    D_A: float  # Angular diameter distance to last scattering [Mpc]
    sigma8: float  # σ₈ normalization
    H0: float  # Hubble constant [km/s/Mpc]

    # Metadata
    success: bool = True
    is_stub: bool = False
    message: str = ""


@dataclass
class CLASSParams:
    """Parameters for CLASS computation."""

    # Cosmological parameters
    h: float = 0.7
    omega_b: float = 0.02237  # Ω_b h²
    omega_cdm: float = 0.12  # Ω_c h²
    tau_reio: float = 0.0544
    A_s: float = 2.1e-9
    n_s: float = 0.9649

    # HRC modifications
    G_eff_ratio: float = 1.0  # G_eff/G at z=0
    G_eff_z_func: Optional[callable] = None  # G_eff(z)/G function

    # Output settings
    l_max: int = 2500
    k_max: float = 10.0  # h/Mpc

    def to_class_dict(self) -> Dict[str, Any]:
        """Convert to CLASS-compatible parameter dictionary."""
        return {
            "h": self.h,
            "omega_b": self.omega_b,
            "omega_cdm": self.omega_cdm,
            "tau_reio": self.tau_reio,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "l_max_scalars": self.l_max,
            "P_k_max_h/Mpc": self.k_max,
            "output": "tCl,pCl,lCl,mPk",
            "lensing": "yes",
        }


def _check_class_available() -> bool:
    """Check if CLASS (classy) is installed."""
    try:
        import classy
        return True
    except ImportError:
        return False


class CLASSInterface:
    """Interface to CLASS Boltzmann code with HRC modifications.

    This class wraps the CLASS cosmology code and modifies the
    background evolution to include epoch-dependent G_eff.
    """

    def __init__(
        self,
        params: HRCParameters,
        background_solution: Optional[BackgroundSolution] = None,
    ):
        """Initialize CLASS interface.

        Args:
            params: HRC parameters
            background_solution: Pre-computed background (for G_eff interpolation)
        """
        self.params = params
        self.background_solution = background_solution
        self._class_available = _check_class_available()

        if not self._class_available:
            warnings.warn(
                "CLASS (classy) not installed. Using analytical stub. "
                "Install with: pip install classy",
                RuntimeWarning,
            )

    def compute(
        self,
        class_params: Optional[CLASSParams] = None,
        verbose: bool = False,
    ) -> CLASSOutput:
        """Compute CMB and matter power spectra.

        Args:
            class_params: CLASS parameters (default: from HRC params)
            verbose: Print progress messages

        Returns:
            CLASSOutput with spectra and derived quantities
        """
        if class_params is None:
            class_params = self._default_class_params()

        if self._class_available:
            return self._compute_with_class(class_params, verbose)
        else:
            return self._compute_stub(class_params, verbose)

    def _default_class_params(self) -> CLASSParams:
        """Create default CLASS parameters from HRC params."""
        # Get G_eff interpolation function if background available
        G_eff_func = None
        if self.background_solution is not None:
            G_eff_func = self.background_solution.G_eff_at

        return CLASSParams(
            h=self.params.h,
            omega_b=self.params.Omega_b * self.params.h**2,
            omega_cdm=self.params.Omega_c * self.params.h**2,
            G_eff_z_func=G_eff_func,
        )

    def _compute_with_class(
        self,
        class_params: CLASSParams,
        verbose: bool,
    ) -> CLASSOutput:
        """Compute using actual CLASS code."""
        try:
            from classy import Class

            cosmo = Class()
            cosmo.set(class_params.to_class_dict())

            # Note: For full HRC implementation, we would need to modify
            # CLASS source code to include G_eff(z). This is a placeholder
            # that runs standard CLASS and then applies approximate corrections.

            cosmo.compute()

            # Get CMB spectra
            cls = cosmo.lensed_cl(class_params.l_max)
            ell = cls["ell"][2:]  # Skip l=0,1
            Cl_TT = cls["tt"][2:] * 1e12  # Convert to μK²
            Cl_EE = cls["ee"][2:] * 1e12
            Cl_TE = cls["te"][2:] * 1e12

            # Get matter power spectrum
            k = np.logspace(-4, np.log10(class_params.k_max), 200)
            Pk = np.array([cosmo.pk(ki, 0.0) for ki in k])

            # Get derived quantities
            theta_star = cosmo.theta_s_100()
            r_s = cosmo.rs_drag()
            derived = cosmo.get_current_derived_parameters(
                ["sigma8", "100*theta_s", "Da_rec"]
            )

            output = CLASSOutput(
                ell=ell,
                Cl_TT=Cl_TT,
                Cl_EE=Cl_EE,
                Cl_TE=Cl_TE,
                k=k,
                Pk=Pk,
                theta_star=derived.get("100*theta_s", PLANCK_2018.theta_star * 100)
                / 100,
                r_s=r_s,
                D_A=derived.get("Da_rec", PLANCK_2018.r_s / PLANCK_2018.theta_star),
                sigma8=derived.get("sigma8", PLANCK_2018.sigma8),
                H0=100 * class_params.h,
                success=True,
                is_stub=False,
                message="Computed with CLASS",
            )

            cosmo.struct_cleanup()
            cosmo.empty()

            # Apply HRC corrections if G_eff function provided
            if class_params.G_eff_z_func is not None:
                output = self._apply_hrc_corrections(output, class_params)

            return output

        except Exception as e:
            warnings.warn(f"CLASS computation failed: {e}. Using stub.")
            return self._compute_stub(class_params, verbose)

    def _compute_stub(
        self,
        class_params: CLASSParams,
        verbose: bool,
    ) -> CLASSOutput:
        """Compute approximate results without CLASS."""
        if verbose:
            print("Using analytical stub (CLASS not available)")

        # Use Planck 2018 values as baseline
        ell = np.arange(2, class_params.l_max + 1)

        # Approximate CMB TT spectrum (simplified fitting function)
        # D_ell = ell(ell+1)C_ell/(2π) ≈ A * (ell/ell_peak)^n * exp(-((ell-ell_peak)/σ)^2)
        ell_peak = 220.0
        A_TT = 6000.0  # μK²
        sigma_TT = 800.0

        D_TT = A_TT * np.exp(-0.5 * ((ell - ell_peak) / sigma_TT) ** 2)
        D_TT *= 1 + 0.1 * np.cos(np.pi * ell / 300)  # Add acoustic oscillations

        Cl_TT = D_TT * 2 * np.pi / (ell * (ell + 1))

        # Approximate EE spectrum (roughly 1/50 of TT)
        Cl_EE = Cl_TT / 50

        # Approximate TE spectrum
        Cl_TE = np.sqrt(Cl_TT * Cl_EE) * np.cos(np.pi * ell / 150)

        # Matter power spectrum (approximate)
        k = np.logspace(-4, np.log10(class_params.k_max), 200)
        k_eq = 0.01  # Equality wavenumber
        Pk = (
            class_params.A_s
            * 1e9
            * k ** class_params.n_s
            / (1 + (k / k_eq) ** 3) ** (2 / 3)
        )

        # Derived quantities
        H0 = 100 * class_params.h
        Omega_m = (class_params.omega_b + class_params.omega_cdm) / class_params.h**2
        theta_star = PLANCK_2018.theta_star * (67.36 / H0) ** 0.3
        r_s = PLANCK_2018.r_s * (67.36 / H0) ** 0.5
        D_A = r_s / theta_star
        sigma8 = PLANCK_2018.sigma8 * (Omega_m / 0.315) ** 0.5

        output = CLASSOutput(
            ell=ell.astype(float),
            Cl_TT=Cl_TT,
            Cl_EE=Cl_EE,
            Cl_TE=Cl_TE,
            k=k,
            Pk=Pk,
            theta_star=theta_star,
            r_s=r_s,
            D_A=D_A,
            sigma8=sigma8,
            H0=H0,
            success=True,
            is_stub=True,
            message="Analytical stub (CLASS not installed)",
        )

        # Apply HRC corrections
        if class_params.G_eff_z_func is not None:
            output = self._apply_hrc_corrections(output, class_params)

        return output

    def _apply_hrc_corrections(
        self,
        output: CLASSOutput,
        class_params: CLASSParams,
    ) -> CLASSOutput:
        """Apply HRC corrections to CLASS output.

        The main effects of G_eff(z) are:
        1. Modified sound horizon: r_s ∝ 1/sqrt(G_eff(z_rec))
        2. Modified angular diameter distance
        3. Shift in acoustic peak positions
        """
        G_eff_func = class_params.G_eff_z_func
        if G_eff_func is None:
            return output

        # G_eff at recombination
        z_rec = 1089.0
        G_eff_rec = G_eff_func(z_rec)
        G_eff_0 = G_eff_func(0.0)

        # Sound horizon scales as r_s ∝ 1/sqrt(G_eff)
        r_s_correction = 1.0 / np.sqrt(G_eff_rec)
        r_s_new = output.r_s * r_s_correction

        # Angular diameter distance correction
        # D_A depends on the integrated expansion history
        # Approximate: D_A ∝ sqrt(G_eff)
        D_A_correction = np.sqrt(G_eff_0)
        D_A_new = output.D_A * D_A_correction

        # New angular scale
        theta_star_new = r_s_new / D_A_new

        # Peak position shift: Δℓ ∝ Δθ_*/θ_*
        delta_theta = (theta_star_new - output.theta_star) / output.theta_star
        ell_shift = 1.0 + delta_theta

        # Shift CMB spectra
        ell_new = output.ell * ell_shift

        # σ₈ correction: growth rate affected by G_eff
        sigma8_correction = (G_eff_0) ** 0.25
        sigma8_new = output.sigma8 * sigma8_correction

        return CLASSOutput(
            ell=ell_new,
            Cl_TT=output.Cl_TT,
            Cl_EE=output.Cl_EE,
            Cl_TE=output.Cl_TE,
            k=output.k,
            Pk=output.Pk * G_eff_0,  # P(k) ∝ G_eff
            theta_star=theta_star_new,
            r_s=r_s_new,
            D_A=D_A_new,
            sigma8=sigma8_new,
            H0=output.H0,
            success=output.success,
            is_stub=output.is_stub,
            message=output.message + " + HRC corrections",
        )

    def write_ini_file(
        self,
        filename: str,
        class_params: Optional[CLASSParams] = None,
    ) -> None:
        """Write CLASS-compatible .ini parameter file.

        Args:
            filename: Output filename
            class_params: Parameters to write
        """
        if class_params is None:
            class_params = self._default_class_params()

        params_dict = class_params.to_class_dict()

        with open(filename, "w") as f:
            f.write("# CLASS parameter file generated by HRC\n")
            f.write("# Note: G_eff modifications require CLASS source modification\n\n")

            for key, value in params_dict.items():
                f.write(f"{key} = {value}\n")

            # Add HRC-specific comments
            f.write("\n# HRC parameters (for modified CLASS):\n")
            f.write(f"# xi = {self.params.xi}\n")
            f.write(f"# phi_0 = {self.params.phi_0}\n")
            f.write(f"# G_eff/G (z=0) = {class_params.G_eff_ratio}\n")


# Convenience alias for when CLASS is not available
CLASSStub = CLASSInterface
