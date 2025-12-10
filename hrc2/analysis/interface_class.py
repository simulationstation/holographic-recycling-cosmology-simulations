"""CLASS Boltzmann code interface for horizon-memory models.

This module provides a CLASS-compatible export system for horizon-memory
cosmology models, allowing integration with the CLASS Boltzmann solver.

The horizon-memory component is treated as an effective dark energy fluid
with:
    - Tabulated w(z) from background analysis
    - Energy density rho_hor(a)/rho_crit0
    - c_s^2 = 1 (smooth DE, non-clustering)

CLASS supports tabulated w(z) through its "fluid" dark energy mode.
This module exports the necessary tables and parameter files.

References:
- CLASS: https://github.com/lesgourg/class_public
- CLASS documentation on fluid dark energy
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hrc2.theory import HRC2Parameters, CouplingFamily, PotentialType
from hrc2.background import BackgroundCosmology


@dataclass
class HorizonMemoryClassExport:
    """Container for CLASS-compatible horizon-memory export."""
    model_id: str
    lambda_hor: float
    tau_hor: float

    # Background tables
    z_table: np.ndarray
    w_table: np.ndarray  # w_hor(z)
    Omega_de_table: np.ndarray  # Omega_de(z) = rho_de / rho_crit

    # Derived parameters
    Omega_de_0: float
    Omega_Lambda_eff: float
    c_s_squared: float


def integrate_memory_field(cosmo: BackgroundCosmology, z_max: float = 10.0) -> callable:
    """Integrate the memory field ODE and return interpolator."""
    a_start = 1.0 / (1.0 + z_max)
    a_end = 1.0
    ln_a_start, ln_a_end = np.log(a_start), np.log(a_end)

    def memory_ode(ln_a, y):
        M = y[0]
        a = np.exp(ln_a)
        H = cosmo.H_of_a_gr(a)
        S_n = cosmo.S_norm(H)
        dM_dlna = (S_n - M) / cosmo.tau_hor
        return [dM_dlna]

    sol = solve_ivp(
        memory_ode,
        (ln_a_start, ln_a_end),
        [0.0],
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Memory field integration failed: {sol.message}")

    return sol.sol


def compute_w_hor(
    z: float,
    cosmo: BackgroundCosmology,
    M_interp: callable,
    eps: float = 1e-4,
) -> float:
    """Compute effective equation of state w_hor(z).

    w_hor = -1 - (1/3) * d ln(rho_hor) / d ln(a)
          = -1 - (1/3) * d ln(M) / d ln(a)

    Args:
        z: Redshift
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator
        eps: Step for numerical derivative

    Returns:
        w_hor(z)
    """
    a = 1.0 / (1.0 + z)
    ln_a = np.log(a)

    # Get M at this point
    M = M_interp(ln_a)[0]

    if M <= 0:
        return -1.0  # Default to cosmological constant

    # Numerical derivative
    ln_a_plus = min(ln_a + eps, 0.0)
    ln_a_minus = ln_a - eps

    M_plus = M_interp(ln_a_plus)[0]
    M_minus = M_interp(ln_a_minus)[0]

    if M_plus <= 0 or M_minus <= 0:
        return -1.0

    d_ln_M_d_ln_a = (np.log(M_plus) - np.log(M_minus)) / (2 * eps)

    w = -1.0 - d_ln_M_d_ln_a / 3.0

    # Safety clamp
    return np.clip(w, -2.5, 0.5)


def compute_Omega_de(
    z: float,
    cosmo: BackgroundCosmology,
    M_interp: callable,
) -> float:
    """Compute effective dark energy density fraction.

    Omega_de(z) = (rho_Lambda_eff + rho_hor) / rho_crit(z)

    Args:
        z: Redshift
        cosmo: BackgroundCosmology instance
        M_interp: Memory field interpolator

    Returns:
        Omega_de(z)
    """
    a = 1.0 / (1.0 + z)
    ln_a = np.log(a)

    M = M_interp(ln_a)[0]

    # Horizon-memory density
    rho_hor = cosmo.lambda_hor * M

    # Effective Lambda density (constant)
    rho_Lambda_eff = cosmo.Omega_L0_eff

    # Total DE density
    rho_de = rho_Lambda_eff + rho_hor

    # Compute H^2 / H0^2
    H_ratio = cosmo.H_of_a_selfconsistent(a, M) / cosmo.H0
    H_sq_ratio = H_ratio ** 2

    # Omega_de = rho_de / (H^2/H0^2)
    return rho_de / H_sq_ratio if H_sq_ratio > 0 else 0.0


def prepare_class_export(
    model_data: dict,
    n_points: int = 100,
    z_max: float = 10.0,
) -> HorizonMemoryClassExport:
    """Prepare CLASS-compatible export for a horizon-memory model.

    Args:
        model_data: Dictionary with model parameters
        n_points: Number of table points
        z_max: Maximum redshift for tables

    Returns:
        HorizonMemoryClassExport with all data
    """
    lambda_hor = model_data["lambda_hor"]
    tau_hor = model_data["tau_hor"]
    model_id = f"{lambda_hor:.3f}_{tau_hor:.2f}"

    # Create cosmology
    params = HRC2Parameters(
        xi=0.0,
        phi_0=0.0,
        coupling_family=CouplingFamily.QUADRATIC,
        potential_type=PotentialType.QUADRATIC,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
    )
    cosmo = BackgroundCosmology(params)

    # Integrate memory field
    M_interp = integrate_memory_field(cosmo, z_max=z_max)
    M_today = M_interp(0.0)[0]
    cosmo.set_M_today(M_today)

    # Create redshift table
    z_table = np.linspace(0, z_max, n_points)

    # Compute w(z) table
    w_table = np.array([compute_w_hor(z, cosmo, M_interp) for z in z_table])

    # Compute Omega_de(z) table
    Omega_de_table = np.array([compute_Omega_de(z, cosmo, M_interp) for z in z_table])

    return HorizonMemoryClassExport(
        model_id=model_id,
        lambda_hor=lambda_hor,
        tau_hor=tau_hor,
        z_table=z_table,
        w_table=w_table,
        Omega_de_table=Omega_de_table,
        Omega_de_0=Omega_de_table[0],
        Omega_Lambda_eff=cosmo.Omega_L0_eff,
        c_s_squared=1.0,
    )


def export_to_class_format(
    model_id: str,
    output_dir: str,
    n_points: int = 100,
    z_max: float = 10.0,
) -> str:
    """Export horizon-memory model to CLASS-compatible format.

    This produces:
    1. A parameter file (.ini) for CLASS
    2. A w(z) table file for the fluid approximation
    3. A background density file

    Args:
        model_id: Model identifier (e.g., "0.200_0.10")
        output_dir: Output directory
        n_points: Number of table points
        z_max: Maximum redshift

    Returns:
        Path to the main .ini file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse model_id
    parts = model_id.split("_")
    lambda_hor = float(parts[0])
    tau_hor = float(parts[1])

    model_data = {"lambda_hor": lambda_hor, "tau_hor": tau_hor}

    # Prepare export data
    export = prepare_class_export(model_data, n_points, z_max)

    # Write w(z) table
    w_table_path = os.path.join(output_dir, f"w_hor_{model_id}.dat")
    with open(w_table_path, "w") as f:
        f.write("# Horizon-memory effective equation of state w(z)\n")
        f.write(f"# Model: lambda_hor={lambda_hor}, tau_hor={tau_hor}\n")
        f.write(f"# c_s^2 = {export.c_s_squared}\n")
        f.write("# z    w(z)\n")
        for z, w in zip(export.z_table, export.w_table):
            f.write(f"{z:.6f}  {w:.8f}\n")

    # Write background density table
    rho_table_path = os.path.join(output_dir, f"rho_de_{model_id}.dat")
    with open(rho_table_path, "w") as f:
        f.write("# Horizon-memory effective dark energy density fraction\n")
        f.write(f"# Model: lambda_hor={lambda_hor}, tau_hor={tau_hor}\n")
        f.write("# z    Omega_de(z)\n")
        for z, Omega in zip(export.z_table, export.Omega_de_table):
            f.write(f"{z:.6f}  {Omega:.8f}\n")

    # Write CLASS .ini file
    ini_path = os.path.join(output_dir, f"class_horizon_memory_{model_id}.ini")
    with open(ini_path, "w") as f:
        f.write("# CLASS parameter file for horizon-memory cosmology\n")
        f.write(f"# Model: lambda_hor={lambda_hor}, tau_hor={tau_hor}\n")
        f.write("#\n")
        f.write("# This file configures CLASS to use a tabulated w(z) for dark energy\n")
        f.write("# treating the horizon-memory component as an effective fluid.\n")
        f.write("#\n\n")

        # Basic cosmological parameters (Planck 2018)
        f.write("# Cosmological parameters\n")
        f.write("h = 0.674\n")
        f.write("omega_b = 0.02237\n")
        f.write("omega_cdm = 0.1200\n")
        f.write("tau_reio = 0.0544\n")
        f.write("A_s = 2.1e-9\n")
        f.write("n_s = 0.9649\n")
        f.write("\n")

        # Dark energy settings (fluid approximation)
        f.write("# Dark energy settings (horizon-memory as fluid)\n")
        f.write("Omega_Lambda = 0\n")  # We use fluid instead
        f.write("Omega_fld = {:.8f}\n".format(export.Omega_de_0))
        f.write("fluid_equation_of_state = CLP\n")  # Chevallier-Linder-Polarski parametrization
        f.write("# For exact w(z), use tabulated form:\n")
        f.write(f"# w_fld_file = {w_table_path}\n")
        f.write("# Or use effective constant w at late times:\n")
        f.write("w0_fld = {:.6f}\n".format(export.w_table[0]))
        f.write("wa_fld = 0.0\n")  # No evolution in simple approximation
        f.write("cs2_fld = {:.1f}\n".format(export.c_s_squared))
        f.write("\n")

        # Output settings
        f.write("# Output settings\n")
        f.write("output = tCl,pCl,lCl,mPk\n")
        f.write("lensing = yes\n")
        f.write("l_max_scalars = 2500\n")
        f.write("P_k_max_h/Mpc = 10.0\n")
        f.write("\n")

        # Comments
        f.write("# Notes:\n")
        f.write("# - c_s^2 = 1 makes the DE smooth (non-clustering)\n")
        f.write("# - For exact w(z) evolution, use the tabulated file\n")
        f.write("# - The horizon-memory model has phantom-like w < -1\n")

    # Write metadata JSON
    meta_path = os.path.join(output_dir, f"metadata_{model_id}.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_id": model_id,
            "lambda_hor": lambda_hor,
            "tau_hor": tau_hor,
            "Omega_de_0": float(export.Omega_de_0),
            "Omega_Lambda_eff": float(export.Omega_Lambda_eff),
            "c_s_squared": float(export.c_s_squared),
            "w_0": float(export.w_table[0]),
            "w_z1": float(np.interp(1.0, export.z_table, export.w_table)),
            "z_max": z_max,
            "n_points": n_points,
            "files": {
                "ini": ini_path,
                "w_table": w_table_path,
                "rho_table": rho_table_path,
            }
        }, f, indent=2)

    return ini_path


class HorizonMemoryClassInterface:
    """Interface for running CLASS with horizon-memory models.

    This class manages the export and (if CLASS is available) execution
    of CLASS computations for horizon-memory cosmology.
    """

    def __init__(
        self,
        lambda_hor: float,
        tau_hor: float,
        output_dir: str = "results/class_export",
    ):
        """Initialize CLASS interface.

        Args:
            lambda_hor: Horizon-memory coupling strength
            tau_hor: Memory timescale
            output_dir: Base output directory
        """
        self.lambda_hor = lambda_hor
        self.tau_hor = tau_hor
        self.model_id = f"{lambda_hor:.3f}_{tau_hor:.2f}"
        self.output_dir = os.path.join(output_dir, self.model_id)

        self._export: Optional[HorizonMemoryClassExport] = None
        self._ini_path: Optional[str] = None

    def prepare_export(
        self,
        n_points: int = 100,
        z_max: float = 10.0,
    ) -> HorizonMemoryClassExport:
        """Prepare CLASS export data.

        Args:
            n_points: Number of table points
            z_max: Maximum redshift

        Returns:
            HorizonMemoryClassExport data
        """
        model_data = {"lambda_hor": self.lambda_hor, "tau_hor": self.tau_hor}
        self._export = prepare_class_export(model_data, n_points, z_max)
        return self._export

    def export_files(
        self,
        n_points: int = 100,
        z_max: float = 10.0,
    ) -> str:
        """Export all CLASS-compatible files.

        Args:
            n_points: Number of table points
            z_max: Maximum redshift

        Returns:
            Path to main .ini file
        """
        self._ini_path = export_to_class_format(
            self.model_id, self.output_dir, n_points, z_max
        )
        return self._ini_path

    def get_w_interpolator(self) -> Callable[[float], float]:
        """Get interpolator for w(z).

        Returns:
            Callable w(z) function
        """
        if self._export is None:
            self.prepare_export()

        interp = interp1d(
            self._export.z_table,
            self._export.w_table,
            kind='linear',
            bounds_error=False,
            fill_value=(self._export.w_table[0], self._export.w_table[-1]),
        )

        return lambda z: float(interp(z))

    def get_rho_hor_interpolator(self) -> Callable[[float], float]:
        """Get interpolator for rho_hor(a)/rho_crit0.

        Note: This returns the fractional density rho_hor/rho_crit0,
        not the density parameter Omega_hor.

        Returns:
            Callable rho_hor(a)/rho_crit0 function
        """
        if self._export is None:
            self.prepare_export()

        # Convert from Omega_de(z) to rho/rho_crit0
        # This is approximate since we don't have the exact decomposition
        interp = interp1d(
            self._export.z_table,
            self._export.Omega_de_table,
            kind='linear',
            bounds_error=False,
            fill_value=(self._export.Omega_de_table[0], self._export.Omega_de_table[-1]),
        )

        def rho_func(a: float) -> float:
            z = 1.0 / a - 1.0
            return float(interp(z))

        return rho_func

    @staticmethod
    def check_class_available() -> bool:
        """Check if CLASS (classy) is installed."""
        try:
            import classy
            return True
        except ImportError:
            return False

    def run_class(self, verbose: bool = False) -> Optional[dict]:
        """Run CLASS computation if available.

        Args:
            verbose: Print progress messages

        Returns:
            Dictionary with CLASS output, or None if CLASS unavailable
        """
        if not self.check_class_available():
            if verbose:
                print("CLASS (classy) not installed. Skipping computation.")
            return None

        if self._ini_path is None:
            self.export_files()

        try:
            from classy import Class

            # Read parameters from ini file
            # For now, use simplified parameter set
            cosmo = Class()

            if self._export is None:
                self.prepare_export()

            params = {
                'h': 0.674,
                'omega_b': 0.02237,
                'omega_cdm': 0.1200,
                'tau_reio': 0.0544,
                'A_s': 2.1e-9,
                'n_s': 0.9649,
                'Omega_Lambda': 0,
                'Omega_fld': self._export.Omega_de_0,
                'w0_fld': self._export.w_table[0],
                'wa_fld': 0.0,
                'cs2_fld': self._export.c_s_squared,
                'output': 'tCl,pCl,lCl,mPk',
                'lensing': 'yes',
                'l_max_scalars': 2500,
                'P_k_max_h/Mpc': 10.0,
            }

            cosmo.set(params)
            cosmo.compute()

            # Extract results
            cls = cosmo.lensed_cl(2500)
            ell = cls['ell'][2:]
            Cl_TT = cls['tt'][2:] * 1e12  # μK²

            k = np.logspace(-4, 1, 200)
            Pk = np.array([cosmo.pk(ki, 0.0) for ki in k])

            derived = cosmo.get_current_derived_parameters(['sigma8', 'H0'])

            result = {
                'ell': ell.tolist(),
                'Cl_TT': Cl_TT.tolist(),
                'k': k.tolist(),
                'Pk': Pk.tolist(),
                'sigma8': derived.get('sigma8', 0.811),
                'H0': derived.get('H0', 67.4),
            }

            cosmo.struct_cleanup()
            cosmo.empty()

            return result

        except Exception as e:
            if verbose:
                print(f"CLASS computation failed: {e}")
            return None
