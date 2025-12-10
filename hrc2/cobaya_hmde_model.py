"""Cobaya-compatible wrapper for HMDE T06D horizon-memory dark energy model.

This module provides a Cobaya Theory class that interfaces the T06D
horizon-memory model with CAMB for MCMC parameter estimation.

Strategy:
    T06D defines w_eff(a) = w_base(a) + Δw / (1 + (a/a_w)^m)

    CAMB accepts (w, wa) for CPL parameterization: w(a) = w0 + wa*(1-a)

    We fit w_eff(a) to w(a) = w0 + wa*(1-a) using least squares over
    a grid of scale factors. This captures the time-varying EoS behavior
    while using CAMB's standard dark energy module.

T06D Parameters:
    - delta_w: EoS shift amplitude (sampled or fixed)
    - a_w: Transition scale factor (sampled or fixed)
    - m_eos: Transition power (fixed at 2.0)
    - lambda_hor: Memory amplitude (sampled or fixed at 0.2)
    - tau_hor: Memory timescale (fixed at 0.1)

Derived quantities:
    - w0: Fitted present-day EoS
    - wa: Fitted EoS time derivative
    - Omega_hor0: Horizon-memory density fraction today

Usage in Cobaya YAML:
    theory:
      hrc2.cobaya_hmde_model.HMDE_T06D:
        stop_at_error: true
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np
from scipy.optimize import curve_fit

try:
    from cobaya.theory import Theory
    HAS_COBAYA = True
except ImportError:
    HAS_COBAYA = False
    Theory = object  # Fallback for import


class HMDE_T06D(Theory if HAS_COBAYA else object):
    """Cobaya Theory class for HMDE T06D horizon-memory dark energy.

    This theory computes effective (w0, wa) parameters from the T06D
    horizon-memory model and passes them to CAMB.

    The T06D EoS is:
        w_eff(a) = w_base + Δw / (1 + (a/a_w)^m)

    where w_base ≈ -1 for late times. We fit this to CPL:
        w(a) = w0 + wa*(1-a)

    Parameters provided to Cobaya sampler:
        - delta_w: EoS shift (-0.5 to 0.1)
        - a_w: Transition scale (0.1 to 0.5)
        - lambda_hor: Memory amplitude (0.1 to 0.4), optional

    Fixed parameters:
        - m_eos: 2.0 (transition sharpness)
        - tau_hor: 0.1 (memory timescale)
        - w_base: -1.0 (cosmological constant baseline)
    """

    # Parameters that this theory can provide to other theories/likelihoods
    # CAMB will read w, wa from here
    _provides = ['w0_fld', 'wa_fld']

    # Parameters this theory needs from the sampler
    params = {
        'delta_w': None,  # Must be provided in YAML
        'a_w': None,      # Must be provided in YAML
    }

    # Fixed model parameters (not sampled)
    m_eos: float = 2.0
    tau_hor: float = 0.1
    w_base: float = -1.0

    # Fitting grid
    n_fit_points: int = 100
    a_min: float = 0.01  # Fit from a=0.01 to a=1
    a_max: float = 1.0

    def initialize(self):
        """Initialize the HMDE T06D theory."""
        super().initialize()
        self.log.info("Initializing HMDE T06D horizon-memory theory")
        self.log.info(f"Fixed parameters: m_eos={self.m_eos}, tau_hor={self.tau_hor}")

        # Pre-compute fitting grid
        self._a_grid = np.linspace(self.a_min, self.a_max, self.n_fit_points)

    def get_can_provide_params(self) -> Sequence[str]:
        """Return list of parameters this theory can provide."""
        return ['w0_fld', 'wa_fld']

    def get_allow_agnostic(self) -> bool:
        """Allow this theory to work with any likelihood."""
        return True

    def w_eff_t06d(self, a: np.ndarray, delta_w: float, a_w: float) -> np.ndarray:
        """Compute T06D effective EoS w_eff(a).

        w_eff(a) = w_base + Δw / (1 + (a/a_w)^m)

        Args:
            a: Scale factor array
            delta_w: EoS shift amplitude
            a_w: Transition scale factor

        Returns:
            w_eff array
        """
        x = a / a_w
        w_modifier = delta_w / (1.0 + x ** self.m_eos)
        return self.w_base + w_modifier

    def fit_w0_wa(self, delta_w: float, a_w: float) -> tuple:
        """Fit T06D w_eff(a) to CPL parameterization w0 + wa*(1-a).

        Uses least squares fitting over the pre-computed a grid.

        Args:
            delta_w: T06D delta_w parameter
            a_w: T06D a_w parameter

        Returns:
            Tuple (w0, wa) fitted parameters
        """
        # Compute T06D w_eff on grid
        w_eff = self.w_eff_t06d(self._a_grid, delta_w, a_w)

        # Define CPL model
        def cpl_model(a, w0, wa):
            return w0 + wa * (1 - a)

        # Fit with reasonable initial guess
        try:
            # Initial guess: w0 ≈ w_eff(a=1), wa ≈ 0
            w0_init = w_eff[-1]  # At a=1
            wa_init = 0.0

            popt, _ = curve_fit(
                cpl_model,
                self._a_grid,
                w_eff,
                p0=[w0_init, wa_init],
                bounds=([-3.0, -5.0], [0.0, 5.0])  # Physical bounds
            )
            w0, wa = popt

        except Exception as e:
            # Fallback: use endpoints for linear approximation
            self.log.warning(f"CPL fit failed: {e}, using linear approximation")
            w0 = w_eff[-1]  # w at a=1
            w_early = w_eff[0]  # w at a=a_min
            wa = (w_early - w0) / (1 - self._a_grid[0])

        return w0, wa

    def calculate(self, state: Dict[str, Any], want_derived: bool = True,
                  **params_values) -> bool:
        """Calculate w0, wa from T06D parameters.

        This method is called by Cobaya for each MCMC sample.

        Args:
            state: State dictionary to store results
            want_derived: Whether to compute derived parameters
            **params_values: Parameter values from sampler

        Returns:
            True if calculation succeeded
        """
        # Extract T06D parameters
        delta_w = params_values.get('delta_w')
        a_w = params_values.get('a_w')

        if delta_w is None or a_w is None:
            self.log.error("Missing required parameters delta_w or a_w")
            return False

        # Fit to CPL
        w0, wa = self.fit_w0_wa(delta_w, a_w)

        # Store in state for CAMB to read
        state['w0_fld'] = w0
        state['wa_fld'] = wa

        # Log for debugging (sparse)
        if hasattr(self, '_n_calls'):
            self._n_calls += 1
        else:
            self._n_calls = 1

        if self._n_calls % 100 == 0:
            self.log.debug(f"T06D→CPL: delta_w={delta_w:.4f}, a_w={a_w:.4f} "
                          f"→ w0={w0:.4f}, wa={wa:.4f}")

        return True

    def get_param(self, param: str) -> Optional[float]:
        """Get a derived parameter value.

        Args:
            param: Parameter name ('w0_fld' or 'wa_fld')

        Returns:
            Parameter value or None
        """
        if param in self.current_state:
            return self.current_state[param]
        return None


def compute_omega_hor0(delta_w: float, a_w: float, m_eos: float = 2.0,
                       lambda_hor: float = 0.2, tau_hor: float = 0.1,
                       z_max: float = 1100.0) -> float:
    """Compute Omega_hor0 for given T06D parameters.

    This runs the full T06D integration to get the present-day
    horizon-memory density fraction. Used for derived parameter output.

    Args:
        delta_w: EoS shift amplitude
        a_w: Transition scale factor
        m_eos: Transition power
        lambda_hor: Memory amplitude
        tau_hor: Memory timescale
        z_max: Integration starting redshift

    Returns:
        Omega_hor0 (dimensionless)
    """
    try:
        from .horizon_models.refinement_d import create_dynamical_eos_model

        model = create_dynamical_eos_model(
            delta_w=delta_w,
            a_w=a_w,
            m_eos=m_eos,
            lambda_hor=lambda_hor,
            tau_hor=tau_hor
        )
        result = model.solve(z_max=z_max)

        if result.success:
            return result.Omega_hor0
        return np.nan

    except Exception:
        return np.nan


def get_t06d_best_fit_params() -> Dict[str, float]:
    """Return the best-fit T06D parameters from refinement analysis.

    These are the parameters identified in T06_refinement_selection
    as giving minimal CMB deviation while maintaining observable effects.

    Returns:
        Dictionary of best-fit parameter values
    """
    return {
        'delta_w': -0.033333333333333326,
        'a_w': 0.2777777777777778,
        'm_eos': 2.0,
        'lambda_hor': 0.2,
        'tau_hor': 0.1,
    }


def get_t06d_priors() -> Dict[str, Dict[str, Any]]:
    """Return recommended prior ranges for T06D parameters.

    Based on the parameter scan results showing viable models.

    Returns:
        Dictionary of prior specifications for Cobaya
    """
    return {
        'delta_w': {
            'prior': {'min': -0.5, 'max': 0.1},
            'ref': {'dist': 'norm', 'loc': -0.033, 'scale': 0.05},
            'latex': r'\Delta w'
        },
        'a_w': {
            'prior': {'min': 0.1, 'max': 0.5},
            'ref': {'dist': 'norm', 'loc': 0.28, 'scale': 0.05},
            'latex': r'a_w'
        },
        'lambda_hor': {
            'prior': {'min': 0.1, 'max': 0.4},
            'ref': 0.2,
            'latex': r'\lambda_{\rm hor}'
        }
    }


# Alternative: Direct w(z) theory (experimental)
class HMDE_T06D_Direct(Theory if HAS_COBAYA else object):
    """Direct T06D theory that provides w(z) table to CAMB.

    This is an experimental alternative that provides a tabulated
    w(z) directly to CAMB rather than fitting to CPL. May give
    more accurate results for highly non-linear w(a) evolution.

    NOT RECOMMENDED for initial runs - use HMDE_T06D instead.
    """

    params = {
        'delta_w': {'prior': {'min': -0.5, 'max': 0.1}},
        'a_w': {'prior': {'min': 0.1, 'max': 0.5}},
    }

    m_eos: float = 2.0
    w_base: float = -1.0
    n_z_points: int = 50
    z_max: float = 10.0

    def initialize(self):
        """Initialize with z grid for w(z) table."""
        super().initialize()
        self._z_grid = np.linspace(0, self.z_max, self.n_z_points)
        self._a_grid = 1.0 / (1.0 + self._z_grid)

    def calculate(self, state: Dict[str, Any], want_derived: bool = True,
                  **params_values) -> bool:
        """Compute w(z) table for CAMB dark_energy_w_z mode."""
        delta_w = params_values.get('delta_w')
        a_w = params_values.get('a_w')

        x = self._a_grid / a_w
        w_z = self.w_base + delta_w / (1.0 + x ** self.m_eos)

        # CAMB expects (z, w) table
        state['w_z_table'] = (self._z_grid, w_z)

        return True


if __name__ == "__main__":
    # Test the T06D → CPL conversion
    print("Testing HMDE T06D → CPL conversion")
    print("=" * 50)

    # Best-fit parameters
    params = get_t06d_best_fit_params()
    print(f"T06D parameters: {params}")

    # Create mock theory instance
    theory = HMDE_T06D.__new__(HMDE_T06D)
    theory.m_eos = params['m_eos']
    theory.w_base = -1.0
    theory.n_fit_points = 100
    theory.a_min = 0.01
    theory.a_max = 1.0
    theory._a_grid = np.linspace(theory.a_min, theory.a_max, theory.n_fit_points)

    # Fit
    w0, wa = theory.fit_w0_wa(params['delta_w'], params['a_w'])
    print(f"\nFitted CPL parameters:")
    print(f"  w0 = {w0:.6f}")
    print(f"  wa = {wa:.6f}")

    # Verify fit quality
    w_t06d = theory.w_eff_t06d(theory._a_grid, params['delta_w'], params['a_w'])
    w_cpl = w0 + wa * (1 - theory._a_grid)
    rms_error = np.sqrt(np.mean((w_t06d - w_cpl)**2))
    print(f"\nFit quality (RMS error): {rms_error:.6f}")

    # Compare at key redshifts
    print("\nw(a) comparison at key points:")
    for a in [1.0, 0.5, 0.3, 0.1]:
        w_t = theory.w_eff_t06d(np.array([a]), params['delta_w'], params['a_w'])[0]
        w_c = w0 + wa * (1 - a)
        z = 1/a - 1
        print(f"  z={z:.1f} (a={a}): T06D={w_t:.4f}, CPL={w_c:.4f}, Δ={w_t-w_c:.4f}")
