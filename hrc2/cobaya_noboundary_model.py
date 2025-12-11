"""Cobaya-compatible wrapper for Hawking-Hartle No-Boundary cosmology.

This module provides a Cobaya Theory class that implements the no-boundary
prior structure over primordial parameters and maps to cosmological observables.

Strategy:
    Sample primordial parameters (Ne, log10_V_scale, phi_init, epsilon_corr)
    directly in the MCMC, with the no-boundary prior encoded in the prior
    section of the Cobaya YAML.

    The theory computes derived cosmological parameters and provides
    a modified sound horizon via epsilon_corr.

No-Boundary Parameters:
    - Ne: Number of inflationary e-folds (50-80)
    - log10_V_scale: log10 of inflaton potential scale (-12 to -8)
    - phi_init: Initial field value (0.001-2.0)
    - epsilon_corr: Early-time H(z) correction (-0.1 to 0.1)

Derived quantities:
    - Omega_k: Curvature from e-folds
    - n_s: Spectral index from slow-roll
    - r_s_correction: Fractional change in sound horizon

Usage in Cobaya YAML:
    theory:
      hrc2.cobaya_noboundary_model.NoBoundaryTheory:
        stop_at_error: true

    params:
      Ne:
        prior:
          min: 50
          max: 80
        ref: 60
        proposal: 2
      epsilon_corr:
        prior:
          dist: norm
          loc: 0
          scale: 0.02
        ref: 0
        proposal: 0.005
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np

try:
    from cobaya.theory import Theory
    HAS_COBAYA = True
except ImportError:
    HAS_COBAYA = False
    Theory = object  # Fallback for import

from hrc2.theory import (
    NoBoundaryParams,
    NoBoundaryHyperparams,
    primordial_to_cosmo,
    compute_curvature_from_efolds,
    compute_primordial_spectrum,
    compute_sound_horizon,
    InflationModel,
)
from hrc2.background import apply_epsilon_corr, compute_sound_horizon_epsilon_effect


class NoBoundaryTheory(Theory if HAS_COBAYA else object):
    """Cobaya Theory class for Hawking-Hartle no-boundary cosmology.

    This theory takes primordial parameters from the no-boundary framework
    and computes derived cosmological quantities that affect observables.

    The main effects are:
    1. Curvature (Omega_k) from number of e-folds
    2. Spectral index (n_s) from slow-roll parameters
    3. Sound horizon modification via epsilon_corr

    Parameters provided to Cobaya sampler:
        - Ne: Number of e-folds (50-80)
        - log10_V_scale: Potential scale (-12 to -8)
        - phi_init: Initial field value (0.001-2.0)
        - epsilon_corr: Early-time H(z) correction (-0.1 to 0.1)

    Derived parameters provided to likelihoods:
        - omegak: Curvature density (for CAMB)
        - nnu: Can modify effective neutrinos if needed
        - r_s_factor: Sound horizon modification factor

    Note: The effect of epsilon_corr on r_s is computed analytically
    and applied as a correction to the CAMB result.
    """

    # Parameters that this theory can provide to other theories/likelihoods
    _provides = ['omegak', 'nnu', 'r_s_factor']

    # Parameters this theory needs from the sampler
    params = {
        'Ne': None,            # Must be provided in YAML
        'epsilon_corr': None,  # Must be provided in YAML
        # These can be fixed or sampled
        'log10_V_scale': {'value': -10.0},  # Default fixed value
        'phi_init': {'value': 0.1},         # Default fixed value
    }

    # Fiducial cosmology for reference calculations
    H0_fid: float = 67.4
    Omega_m_fid: float = 0.315
    Omega_b_fid: float = 0.0493

    # Transition redshift for epsilon correction
    z_transition: float = 3000.0

    # Reference for curvature calculation
    Ne_reference: float = 60.0

    def initialize(self):
        """Initialize the no-boundary theory."""
        if HAS_COBAYA:
            super().initialize()
            self.log.info("Initializing Hawking-Hartle No-Boundary Theory")
            self.log.info(f"Reference: Ne={self.Ne_reference}, z_trans={self.z_transition}")
        else:
            print("Warning: Cobaya not available, running in standalone mode")

        # Pre-create inflation model
        self._inflation_model = InflationModel(potential_type="quadratic")

    def get_can_provide_params(self) -> Sequence[str]:
        """Return list of parameters this theory can provide."""
        return ['omegak', 'nnu', 'r_s_factor', 'Omega_k_derived', 'n_s_derived']

    def get_allow_agnostic(self) -> bool:
        """Allow this theory to work with any likelihood."""
        return True

    def compute_omegak(self, Ne: float) -> float:
        """Compute curvature from e-folds using no-boundary prescription.

        More e-folds of inflation -> flatter universe.
        Omega_k ~ exp(-alpha * (Ne - Ne_ref)) for large Ne.

        Args:
            Ne: Number of inflationary e-folds

        Returns:
            Omega_k curvature parameter
        """
        return compute_curvature_from_efolds(Ne, self.Ne_reference)

    def compute_ns_from_slowroll(
        self,
        log10_V_scale: float,
        phi_init: float
    ) -> float:
        """Compute spectral index from slow-roll parameters.

        n_s = 1 - 6*epsilon + 2*eta

        For quadratic potential: epsilon = eta = 2/phi^2

        Args:
            log10_V_scale: log10 of potential scale
            phi_init: Initial field value

        Returns:
            Spectral index n_s
        """
        # Use the mapping function
        _, n_s, _ = compute_primordial_spectrum(
            Ne=60.0,  # Doesn't affect n_s much for slow-roll
            V_scale=log10_V_scale,
            phi_init=phi_init,
            model=self._inflation_model
        )
        return n_s

    def compute_rs_factor(self, epsilon_corr: float) -> float:
        """Compute sound horizon modification factor from epsilon_corr.

        If H(z) is modified at high-z, the sound horizon changes:
            r_s_modified / r_s_standard = 1 + delta_r_s / r_s

        where delta_r_s/r_s is approximately:
            delta_r_s/r_s ~ -epsilon_corr * (high-z integral fraction)

        Args:
            epsilon_corr: Fractional H(z) correction at high-z

        Returns:
            Factor by which sound horizon is modified
        """
        delta_rs_frac = compute_sound_horizon_epsilon_effect(
            epsilon_corr,
            z_transition=self.z_transition
        )
        return 1.0 + delta_rs_frac

    def calculate(self, state: Dict[str, Any], want_derived: bool = True, **params_values):
        """Calculate derived parameters from primordial inputs.

        This is called by Cobaya during MCMC sampling.

        Args:
            state: State dictionary to store results
            want_derived: Whether to compute derived parameters
            **params_values: Parameter values from sampler
        """
        # Extract primordial parameters
        Ne = params_values.get('Ne', 60.0)
        epsilon_corr = params_values.get('epsilon_corr', 0.0)
        log10_V_scale = params_values.get('log10_V_scale', -10.0)
        phi_init = params_values.get('phi_init', 0.1)

        # Compute derived quantities
        Omega_k = self.compute_omegak(Ne)
        n_s = self.compute_ns_from_slowroll(log10_V_scale, phi_init)
        r_s_factor = self.compute_rs_factor(epsilon_corr)

        # Store in state for other theories/likelihoods
        # These will be available to CAMB
        state['omegak'] = Omega_k
        state['nnu'] = 3.046  # Standard neutrino number (can be modified)
        state['r_s_factor'] = r_s_factor

        # Store derived parameters
        if want_derived:
            state['derived'] = {
                'Omega_k_derived': Omega_k,
                'n_s_derived': n_s,
                'r_s_factor_derived': r_s_factor,
            }

    def get_requirements(self) -> Dict[str, Any]:
        """Specify what this theory requires from other theories.

        For the no-boundary model, we don't require anything from other
        theories - we provide primordial quantities to CAMB.
        """
        return {}

    def get_param(self, param_name: str) -> float:
        """Get a parameter value that was computed.

        Args:
            param_name: Name of parameter to retrieve

        Returns:
            Parameter value from the current state
        """
        if hasattr(self, '_current_state') and param_name in self._current_state:
            return self._current_state[param_name]
        raise ValueError(f"Parameter {param_name} not available")


class NoBoundaryPriorLikelihood:
    """Custom likelihood that adds the no-boundary prior term.

    While the flat/Gaussian priors on Ne, epsilon_corr, etc. are specified
    in the Cobaya YAML, the theoretical weighting from the no-boundary
    wave function can be added as a likelihood term.

    The Hawking-Hartle weighting is:
        |Psi|^2 ~ exp(24 * pi^2 / V)

    where V is the inflaton potential. This strongly suppresses
    large-field inflation.

    Usage:
        likelihood:
          hrc2.cobaya_noboundary_model.NoBoundaryPriorLikelihood:
            use_semiclassical: true
    """

    _provides = []
    params = {
        'log10_V_scale': None,
        'phi_init': None,
        'Ne': None,
    }

    # Whether to include semiclassical weighting
    use_semiclassical: bool = False
    # Cap on log-likelihood to prevent numerical issues
    max_loglike: float = 100.0

    def initialize(self):
        """Initialize the prior likelihood."""
        if HAS_COBAYA:
            self.log.info(f"No-boundary prior likelihood initialized")
            self.log.info(f"Semiclassical weighting: {self.use_semiclassical}")

    def get_requirements(self) -> Dict[str, Any]:
        """No requirements from other theories."""
        return {}

    def logp(self, **params_values) -> float:
        """Compute log-likelihood contribution from no-boundary weighting.

        Args:
            **params_values: Parameter values

        Returns:
            Log-likelihood contribution
        """
        if not self.use_semiclassical:
            return 0.0

        log10_V_scale = params_values.get('log10_V_scale', -10.0)
        V_scale = 10**log10_V_scale

        if V_scale <= 0:
            return -np.inf

        # Semiclassical weighting: log|Psi|^2 ~ 24*pi^2 / V
        # This is VERY large for small V, so we cap it
        log_weight = 24 * np.pi**2 / V_scale

        # Cap to prevent numerical issues
        log_weight = min(log_weight, self.max_loglike)

        return log_weight


# For standalone testing
def test_no_boundary_theory():
    """Test the no-boundary theory calculations."""
    theory = NoBoundaryTheory()
    theory.initialize()

    # Test cases
    test_params = [
        {'Ne': 50, 'epsilon_corr': 0.0, 'log10_V_scale': -10.0, 'phi_init': 0.1},
        {'Ne': 60, 'epsilon_corr': 0.02, 'log10_V_scale': -10.0, 'phi_init': 0.1},
        {'Ne': 70, 'epsilon_corr': -0.02, 'log10_V_scale': -10.0, 'phi_init': 0.1},
        {'Ne': 80, 'epsilon_corr': 0.05, 'log10_V_scale': -10.0, 'phi_init': 0.1},
    ]

    print("No-Boundary Theory Test Results")
    print("=" * 60)
    print(f"{'Ne':>6} {'eps_corr':>10} {'Omega_k':>12} {'r_s_factor':>12}")
    print("-" * 60)

    for params in test_params:
        state = {}
        theory.calculate(state, want_derived=True, **params)
        print(f"{params['Ne']:>6} {params['epsilon_corr']:>10.3f} "
              f"{state['omegak']:>12.6f} {state['r_s_factor']:>12.6f}")

    print("=" * 60)


if __name__ == "__main__":
    test_no_boundary_theory()
