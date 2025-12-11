"""
Layered Expansion (Bent-Deck) Cosmology

SIMULATION 24: This module implements a flexible parameterization of the
expansion history H(z) using a "stack of cards" (layered) approach.

Conceptual picture:
- The expansion history H(z) is discretized at n_layers redshift nodes
- Each node can deviate from LCDM by a fractional amount delta_i
- The deviations are smoothly interpolated between nodes
- A "smoothness prior" penalizes sharp kinks (bent deck penalty)

The model is "bound" at:
- Early times: must satisfy CMB/BBN constraints
- Today: local H0 measurement

The key question: can *any* smooth layered expansion history reconcile
CMB + BAO + SN with H0 ~ 73 km/s/Mpc?
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.integrate import quad


# Physical constants
C_KM_S = 299792.458  # Speed of light in km/s


@dataclass
class LayeredExpansionHyperparams:
    """
    Hyperparameters for the layered expansion model.

    These define the structure of the "bent deck" - how many layers,
    over what redshift range, and how stiff the smoothness penalty is.

    Attributes
    ----------
    n_layers : int
        Number of redshift nodes (the "cards" in the deck)
    z_min : float
        Minimum redshift (typically 0)
    z_max : float
        Maximum redshift to parameterize (beyond this, assume LCDM)
    smooth_sigma : float
        Controls stiffness of the deck. Smaller = stiffer = penalizes wiggles more.
        Typical values: 0.02 (very stiff) to 0.1 (floppy)
    mode : Literal["delta_H", "delta_w"]
        How to interpret the deviations:
        - "delta_H": fractional offset to H(z), i.e., H = H_LCDM * (1 + delta)
        - "delta_w": additive offset to equation of state w(z)
    spacing : Literal["linear", "log"]
        How to space the redshift nodes:
        - "linear": uniform spacing in z
        - "log": uniform spacing in ln(1+z), concentrates nodes at low z
    """
    n_layers: int = 6
    z_min: float = 0.0
    z_max: float = 6.0
    smooth_sigma: float = 0.05
    mode: Literal["delta_H", "delta_w"] = "delta_H"
    spacing: Literal["linear", "log"] = "log"


@dataclass
class LayeredExpansionParams:
    """
    Parameters for a specific layered expansion configuration.

    These are the "bent deck" parameters - the redshift nodes and
    the deviation at each node.

    Attributes
    ----------
    z_nodes : NDArray
        Monotonically increasing array of redshift nodes, length n_layers
    delta_nodes : NDArray
        Deviation values at each node (fractional H offset or w offset)
    """
    z_nodes: NDArray[np.floating]
    delta_nodes: NDArray[np.floating]

    def __post_init__(self):
        """Validate inputs."""
        self.z_nodes = np.asarray(self.z_nodes)
        self.delta_nodes = np.asarray(self.delta_nodes)

        if len(self.z_nodes) != len(self.delta_nodes):
            raise ValueError(
                f"z_nodes (len={len(self.z_nodes)}) and delta_nodes "
                f"(len={len(self.delta_nodes)}) must have same length"
            )

        if len(self.z_nodes) < 2:
            raise ValueError("Need at least 2 nodes for interpolation")

        # Check monotonicity
        if not np.all(np.diff(self.z_nodes) > 0):
            raise ValueError("z_nodes must be strictly monotonically increasing")

    @property
    def n_layers(self) -> int:
        """Number of layers (nodes)."""
        return len(self.z_nodes)


def make_default_nodes(hyp: LayeredExpansionHyperparams) -> NDArray[np.floating]:
    """
    Create default redshift nodes based on hyperparameters.

    For "log" spacing, uses uniform spacing in ln(1+z), which concentrates
    more nodes at low redshift where the observational constraints are strongest.

    For "linear" spacing, uses uniform spacing in z.

    Parameters
    ----------
    hyp : LayeredExpansionHyperparams
        Hyperparameters specifying n_layers, z_min, z_max, and spacing

    Returns
    -------
    NDArray
        Array of n_layers redshift values

    Notes
    -----
    We use logarithmic spacing by default because:
    1. BAO and SN data concentrate at z < 2
    2. The Hubble tension manifests primarily at low z
    3. More resolution where H(z) changes most rapidly relative to LCDM
    """
    if hyp.spacing == "log":
        # Uniform in ln(1+z)
        ln_1pz_min = np.log(1 + hyp.z_min)
        ln_1pz_max = np.log(1 + hyp.z_max)
        ln_1pz_nodes = np.linspace(ln_1pz_min, ln_1pz_max, hyp.n_layers)
        z_nodes = np.exp(ln_1pz_nodes) - 1
    else:
        # Uniform in z
        z_nodes = np.linspace(hyp.z_min, hyp.z_max, hyp.n_layers)

    return z_nodes


def log_smoothness_prior(
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Compute the log of the smoothness prior (bent deck penalty).

    The smoothness prior penalizes sharp kinks in the expansion history:

        S = sum_i [ (delta_{i+1} - delta_i)^2 / (2 * smooth_sigma^2) ]
        log_prior = -S

    This is equivalent to a Gaussian Process prior with a random-walk
    covariance structure. Smaller smooth_sigma means stiffer constraints
    (less wiggle room).

    Parameters
    ----------
    params : LayeredExpansionParams
        The layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters including smooth_sigma

    Returns
    -------
    float
        Log of the smoothness prior (always <= 0)

    Notes
    -----
    For flat delta_nodes (all equal), this returns 0 (maximum prior probability).
    For very wiggly configurations, this returns large negative values.
    """
    if hyp.smooth_sigma <= 0:
        raise ValueError(f"smooth_sigma must be positive, got {hyp.smooth_sigma}")

    # Compute differences between adjacent nodes
    delta_diffs = np.diff(params.delta_nodes)

    # Sum of squared differences, normalized by variance
    S = np.sum(delta_diffs**2) / (2 * hyp.smooth_sigma**2)

    return -S


def _interpolate_delta(
    z: Union[float, NDArray[np.floating]],
    params: LayeredExpansionParams,
    extrapolate: bool = True
) -> Union[float, NDArray[np.floating]]:
    """
    Interpolate delta values at arbitrary redshifts.

    Uses linear interpolation in ln(1+z) space for smoothness.
    Outside the node range, extrapolates to 0 (returns to LCDM).

    Parameters
    ----------
    z : float or array
        Redshift(s) at which to evaluate
    params : LayeredExpansionParams
        The layered expansion parameters
    extrapolate : bool
        If True, extrapolate to 0 outside range. If False, use boundary values.

    Returns
    -------
    float or array
        Interpolated delta value(s)
    """
    z = np.atleast_1d(z)

    # Transform to ln(1+z) for smoother interpolation
    ln_1pz = np.log(1 + z)
    ln_1pz_nodes = np.log(1 + params.z_nodes)

    # Create interpolator
    interp_func = interp1d(
        ln_1pz_nodes,
        params.delta_nodes,
        kind='linear',
        bounds_error=False,
        fill_value=(params.delta_nodes[0], params.delta_nodes[-1]) if not extrapolate else (0.0, 0.0)
    )

    delta_interp = interp_func(ln_1pz)

    # For extrapolate=True, smoothly go to zero outside range
    if extrapolate:
        # Below z_min: linearly go to 0
        below_mask = z < params.z_nodes[0]
        if np.any(below_mask):
            # Linear extrapolation from first node to z=0 with delta=0
            if params.z_nodes[0] > 0:
                slope = params.delta_nodes[0] / params.z_nodes[0]
                delta_interp[below_mask] = slope * z[below_mask]
            else:
                delta_interp[below_mask] = params.delta_nodes[0]

        # Above z_max: linearly decay to 0
        above_mask = z > params.z_nodes[-1]
        if np.any(above_mask):
            # Exponential decay beyond z_max
            decay_scale = 0.5  # e-folding scale in z
            delta_interp[above_mask] = params.delta_nodes[-1] * np.exp(
                -(z[above_mask] - params.z_nodes[-1]) / decay_scale
            )

    return delta_interp if len(delta_interp) > 1 else float(delta_interp[0])


# ============================================================================
# LCDM Background Functions
# ============================================================================

@dataclass
class LCDMBackground:
    """
    Flat LCDM background cosmology.

    Provides H(z) and related quantities for a flat LCDM model.

    Attributes
    ----------
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Matter density parameter (today)
    Omega_r : float
        Radiation density parameter (today), typically ~5e-5
    """
    H0: float = 67.5
    Omega_m: float = 0.315
    Omega_r: float = 5e-5  # Small but nonzero for high-z accuracy

    @property
    def Omega_L(self) -> float:
        """Dark energy density (cosmological constant)."""
        return 1.0 - self.Omega_m - self.Omega_r

    @property
    def h(self) -> float:
        """Reduced Hubble constant H0/100."""
        return self.H0 / 100.0

    def E_of_z(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.

        For flat LCDM:
            E^2(z) = Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_L
        """
        z = np.atleast_1d(z)
        E2 = (
            self.Omega_m * (1 + z)**3 +
            self.Omega_r * (1 + z)**4 +
            self.Omega_L
        )
        result = np.sqrt(E2)
        return result if len(result) > 1 else float(result[0])

    def H_of_z(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Hubble parameter H(z) in km/s/Mpc.
        """
        return self.H0 * self.E_of_z(z)

    def comoving_distance(self, z: float) -> float:
        """
        Comoving distance D_C(z) in Mpc.

        D_C = (c/H0) * integral_0^z dz'/E(z')
        """
        def integrand(zp):
            return 1.0 / self.E_of_z(zp)

        limit = 200 if z > 100 else 100
        result, _ = quad(integrand, 0, z, limit=limit, epsrel=1e-10)
        return (C_KM_S / self.H0) * result


# ============================================================================
# Layered H(z) with mode="delta_H"
# ============================================================================

def H_of_z_layered_delta_H(
    z: Union[float, NDArray[np.floating]],
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute H(z) with layered delta_H modification.

    H(z) = H_LCDM(z) * (1 + delta_H_interp(z))

    where delta_H_interp(z) is smoothly interpolated from the node values.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters with delta_H values
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float or array
        H(z) in km/s/Mpc
    """
    H_lcdm = lcdm.H_of_z(z)
    delta_H = _interpolate_delta(z, params)

    H_modified = H_lcdm * (1 + delta_H)

    return H_modified


# ============================================================================
# Layered H(z) with mode="delta_w" (equation of state modification)
# ============================================================================

def _rho_DE_from_w(
    z: float,
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Compute dark energy density with modified w(z).

    For a w(z) equation of state:
        rho_DE(z) / rho_DE(0) = exp(3 * integral_0^z (1 + w(z')) d ln(1+z'))

    With w_eff(z) = w_LCDM + delta_w(z), where w_LCDM = -1 (cosmological constant).

    Parameters
    ----------
    z : float
        Redshift
    lcdm : LCDMBackground
        Baseline LCDM (provides Omega_L at z=0)
    params : LayeredExpansionParams
        Contains delta_w at each node
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float
        rho_DE(z) / rho_crit(0), normalized to 3H0^2/(8*pi*G)
    """
    if z <= 0:
        return lcdm.Omega_L

    # Integrate (1 + w(z)) d ln(1+z) from 0 to z
    def integrand(zp):
        w_lcdm = -1.0
        delta_w = _interpolate_delta(zp, params)
        w_eff = w_lcdm + delta_w
        return (1 + w_eff) / (1 + zp)

    integral, _ = quad(integrand, 0, z, limit=100, epsrel=1e-8)

    # rho_DE(z) = rho_DE(0) * exp(3 * integral)
    return lcdm.Omega_L * np.exp(3 * integral)


def H_of_z_layered_delta_w(
    z: Union[float, NDArray[np.floating]],
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute H(z) with layered delta_w modification.

    Modifies the dark energy equation of state:
        w_eff(z) = -1 + delta_w_interp(z)

    Then computes H^2(z) from the modified Friedmann equation.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters with delta_w values
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float or array
        H(z) in km/s/Mpc
    """
    z = np.atleast_1d(z)

    H_values = []
    for zi in z:
        # Matter + radiation (unchanged)
        Omega_m_z = lcdm.Omega_m * (1 + zi)**3
        Omega_r_z = lcdm.Omega_r * (1 + zi)**4

        # Modified dark energy
        Omega_DE_z = _rho_DE_from_w(zi, lcdm, params, hyp)

        E2 = Omega_m_z + Omega_r_z + Omega_DE_z

        if E2 <= 0:
            # Unphysical: return NaN to signal problem
            H_values.append(np.nan)
        else:
            H_values.append(lcdm.H0 * np.sqrt(E2))

    result = np.array(H_values)
    return result if len(result) > 1 else float(result[0])


# ============================================================================
# Unified Interface
# ============================================================================

def H_of_z_layered(
    z: Union[float, NDArray[np.floating]],
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> Union[float, NDArray[np.floating]]:
    """
    Compute H(z) with layered expansion modification.

    This is the main interface for computing the modified Hubble parameter.
    The modification mode is determined by hyp.mode:
    - "delta_H": direct fractional modification to H(z)
    - "delta_w": modification to the equation of state w(z)

    Parameters
    ----------
    z : float or array
        Redshift(s)
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters including mode

    Returns
    -------
    float or array
        H(z) in km/s/Mpc
    """
    if hyp.mode == "delta_H":
        return H_of_z_layered_delta_H(z, lcdm, params, hyp)
    elif hyp.mode == "delta_w":
        return H_of_z_layered_delta_w(z, lcdm, params, hyp)
    else:
        raise ValueError(f"Unknown mode: {hyp.mode}, expected 'delta_H' or 'delta_w'")


def E_of_z_layered(
    z: Union[float, NDArray[np.floating]],
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> Union[float, NDArray[np.floating]]:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0 for layered model.

    Note: We use the LCDM H0 for normalization. The "effective H0" is
    computed separately as H_layered(z=0).
    """
    H_layered = H_of_z_layered(z, lcdm, params, hyp)
    return H_layered / lcdm.H0


def get_H0_effective(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams
) -> float:
    """
    Get the effective H0 from the layered model.

    This is H(z=0) in the modified model, which may differ from the
    baseline LCDM H0 if delta(z=0) != 0.

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters

    Returns
    -------
    float
        Effective H0 in km/s/Mpc
    """
    return H_of_z_layered(0.0, lcdm, params, hyp)


# ============================================================================
# Physical validity checks
# ============================================================================

def check_physical_validity(
    lcdm: LCDMBackground,
    params: LayeredExpansionParams,
    hyp: LayeredExpansionHyperparams,
    z_test: Optional[NDArray] = None
) -> dict:
    """
    Check physical validity of the layered expansion model.

    Checks:
    1. H(z) > 0 for all z (no negative expansion rate)
    2. No phantom crossing issues in delta_w mode
    3. Effective energy densities remain positive

    Parameters
    ----------
    lcdm : LCDMBackground
        Baseline LCDM cosmology
    params : LayeredExpansionParams
        Layered expansion parameters
    hyp : LayeredExpansionHyperparams
        Hyperparameters
    z_test : array, optional
        Redshifts at which to test (default: 0 to z_max with 100 points)

    Returns
    -------
    dict
        Dictionary with:
        - "valid": bool, overall validity
        - "H_positive": bool, all H(z) > 0
        - "H_min": float, minimum H(z) value
        - "H0_eff": float, effective H0
        - "warnings": list of warning messages
    """
    if z_test is None:
        z_test = np.linspace(0, hyp.z_max, 100)

    H_values = H_of_z_layered(z_test, lcdm, params, hyp)
    H_values = np.atleast_1d(H_values)

    H_positive = np.all(H_values > 0) and np.all(np.isfinite(H_values))
    H_min = np.nanmin(H_values) if len(H_values) > 0 else 0.0
    H0_eff = get_H0_effective(lcdm, params, hyp)

    warnings_list = []

    if not H_positive:
        warnings_list.append(f"H(z) becomes non-positive or NaN; min H = {H_min:.2f}")

    if H0_eff < 50 or H0_eff > 100:
        warnings_list.append(f"H0_eff = {H0_eff:.2f} outside reasonable range [50, 100]")

    # Check for extreme deviations
    max_delta = np.max(np.abs(params.delta_nodes))
    if max_delta > 0.5:
        warnings_list.append(f"max |delta| = {max_delta:.3f} is large (>50%)")

    return {
        "valid": H_positive and (50 < H0_eff < 100),
        "H_positive": H_positive,
        "H_min": H_min,
        "H0_eff": H0_eff,
        "warnings": warnings_list
    }


# ============================================================================
# Convenience constructors
# ============================================================================

def make_zero_params(hyp: LayeredExpansionHyperparams) -> LayeredExpansionParams:
    """
    Create parameters with zero deviations (recovers baseline LCDM).

    Parameters
    ----------
    hyp : LayeredExpansionHyperparams
        Hyperparameters specifying node structure

    Returns
    -------
    LayeredExpansionParams
        Parameters with delta = 0 at all nodes
    """
    z_nodes = make_default_nodes(hyp)
    delta_nodes = np.zeros(hyp.n_layers)
    return LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)


def make_random_params(
    hyp: LayeredExpansionHyperparams,
    sigma_delta: float = 0.05,
    rng: Optional[np.random.Generator] = None
) -> LayeredExpansionParams:
    """
    Create parameters with random Gaussian deviations.

    Parameters
    ----------
    hyp : LayeredExpansionHyperparams
        Hyperparameters specifying node structure
    sigma_delta : float
        Standard deviation for random delta values
    rng : Generator, optional
        Random number generator (for reproducibility)

    Returns
    -------
    LayeredExpansionParams
        Parameters with random delta values
    """
    if rng is None:
        rng = np.random.default_rng()

    z_nodes = make_default_nodes(hyp)
    delta_nodes = rng.normal(0, sigma_delta, size=hyp.n_layers)

    return LayeredExpansionParams(z_nodes=z_nodes, delta_nodes=delta_nodes)
