"""
Holographic Recycling Cosmology (HRC) - Theoretical Foundation

This module implements the Lagrangian formulation and field equations for HRC,
a speculative cosmological model where:
- Black hole evaporation partially recycles into Planck-mass remnants
- These remnants may constitute dark matter
- A scalar "recycling field" φ mediates the process
- Modified Friedmann equations arise from the new physics

Units: Natural units with ℏ = c = 1, but G kept explicit.
Metric signature: (-,+,+,+)

Author: HRC Theory Development
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import sympy as sp
from sympy import symbols, Function, sqrt, diff, simplify, expand
from sympy import Matrix, eye, diag, Rational, pi, exp, cos, sin
from sympy.diffgeom import Manifold, Patch, CoordSystem, metric_to_Christoffel_2nd
from sympy.diffgeom import metric_to_Riemann_components, metric_to_Ricci_components


# =============================================================================
# PART A: ACTION DEFINITION
# =============================================================================

class ActionComponents:
    """
    Symbolic representation of the HRC action components.

    The total action is:
        S = S_EH + S_m + S_recycle + S_rem

    where:
        S_EH = (1/16πG) ∫ d⁴x √(-g) (R - 2Λ)           [Einstein-Hilbert]
        S_m = -∫ d⁴x √(-g) ρ(1 + Π)                     [Matter]
        S_recycle = ∫ d⁴x √(-g) L_φ                     [Recycling field]
        S_rem = -∫ d⁴x √(-g) ρ_rem(1 + αφ)             [Remnants]

    The recycling field Lagrangian density is:
        L_φ = -½ g^μν ∂_μφ ∂_νφ - V(φ) - ξφR - λφn_rem
    """

    def __init__(self):
        """Initialize symbolic variables for the action."""
        # Coordinates
        self.t, self.x, self.y, self.z = symbols('t x y z', real=True)

        # Scale factor (for FLRW)
        self.a = Function('a')(self.t)

        # Fields
        self.phi = Function('phi')(self.t)  # Recycling scalar field (homogeneous)

        # Densities
        self.rho_m = Function('rho_m')(self.t)      # Matter density
        self.rho_rem = Function('rho_rem')(self.t)  # Remnant density
        self.n_rem = Function('n_rem')(self.t)      # Remnant number density
        self.p_m = Function('p_m')(self.t)          # Matter pressure

        # Parameters
        self.G = symbols('G', positive=True)           # Newton's constant
        self.Lambda = symbols('Lambda', real=True)     # Cosmological constant
        self.xi = symbols('xi', real=True)             # Non-minimal coupling
        self.lambda_r = symbols('lambda_r', real=True) # Recycling coupling
        self.alpha = symbols('alpha', real=True)       # Remnant-field coupling
        self.m_phi = symbols('m_phi', positive=True)   # Scalar field mass

        # Derived quantities
        self.H = diff(self.a, self.t) / self.a  # Hubble parameter

    def potential_V(self, phi_val=None):
        """
        Scalar field potential V(φ).

        Default: Quadratic potential V = m²φ²/2

        Physical interpretation:
        - m_phi sets the mass scale for the recycling field
        - For m_phi ~ H_0, field evolves on cosmological timescales
        - Potential minimum at φ=0 (no recycling in vacuum)
        """
        phi = phi_val if phi_val is not None else self.phi
        return Rational(1, 2) * self.m_phi**2 * phi**2

    def dV_dphi(self, phi_val=None):
        """Derivative of potential: dV/dφ = m²φ"""
        phi = phi_val if phi_val is not None else self.phi
        return self.m_phi**2 * phi

    def lagrangian_EH(self, R):
        """
        Einstein-Hilbert Lagrangian density (without √(-g)).

        L_EH = (1/16πG)(R - 2Λ)

        Returns the Lagrangian density.
        """
        return (R - 2*self.Lambda) / (16 * pi * self.G)

    def lagrangian_matter(self, rho, Pi):
        """
        Matter Lagrangian density.

        L_m = -ρ(1 + Π)

        where Π is the specific internal energy, related to pressure via:
        p = ρ²(∂Π/∂ρ) for adiabatic processes

        For dust (Π=0): L_m = -ρ
        For radiation (Π = 3p/ρ): includes internal energy contribution
        """
        return -rho * (1 + Pi)

    def lagrangian_recycling(self, kinetic_term, R):
        """
        Recycling field Lagrangian density.

        L_φ = -½ g^μν ∂_μφ ∂_νφ - V(φ) - ξφR - λφn_rem

        Parameters:
        -----------
        kinetic_term : symbolic
            The quantity g^μν ∂_μφ ∂_νφ (computed from metric)
        R : symbolic
            Ricci scalar

        Physical interpretation:
        -----------------------
        - Kinetic term: standard scalar field dynamics
        - V(φ): potential energy, drives field toward minimum
        - ξφR: non-minimal coupling to gravity (like Brans-Dicke)
        - λφn_rem: direct coupling to remnant population

        The ξφR term is crucial:
        - ξ = 0: minimal coupling (standard scalar field)
        - ξ = 1/6: conformal coupling in 4D
        - Other values: modify effective gravitational strength
        """
        V = self.potential_V()
        return (-Rational(1, 2) * kinetic_term
                - V
                - self.xi * self.phi * R
                - self.lambda_r * self.phi * self.n_rem)

    def lagrangian_remnant(self):
        """
        Remnant Lagrangian density.

        L_rem = -ρ_rem(1 + αφ)

        Physical interpretation:
        -----------------------
        The remnants are modeled as:
        - Planck-mass objects (M ~ M_Planck)
        - Large interior volume (holographic storage)
        - Effectively pressureless (dust-like)
        - Interact with recycling field via αφ coupling

        The αφ term means:
        - Remnant effective mass varies with φ
        - M_eff = M_Planck(1 + αφ)
        - Positive α: remnants heavier when φ > 0
        - This creates feedback: more recycling → heavier remnants

        IMPORTANT: For energy conservation, this coupling induces
        an effective pressure in the remnant sector.
        """
        return -self.rho_rem * (1 + self.alpha * self.phi)


# =============================================================================
# PART B: FIELD EQUATIONS (Symbolic Derivation)
# =============================================================================

class FieldEquations:
    """
    Derives field equations from the HRC action.

    Variation procedure:
    1. δS/δg^μν = 0 → Modified Einstein equations
    2. δS/δφ = 0 → Scalar field equation
    3. Bianchi identity → Conservation equations
    """

    def __init__(self, action: ActionComponents):
        self.action = action
        self._derive_einstein_equations()
        self._derive_scalar_equation()

    def _derive_einstein_equations(self):
        """
        Derive modified Einstein equations from δS/δg^μν = 0.

        The variation of each action component contributes to T_μν:

        G_μν + Λg_μν = 8πG (T^m_μν + T^φ_μν + T^rem_μν)

        where:

        T^m_μν = (ρ + p)u_μu_ν + p g_μν        [Perfect fluid]

        T^φ_μν = ∂_μφ∂_νφ - g_μν[½(∂φ)² + V]   [Scalar field]
                 + ξ(g_μν□ - ∇_μ∇_ν + G_μν)φ   [Non-minimal coupling]

        T^rem_μν = ρ_rem(1+αφ) u_μu_ν          [Dust remnants]

        NOTE: The non-minimal coupling ξφR modifies both left and right
        sides of Einstein equations. It's conventional to absorb this into
        an effective stress-energy tensor.
        """
        self.einstein_eq_description = """
        Modified Einstein Equations for HRC:

        G_μν + Λg_μν = 8πG T^(total)_μν

        where T^(total) = T^m + T^φ + T^rem with:

        1. Matter (perfect fluid):
           T^m_μν = (ρ_m + p_m)u_μu_ν + p_m g_μν

        2. Recycling field (with non-minimal coupling):
           T^φ_μν = ∂_μφ∂_νφ - ½g_μν(∂φ)² - g_μν V(φ)
                    + ξ[G_μν φ + g_μν □φ - ∇_μ∇_ν φ]
                    + ½g_μν λφn_rem - ...

           Note: Full expression includes connection terms

        3. Remnants:
           T^rem_μν = ρ_rem(1 + αφ)u_μu_ν

        The ξφR term is handled by moving it to effective T_μν:

        (1 - 8πGξφ)G_μν = 8πG[T^rest_μν - ξ(g_μν□ - ∇_μ∇_ν)φ] + Λg_μν

        This defines an effective Newton's constant:
        G_eff = G / (1 - 8πGξφ)

        PHYSICAL CONSTRAINT: Require 1 - 8πGξφ > 0 to maintain
        attractive gravity (or allow repulsive phase if desired).
        """

    def _derive_scalar_equation(self):
        """
        Derive scalar field equation from δS/δφ = 0.

        Varying the action with respect to φ:

        δS_recycle/δφ = √(-g)[□φ - dV/dφ - ξR - λn_rem]
        δS_rem/δφ = -√(-g) α ρ_rem

        Field equation:
        □φ - dV/dφ - ξR = λn_rem + αρ_rem

        In component form:
        g^μν∇_μ∇_νφ - m²φ - ξR = λn_rem + αρ_rem

        Physical interpretation:
        - □φ: wave operator (propagation)
        - m²φ: mass term (oscillations)
        - ξR: curvature coupling (sourced by spacetime geometry)
        - λn_rem: direct coupling to remnant count
        - αρ_rem: coupling to remnant energy density

        The source terms create the "recycling" effect:
        - More remnants → larger φ → modified expansion
        - Modified expansion → changed remnant production
        - This feedback loop is the core of HRC
        """
        self.scalar_eq_description = """
        Scalar Field Equation:

        □φ - m²φ - ξR = λn_rem + αρ_rem

        Or equivalently:
        □φ = m²φ + ξR + λn_rem + αρ_rem

        In FLRW (flat, homogeneous φ):
        φ̈ + 3Hφ̇ + m²φ + ξR = λn_rem + αρ_rem

        where R = 6(2H² + Ḣ) for flat FLRW.
        """


# =============================================================================
# PART C: FLRW REDUCTION
# =============================================================================

class FLRWCosmology:
    """
    Specialization of HRC to flat FLRW cosmology.

    Metric: ds² = -dt² + a(t)²(dx² + dy² + dz²)

    Assumptions:
    - Homogeneity and isotropy
    - Flat spatial sections (k=0)
    - All fields depend only on t
    - Matter and remnants comoving (u^μ = (1,0,0,0))
    """

    def __init__(self, action: ActionComponents):
        self.action = action
        self.t = action.t
        self.a = action.a
        self.phi = action.phi

        # Compute FLRW geometric quantities
        self._compute_flrw_geometry()

    def _compute_flrw_geometry(self):
        """
        Compute geometric quantities for flat FLRW.

        For ds² = -dt² + a²(dx² + dy² + dz²):

        Non-zero Christoffel symbols:
        Γ^0_ij = aȧ δ_ij
        Γ^i_0j = (ȧ/a) δ^i_j = H δ^i_j

        Ricci tensor:
        R_00 = -3(Ḧ + H²) = -3ä/a
        R_ij = a²(ä/a + 2H²) δ_ij

        Ricci scalar:
        R = 6(ä/a + H²) = 6(2H² + Ḣ)

        Note: Ḣ = ä/a - H² so ä/a = Ḣ + H²
        """
        t = self.t
        a = self.a

        # Hubble parameter and its derivative
        self.H = diff(a, t) / a
        self.H_dot = diff(self.H, t)

        # Acceleration
        self.a_ddot = diff(a, t, 2)
        self.a_ddot_over_a = self.a_ddot / a

        # Ricci scalar
        # R = 6(ä/a + H²) = 6(Ḣ + 2H²)
        self.R = 6 * (self.a_ddot_over_a + self.H**2)
        self.R_alt = 6 * (self.H_dot + 2*self.H**2)  # Alternative form

        # Store metric components
        self.g_00 = -1
        self.g_ij = a**2  # Spatial components (diagonal)

        # d'Alembertian of scalar in FLRW (homogeneous field)
        # □φ = -φ̈ - 3Hφ̇ (with our signature)
        # Note: With (-,+,+,+) signature and homogeneous φ(t):
        # □φ = g^μν∇_μ∇_νφ = -∂_t²φ - 3H∂_tφ
        phi = self.phi
        self.phi_dot = diff(phi, t)
        self.phi_ddot = diff(phi, t, 2)
        self.box_phi = -self.phi_ddot - 3*self.H*self.phi_dot

    def modified_friedmann_1(self):
        """
        Derive the modified first Friedmann equation from G_00 component.

        Standard: H² = (8πG/3)ρ + Λ/3

        Modified (HRC):

        H² = (8πG/3) × [ρ_m + ρ_φ + ρ_rem(1+αφ)] / (1 - 8πGξφ) + Λ/3

        where the scalar field energy density is:
        ρ_φ = ½φ̇² + V(φ) + 3ξHφ̇ + ½λφn_rem

        DERIVATION:
        -----------
        The 00-component of Einstein equations:
        G_00 = 3H² (in FLRW)

        From T^(total)_00:
        - T^m_00 = ρ_m (comoving)
        - T^φ_00 = ½φ̇² + V + ξ(G_00φ + 3Hφ̇) + ½λφn_rem
        - T^rem_00 = ρ_rem(1+αφ)

        The ξφG_00 term couples to geometry, giving effective G:
        (1 - 8πGξφ)G_00 = 8πG(T^rest_00 + ξ·3Hφ̇) + Λg_00

        Solving for H²:
        H² = [8πG(ρ_m + ρ_φ^eff + ρ_rem(1+αφ)) + Λ] / [3(1 - 8πGξφ)]
        """
        G = self.action.G
        Lambda = self.action.Lambda
        xi = self.action.xi
        lambda_r = self.action.lambda_r
        alpha = self.action.alpha
        rho_m = self.action.rho_m
        rho_rem = self.action.rho_rem
        n_rem = self.action.n_rem
        phi = self.phi

        # Effective scalar field energy density
        rho_phi = (Rational(1,2) * self.phi_dot**2
                   + self.action.potential_V(phi)
                   + 3*xi*self.H*self.phi_dot*phi  # From non-minimal coupling
                   + Rational(1,2)*lambda_r*phi*n_rem)

        # Effective remnant density (includes coupling)
        rho_rem_eff = rho_rem * (1 + alpha*phi)

        # Effective Newton's constant factor
        G_eff_factor = 1 - 8*pi*G*xi*phi

        # Total effective density
        rho_total = rho_m + rho_phi + rho_rem_eff

        # First Friedmann equation
        # H² = (8πG ρ_total + Λ) / (3 × G_eff_factor)
        H_squared = (8*pi*G*rho_total + Lambda) / (3 * G_eff_factor)

        return {
            'H_squared': H_squared,
            'rho_phi': rho_phi,
            'rho_rem_eff': rho_rem_eff,
            'G_eff_factor': G_eff_factor,
            'equation': sp.Eq(self.H**2, H_squared)
        }

    def modified_friedmann_2(self):
        """
        Derive the modified second Friedmann equation (acceleration equation).

        Standard: ä/a = -(4πG/3)(ρ + 3p) + Λ/3

        Modified (HRC):

        ä/a = -[(4πG/3)(ρ_eff + 3p_eff) - Λ/3] / (1 - 8πGξφ)
              + correction terms from ξ coupling

        The scalar field contributes effective pressure:
        p_φ = ½φ̇² - V(φ) - ξ(2ä/a + H² + 2Ḣ)φ - ξφ̈ - 2ξHφ̇ - ½λφn_rem

        Note: Remnants are dust-like but coupling to φ induces effective pressure.

        DERIVATION:
        -----------
        Using the trace-reversed Einstein equations and spatial components.
        Alternatively, from first Friedmann + energy conservation.
        """
        G = self.action.G
        Lambda = self.action.Lambda
        xi = self.action.xi
        lambda_r = self.action.lambda_r
        alpha = self.action.alpha
        rho_m = self.action.rho_m
        p_m = self.action.p_m
        rho_rem = self.action.rho_rem
        n_rem = self.action.n_rem
        phi = self.phi

        # Scalar field pressure (complicated due to non-minimal coupling)
        # For minimal coupling (ξ=0): p_φ = ½φ̇² - V
        p_phi_minimal = Rational(1,2)*self.phi_dot**2 - self.action.potential_V(phi)

        # Non-minimal coupling corrections to pressure
        # These come from the spatial components of T^φ_μν
        p_phi_nm = -xi * (2*self.H_dot + 3*self.H**2) * phi
        p_phi_nm += -xi * self.phi_ddot - 2*xi*self.H*self.phi_dot

        p_phi = p_phi_minimal + p_phi_nm - Rational(1,2)*lambda_r*phi*n_rem

        # Remnants: dust (p=0) but coupling gives effective contribution
        # The αφρ_rem term in action gives effective pressure
        p_rem_eff = alpha * phi * rho_rem * diff(phi, self.t) / (3*self.H)
        # Actually, more careful: this needs derivation from conservation
        # For now, set to zero and note in documentation
        p_rem_eff = 0

        # Total pressure
        p_total = p_m + p_phi + p_rem_eff

        # Effective factor
        G_eff_factor = 1 - 8*pi*G*xi*phi

        # Get rho from first Friedmann
        friedmann1 = self.modified_friedmann_1()
        rho_total = rho_m + friedmann1['rho_phi'] + friedmann1['rho_rem_eff']

        # Second Friedmann equation
        # ä/a = -(4πG/3)(ρ + 3p)/G_eff_factor + Λ/3
        a_ddot_over_a = (-(4*pi*G/3)*(rho_total + 3*p_total) + Lambda/3) / G_eff_factor

        return {
            'a_ddot_over_a': a_ddot_over_a,
            'p_phi': p_phi,
            'p_total': p_total,
            'equation': sp.Eq(self.a_ddot/self.a, a_ddot_over_a)
        }

    def scalar_field_equation(self):
        """
        Scalar field evolution equation in FLRW.

        □φ - m²φ - ξR = λn_rem + αρ_rem

        In FLRW with signature (-,+,+,+):
        □φ = -φ̈ - 3Hφ̇
        R = 6(Ḣ + 2H²)

        So:
        φ̈ + 3Hφ̇ + m²φ + 6ξ(Ḣ + 2H²) = -λn_rem - αρ_rem

        Or:
        φ̈ = -3Hφ̇ - m²φ - 6ξ(Ḣ + 2H²) - λn_rem - αρ_rem

        Physical interpretation:
        - 3Hφ̇: Hubble friction (expansion dilutes kinetic energy)
        - m²φ: restoring force toward φ=0
        - ξR: curvature drives φ (positive R → negative source)
        - λn_rem, αρ_rem: remnants source the field

        IMPORTANT NOTE:
        The sign conventions matter here. With our action:
        S_φ ∝ ∫[-½(∂φ)² - V - ξφR - λφn_rem]

        The equation of motion is:
        □φ = ∂V/∂φ + ξR + λn_rem (from δS_φ/δφ = 0)

        Adding the remnant action contribution (δS_rem/δφ = -αρ_rem):
        □φ = m²φ + ξR + λn_rem + αρ_rem
        """
        m_phi = self.action.m_phi
        xi = self.action.xi
        lambda_r = self.action.lambda_r
        alpha = self.action.alpha
        n_rem = self.action.n_rem
        rho_rem = self.action.rho_rem
        phi = self.phi

        # Source terms
        source = (m_phi**2 * phi
                  + xi * self.R
                  + lambda_r * n_rem
                  + alpha * rho_rem)

        # φ̈ + 3Hφ̇ = source (since □φ = -φ̈ - 3Hφ̇ with our convention)
        phi_ddot_expr = -3*self.H*self.phi_dot + source

        return {
            'phi_ddot': phi_ddot_expr,
            'equation': sp.Eq(self.phi_ddot, phi_ddot_expr),
            'source': source
        }

    def conservation_equations(self):
        """
        Energy conservation equations from ∇_μT^μν = 0.

        In FLRW, the 0-component gives continuity equations:

        1. Matter (standard):
           ρ̇_m + 3H(ρ_m + p_m) = 0

        2. Remnants (modified by φ coupling):
           ρ̇_rem + 3Hρ_rem = -αρ_rem φ̇

           Interpretation: remnant energy is not separately conserved
           when coupled to φ. Energy flows between φ and remnant sectors.

        3. Scalar field (modified by couplings):
           The scalar field equation itself encodes conservation.

        4. Total energy (Bianchi identity):
           ∇_μT^μν_total = 0 is guaranteed by Bianchi identity
           applied to Einstein equations.

        DERIVATION for remnants:
        From S_rem = -∫√(-g)ρ_rem(1+αφ), varying and using
        covariant conservation, we get modified continuity.
        """
        H = self.H
        rho_m = self.action.rho_m
        p_m = self.action.p_m
        rho_rem = self.action.rho_rem
        alpha = self.action.alpha
        phi = self.phi

        # Matter continuity (standard)
        rho_m_dot = -3*H*(rho_m + p_m)

        # Remnant continuity (modified)
        # From coupled system: ρ̇_rem(1+αφ) + 3Hρ_rem(1+αφ) + αρ_rem·φ̇ = 0
        # Simplifies to: ρ̇_rem + 3Hρ_rem = -αρ_rem·φ̇·(1+αφ)⁻¹
        # For small αφ: ρ̇_rem ≈ -3Hρ_rem - αρ_rem·φ̇
        rho_rem_dot = -3*H*rho_rem - alpha*rho_rem*self.phi_dot

        return {
            'matter_continuity': sp.Eq(diff(rho_m, self.t), rho_m_dot),
            'remnant_continuity': sp.Eq(diff(rho_rem, self.t), rho_rem_dot),
            'rho_m_dot': rho_m_dot,
            'rho_rem_dot': rho_rem_dot
        }


# =============================================================================
# PART D: NUMERICAL IMPLEMENTATION
# =============================================================================

@dataclass
class HRCParameters:
    """
    Parameter container for HRC model.

    Physical parameter ranges for sensible behavior:

    - G: Newton's constant = 6.674e-11 m³/(kg·s²)
      In natural units with M_Planck = 1: G = 1

    - Lambda: Cosmological constant ~ 10⁻¹²² M_Planck⁴ (observed)
      In units where H₀ = 1: Λ ~ 3Ω_Λ where Ω_Λ ~ 0.7

    - xi: Non-minimal coupling
      ξ = 0: minimal coupling (standard scalar)
      ξ = 1/6: conformal coupling
      |ξ| << 1/(8πGφ) required to keep G_eff > 0

    - lambda_r: Recycling coupling
      Sets strength of φ-remnant number coupling
      Dimensionally: [λ] = energy

    - alpha: Remnant-field coupling (dimensionless)
      |α| << 1 for perturbative regime

    - m_phi: Scalar field mass
      m ~ H₀ for cosmologically relevant dynamics
      Larger m: faster oscillations, less cosmological impact
    """
    G: float = 1.0               # Natural units
    Lambda: float = 0.0          # Cosmological constant
    xi: float = 0.0              # Non-minimal coupling
    lambda_r: float = 0.0        # Recycling coupling
    alpha: float = 0.0           # Remnant-field coupling
    m_phi: float = 1.0           # Scalar field mass

    def to_dict(self) -> dict:
        return {
            'G': self.G,
            'Lambda': self.Lambda,
            'xi': self.xi,
            'lambda_r': self.lambda_r,
            'alpha': self.alpha,
            'm_phi': self.m_phi
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'HRCParameters':
        return cls(**d)

    def is_valid(self) -> Tuple[bool, str]:
        """Check if parameters are physically sensible."""
        if self.G <= 0:
            return False, "G must be positive"
        if self.m_phi < 0:
            return False, "m_phi must be non-negative"
        return True, "OK"


class HRCTheory:
    """
    Main class implementing HRC theoretical framework.

    Provides both symbolic derivations and numerical computations
    for the modified cosmological equations.

    Usage:
    ------
    >>> params = HRCParameters(G=1, Lambda=0.7, xi=0.01,
    ...                        lambda_r=0.1, alpha=0.05, m_phi=1.0)
    >>> theory = HRCTheory(params)
    >>> H2 = theory.friedmann_H_squared(a=1.0, rho_m=0.3, rho_rem=0.1,
    ...                                  phi=0.1, dphi_dt=0.01)
    """

    def __init__(self, params: HRCParameters):
        """
        Initialize HRC theory with given parameters.

        Parameters
        ----------
        params : HRCParameters
            Physical parameters of the model

        Raises
        ------
        ValueError
            If parameters are unphysical
        """
        valid, msg = params.is_valid()
        if not valid:
            raise ValueError(f"Invalid parameters: {msg}")

        self.params = params
        self.G = params.G
        self.Lambda = params.Lambda
        self.xi = params.xi
        self.lambda_r = params.lambda_r
        self.alpha = params.alpha
        self.m_phi = params.m_phi

        # Initialize symbolic framework
        self._action = ActionComponents()
        self._flrw = FLRWCosmology(self._action)
        self._field_eqs = FieldEquations(self._action)

    # -------------------------------------------------------------------------
    # Geometric quantities
    # -------------------------------------------------------------------------

    def ricci_scalar_flrw(self, H: float, H_dot: float) -> float:
        """
        Compute Ricci scalar for flat FLRW.

        R = 6(Ḣ + 2H²)

        Parameters
        ----------
        H : float
            Hubble parameter
        H_dot : float
            Time derivative of H

        Returns
        -------
        float
            Ricci scalar R
        """
        return 6.0 * (H_dot + 2.0 * H**2)

    def einstein_tensor_flrw(self, H: float, H_dot: float) -> Dict[str, float]:
        """
        Compute Einstein tensor components for flat FLRW.

        Non-zero components:
        G_00 = 3H²
        G_ij = -(2Ḣ + 3H²)g_ij  (spatial, diagonal)

        Parameters
        ----------
        H : float
            Hubble parameter
        H_dot : float
            Time derivative of H

        Returns
        -------
        dict
            Dictionary with 'G_00' and 'G_ii' (spatial diagonal)
        """
        G_00 = 3.0 * H**2
        G_ii = -(2.0 * H_dot + 3.0 * H**2)  # Note: this is G^i_i not G_ij

        return {
            'G_00': G_00,
            'G_ii_mixed': G_ii,
            'trace': -G_00 + 3*G_ii  # G = g^μν G_μν
        }

    # -------------------------------------------------------------------------
    # Stress-energy tensors
    # -------------------------------------------------------------------------

    def stress_energy_matter(self, rho: float, p: float) -> Dict[str, float]:
        """
        Compute stress-energy tensor for perfect fluid (comoving).

        T^m_μν = (ρ + p)u_μu_ν + p g_μν

        For comoving fluid (u^μ = (1,0,0,0)):
        T^m_00 = ρ
        T^m_ij = p g_ij

        Parameters
        ----------
        rho : float
            Energy density
        p : float
            Pressure

        Returns
        -------
        dict
            Stress-energy components
        """
        return {
            'T_00': rho,
            'T_ii': p,  # Spatial diagonal (in comoving coords)
            'rho': rho,
            'p': p,
            'w': p / rho if rho != 0 else 0  # Equation of state
        }

    def stress_energy_recycling(self, phi: float, dphi_dt: float,
                                 H: float, H_dot: float,
                                 d2phi_dt2: float,
                                 n_rem: float = 0.0) -> Dict[str, float]:
        """
        Compute stress-energy tensor for recycling field.

        For a scalar field with non-minimal coupling:

        T^φ_00 = ½φ̇² + V(φ) + 3ξHφφ̇ + ½λφn_rem
        T^φ_ii = ½φ̇² - V(φ) + ξ-terms + ... (pressure-like)

        Effective density and pressure:
        ρ_φ = T^φ_00
        p_φ = T^φ_ii / 3  (isotropic)

        Parameters
        ----------
        phi : float
            Field value
        dphi_dt : float
            Field time derivative
        H : float
            Hubble parameter
        H_dot : float
            Hubble parameter derivative
        d2phi_dt2 : float
            Field second derivative
        n_rem : float
            Remnant number density (optional)

        Returns
        -------
        dict
            Stress-energy components and derived quantities
        """
        xi = self.xi
        m = self.m_phi
        lam = self.lambda_r

        # Potential and derivative
        V = 0.5 * m**2 * phi**2
        dV = m**2 * phi

        # Kinetic energy
        kinetic = 0.5 * dphi_dt**2

        # Ricci scalar for non-minimal terms
        R = self.ricci_scalar_flrw(H, H_dot)

        # Energy density (00 component)
        rho_phi = kinetic + V
        rho_phi += 3.0 * xi * H * phi * dphi_dt  # Non-minimal contribution
        rho_phi += 0.5 * lam * phi * n_rem  # Coupling to remnants

        # Pressure (from spatial components)
        # Minimal coupling part
        p_phi = kinetic - V

        # Non-minimal coupling contributions to pressure
        p_phi += -xi * (2.0*H_dot + 3.0*H**2) * phi
        p_phi += -xi * d2phi_dt2 * phi  # ξφ̈ term
        p_phi += -2.0 * xi * H * dphi_dt * phi
        p_phi += -0.5 * lam * phi * n_rem

        return {
            'T_00': rho_phi,
            'T_ii': p_phi,
            'rho': rho_phi,
            'p': p_phi,
            'V': V,
            'kinetic': kinetic,
            'w_eff': p_phi / rho_phi if rho_phi != 0 else 0
        }

    def stress_energy_remnant(self, rho_rem: float, phi: float) -> Dict[str, float]:
        """
        Compute stress-energy tensor for remnants.

        Remnants are dust-like with modified coupling:
        T^rem_00 = ρ_rem(1 + αφ)
        T^rem_ij = 0  (pressureless)

        The effective density includes the field coupling.

        Parameters
        ----------
        rho_rem : float
            Remnant rest-frame density
        phi : float
            Recycling field value

        Returns
        -------
        dict
            Stress-energy components
        """
        rho_eff = rho_rem * (1.0 + self.alpha * phi)

        return {
            'T_00': rho_eff,
            'T_ii': 0.0,
            'rho': rho_eff,
            'p': 0.0,
            'rho_bare': rho_rem,
            'coupling_factor': 1.0 + self.alpha * phi
        }

    # -------------------------------------------------------------------------
    # Modified Friedmann equations
    # -------------------------------------------------------------------------

    def G_effective_factor(self, phi: float) -> float:
        """
        Compute the effective gravitational coupling factor.

        G_eff = G / (1 - 8πGξφ)

        Returns the denominator (1 - 8πGξφ).

        Parameters
        ----------
        phi : float
            Recycling field value

        Returns
        -------
        float
            The factor (1 - 8πGξφ)

        Raises
        ------
        ValueError
            If factor becomes non-positive (gravity becomes repulsive/singular)
        """
        factor = 1.0 - 8.0 * np.pi * self.G * self.xi * phi

        if factor <= 0:
            raise ValueError(
                f"Effective gravitational factor non-positive: {factor}. "
                f"Field value φ={phi} too large for ξ={self.xi}. "
                "This indicates breakdown of perturbative regime."
            )

        return factor

    def friedmann_H_squared(self, a: float, rho_m: float, rho_rem: float,
                            phi: float, dphi_dt: float,
                            n_rem: float = 0.0) -> float:
        """
        Compute H² from the modified first Friedmann equation.

        H² = [8πG(ρ_m + ρ_φ + ρ_rem(1+αφ)) + Λ] / [3(1 - 8πGξφ)]

        Parameters
        ----------
        a : float
            Scale factor (not used directly but included for interface)
        rho_m : float
            Matter energy density
        rho_rem : float
            Remnant energy density (bare, before φ coupling)
        phi : float
            Recycling field value
        dphi_dt : float
            Field time derivative
        n_rem : float
            Remnant number density (for λφn_rem coupling)

        Returns
        -------
        float
            H² (Hubble parameter squared)

        Notes
        -----
        - In standard ΛCDM limit (φ→0, ξ→0, etc.), recovers H² = 8πGρ/3 + Λ/3
        - For n_rem, if not provided separately, estimate as ρ_rem/m_Planck
        """
        G = self.G
        Lambda = self.Lambda
        xi = self.xi
        m = self.m_phi
        lam = self.lambda_r
        alpha = self.alpha

        # Scalar field energy density
        V = 0.5 * m**2 * phi**2
        kinetic = 0.5 * dphi_dt**2

        # We need H for the ξHφφ̇ term, but H² is what we're solving for
        # This creates an implicit equation. For now, neglect this term
        # in the energy density (valid when ξ is small or φ̇ is small)
        # TODO: Implement iterative solution for full non-minimal case

        rho_phi = kinetic + V + 0.5 * lam * phi * n_rem

        # Effective remnant density
        rho_rem_eff = rho_rem * (1.0 + alpha * phi)

        # Total density
        rho_total = rho_m + rho_phi + rho_rem_eff

        # Effective G factor
        G_eff_factor = self.G_effective_factor(phi)

        # First Friedmann equation
        H_squared = (8.0 * np.pi * G * rho_total + Lambda) / (3.0 * G_eff_factor)

        return H_squared

    def friedmann_a_ddot(self, a: float, rho_m: float, p_m: float,
                         rho_rem: float, phi: float, dphi_dt: float,
                         d2phi_dt2: float, n_rem: float = 0.0) -> float:
        """
        Compute ä/a from the modified second Friedmann equation.

        ä/a = -[(4πG/3)(ρ + 3p) - Λ/3] / (1 - 8πGξφ)
              + corrections from non-minimal coupling

        Parameters
        ----------
        a : float
            Scale factor
        rho_m : float
            Matter energy density
        p_m : float
            Matter pressure
        rho_rem : float
            Remnant energy density
        phi : float
            Field value
        dphi_dt : float
            Field first derivative
        d2phi_dt2 : float
            Field second derivative
        n_rem : float
            Remnant number density

        Returns
        -------
        float
            ä/a (acceleration per scale factor)
        """
        G = self.G
        Lambda = self.Lambda
        xi = self.xi
        m = self.m_phi
        lam = self.lambda_r
        alpha = self.alpha

        # Need H and H_dot for full calculation
        # Compute H from first Friedmann
        H_squared = self.friedmann_H_squared(a, rho_m, rho_rem, phi, dphi_dt, n_rem)
        H = np.sqrt(H_squared)

        # Estimate H_dot from energy conservation (approximate)
        # For now, use simplified version

        # Scalar field contributions
        V = 0.5 * m**2 * phi**2
        kinetic = 0.5 * dphi_dt**2

        rho_phi = kinetic + V + 0.5 * lam * phi * n_rem
        p_phi = kinetic - V - 0.5 * lam * phi * n_rem

        # Non-minimal coupling corrections (simplified)
        if xi != 0:
            p_phi += -xi * d2phi_dt2 * phi - 2.0 * xi * H * dphi_dt * phi

        # Remnants (pressureless but with coupling)
        rho_rem_eff = rho_rem * (1.0 + alpha * phi)
        p_rem_eff = 0.0

        # Totals
        rho_total = rho_m + rho_phi + rho_rem_eff
        p_total = p_m + p_phi + p_rem_eff

        # Effective G factor
        G_eff_factor = self.G_effective_factor(phi)

        # Second Friedmann equation
        a_ddot_over_a = (-(4.0*np.pi*G/3.0)*(rho_total + 3.0*p_total) + Lambda/3.0) / G_eff_factor

        return a_ddot_over_a

    def phi_equation_of_motion(self, phi: float, a: float, H: float,
                                rho_rem: float, n_rem: float,
                                dphi_dt: float) -> float:
        """
        Compute d²φ/dt² from the scalar field equation.

        φ̈ = -3Hφ̇ + m²φ + ξR + λn_rem + αρ_rem

        Note the sign: with our conventions, the source terms drive φ away from zero.

        Parameters
        ----------
        phi : float
            Current field value
        a : float
            Scale factor (not directly used)
        H : float
            Hubble parameter
        rho_rem : float
            Remnant energy density
        n_rem : float
            Remnant number density
        dphi_dt : float
            Current field velocity

        Returns
        -------
        float
            d²φ/dt² (field acceleration)

        Notes
        -----
        The Ricci scalar R requires knowledge of H_dot, which requires
        solving the coupled system. Here we use H² from first Friedmann
        and approximate H_dot from continuity.

        For a complete solution, this should be integrated simultaneously
        with the Friedmann equations.
        """
        m = self.m_phi
        xi = self.xi
        lam = self.lambda_r
        alpha = self.alpha

        # Hubble friction
        friction = -3.0 * H * dphi_dt

        # Mass term (restoring force)
        mass_term = -m**2 * phi

        # Estimate R from H (assuming matter + remnant dominated)
        # R ≈ 6(Ḣ + 2H²) ≈ 6(2H² - 3H²/2) = 3H² for dust
        # More careful: R = 12H² - 6ä/a
        R_approx = 6.0 * H**2  # Simplified estimate

        # Curvature coupling
        curvature_term = -xi * R_approx

        # Source from remnants
        remnant_source = lam * n_rem + alpha * rho_rem

        # Total acceleration
        d2phi_dt2 = friction + mass_term + curvature_term + remnant_source

        return d2phi_dt2

    # -------------------------------------------------------------------------
    # Continuity equations
    # -------------------------------------------------------------------------

    def matter_density_dot(self, rho_m: float, p_m: float, H: float) -> float:
        """
        Time derivative of matter density from continuity.

        ρ̇_m = -3H(ρ_m + p_m)

        Standard continuity equation (matter decoupled from φ).
        """
        return -3.0 * H * (rho_m + p_m)

    def remnant_density_dot(self, rho_rem: float, phi: float,
                            dphi_dt: float, H: float) -> float:
        """
        Time derivative of remnant density from modified continuity.

        ρ̇_rem = -3Hρ_rem - αρ_rem·φ̇

        The αφ coupling causes energy transfer between φ and remnant sectors.
        """
        return -3.0 * H * rho_rem - self.alpha * rho_rem * dphi_dt

    # -------------------------------------------------------------------------
    # Limit recovery
    # -------------------------------------------------------------------------

    def standard_friedmann_limit(self, rho: float) -> float:
        """
        Standard Friedmann equation for comparison.

        H² = 8πGρ/3 + Λ/3

        Recovered when φ = 0, ξ = 0, α = 0, λ = 0.
        """
        return (8.0 * np.pi * self.G * rho + self.Lambda) / 3.0

    def verify_lcdm_limit(self, rho_m: float, rho_rem: float) -> Dict[str, float]:
        """
        Verify recovery of ΛCDM in the limit of no recycling physics.

        Sets φ = φ̇ = 0 and checks Friedmann equations match standard form.

        Returns
        -------
        dict
            Comparison of HRC vs standard results
        """
        # HRC with φ=0
        H2_hrc = self.friedmann_H_squared(
            a=1.0, rho_m=rho_m, rho_rem=rho_rem,
            phi=0.0, dphi_dt=0.0, n_rem=0.0
        )

        # Standard ΛCDM
        H2_lcdm = self.standard_friedmann_limit(rho_m + rho_rem)

        return {
            'H2_hrc': H2_hrc,
            'H2_lcdm': H2_lcdm,
            'relative_difference': abs(H2_hrc - H2_lcdm) / H2_lcdm if H2_lcdm != 0 else 0,
            'match': np.isclose(H2_hrc, H2_lcdm)
        }

    # -------------------------------------------------------------------------
    # Symbolic access
    # -------------------------------------------------------------------------

    def get_symbolic_friedmann_1(self):
        """Return symbolic first Friedmann equation."""
        return self._flrw.modified_friedmann_1()

    def get_symbolic_friedmann_2(self):
        """Return symbolic second Friedmann equation."""
        return self._flrw.modified_friedmann_2()

    def get_symbolic_scalar_equation(self):
        """Return symbolic scalar field equation."""
        return self._flrw.scalar_field_equation()

    def get_symbolic_conservation(self):
        """Return symbolic conservation equations."""
        return self._flrw.conservation_equations()


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def compute_remnant_production_rate(H: float, T_hawking: float,
                                    recycling_efficiency: float = 0.01) -> float:
    """
    Estimate remnant production rate from black hole evaporation.

    This is a phenomenological model connecting to the microscopic picture:
    - Black holes evaporate via Hawking radiation
    - Some fraction η of evaporating BHs leave Planck-mass remnants
    - Production rate depends on BH population and evaporation timescale

    Parameters
    ----------
    H : float
        Hubble parameter (sets cosmological timescale)
    T_hawking : float
        Characteristic Hawking temperature of BH population
    recycling_efficiency : float
        Fraction η of BHs that form remnants (0 < η < 1)

    Returns
    -------
    float
        Remnant production rate dn_rem/dt

    Notes
    -----
    This is highly speculative and model-dependent. The actual rate
    would depend on:
    - Primordial black hole abundance
    - BH mass distribution
    - Details of quantum gravity near Planck scale
    """
    # Placeholder implementation
    # Real model would require PBH population dynamics
    return recycling_efficiency * H * T_hawking


def hubble_tension_diagnostic(H0_local: float, H0_cmb: float,
                              phi_late: float, phi_early: float,
                              xi: float, G: float) -> Dict[str, float]:
    """
    Analyze whether HRC can address the Hubble tension.

    The Hubble tension: H0 measured locally (~73 km/s/Mpc) differs from
    CMB-inferred value (~67 km/s/Mpc) by ~9%.

    In HRC, if φ evolves between recombination and today:
    - G_eff changes: G_eff(early) ≠ G_eff(late)
    - This modifies the distance-redshift relation
    - Could reconcile local vs. CMB H0

    Parameters
    ----------
    H0_local : float
        Locally measured Hubble constant
    H0_cmb : float
        CMB-inferred Hubble constant
    phi_late : float
        Field value today
    phi_early : float
        Field value at recombination
    xi : float
        Non-minimal coupling
    G : float
        Newton's constant

    Returns
    -------
    dict
        Diagnostic information
    """
    tension = (H0_local - H0_cmb) / H0_cmb

    G_eff_factor_late = 1.0 - 8.0 * np.pi * G * xi * phi_late
    G_eff_factor_early = 1.0 - 8.0 * np.pi * G * xi * phi_early

    # If H² ∝ G_eff, then H ∝ sqrt(G_eff)
    # H_ratio ≈ sqrt(G_eff_late / G_eff_early)
    predicted_ratio = np.sqrt(G_eff_factor_early / G_eff_factor_late)

    return {
        'observed_tension': tension,
        'predicted_H_ratio': predicted_ratio,
        'phi_evolution': phi_late - phi_early,
        'G_eff_ratio': G_eff_factor_late / G_eff_factor_early,
        'can_explain': abs(predicted_ratio - (1 + tension)) < 0.01
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Holographic Recycling Cosmology (HRC) Theory Module")
    print("=" * 50)

    # Create theory with sample parameters
    params = HRCParameters(
        G=1.0,
        Lambda=0.7,
        xi=0.01,
        lambda_r=0.1,
        alpha=0.05,
        m_phi=1.0
    )

    theory = HRCTheory(params)

    # Test ΛCDM limit
    print("\nVerifying ΛCDM limit recovery:")
    limit_check = theory.verify_lcdm_limit(rho_m=0.3, rho_rem=0.0)
    print(f"  H² (HRC with φ=0): {limit_check['H2_hrc']:.6f}")
    print(f"  H² (standard):     {limit_check['H2_lcdm']:.6f}")
    print(f"  Match: {limit_check['match']}")

    # Compute modified Friedmann with non-zero φ
    print("\nModified Friedmann equation with recycling:")
    H2_modified = theory.friedmann_H_squared(
        a=1.0, rho_m=0.3, rho_rem=0.1,
        phi=0.1, dphi_dt=0.01, n_rem=0.05
    )
    print(f"  H² (with recycling): {H2_modified:.6f}")

    # Scalar field equation
    print("\nScalar field acceleration:")
    phi_ddot = theory.phi_equation_of_motion(
        phi=0.1, a=1.0, H=np.sqrt(H2_modified),
        rho_rem=0.1, n_rem=0.05, dphi_dt=0.01
    )
    print(f"  φ̈ = {phi_ddot:.6f}")

    print("\n" + "=" * 50)
    print("Module loaded successfully.")
