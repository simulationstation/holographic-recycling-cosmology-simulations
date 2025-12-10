"""
Habitability Parameters for BHFC Cosmology

This module defines the habitability-optimal parameter A_HAB_STAR and provides
a fixed mapping to the cosmological BH formation sharpness parameter A_eff.

The habitability functional prefers certain cosmological conditions for the
emergence of complex structures and life. The parameter A_hab characterizes
the "sharpness" or efficiency of structure formation processes that are
conducive to habitability.

Physical interpretation:
- A_hab ~ 1.0 corresponds to a moderate, gradual structure formation process
  that allows sufficient time for galaxy formation while not being so slow
  that the universe becomes too dilute before structures can form.
- Higher A_hab would mean more abrupt transitions (less time for gradual
  enrichment and cooling).
- Lower A_hab would mean overly extended transitions (structures form too
  late or too diffusely).

The mapping A_eff(A_hab) connects this habitability-optimal value to the
cosmological parameter controlling BH formation window sharpness.
"""

# =============================================================================
# HABITABILITY-OPTIMAL PARAMETER
# =============================================================================

# A_HAB_STAR: The habitability-optimal value of the formation sharpness.
#
# This value is derived from the requirement that:
# 1. BH formation occurs on timescales compatible with subsequent galaxy formation
# 2. The early BH-dominated phase does not prevent baryonic structure formation
# 3. The transition from BH-dominated to standard cosmology is neither too
#    abrupt (disrupting structure) nor too gradual (diluting the BH effect)
#
# From habitability considerations in the Holographic framework, the optimal
# value corresponds to a moderate sharpness that balances these competing effects.
#
# Placeholder value: 1.0 (can be updated based on detailed habitability modeling)
A_HAB_STAR: float = 1.0


# =============================================================================
# MAPPING FUNCTION: A_hab -> A_eff
# =============================================================================

def map_A_hab_to_A_eff(A_hab: float) -> float:
    """
    Map habitability-optimal A_hab to cosmological A_eff.

    This mapping is FIXED and deterministic - no additional free parameters.
    The mapping is designed to be simple (identity for the reference case)
    while allowing for future refinement if the habitability model evolves.

    Physical motivation:
    - The cosmological A_eff controls the sharpness of the BH formation window
      function: W(a) = sigmoid(A_eff * ln(a/a_form))
    - The habitability parameter A_hab characterizes optimal structure formation
    - For the baseline habitability model, we assume direct proportionality

    Args:
        A_hab: Habitability parameter value

    Returns:
        A_eff: Cosmological BH formation sharpness parameter

    Notes:
        The mapping uses a simple power-law form:
            A_eff = A_0 * (A_hab / A_ref)^p

        With default values:
            A_ref = 1.0 (reference habitability value)
            p = 1.0 (linear mapping)
            A_0 = 1.0 (normalization)

        This gives A_eff = A_hab for the baseline case.
    """
    # Mapping parameters (fixed, not scanned)
    A_ref = 1.0   # Reference habitability value
    p = 1.0       # Power-law exponent (linear mapping)
    A_0 = 1.0     # Normalization constant

    # Simple power-law mapping
    A_eff = A_0 * (A_hab / A_ref) ** p

    return A_eff


def get_fixed_A_eff() -> float:
    """
    Get the fixed A_eff value derived from habitability constraints.

    This is the primary interface for SIMULATION 9B - it returns the
    A_eff value that should be used for ALL BHFC models when A_eff
    is no longer a free scan parameter.

    Returns:
        A_eff: Fixed cosmological BH formation sharpness (= 1.0 for baseline)
    """
    return map_A_hab_to_A_eff(A_HAB_STAR)


# =============================================================================
# DOCUMENTATION
# =============================================================================

def describe_habitability_mapping() -> str:
    """Return a description of the habitability-to-cosmology mapping."""
    A_eff_fixed = get_fixed_A_eff()
    return f"""
Habitability-Cosmology Parameter Mapping
========================================

Habitability-optimal parameter:
    A_HAB_STAR = {A_HAB_STAR}

Mapping function:
    A_eff = A_0 * (A_hab / A_ref)^p

    with A_ref = 1.0, p = 1.0, A_0 = 1.0 (linear identity mapping)

Result:
    A_eff (fixed) = {A_eff_fixed}

Physical interpretation:
    - A_eff controls the sharpness of BH formation window
    - Higher A_eff = more abrupt formation (step-like)
    - Lower A_eff = more gradual formation (smooth transition)
    - A_eff = 1.0 gives moderate sharpness compatible with structure formation
"""


if __name__ == '__main__':
    print(describe_habitability_mapping())
