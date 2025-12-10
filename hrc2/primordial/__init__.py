"""
Primordial power spectrum modifications for HRC cosmology.

This module implements WHBC-motivated modifications to the primordial
curvature power spectrum P(k).
"""

from .whbc_primordial import (
    WHBCPrimordialParameters,
    WHBCPrimordialResult,
    primordial_ratio,
    primordial_PK_whbc,
    primordial_PK_lcdm,
    analyze_whbc_primordial,
    generate_class_pk_file,
    PRESETS,
)

from .class_interface import (
    generate_whbc_pk_file,
    approximate_cmb_effects,
    run_camb_with_whbc,
    run_boltzmann_with_whbc,
    compute_cmb_chi2,
    HAS_CLASS,
    HAS_CAMB,
)

__all__ = [
    'WHBCPrimordialParameters',
    'WHBCPrimordialResult',
    'primordial_ratio',
    'primordial_PK_whbc',
    'primordial_PK_lcdm',
    'analyze_whbc_primordial',
    'generate_class_pk_file',
    'approximate_cmb_effects',
    'HAS_CLASS',
    'PRESETS',
]
