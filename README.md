# Holographic Recycling Cosmology (HRC)

**Version 2.0** - A rigorous, modular Python framework for testing Holographic Recycling Cosmology.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HRC is a speculative cosmological model where:
- Black hole evaporation produces stable **Planck-mass remnants**
- A scalar "recycling field" φ couples non-minimally to spacetime curvature
- The effective gravitational constant varies as **G_eff = G/(1 - 8πGξφ)**
- The **Hubble tension** arises naturally from epoch-dependent G_eff

### Key Result
HRC predicts different H₀ values for different cosmological probes, matching the observed 5σ discrepancy between local (SH0ES) and early-universe (Planck CMB) measurements.

## Scientific Assumptions

### Core Physics

1. **Modified Gravity**: Scalar-tensor theory with action
   ```
   S = S_EH + ∫d⁴x√(-g)[-½(∂φ)² - V(φ) - ξφR]
   ```
   The non-minimal coupling ξφR leads to G_eff ≠ G.

2. **Black Hole Remnants**: Hawking evaporation halts at the Planck mass (~2×10⁻⁸ kg), producing stable remnants that may contribute to dark matter.

3. **Cosmological Evolution**: The scalar field φ evolves according to the Klein-Gordon equation with curvature coupling:
   ```
   φ̈ + 3Hφ̇ + V'(φ) + ξR = 0
   ```

### Numerical Implementation

- **ODE-based cosmology**: No parametric shortcuts (e.g., φ(z) = φ₀(1+z)^α). Full numerical integration of coupled Friedmann + scalar field equations.
- **Ricci scalar**: Computed exactly from R = 6(2H² + Ḣ)
- **Adaptive integration**: Uses `scipy.integrate.solve_ivp` with automatic step control

### Theoretical Consistency

The code enforces:
- **No-ghost condition**: M_eff² > 0 (positive kinetic term)
- **Gradient stability**: c_s² > 0 (no exponential growth of perturbations)
- **Tensor stability**: Luminal GW propagation, massless graviton

### Observational Constraints

Implemented constraints include:
- **BBN**: |ΔG/G| < 10% at z ~ 10⁹
- **Solar System (PPN)**: |Ġ/G| < 1.5×10⁻¹² yr⁻¹, |γ-1| < 2.3×10⁻⁵
- **Stellar Evolution**: Constraints from helioseismology, white dwarfs, globular clusters
- **Structure Growth**: fσ₈(z) measurements

## Installation

```bash
# Clone the repository
git clone https://github.com/simulationstation/holographic-recycling-cosmology-simulations.git
cd holographic-recycling-cosmology-simulations

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib

# Optional: Install for MCMC
pip install emcee corner

# Optional: Install for development
pip install pytest

# Install package
pip install -e .
```

## Quick Start

```python
from hrc import HRCParameters, BackgroundCosmology, quick_summary

# Default parameters that resolve the Hubble tension
params = HRCParameters(xi=0.03, phi_0=0.2)

# Quick summary of predictions
quick_summary(params)

# Full analysis
from hrc import run_full_analysis
results = run_full_analysis(params)
```

## Package Structure

```
hrc/
├── __init__.py              # Package initialization
├── background.py            # ODE-based Friedmann solver
├── scalar_field.py          # Scalar field dynamics
├── effective_gravity.py     # G_eff computation
├── remnants.py              # Planck-mass remnant physics
│
├── perturbations/           # Perturbation theory
│   ├── stability_checks.py  # Ghost, gradient, tensor stability
│   └── interface_class.py   # CLASS Boltzmann code interface
│
├── observables/             # Observational predictions
│   ├── distances.py         # Cosmological distances
│   ├── h0_likelihoods.py    # SH0ES, TRGB, CMB priors
│   ├── bao.py               # DESI/BOSS BAO
│   ├── supernovae.py        # Pantheon+ SNe Ia
│   └── standard_sirens.py   # GW standard sirens
│
├── constraints/             # Theoretical constraints
│   ├── bbn.py               # Big Bang Nucleosynthesis
│   ├── ppn.py               # Solar system (PPN)
│   ├── stellar.py           # Stellar evolution
│   └── structure_growth.py  # fσ₈ growth rate
│
├── sampling/                # Parameter inference
│   ├── priors.py            # Prior distributions
│   └── mcmc.py              # MCMC sampler (emcee interface)
│
├── plots/                   # Visualization
│   ├── geff_evolution.py
│   ├── w_of_z.py
│   └── phase_diagram.py
│
└── utils/                   # Utilities
    ├── constants.py         # Physical constants
    ├── config.py            # Configuration classes
    └── numerics.py          # Numerical utilities

tests/                       # Unit tests
docs/                        # Documentation
```

## Key Results: HMDE T06D Falsification Tests

### Summary: Model Does NOT Resolve Hubble Tension

After extensive testing of the Horizon-Memory Dark Energy (HMDE) T06D model, we find:

| Test | Result | Implication |
|------|--------|-------------|
| Parameter Scan | **0/2250 pass** | No valid parameter space |
| High-z Impact | **< 0.01%** | Negligible at recombination |
| H₀ Best Fit | **67.43 ± 0.10** | No improvement over ΛCDM |
| ΔAIC vs ΛCDM | **+4.3** | ΛCDM slightly preferred |

### Test Results Detail

**TEST 1: Direct MCMC (emcee)**
- Sampled: H₀, Ω_m, δw, a_w, λ_hor
- Result: H₀ = 67.56 ± 0.43 km/s/Mpc
- HMDE parameters: δw = -0.09 ± 0.05, a_w = 0.31 ± 0.04

**TEST 2: High-Redshift Sensitivity Scan**
- 2250 parameter combinations tested
- Constraints: θ_s < 0.1%, D_L < 2%, BAO < 1%
- **Result: 0 points pass all constraints simultaneously**
- Maximum H(z=1089) deviation: 0.000%

**TEST 3: Aggressive H₀ Falsification**
- Target: Push H₀ to 73 km/s/Mpc while matching CMB θ_s
- CPL fit: w₀ = -0.986, wₐ = 0.075 (nearly ΛCDM)
- **Result: Cannot achieve high H₀ within observational bounds**

### Physical Interpretation

The HMDE model modifies the dark energy equation of state as:
```
w_eff(a) = w_base + Δw / (1 + (a/a_w)^m)
```

This late-time modification **cannot** significantly alter early-universe physics (z > 1000) while satisfying:
- CMB sound horizon (θ_s from Planck)
- BAO standard ruler (r_drag)
- Type Ia SNe distances (Pantheon+)

The Hubble tension requires correlated modifications at **both** early AND late times, which this model architecture cannot provide.

### Falsification Status

| Criterion | Status |
|-----------|--------|
| Standard sirens → H₀ ≈ 67 | Pending data |
| w = -1 ± 0.02 confirmed | **Supported by HMDE fit** |
| Tension resolved by systematics | Possible |
| HMDE improves over ΛCDM | **NO** |

---

## Original Key Predictions

| Observable | ΛCDM | HRC | Status |
|------------|------|-----|--------|
| H₀ tension | 5σ unexplained | ΔH₀ ≈ 6 km/s/Mpc | **NOT ACHIEVED** |
| w₀ | -1.00 | -0.88 | HMDE: -0.99 |
| wₐ | 0 | -0.5 | HMDE: +0.08 |
| GW echoes | None | t ≈ 27 ms (30 M☉) | Testable |
| DM mass | Unknown | M_Planck | Theory |

## Model Parameters

| Parameter | Symbol | Fiducial | Description |
|-----------|--------|----------|-------------|
| Non-minimal coupling | ξ | 0.03 | Scalar-curvature coupling |
| Field value today | φ₀ | 0.2 | In Planck units |
| Scalar mass | m_φ | ~H₀ | Determines evolution rate |
| Remnant fraction | f_rem | 0.2 | Fraction of DM in remnants |

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

- [Theory Guide](docs/source/theory.md) - Mathematical derivations
- [API Reference](docs/source/api.md) - Full API documentation
- [Examples](docs/examples/README.md) - Usage examples

## Falsification Criteria

### HMDE T06D Model: EFFECTIVELY FALSIFIED

The horizon-memory dark energy approach has been tested and found **unable to resolve the Hubble tension**:

- **0 of 2250 parameter combinations** pass combined CMB + BAO + SNe constraints
- Best-fit H₀ = 67.43 km/s/Mpc (identical to ΛCDM)
- ΔAIC = +4.3 (model complexity not justified by improved fit)

### Original HRC Criteria

HRC would be **falsified** if:
- Standard sirens converge to H₀ ≈ 67 km/s/Mpc
- w = -1 ± 0.02 confirmed with high precision ← **HMDE gives w₀ = -0.99**
- Hubble tension resolved by identified systematics
- GW echo searches definitively negative at <1% amplitude

HRC would be **strongly supported** if:
- Standard sirens match local H₀ (~73)
- w(z) trajectory matches HRC predictions ← **NOT ACHIEVED**
- GW echoes detected at predicted delays

## Citation

```bibtex
@software{hrc2025,
  title={Holographic Recycling Cosmology: A Rigorous Framework for Resolving the Hubble Tension},
  author={HRC Collaboration},
  year={2025},
  version={2.0},
  url={https://github.com/simulationstation/holographic-recycling-cosmology-simulations}
}
```

## Contributing

Contributions welcome! Key areas:
- Full CMB Boltzmann code implementation (CLASS/CAMB modification)
- Extended MCMC analysis with more parameters
- Gravitational wave template development
- Additional observational likelihoods

## License

MIT License - see LICENSE file.

## References

1. Planck Collaboration (2020), A&A 641, A6
2. Riess et al. (2024), ApJ (SH0ES)
3. DESI Collaboration (2024), arXiv:2404.03002
4. Rovelli (2018), arXiv:1805.03872
5. Chen, Ong & Yeom (2015), Physics Reports 603, 1-45

---

*Version 2.0 - December 2025*

---

## Appendix: HMDE T06D Test Scripts

The falsification tests can be reproduced using:

```bash
# TEST 1: Direct MCMC sampling
python scripts/run_hmde_background_mcmc.py --n-walkers 24 --n-steps 1000

# TEST 2: High-redshift sensitivity scan
python scripts/run_hmde_highz_sensitivity_scan.py

# TEST 3: Aggressive H0 falsification
python scripts/run_hmde_aggressive_falsification.py
```

Results are saved in `results/test1_background_mcmc/`, `results/test2_highz_sensitivity/`, and `results/test3_aggressive_falsification/`.
