# Holographic Recycling Cosmology (HRC) Documentation

Welcome to the HRC documentation. This package implements a rigorous framework for testing Holographic Recycling Cosmology, a theoretical model that aims to resolve the Hubble tension through epoch-dependent gravitational coupling.

## Quick Links

- [Getting Started](getting_started.md)
- [Theory Guide](source/theory.md)
- [API Reference](source/api.md)
- [Examples](examples/README.md)

## Overview

HRC is based on three key physical ingredients:

1. **Planck-mass remnants**: Black hole evaporation halts at the Planck mass, producing stable remnants
2. **Scalar recycling field**: A field φ mediates the recycling process and couples to curvature
3. **Epoch-dependent gravity**: The effective Newton's constant varies as:

   $$G_{\rm eff}(\phi) = \frac{G}{1 - 8\pi G\xi\phi}$$

## Installation

```bash
pip install -e .
```

Or with optional dependencies:
```bash
pip install -e ".[dev,docs]"
```

## Quick Start

```python
from hrc import HRCParameters, BackgroundCosmology, quick_summary

# Default parameters that resolve the Hubble tension
params = HRCParameters(xi=0.03, phi_0=0.2)

# Get a quick summary
quick_summary(params)

# Run full analysis
from hrc import run_full_analysis
results = run_full_analysis(params)
```

## Package Structure

```
hrc/
├── background.py          # Friedmann equation solver
├── scalar_field.py        # Scalar field dynamics
├── effective_gravity.py   # G_eff computation
├── remnants.py            # Black hole remnant physics
├── perturbations/         # Stability and CLASS interface
├── observables/           # Distance and likelihood calculations
├── constraints/           # BBN, PPN, stellar constraints
├── sampling/              # MCMC parameter inference
└── plots/                 # Visualization tools
```

## Key Features

- **ODE-based evolution**: No parametric assumptions; full numerical integration
- **Stability checks**: Automatic ghost/gradient instability detection
- **Observational likelihoods**: SH0ES, BAO, SNe, CMB, standard sirens
- **Constraint checking**: BBN, PPN, stellar evolution limits
- **MCMC sampling**: Full posterior exploration with emcee
- **Modular design**: Easy to extend with new physics

## Physical Assumptions

See [Theory Guide](source/theory.md) for detailed derivations and assumptions.

## Citation

If you use this code, please cite:

```bibtex
@software{hrc2025,
  title={Holographic Recycling Cosmology: A Rigorous Framework},
  author={HRC Collaboration},
  year={2025},
  url={https://github.com/simulationstation/holographic-recycling-cosmology-simulations}
}
```

## License

MIT License - see LICENSE file.
