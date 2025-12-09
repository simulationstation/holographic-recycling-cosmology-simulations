# Holographic Recycling Cosmology (HRC)

A comprehensive Python framework for modeling black hole evaporation, Planck-mass remnant formation, and their cosmological consequences.

## Overview

HRC is a speculative cosmological model where:
- Black hole evaporation produces stable Planck-mass remnants
- A scalar "recycling field" φ couples matter to curvature
- The effective gravitational constant varies with cosmic epoch
- The Hubble tension arises naturally from epoch-dependent G_eff

**Key Result:** HRC predicts different H₀ values for different probes, matching the observed 5σ Hubble tension.

## Installation

```bash
# Clone the repository
git clone https://github.com/hrc-cosmology/hrc.git
cd hrc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib pandas
```

## Quick Start

```python
from hrc import quick_summary, run_full_analysis

# Print quick summary of HRC predictions
quick_summary()

# Run full analysis pipeline
results = run_full_analysis()
```

## Package Structure

```
hrc/
├── __init__.py          # Package initialization and convenience functions
├── run_analysis.py      # Full analysis pipeline
├── generate_figures.py  # Publication figure generation
│
hrc_theory.py           # Theoretical foundations (action, field equations)
hrc_dynamics.py         # Black hole dynamics and remnant formation
hrc_observations.py     # Observational constraints and MCMC
hrc_signatures.py       # Unique predictions (CMB, GW, DM, H(z))
│
hrc_paper.md            # Draft scientific paper
predictions_summary.md  # Summary of quantitative predictions
observational_tests.md  # Prioritized test descriptions
model_comparison.md     # HRC vs ΛCDM comparison
theory_derivations.md   # Detailed theoretical derivations
```

## Key Predictions

| Observable | ΛCDM | HRC | Status |
|------------|------|-----|--------|
| H₀ tension | 5σ unexplained | Resolved (ΔH₀≈6) | **OBSERVED** |
| w₀ | -1.00 | -0.88 | DESI hints |
| wₐ | 0 | -0.5 | DESI hints |
| GW echoes | None | t≈27 ms (30 M☉) | Testable |
| DM mass | Unknown | M_Planck | Theory |

## Running the Analysis

### Full Pipeline
```bash
python -m hrc.run_analysis --output results/
```

### Generate Figures
```bash
python -m hrc.generate_figures --output figures/
```

### Interactive Analysis
```bash
jupyter notebook signature_calculations.ipynb
```

## Model Parameters

Default parameters that resolve the Hubble tension:

| Parameter | Value | Description |
|-----------|-------|-------------|
| ξ | 0.03 | Non-minimal coupling |
| φ₀ | 0.2 | Scalar field today (Planck units) |
| α | 0.01 | Evolution exponent |
| f_rem | 0.2 | Remnant fraction of DM |

## Observational Tests

### Near-term (1-3 years)
1. **Multi-probe H₀ comparison** - Check if pattern matches HRC predictions
2. **DESI BAO** - Constrain w(z) trajectory

### Medium-term (3-5 years)
3. **Standard siren H₀** - GW170817-like events with EM counterparts
4. **GW ringdown echoes** - LIGO/Virgo O5 sensitivity

### Long-term (5-10 years)
5. **CMB-S4** - Precision θ* measurement
6. **Third-generation GW detectors** - QNM frequency precision

## Falsification Criteria

HRC would be **falsified** if:
- Standard sirens converge to H₀ ≈ 67 km/s/Mpc
- w = -1 ± 0.02 confirmed with high precision
- Hubble tension resolved by identified systematics
- Echo searches definitively negative at <1% amplitude

HRC would be **strongly supported** if:
- Standard sirens match local H₀ (~73)
- w(z) trajectory matches HRC predictions
- GW echoes detected at predicted delays

## Documentation

- [Scientific Paper](hrc_paper.md) - Full manuscript
- [Theory Derivations](theory_derivations.md) - Mathematical details
- [Predictions Summary](predictions_summary.md) - Quantitative predictions
- [Observational Tests](observational_tests.md) - Test descriptions and timeline
- [Model Comparison](model_comparison.md) - HRC vs ΛCDM

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hrc2025,
  title={Holographic Recycling Cosmology: A Framework for Resolving the Hubble Tension},
  author={HRC Collaboration},
  year={2025},
  url={https://github.com/hrc-cosmology/hrc}
}
```

## Contributing

Contributions welcome! Key areas:
- Full CMB Boltzmann code implementation (CLASS/CAMB modification)
- Extended MCMC analysis with more parameters
- Gravitational wave template development

## License

MIT License - see LICENSE file.

## References

1. Planck Collaboration (2020), A&A 641, A6
2. Riess et al. (2024), ApJ [SH0ES]
3. DESI Collaboration (2024), arXiv:2404.03002
4. Rovelli (2018), arXiv:1805.03872
5. Chen, Ong & Yeom (2015), Physics Reports 603, 1-45

---

*Developed December 2025*
