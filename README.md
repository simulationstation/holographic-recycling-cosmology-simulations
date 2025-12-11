# Hubble Tension Analysis Framework

**A comprehensive Python framework for investigating the Hubble tension through cosmological simulations and MCMC inference.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a rigorous, modular framework for quantitatively assessing whether the ~5σ Hubble tension between local (SH0ES: H₀ ≈ 73 km/s/Mpc) and early-universe (Planck CMB: H₀ ≈ 67 km/s/Mpc) measurements can be explained by:

1. **Known systematic errors** in the cosmic distance ladder
2. **Modified gravity / dark energy** models
3. **Alternative cosmological priors** (e.g., Hawking-Hartle no-boundary)
4. **JWST observations** improving distance calibration

### Key Result

**P(H₀ ≥ 73 km/s/Mpc | ΛCDM + all known systematics) ≈ 0.27%**

Even when marginalizing over generous priors on all identified systematic uncertainties in the Type Ia supernova distance ladder, the probability of obtaining the SH0ES value of H₀ remains below 0.3%. This strongly suggests that:
- The Hubble tension is not a result of known measurement systematics
- JWST improvements to Cepheid photometry will not resolve the tension
- New physics or unidentified systematics are required

## Simulation Suite

### Distance Ladder Systematics (SIM 11-15)

| Simulation | Description | Key Result |
|------------|-------------|------------|
| **SIM 11A** | Baseline ΛCDM fitting | H₀ = 67.5 ± 0.3, unbiased recovery |
| **SIM 11B** | Host galaxy mass step | δM_step = 0.02 ± 0.01 mag |
| **SIM 11C** | Combined SN systematics | Multiple nuisance parameters |
| **SIM 12** | Cepheid calibration | P-L relation, metallicity effects |
| **SIM 13** | JWST crowding improvements | 50% reduction in crowding errors |
| **SIM 14** | Anchor galaxy systematics | LMC, NGC 4258, MW anchors |
| **SIM 15** | **Joint hierarchical model** | **P(H₀ ≥ 73) = 0.27%** |

### Hawking-Hartle No-Boundary Cosmology (SIM 16)

Explores whether structured priors from fundamental physics can affect H₀ inference:

| Simulation | Description | Status |
|------------|-------------|--------|
| **SIM 16A** | Prior predictive sampling | Ready |
| **SIM 16B** | MCMC with epsilon_corr | Ready |

### Modified Gravity / Dark Energy (HRC 2.0)

Tests whether scalar-tensor theories can resolve the tension:

| Model | Coupling | Result |
|-------|----------|--------|
| Linear | F(φ) = 1 - ξφ | Max ΔH₀ ~ 3.5 km/s/Mpc |
| Quadratic | F(φ) = 1 - ξφ² | Similar ceiling |
| Exponential | F(φ) = exp(-ξφ) | Constrained by BBN/PPN |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hubble-tension-analysis.git
cd hubble-tension-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib emcee corner tqdm

# Optional: Cobaya for full MCMC
pip install cobaya camb

# Install package in development mode
pip install -e .
```

## Quick Start

### Run the main result (SIM 15)

```bash
# Joint hierarchical systematics analysis
python scripts/run_sim15_joint_hierarchical.py --n-walkers 40 --n-steps 5000

# Analyze results
python scripts/analyze_sim15_joint_systematics_results.py
```

### Prior predictive sampling (SIM 16A)

```bash
# Sample from Hawking-Hartle no-boundary prior
python scripts/run_sim16a_prior_predictive.py --n-samples 1000
```

### Compare with/without JWST

```bash
# Pre-JWST systematics
python scripts/run_sim15_joint_hierarchical.py --sigma-crowd 0.05

# Post-JWST (improved crowding)
python scripts/run_sim15_joint_hierarchical.py --sigma-crowd 0.025
```

## Project Structure

```
.
├── hrc2/                          # Core cosmology modules
│   ├── ladder/                    # Distance ladder likelihood
│   │   ├── cosmology_baseline.py  # ΛCDM background
│   │   ├── sn_systematics.py      # SN Ia systematics
│   │   ├── cepheid_model.py       # Cepheid P-L relation
│   │   └── joint_likelihood.py    # Full hierarchical model
│   │
│   ├── theory/                    # No-boundary cosmology
│   │   ├── no_boundary_prior.py   # Hawking-Hartle prior
│   │   └── no_boundary_to_cosmo.py # Parameter mapping
│   │
│   ├── background.py              # ODE-based Friedmann solver
│   ├── cobaya_noboundary_model.py # Cobaya Theory wrapper
│   └── ...
│
├── scripts/                       # Simulation scripts
│   ├── run_sim11a_*.py           # SIM 11 variants
│   ├── run_sim12_*.py            # SIM 12 Cepheid
│   ├── run_sim13_*.py            # SIM 13 JWST
│   ├── run_sim14_*.py            # SIM 14 anchors
│   ├── run_sim15_*.py            # SIM 15 joint hierarchical
│   ├── run_sim16a_*.py           # SIM 16A prior predictive
│   ├── run_sim16b_*.py           # SIM 16B MCMC
│   └── analyze_*.py              # Analysis scripts
│
├── paper/                         # LaTeX paper and figures
│   ├── main.tex                  # "JWST won't solve Hubble tension"
│   ├── images/                   # PNG figures
│   └── generate_figures.py       # Figure generation
│
├── results/                       # Output directories
│   ├── simulation_15_*/          # SIM 15 MCMC chains
│   ├── simulation_16a_*/         # SIM 16A samples
│   └── ...
│
└── cobaya_configs/                # Cobaya YAML files
```

## Key Findings

### 1. Systematics Cannot Explain the Tension

From SIM 15 (joint hierarchical analysis with 9 nuisance parameters):

```
H₀ posterior: 67.85 ± 1.76 km/s/Mpc

P(H₀ ≥ 73) = 0.0027 (0.27%)
P(H₀ ≥ 70) = 0.11 (11%)

H₀ = 73 is at 2.9σ from the posterior mean
```

### 2. JWST Improvements Are Insufficient

Even with JWST reducing crowding systematics by 50%:
- The posterior shifts by only ~0.1 km/s/Mpc
- P(H₀ ≥ 73) remains below 1%
- The tension persists at >2.5σ

### 3. Modified Gravity Has a Ceiling

HRC 2.0 scalar-tensor analysis shows:
- Maximum achievable ΔH₀ ~ 3.5 km/s/Mpc (insufficient for 6 km/s/Mpc gap)
- BBN and solar system constraints limit G_eff variation
- The "no-go theorem" prevents single-field scalar-tensor resolution

### 4. No-Boundary Prior Effects

SIM 16 explores whether Hawking-Hartle priors affect H₀:
- epsilon_corr modifies H(z) at z > 3000 (pre-recombination)
- This affects the sound horizon and CMB-inferred H₀
- Results pending...

## Reproducibility

All simulations use fixed random seeds for reproducibility:

```python
# Default seeds
SIM_15_SEED = 20241215
SIM_16_SEED = 20241210
```

Results can be exactly reproduced by running the scripts with default parameters.

## Dependencies

Core:
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.5
- emcee >= 3.0
- tqdm

Optional:
- cobaya >= 3.0 (for Planck/BAO likelihoods)
- camb >= 1.0 (for CMB calculations)
- corner (for triangle plots)

## Citation

```bibtex
@software{hubble_tension_analysis_2025,
  title={Hubble Tension Analysis Framework: JWST, Systematics, and Beyond},
  author={Smith, Aiden B.},
  year={2025},
  url={https://github.com/your-repo/hubble-tension-analysis}
}
```

## References

1. Riess, A. et al. (2024). SH0ES: H₀ = 73.04 ± 1.04 km/s/Mpc
2. Planck Collaboration (2020). H₀ = 67.4 ± 0.5 km/s/Mpc
3. Freedman, W. et al. (2024). JWST Cepheid observations
4. Hartle, J.B. & Hawking, S.W. (1983). Wave function of the Universe

## License

MIT License - see LICENSE file.

---

*Last updated: December 2025*
