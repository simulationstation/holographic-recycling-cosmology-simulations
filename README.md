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
5. **Phenomenological expansion history modifications**

---

## Executive Summary: Key Results

| Hypothesis Tested | Result | P(H₀ ≥ 73) |
|-------------------|--------|------------|
| Known distance ladder systematics | **NEGATIVE** | 0.27% |
| JWST crowding improvements | **NEGATIVE** | < 1% |
| Modified gravity (scalar-tensor) | **NEGATIVE** | Max ΔH₀ ~ 3.5 km/s/Mpc |
| Hawking-Hartle no-boundary prior | **NEGATIVE** | 0.14% |
| Black-hole interior cosmology (BITCC) | **NEGATIVE** | 0.00% |
| Multi-mode expansion modifications | **NEGATIVE** | 0.00% |
| ALP-EDE (axion-like particles) | **PARTIAL** | Requires f_EDE > 10% |
| Entropic/emergent metric effects | **NEGATIVE** | Pushes H₀ lower |

**Bottom Line:** No tested mechanism can reconcile CMB + BAO + SN data with H₀ ≥ 73 km/s/Mpc. The Hubble tension appears to require either:
- Unidentified systematic errors in local distance measurements
- New physics not captured by these models
- A statistical fluctuation (unlikely at ~5σ)

---

## Complete Simulation Suite

### Distance Ladder Systematics (SIM 11-15)

| Simulation | Description | Key Result |
|------------|-------------|------------|
| **SIM 11A** | Baseline ΛCDM fitting | H₀ = 67.5 ± 0.3, unbiased recovery |
| **SIM 11B** | Host galaxy mass step | δM_step = 0.02 ± 0.01 mag |
| **SIM 11C** | Combined SN systematics | Multiple nuisance parameters |
| **SIM 12** | Cepheid calibration | P-L relation, metallicity effects |
| **SIM 13** | JWST crowding improvements | 50% reduction ≠ tension resolution |
| **SIM 14** | Rest-frame misalignment bias | Small effect on H₀ |
| **SIM 15** | **Joint hierarchical model** | **P(H₀ ≥ 73) = 0.27%** |

### Fundamental Physics Priors (SIM 16-19)

| Simulation | Description | Key Result |
|------------|-------------|------------|
| **SIM 16** | Hawking-Hartle no-boundary prior | P(H₀ ≥ 73) = 0.14%, prior pushes H₀ ↓ |
| **SIM 17** | Unified distance ladder model | Systematics insufficient |
| **SIM 18** | Boundary kernel effects | Negligible H₀ shift |
| **SIM 19** | Boundary monopole viability | No viable high-H₀ configurations |

### Beyond-ΛCDM Physics (SIM 20-25)

| Simulation | Description | Key Result |
|------------|-------------|------------|
| **SIM 20** | ALP/EDE (axion-like particles) | Requires f_EDE > 10%, tension with BAO |
| **SIM 21** | EMIC (emergent metric inflation) | H₀_eff ~ 65-68, pushes lower |
| **SIM 22** | ERC (entropic response cosmology) | Cannot raise H₀ sufficiently |
| **SIM 23** | SSHR (stochastic sound horizon) | Limited H₀ range achievable |
| **SIM 24** | BITCC (black-hole interior cosmology) | P(H₀ ≥ 73) = 0.00% |
| **SIM 25** | Multi-mode terminal spectrum | **No allowed configurations** |

### Modified Gravity (HRC 2.0)

| Model | Coupling | Result |
|-------|----------|--------|
| Linear | F(φ) = 1 - ξφ | Max ΔH₀ ~ 3.5 km/s/Mpc |
| Quadratic | F(φ) = 1 - ξφ² | Similar ceiling |
| Exponential | F(φ) = exp(-ξφ) | Constrained by BBN/PPN |

---

## Detailed Findings

### 1. Distance Ladder Systematics Cannot Explain the Tension

From SIM 15 (joint hierarchical analysis with 9 nuisance parameters):

```
H₀ posterior: 67.85 ± 1.76 km/s/Mpc

P(H₀ ≥ 73) = 0.0027 (0.27%)
P(H₀ ≥ 70) = 0.11 (11%)

H₀ = 73 is at 2.9σ from the posterior mean
```

Even marginalizing over all known systematics (host mass step, Cepheid metallicity, crowding, calibration uncertainties), the probability of H₀ ≥ 73 remains below 0.3%.

### 2. JWST Improvements Are Insufficient

Even with JWST reducing crowding systematics by 50%:
- The posterior shifts by only ~0.1 km/s/Mpc
- P(H₀ ≥ 73) remains below 1%
- The tension persists at >2.5σ

### 3. Modified Gravity Has a Ceiling

HRC 2.0 scalar-tensor analysis shows:
- Maximum achievable ΔH₀ ~ 3.5 km/s/Mpc (insufficient for 6 km/s/Mpc gap)
- BBN and solar system constraints limit G_eff variation
- Single-field scalar-tensor theories cannot resolve the tension

### 4. Fundamental Physics Priors Push H₀ Lower

**SIM 16 (Hawking-Hartle No-Boundary):**
- Prior naturally prefers low H₀ values
- P(H₀ ≥ 73) = 0.14% under no-boundary prior
- The cosmological measure problem disfavors high-H₀ universes

**SIM 24 (BITCC - Black-Hole Interior Cosmology):**
- χ_trans (computational residue) maps to H_init
- Prior mean H₀ = 64.4 km/s/Mpc
- P(H₀ ≥ 73) = 0.14% prior, 0.00% posterior
- Data compatibility pushes toward Planck value (67.5)

### 5. Phenomenological Expansion Modifications Are Highly Constrained

**SIM 25 (Multi-Mode Terminal Spectrum):**
```
Model: δH/H(a) = Σ_i A_i * f_i(ln a; μ_i, σ_i)
```
- 3-mode template at z ~ 3000, 100, 1
- Amplitude range ±5%
- **Result: ZERO configurations pass all constraints (θ*, BAO, SN)**
- The CMB acoustic angle θ* is extremely constraining

The tightly measured CMB acoustic scale (θ* = 0.01041 ± 0.00003 rad) leaves essentially no room for modifications to H(z) that would shift the inferred H₀.

### 6. Early Dark Energy Requires Fine-Tuning

**SIM 20 (ALP-EDE):**
- Axion-like early dark energy can raise H₀
- But requires f_EDE > 10% (fraction at z ~ 3000)
- Creates tension with BAO measurements
- Not a clean solution

---

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

### Run the main systematics result (SIM 15)

```bash
python scripts/run_sim15_joint_hierarchical.py --n-walkers 40 --n-steps 5000
python scripts/analyze_sim15_joint_systematics_results.py
```

### Run phenomenological expansion scan (SIM 25)

```bash
python scripts/run_sim25a_mode_spectrum_scan.py --n-grid 9
python scripts/run_sim25b_inverse_mode_fit.py --n-samples 2000
python scripts/analyze_sim25_results.py
```

### Run no-boundary cosmology (SIM 16)

```bash
python scripts/run_sim16a_prior_predictive.py --n-samples 1000
```

## Project Structure

```
.
├── hrc2/                          # Core cosmology modules
│   ├── ladder/                    # Distance ladder likelihood
│   ├── noboundary/                # Hawking-Hartle no-boundary
│   ├── terminal_spectrum/         # Multi-mode expansion (SIM 25)
│   ├── bitcc/                     # Black-hole interior cosmology (SIM 24)
│   ├── layered/                   # Layered expansion model
│   ├── alp/                       # ALP-EDE model (SIM 20)
│   ├── metric/                    # Emergent metric (SIM 21)
│   ├── entropy/                   # Entropic response (SIM 22)
│   └── background.py              # Friedmann solver
│
├── scripts/                       # Simulation scripts
│   ├── run_sim*.py               # Simulation drivers
│   └── analyze_sim*.py           # Analysis scripts
│
├── tests/                         # Unit tests
│   └── test_*.py                 # pytest test files
│
├── results/                       # Output directories
│   └── simulation_*/             # Per-simulation results
│
└── figures/                       # Generated plots
    └── simulation_*/             # Per-simulation figures
```

## Reproducibility

All simulations use fixed random seeds for reproducibility:

```python
# Default seeds
SIM_15_SEED = 20241215
SIM_16_SEED = 20241210
SIM_25_SEED = 12345
```

Results can be exactly reproduced by running scripts with default parameters.

## Conclusions

After extensive testing of:
- **9 systematic error parameters** in the distance ladder
- **5 beyond-ΛCDM physics models**
- **3 fundamental physics prior frameworks**
- **1 general phenomenological expansion parameterization**

We find **no mechanism that can reconcile CMB + BAO + SN + local H₀ data**. The Hubble tension remains robust at ~5σ.

**Implications:**
1. The tension is unlikely to be resolved by JWST or improved measurements
2. Single-field modified gravity cannot bridge the gap
3. Alternative priors from fundamental physics (no-boundary, BITCC) make the tension worse
4. The CMB acoustic scale θ* is extremely constraining on any H(z) modifications

The most likely resolutions are:
- Unidentified systematic errors in SH0ES (not in our model space)
- Multi-field new physics with specific features (e.g., interacting dark sector)
- The tension is a genuine ~5σ statistical fluctuation

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

## References

1. Riess, A. et al. (2024). SH0ES: H₀ = 73.04 ± 1.04 km/s/Mpc
2. Planck Collaboration (2020). H₀ = 67.4 ± 0.5 km/s/Mpc
3. Freedman, W. et al. (2024). JWST Cepheid observations
4. Hartle, J.B. & Hawking, S.W. (1983). Wave function of the Universe

## License

MIT License - see LICENSE file.

---

*Last updated: December 2025*
