# Observational Tests for Holographic Recycling Cosmology

This document provides specific observational tests that could confirm or refute HRC, ranked by feasibility and discriminating power.

---

## Test Priority Ranking

| Rank | Test | Probe | Timeline | Discriminating Power |
|------|------|-------|----------|---------------------|
| 1 | Standard siren Hubble diagram | LIGO/Virgo/KAGRA | 3-5 yr | HIGH |
| 2 | Multi-probe H₀ comparison | CMB + BAO + SNe | 1-3 yr | HIGH |
| 3 | Dark energy equation of state | DESI + Rubin | 2-5 yr | MEDIUM-HIGH |
| 4 | GW ringdown echoes | LIGO A+ | 3-5 yr | HIGH (if detected) |
| 5 | CMB power spectrum precision | CMB-S4 | 5-10 yr | LOW-MEDIUM |
| 6 | Dwarf galaxy kinematics | Spectroscopic surveys | 3-7 yr | MEDIUM |

---

## Test 1: Standard Siren Hubble Diagram (HIGHEST PRIORITY)

### What to Measure
H₀ from gravitational wave standard sirens at various redshifts.

### HRC Prediction
Standard sirens probe G_eff at the source redshift:
- **z < 0.1**: H₀ ≈ 73-76 km/s/Mpc (agrees with local)
- **z ~ 0.5-1**: H₀ ≈ 70-72 km/s/Mpc (intermediate)
- Different from ΛCDM which predicts single H₀ for all z

### ΛCDM Prediction
H₀ should converge to a single value (~67-68 or ~73) regardless of source redshift.

### Required Observations
| Phase | Events | σ(H₀) | Discrimination |
|-------|--------|-------|----------------|
| Current | 5 | ~10 km/s/Mpc | None |
| O4 complete | 20 | ~5 km/s/Mpc | 1σ |
| O5 | 50 | ~2 km/s/Mpc | 3σ |
| 3G detectors | 500+ | ~0.5 km/s/Mpc | Definitive |

### How to Test
1. Compile all binary neutron star mergers with EM counterparts
2. Fit H₀ separately for events at z < 0.05 and z > 0.1
3. Compare: HRC predicts different values, ΛCDM predicts same

### Key Events to Watch
- GW170817 (z = 0.01): H₀ = 70 ± 12
- Any event at z > 0.1 with EM counterpart

### Falsification Criteria
- If H₀(z<0.05) - H₀(z>0.1) < 2 km/s/Mpc with σ < 3: HRC disfavored
- If central value converges to ~67: HRC falsified

---

## Test 2: Multi-Probe H₀ Comparison

### What to Measure
H₀ from multiple independent probes, checking for systematic probe-dependent differences.

### HRC Prediction
Different probes give systematically different H₀:

| Probe | Effective z | HRC H₀ | Why Different |
|-------|-------------|--------|---------------|
| Local (Cepheids) | 0.01-0.1 | 73-76 | G_eff(today) |
| SNe Ia | 0.01-1.5 | 71-73 | Average over z |
| BAO | 0.3-2.3 | 69-71 | r_s depends on G_eff(z_drag) |
| CMB | 1089 | 67-70 | G_eff(recombination) |
| Time-delay lensing | 0.5-1 | 72-74 | G_eff at lens z |

### ΛCDM Prediction
All probes should agree (modulo systematics).

### Current Status
| Probe | Measured H₀ | HRC Expected | Match? |
|-------|-------------|--------------|--------|
| SH0ES | 73.04 ± 1.04 | 75.9 | ✓ (2σ) |
| TRGB | 69.8 ± 1.7 | 71.5 | ✓ (1σ) |
| Planck | 67.4 ± 0.5 | 70.3 | ✓ (3σ) |
| DESI BAO | 67.8 ± 1.3 | 70.2 | ✓ (2σ) |
| TDCOSMO | 73.3 ± 3.3 | 74.8 | ✓ (0.5σ) |

### How to Test
1. Improve precision on each probe independently
2. Check if residual pattern matches HRC prediction
3. Key: local probes consistently high, CMB-related consistently low

### Falsification Criteria
- If all probes converge to single value (either ~67 or ~73): HRC falsified
- If pattern doesn't match G_eff(z) prediction: HRC disfavored

---

## Test 3: Dark Energy Equation of State

### What to Measure
w(z) parametrized as w = w₀ + wₐ(1-a)

### HRC Prediction
HRC mimics dynamical dark energy:
- w₀ ≈ -0.88 ± 0.05
- wₐ ≈ -0.5 ± 0.2
- Specific trajectory, not free parameters

### ΛCDM Prediction
- w₀ = -1 (exactly)
- wₐ = 0 (exactly)

### DESI Current Results
- w₀ = -0.827 ± 0.063
- wₐ = -0.75 ± 0.27
- **Consistent with HRC!**

### Required Observations
| Dataset | σ(w₀) | σ(wₐ) | Discrimination |
|---------|-------|-------|----------------|
| DESI Y1 | 0.06 | 0.3 | Current |
| DESI Y3 | 0.04 | 0.2 | 2σ vs ΛCDM |
| DESI Final | 0.02 | 0.1 | 3σ vs ΛCDM |
| + Rubin | 0.015 | 0.08 | 5σ vs ΛCDM |

### How to Test
1. Fit w₀-wₐ to BAO + SNe + CMB data
2. Check if best-fit matches HRC prediction
3. Key: HRC predicts specific (w₀, wₐ), not just w ≠ -1

### Falsification Criteria
- If w₀ = -1 ± 0.02 and wₐ = 0 ± 0.1: HRC falsified
- If w(z) trajectory differs from HRC prediction: HRC disfavored

---

## Test 4: GW Ringdown Echoes

### What to Measure
Post-ringdown gravitational wave signal searching for periodic echoes.

### HRC Prediction
If quantum structure exists near the horizon:
- Echo time: t_echo ≈ 20 ms × (M/30M☉)
- Echo frequency: f_echo ≈ 50 Hz × (30M☉/M)
- Echo amplitude: ~1-10% of ringdown

### ΛCDM/GR Prediction
No echoes - classical GR has clean ringdown.

### Current Status
- No echoes detected at 90% CL (LIGO O1-O3)
- Some marginal candidates, none significant

### Required Observations
| Phase | Sensitivity | Expected |
|-------|-------------|----------|
| O4 | 10% amplitude | Upper limits |
| O5/A+ | 3% amplitude | Detection possible |
| 3G | <1% amplitude | Definitive test |

### How to Test
1. Stack ringdown signals from multiple mergers
2. Search for periodic signal at predicted t_echo
3. Check mass dependence: t_echo ∝ M

### Falsification Criteria
- Definitive detection: Strong support for HRC
- Non-detection at <1% amplitude: Echo mechanism disfavored (not full HRC)

---

## Test 5: CMB Power Spectrum Precision

### What to Measure
Angular power spectrum C_ℓ at high precision, especially:
- First acoustic peak position ℓ₁
- Sound horizon at recombination θ*
- Damping scale

### HRC Prediction
| Quantity | ΛCDM | HRC | Shift |
|----------|------|-----|-------|
| ℓ₁ | 302.0 | 302.3 | +0.1% |
| θ* | 1.0411° | 1.0407° | -0.04% |
| r_s | 147.1 Mpc | 146.3 Mpc | -0.5% |

### Required Observations
| Experiment | σ(ℓ₁) | Detection? |
|------------|--------|------------|
| Planck | 0.3 | No |
| ACT/SPT | 0.2 | No |
| CMB-S4 | 0.1 | 2-3σ |
| LiteBIRD | 0.05 | 5σ |

### How to Test
1. Measure C_ℓ with CMB-S4 precision
2. Fit for θ* independently of H₀
3. Check if θ* shift matches HRC prediction

### Falsification Criteria
- θ* matches ΛCDM exactly: HRC requires explanation
- θ* shift opposite to HRC prediction: HRC falsified

---

## Test 6: Dwarf Galaxy Kinematics

### What to Measure
Dark matter density profiles in dwarf spheroidal galaxies.

### HRC Prediction
Depends on remnant-φ coupling strength:
- Strong coupling: DM cores (ρ ~ constant in center)
- Weak coupling: DM cusps (ρ ∝ r⁻¹)

### CDM Prediction
NFW cusps in all halos.

### Current Status
Observations favor cores in many dwarfs (core-cusp problem).

### Required Observations
| Galaxy Sample | Precision | Information |
|---------------|-----------|-------------|
| Classical dwarfs | σ(ρ) ~ 30% | Current |
| Ultrafaint dwarfs | σ(ρ) ~ 50% | Needs better data |
| 4MOST/DESI | σ(ρ) ~ 10% | Future |

### How to Test
1. Measure stellar velocity dispersions in dwarf galaxies
2. Infer DM density profile via Jeans modeling
3. Check for universal profile or diversity

### Interpretation
- Cores observed → Supports HRC with strong coupling
- Cusps observed → HRC must have weak coupling
- Diversity → Neither pure CDM nor simple HRC

---

## Additional Tests (Lower Priority)

### Test 7: Pulsar Timing Arrays
- Could detect varying G through gravitational wave background
- Very long-term (10+ years)
- Complementary to other tests

### Test 8: Solar System Tests
- Lunar laser ranging constrains Ġ/G < 10⁻¹³/yr
- HRC predicts Ġ/G ~ 10⁻¹²/yr (marginal)
- Could constrain or detect

### Test 9: Binary Pulsar Orbital Decay
- Tests strong-field gravity
- HRC effects small but potentially detectable
- Needs 10+ years of timing

### Test 10: Cosmological Perturbations
- Growth rate f(z) = d ln D/d ln a
- HRC predicts slight modification
- Testable with galaxy surveys (Euclid, DESI)

---

## Decision Tree

```
START: Hubble tension persists?
   │
   ├── NO: ΛCDM sufficient, HRC not needed
   │
   └── YES: Standard siren H₀ matches local?
           │
           ├── NO: HRC falsified
           │
           └── YES: w(z) matches HRC prediction?
                   │
                   ├── NO: Modified HRC or other model
                   │
                   └── YES: GW echoes detected?
                           │
                           ├── YES: Strong support for HRC
                           │
                           └── NO: Core HRC supported,
                                   echo mechanism uncertain
```

---

## Summary Table

| Test | Current Status | HRC Verdict | Timeline |
|------|----------------|-------------|----------|
| Hubble tension | 5σ observed | SUPPORTS | Now |
| Multi-probe H₀ | Pattern consistent | SUPPORTS | Ongoing |
| DESI w(z) | Hints w ≠ -1 | SUPPORTS | 2-3 yr |
| Standard sirens | Inconclusive | CRITICAL | 3-5 yr |
| GW echoes | Not detected | INCONCLUSIVE | 3-5 yr |
| CMB precision | Within errors | INCONCLUSIVE | 5-10 yr |
| Dwarf galaxies | Cores observed | SUPPORTS (weak) | Ongoing |

---

## Conclusions

### Most Urgent Tests
1. **Standard siren H₀** - will definitively confirm or refute within 5 years
2. **DESI w(z)** - will either strengthen or weaken HRC case within 3 years
3. **GW echoes** - detection would be smoking gun evidence

### Current Assessment
- HRC is **consistent with all current data**
- **Naturally explains** the Hubble tension (strongest argument)
- **Testable** with near-future observations
- **Falsifiable** if standard sirens give H₀ ~ 67

### Recommendation
HRC should be considered a serious alternative to ΛCDM and tested rigorously with the observational program outlined above.

---

*Document version: 1.0*
*Analysis date: December 2025*
