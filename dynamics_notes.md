# HRC Dynamics: Physical Assumptions and Limitations

This document describes the physical foundations, approximations, and limitations of the black hole evaporation and recycling dynamics implemented in `hrc_dynamics.py`.

## Table of Contents

1. [Established Physics](#1-established-physics)
2. [Speculative Extensions](#2-speculative-extensions)
3. [Key Approximations](#3-key-approximations)
4. [Numerical Considerations](#4-numerical-considerations)
5. [Known Limitations](#5-known-limitations)
6. [Parameter Space](#6-parameter-space)
7. [Observational Constraints](#7-observational-constraints)

---

## 1. Established Physics

### 1.1 Hawking Radiation

The Hawking effect is well-established semiclassical physics:

**Temperature:**
$$T_H = \frac{\hbar c^3}{8\pi G M k_B} \approx 6.2 \times 10^{-8} \left(\frac{M_\odot}{M}\right) \text{ K}$$

**Luminosity** (Stefan-Boltzmann for effective 2D emission):
$$L = \frac{\hbar c^6}{15360 \pi G^2 M^2} \approx 9.0 \times 10^{-29} \left(\frac{M_\odot}{M}\right)^2 \text{ W}$$

**Evaporation time:**
$$\tau_{\rm evap} = \frac{5120 \pi G^2 M^3}{\hbar c^4} \approx 8.4 \times 10^{-17} \left(\frac{M}{\rm kg}\right)^3 \text{ s}$$

**Implementation fidelity:** HIGH

The module uses the standard semiclassical formulas. Corrections from:
- Greybody factors (~O(1) modifications)
- Particle species (we use the 2D approximation)
- Rotation (Kerr BHs radiate faster)

...are neglected but would only change results by factors of order unity.

### 1.2 Bekenstein-Hawking Entropy

$$S_{BH} = \frac{A}{4\ell_P^2} = \frac{4\pi G^2 M^2}{\hbar c}$$

This is perhaps the most robust result in quantum gravity, supported by multiple independent derivations.

**Implementation fidelity:** HIGH

### 1.3 Primordial Black Hole Mass Functions

The log-normal distribution is standard in the PBH literature:
$$\frac{dn}{dM} = \frac{f_{\rm PBH}\rho_{\rm DM}}{M\sqrt{2\pi}\sigma_M}\exp\left[-\frac{(\ln(M/M_c))^2}{2\sigma_M^2}\right]$$

This is motivated by:
- Inflationary perturbation spectra
- Critical collapse dynamics
- Press-Schechter formalism

**Implementation fidelity:** MEDIUM

The true mass function depends on unknown early-universe physics. The log-normal is a reasonable parametrization, but:
- Extended mass functions exist
- Clustering effects are neglected
- Formation mechanisms are uncertain

---

## 2. Speculative Extensions

### 2.1 Remnant Formation Hypothesis

**Claim:** Black holes don't fully evaporate but leave Planck-mass remnants.

**Physical motivation:**
1. **Information paradox:** Complete evaporation seems to destroy quantum information
2. **Uncertainty principle:** Localizing to < ℓ_P costs > M_P energy
3. **Quantum gravity:** Semiclassical approximation breaks at Planck scale

**Supporting arguments:**
- 't Hooft's S-matrix approach suggests remnants
- Loop quantum gravity predicts bounce to white hole
- String theory has candidate remnant states

**Against:**
- No experimental evidence
- Potential information overcount problem
- Unclear stability mechanism

**Implementation:** We assume remnants form at M = M_Planck and are stable.

**Fidelity:** SPECULATIVE but physically motivated

### 2.2 Remnant Interior Volume (Erebon Hypothesis)

**Claim:** Remnants have Planck-scale exterior but large interior volume.

**Physical motivation:**
- Holographic principle allows interior >> exterior
- Rovelli's "white hole erebons" model
- Information storage capacity requirements

**Implementation:** Parametrized by `remnant_interior_volume`

**Fidelity:** HIGHLY SPECULATIVE

### 2.3 Recycling Mechanism

**Claim:** Hawking radiation is partially reabsorbed by remnants.

**Physical model:**
1. BH emits Hawking quanta isotropically
2. Quanta propagate through remnant-filled space
3. Remnants have absorption cross-section σ_abs
4. Absorption probability: P = 1 - exp(-n_rem × σ_abs × L)

**Assumptions:**
- Remnants behave as classical absorbers
- Cross-section is geometric (~πR²)
- Absorption is complete (no partial scattering)

**Implementation:** RecyclingDynamics class

**Fidelity:** HIGHLY SPECULATIVE - no established physics supports this mechanism

### 2.4 Coupling to Scalar Field φ

**Claim:** Absorbed energy sources the recycling field.

This is the core HRC hypothesis connecting microphysics to cosmology. The coupling is postulated, not derived.

**Implementation:** Source term Γ = η × (recycled power / volume)

**Fidelity:** ENTIRELY PHENOMENOLOGICAL

---

## 3. Key Approximations

### 3.1 Geometric Optics Limit

We treat Hawking radiation as classical rays. This is valid when:
- Wavelength λ << curvature scale R
- For typical Hawking radiation: λ ~ r_s (marginal)

**Error estimate:** O(1) factors in luminosity

### 3.2 No Accretion

PBHs are assumed to only lose mass, never gain it. Valid when:
- Background matter density is low
- BH is not in a dense environment

For cosmological PBHs in the current epoch, this is reasonable.

### 3.3 Homogeneous Distribution

Remnants are assumed uniformly distributed. In reality:
- Clustering from gravitational instability
- Correlation with BH locations
- Large-scale structure effects

**Impact:** Recycling probability could vary spatially by orders of magnitude

### 3.4 Single Mass Remnants

All remnants have exactly M_Planck mass. More realistic:
- Mass distribution from formation process
- Time evolution from absorption

### 3.5 Instantaneous Thermalization

Absorbed radiation instantly thermalizes and contributes to φ source. In reality:
- Finite absorption timescale
- Possible re-emission
- Interior dynamics unknown

---

## 4. Numerical Considerations

### 4.1 Timescale Hierarchy

The problem spans enormous timescales:

| Process | Timescale |
|---------|-----------|
| Planck time | 5.4 × 10⁻⁴⁴ s |
| Stellar BH evaporation | 10⁶⁷ years |
| 10¹² kg PBH evaporation | 10¹⁷ s |
| Hubble time | 10¹⁷ s |
| Proton decay (GUT) | 10³⁶ years |

**Challenge:** Simultaneously resolving Planck-scale and Hubble-scale dynamics.

**Solution:** Use dimensionless variables in Planck units. The ODE system becomes:
- ã = a/a_0
- t̃ = t/t_P
- ρ̃ = ρ × ℓ_P³/M_P

### 4.2 Stiffness

The system is stiff due to:
- Different decay rates for matter vs. remnants
- Scalar field oscillations
- Nonlinear couplings

**Solution:** Use implicit methods (Radau, BDF) for production runs.

### 4.3 Conservation Monitoring

Energy conservation provides a check:
$$\frac{d}{dt}(\rho a^3) = -3pHa^3 + \text{sources}$$

The module tracks comoving energy E = ρa³ to verify conservation.

### 4.4 Singularity Avoidance

Potential singularities at:
- a → 0 (Big Bang)
- M → 0 (BH evaporation endpoint)
- G_eff → 0 (strong coupling)

**Mitigation:** Regularization with small cutoffs; remnant formation prevents M → 0.

---

## 5. Known Limitations

### 5.1 Theoretical Limitations

1. **No quantum gravity:** We use semiclassical approximations everywhere
2. **Linear φ coupling:** The ξφR term is unusual (compared to ξφ²R)
3. **Phenomenological sources:** Remnant-to-φ coupling is ad hoc
4. **No perturbations:** Only background cosmology, no structure formation

### 5.2 Computational Limitations

1. **Mass function resolution:** Coarse binning misses rapid evaporators
2. **Spatial averaging:** No position-dependent recycling
3. **Fixed parameters:** No parameter evolution (running couplings)

### 5.3 Physical Limitations

1. **Remnant stability:** Assumed but not justified
2. **Cross-section model:** Geometric optics may not apply at Planck scale
3. **Absorption mechanism:** Hand-waved, not derived
4. **Interior physics:** Completely unknown

---

## 6. Parameter Space

### 6.1 Constrained Parameters

| Parameter | Constraint | Source |
|-----------|------------|--------|
| G | 6.674 × 10⁻¹¹ m³/kg/s² | Lab measurements |
| Λ | ~10⁻¹²² M_P⁴ | Cosmological observations |
| f_PBH | < 1 (various M-dependent) | Gravitational lensing, CMB |

### 6.2 Weakly Constrained

| Parameter | Plausible Range | Notes |
|-----------|-----------------|-------|
| M_c (PBH) | 10⁹ - 10¹⁵ kg | Evaporating now to asteroid mass |
| σ_M | 0.5 - 2.0 | Mass function width |
| ξ | 10⁻³ - 1 | Non-minimal coupling |

### 6.3 Unconstrained (Phenomenological)

| Parameter | Default | Physical meaning |
|-----------|---------|------------------|
| σ_abs | πℓ_P² | Remnant cross-section |
| V_interior | ℓ_P³ | Interior storage volume |
| λ_r | 10⁻⁶⁰ | φ-remnant coupling |
| α | 0.01 | Remnant-energy coupling |
| m_φ | ~H_0 | Scalar field mass |

---

## 7. Observational Constraints

### 7.1 Direct Constraints on PBHs

**Evaporating PBHs (M < 10¹⁵ g):**
- Gamma-ray background limits f_PBH < 10⁻⁸ for M ~ 10¹⁴ g
- 511 keV line constraints

**Larger PBHs (M > 10¹⁵ g):**
- Microlensing: f_PBH < 0.1 for 10⁻⁷ - 10 M_☉
- CMB: f_PBH < 0.01 for > 10⁴ M_☉
- Gravitational waves: constraints on merging PBH binaries

### 7.2 Indirect Constraints on Remnants

If remnants constitute dark matter:
- Must satisfy DM density: Ω_rem ~ 0.26
- Must not overclose universe
- Must be consistent with structure formation

**Number density estimate:**
$$n_{\rm rem} = \frac{\rho_{\rm DM}}{M_P} \approx \frac{2.3 \times 10^{-27} \text{ kg/m}^3}{2.2 \times 10^{-8} \text{ kg}} \sim 10^{-19} \text{ m}^{-3}$$

This is very sparse: mean separation ~ 10⁶ m.

### 7.3 Cosmological Constraints

The modified Friedmann equations must satisfy:
- BBN: G_eff within ~10% of G at T ~ MeV
- CMB: Acoustic peaks unperturbed
- BAO: Sound horizon correctly predicted
- H₀: Consistent with either local or CMB value (or explain tension)

### 7.4 Theoretical Consistency

- **Second Law:** Total entropy must not decrease
- **Energy conservation:** Up to Λ contributions
- **Causality:** No superluminal information transfer

---

## Appendix A: Extrapolations Beyond Established Physics

| Concept | Status | Confidence |
|---------|--------|------------|
| Hawking radiation | Established theory | HIGH |
| BH entropy | Established theory | HIGH |
| PBH existence | Plausible | MEDIUM |
| Remnant formation | Speculative | LOW |
| Remnant stability | Assumed | VERY LOW |
| Interior volume >> exterior | Highly speculative | VERY LOW |
| Recycling mechanism | Phenomenological | VERY LOW |
| φ field existence | Postulated | VERY LOW |
| Hubble tension resolution | Hoped for | UNKNOWN |

---

## Appendix B: Comparison with Standard ΛCDM

In the limit where recycling physics is turned off:
- σ_abs → 0: P_recycle → 0
- α → 0: Remnants decouple from φ
- λ_r → 0: φ not sourced by remnants
- ξ → 0: Minimal coupling
- φ → 0: Field vanishes

The system should recover:
1. Standard Friedmann equations
2. Matter continuity: ρ̇ + 3Hρ = 0
3. Remnants behave as cold dark matter

This limit is tested in `test_dynamics.py`.

---

## Appendix C: Future Improvements

1. **Perturbation theory:** Linear perturbations for CMB/LSS predictions
2. **Spatial dependence:** Inhomogeneous recycling
3. **Better mass functions:** Extended, clustered, spinning PBHs
4. **Quantum corrections:** Beyond semiclassical Hawking
5. **Remnant dynamics:** Formation, merger, interaction physics
6. **Observational signatures:** Specific predictions for detection

---

*Document version: 1.0*
*Generated for HRC Dynamics Module*
