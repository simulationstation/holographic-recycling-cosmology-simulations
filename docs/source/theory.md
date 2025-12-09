# Theoretical Foundations of HRC

This document provides the mathematical derivations underlying the HRC model.

## 1. The HRC Action

The total action is:

$$S = S_{\rm EH} + S_\phi + S_m + S_{\rm rem}$$

where:

### Einstein-Hilbert with cosmological constant
$$S_{\rm EH} = \frac{1}{16\pi G}\int d^4x \sqrt{-g}(R - 2\Lambda)$$

### Scalar field with non-minimal coupling
$$S_\phi = \int d^4x \sqrt{-g}\left[-\frac{1}{2}(\partial\phi)^2 - V(\phi) - \xi\phi R\right]$$

The non-minimal coupling $\xi\phi R$ is the crucial new physics ingredient.

### Matter sector
$$S_m = -\int d^4x \sqrt{-g}\rho_m(1 + \Pi)$$

### Remnant sector
$$S_{\rm rem} = -\int d^4x \sqrt{-g}\rho_{\rm rem}(1 + \alpha\phi)$$

## 2. Field Equations

Varying with respect to the metric yields modified Einstein equations:

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G_{\rm eff}\left(T^{(m)}_{\mu\nu} + T^{(\phi)}_{\mu\nu} + T^{(\rm rem)}_{\mu\nu}\right)$$

where the **effective gravitational coupling** is:

$$\boxed{G_{\rm eff} = \frac{G}{1 - 8\pi G\xi\phi}}$$

For $\xi\phi > 0$, gravity is effectively stronger ($G_{\rm eff} > G$).

## 3. Scalar Field Equation of Motion

The scalar field satisfies:

$$\Box\phi - V'(\phi) - \xi R = 0$$

In an expanding FLRW universe:

$$\ddot{\phi} + 3H\dot{\phi} + V'(\phi) + \xi R = 0$$

where the Ricci scalar is:

$$R = 6\left(2H^2 + \dot{H}\right)$$

## 4. Modified Friedmann Equations

In a flat FLRW universe:

$$H^2 = \frac{8\pi G_{\rm eff}}{3}\left(\rho_m + \rho_r + \rho_\phi\right) + \frac{\Lambda}{3}$$

with scalar field energy density and pressure:

$$\rho_\phi = \frac{1}{2}\dot{\phi}^2 + V(\phi)$$
$$P_\phi = \frac{1}{2}\dot{\phi}^2 - V(\phi)$$

## 5. Resolution of the Hubble Tension

The key insight is that different cosmological probes measure $H_0$ through different physical processes:

### Local measurements (z ≈ 0)
Directly probe the Hubble flow, sensitive to $H(z\approx 0)$ which depends on $G_{\rm eff}(\text{today})$:

$$H_0^{\rm local} \approx \frac{H_0^{\rm true}}{\sqrt{G_{\rm eff}(z=0)/G}}$$

### CMB inference (z ≈ 1100)
Measures the angular scale $\theta_* = r_s/D_A$ and assumes standard physics. If $G_{\rm eff}$ was different at recombination:

$$H_0^{\rm CMB} \approx H_0^{\rm true}\left(1 + 0.4\frac{G_{\rm eff}(0) - G_{\rm eff}(z_{\rm rec})}{G_{\rm eff}(z_{\rm rec})}\right)$$

With typical HRC parameters ($\xi = 0.03$, $\phi_0 = 0.2$):
- $\Delta H_0 \approx 6$ km/s/Mpc
- Matching the observed Hubble tension!

## 6. Stability Conditions

For a theoretically consistent model, we require:

### No-ghost condition (positive kinetic term)
$$Q_s = M_{\rm eff}^2 > 0 \quad \Rightarrow \quad 1 - 8\pi\xi\phi > 0$$

### Gradient stability (positive sound speed squared)
$$c_s^2 = \frac{P_s}{Q_s} > 0$$

### Tensor stability
- No tensor ghost: $G_T > 0$
- Luminal propagation: $c_T^2 = 1$
- Massless graviton: $m_T^2 = 0$

## 7. Observational Constraints

### BBN (Big Bang Nucleosynthesis)
$$\left|\frac{\Delta G}{G}\right|_{\rm BBN} < 10\%$$

### Solar System (PPN)
$$\left|\frac{\dot{G}}{G}\right|_{\rm today} < 1.5 \times 10^{-12} \text{ yr}^{-1}$$
$$|γ - 1| < 2.3 \times 10^{-5}$$

### Stellar Evolution
$$\left|\frac{\Delta G}{G}\right|_{\rm solar} < 2\%$$

## 8. Numerical Implementation

The code solves the coupled system:

1. **Background**: Integrate Friedmann + scalar field ODEs from $z=0$ to $z_{\rm max}$
2. **G_eff**: Compute at each step and check for divergences
3. **Stability**: Verify all conditions at each point
4. **Observables**: Compute distances, likelihoods, predictions

The integration uses `scipy.integrate.solve_ivp` with adaptive step control.

## 9. Model Parameters

| Parameter | Symbol | Fiducial Value | Description |
|-----------|--------|----------------|-------------|
| Non-minimal coupling | ξ | 0.03 | Coupling strength |
| Field value today | φ₀ | 0.2 | In Planck units |
| Scalar mass | m_φ | ~H₀ | Mass scale |
| Remnant fraction | f_rem | 0.2 | Fraction of DM |
| Remnant coupling | α | 0.01 | Remnant-φ coupling |

## References

1. Planck Collaboration (2020), A&A 641, A6
2. Riess et al. (2024), SH0ES
3. DESI Collaboration (2024), arXiv:2404.03002
4. Rovelli (2018), arXiv:1805.03872
5. Chen, Ong & Yeom (2015), Physics Reports 603, 1-45
