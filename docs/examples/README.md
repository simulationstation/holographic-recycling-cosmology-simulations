# HRC Examples

This directory contains example scripts and notebooks for using the HRC package.

## Quick Start Examples

### 1. Basic Usage

```python
from hrc import HRCParameters, BackgroundCosmology, quick_summary

# Create parameters
params = HRCParameters(xi=0.03, phi_0=0.2)

# Get quick summary
quick_summary(params)
```

### 2. Solve Background Cosmology

```python
from hrc import HRCParameters, BackgroundCosmology

params = HRCParameters(xi=0.03, phi_0=0.2, h=0.7)
cosmo = BackgroundCosmology(params)

# Solve from z=0 to z=1100 (recombination)
solution = cosmo.solve(z_max=1100, z_points=500)

# Print key results
print(f"Integration success: {solution.success}")
print(f"G_eff/G at z=0: {solution.G_eff_at(0):.4f}")
print(f"G_eff/G at z=1089: {solution.G_eff_at(1089):.4f}")
print(f"φ at z=0: {solution.phi[0]:.4f}")
print(f"φ at z=1089: {solution.phi_at(1089):.4f}")
```

### 3. Check Hubble Tension Resolution

```python
from hrc import HRCParameters, BackgroundCosmology, compute_hubble_tension

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)
solution = cosmo.solve(z_max=1100)

tension = compute_hubble_tension(solution, params)

print(f"H₀ (local): {tension['H0_local']:.1f} km/s/Mpc")
print(f"H₀ (CMB): {tension['H0_cmb']:.1f} km/s/Mpc")
print(f"ΔH₀: {tension['Delta_H0']:.1f} km/s/Mpc")
print(f"Tension resolved: {tension['resolves_tension']}")
```

### 4. Run Stability Checks

```python
from hrc import HRCParameters, BackgroundCosmology
from hrc.perturbations import StabilityChecker

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)
solution = cosmo.solve(z_max=100)

checker = StabilityChecker(params)
stable, mask, messages = checker.check_solution(solution)

print(f"All stable: {stable}")
if not stable:
    for msg in messages:
        print(f"  {msg}")
```

### 5. Check Observational Constraints

```python
from hrc import HRCParameters, BackgroundCosmology
from hrc.constraints import (
    check_bbn_constraint,
    check_ppn_constraints,
    check_stellar_constraints,
)

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)
solution = cosmo.solve(z_max=1100)

# BBN
bbn = check_bbn_constraint(solution)
print(f"BBN: {'✓' if bbn.allowed else '✗'} {bbn.message}")

# Solar system
ppn_passed, ppn_results = check_ppn_constraints(solution, params=params)
print(f"PPN: {'✓' if ppn_passed else '✗'}")

# Stellar
stellar_passed, stellar_results = check_stellar_constraints(solution)
print(f"Stellar: {'✓' if stellar_passed else '✗'}")
```

### 6. Compute Likelihoods

```python
from hrc import HRCParameters, BackgroundCosmology
from hrc.observables import SH0ESLikelihood, BAOLikelihood

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)
solution = cosmo.solve(z_max=10)

# SH0ES likelihood
shoes = SH0ESLikelihood()
H0_local = 70 * solution.G_eff_at(0)**0.5  # Approximate
logL_shoes = shoes.log_likelihood(H0_local)
print(f"SH0ES log-likelihood: {logL_shoes:.2f}")

# BAO likelihood
bao = BAOLikelihood()
logL_bao = bao.log_likelihood(params, solution)
print(f"BAO log-likelihood: {logL_bao:.2f}")
```

### 7. Create Plots

```python
from hrc import HRCParameters, BackgroundCosmology
from hrc.plots import plot_geff_evolution, plot_hubble_tension_resolution
import matplotlib.pyplot as plt

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)
solution = cosmo.solve(z_max=1100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_geff_evolution(solution, params, ax=axes[0])
plot_hubble_tension_resolution(solution, params, ax=axes[1])

plt.tight_layout()
plt.savefig("hrc_plots.pdf")
plt.show()
```

### 8. Parameter Space Exploration

```python
from hrc import HRCParameters, BackgroundCosmology, compute_hubble_tension
import numpy as np

xi_values = [0.01, 0.02, 0.03, 0.05]
phi_values = [0.1, 0.15, 0.2, 0.25, 0.3]

results = []
for xi in xi_values:
    for phi in phi_values:
        params = HRCParameters(xi=xi, phi_0=phi)
        valid, _ = params.validate()
        if not valid:
            continue

        cosmo = BackgroundCosmology(params)
        solution = cosmo.solve(z_max=1100)

        if solution.success:
            tension = compute_hubble_tension(solution, params)
            if tension['valid']:
                results.append({
                    'xi': xi,
                    'phi_0': phi,
                    'Delta_H0': tension['Delta_H0'],
                    'resolves': tension['resolves_tension'],
                })

# Print results
print("xi\tφ₀\tΔH₀\tResolves?")
for r in results:
    print(f"{r['xi']:.2f}\t{r['phi_0']:.2f}\t{r['Delta_H0']:.1f}\t{r['resolves']}")
```

## Jupyter Notebooks

- `basic_analysis.ipynb`: Step-by-step HRC analysis
- `parameter_scan.ipynb`: Explore parameter space
- `mcmc_inference.ipynb`: Full MCMC parameter inference
- `comparison_lcdm.ipynb`: HRC vs ΛCDM comparison
