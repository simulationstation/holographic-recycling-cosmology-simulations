# API Reference

## Core Classes

### HRCParameters

```python
from hrc import HRCParameters

params = HRCParameters(
    xi=0.03,        # Non-minimal coupling
    phi_0=0.2,      # Present field value
    m_phi=1.0,      # Scalar mass (in H0 units)
    phi_dot_0=0.0,  # Present field velocity
    f_rem=0.2,      # Remnant DM fraction
    h=0.7,          # Dimensionless Hubble
    Omega_b=0.05,   # Baryon density
    Omega_c=0.25,   # CDM density
)

# Validate parameters
valid, errors = params.validate()
```

### BackgroundCosmology

```python
from hrc import BackgroundCosmology, HRCParameters

params = HRCParameters(xi=0.03, phi_0=0.2)
cosmo = BackgroundCosmology(params)

# Solve from z=0 to z=1100
solution = cosmo.solve(z_max=1100, z_points=500)

# Access solution data
print(f"H(z=0) = {solution.H[0]:.3f}")
print(f"φ(z=1000) = {solution.phi_at(1000):.4f}")
print(f"G_eff(z=0)/G = {solution.G_eff_at(0):.4f}")
```

### EffectiveGravity

```python
from hrc import EffectiveGravity, HRCParameters

params = HRCParameters(xi=0.03, phi_0=0.2)
grav = EffectiveGravity(params)

# Compute G_eff at specific field value
result = grav.G_eff_ratio(0.2)
print(f"G_eff/G = {result.G_eff_ratio:.4f}")
print(f"Physical: {result.is_physical}")

# Critical field value (where G_eff diverges)
phi_crit = grav.critical_phi()
```

## Stability Checks

```python
from hrc.perturbations import (
    check_all_stability,
    StabilityChecker,
)

# Check at a single point
all_passed, results = check_all_stability(
    phi=0.2,
    phi_dot=0.0,
    H=1.0,
    params=params,
)

# Check over full solution
checker = StabilityChecker(params)
stable, stable_mask, messages = checker.check_solution(solution)
```

## Constraints

```python
from hrc.constraints import (
    check_bbn_constraint,
    check_ppn_constraints,
    check_stellar_constraints,
    check_growth_constraints,
)

# BBN constraint
bbn = check_bbn_constraint(solution)
print(f"BBN allowed: {bbn.allowed}")

# PPN (solar system) constraints
ppn_passed, ppn_results = check_ppn_constraints(
    solution=solution, params=params
)

# Stellar constraints
stellar_passed, stellar_results = check_stellar_constraints(solution)

# Structure growth
from hrc.constraints.structure_growth import GrowthCalculator
growth_calc = GrowthCalculator(params, solution)
growth = growth_calc.solve()
passed, results, chi2 = check_growth_constraints(growth)
```

## Observational Likelihoods

### H₀ Likelihoods

```python
from hrc.observables import (
    SH0ESLikelihood,
    TRGBLikelihood,
    CMBDistanceLikelihood,
)

shoes = SH0ESLikelihood()
logL = shoes.log_likelihood(H0_predicted=73.0)

cmb = CMBDistanceLikelihood()
logL_cmb = cmb.log_likelihood(H0_predicted=67.4, params=params)
```

### BAO Likelihood

```python
from hrc.observables import BAOLikelihood

bao = BAOLikelihood()  # Uses DESI + BOSS data
logL = bao.log_likelihood(params, solution)
chi2 = bao.chi2(params, solution)
```

### Supernovae Likelihood

```python
from hrc.observables import PantheonPlusLikelihood

sne = PantheonPlusLikelihood()
logL = sne.log_likelihood(params, solution)
```

### Standard Sirens

```python
from hrc.observables import StandardSirenLikelihood

sirens = StandardSirenLikelihood()
H0_result = sirens.infer_H0()
print(f"H0 = {H0_result.H0:.1f} +{H0_result.H0_err_high:.1f} -{H0_result.H0_err_low:.1f}")
```

## MCMC Sampling

```python
from hrc.sampling import MCMCSampler, PriorSet, UniformPrior

# Define priors
priors = PriorSet()
priors.add("xi", UniformPrior(0.01, 0.1))
priors.add("phi_0", UniformPrior(0.1, 0.4))

# Define likelihood function
def log_likelihood(params_dict):
    params = HRCParameters(**params_dict)
    cosmo = BackgroundCosmology(params)
    solution = cosmo.solve()
    # Compute likelihood...
    return logL

# Run MCMC
sampler = MCMCSampler(log_likelihood, priors)
result = sampler.run(n_steps=1000, n_walkers=32)

# Get results
samples = result.get_flat_samples(burn=200)
summary = result.summary(burn=200)
```

## Plotting

```python
from hrc.plots import (
    plot_geff_evolution,
    plot_hubble_tension_resolution,
    plot_effective_w,
    plot_phase_diagram,
    plot_stability_region,
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_geff_evolution(solution, params, ax=ax)
plt.savefig("geff_evolution.pdf")
```

## Convenience Functions

```python
from hrc import quick_summary, run_full_analysis

# Quick summary of predictions
quick_summary(params)

# Full analysis pipeline
results = run_full_analysis(
    params,
    z_max=1100,
    check_constraints=True,
    verbose=True,
)
```

## CLASS Interface

```python
from hrc.perturbations import CLASSInterface

interface = CLASSInterface(params, solution)

# Compute spectra (uses stub if CLASS not installed)
output = interface.compute()

print(f"θ* = {output.theta_star:.6f}")
print(f"r_s = {output.r_s:.2f} Mpc")
print(f"σ₈ = {output.sigma8:.3f}")

# Write CLASS parameter file
interface.write_ini_file("hrc_class_params.ini")
```
