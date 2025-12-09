# Observational Data for HRC Analysis

This directory contains cosmological data files used for constraining HRC parameters.

## Data Sources

### 1. Planck CMB (2018 + 2020)
- **Source**: Planck Collaboration, A&A 641, A6 (2020)
- **Data**: ΛCDM parameter chains and best-fit values
- **URL**: https://pla.esac.esa.int/

### 2. ACT DR6 (2024)
- **Source**: ACT Collaboration (2024)
- **Data**: CMB power spectrum and derived parameters
- **URL**: https://lambda.gsfc.nasa.gov/product/act/

### 3. DESI BAO (2024)
- **Source**: DESI Collaboration, arXiv:2404.03002
- **Data**: D_V/r_d measurements at z = 0.3-2.3
- **URL**: https://data.desi.lbl.gov/

### 4. SH0ES (2024)
- **Source**: Riess et al., ApJ (2024)
- **Data**: H0 = 73.04 ± 1.04 km/s/Mpc
- **URL**: https://www.stsci.edu/~ariess/

### 5. Pantheon+ (2022)
- **Source**: Scolnic et al., ApJ 938, 113 (2022)
- **Data**: 1701 Type Ia SN light curves
- **URL**: https://github.com/PantheonPlusSH0ES/DataRelease

### 6. BBN Constraints
- **Source**: Particle Data Group (2024)
- **Data**: Primordial abundances Y_p, D/H

## File Descriptions

- `planck_2018_params.json`: Planck best-fit parameters with errors
- `desi_bao_2024.csv`: DESI BAO measurements
- `pantheon_plus_binned.csv`: Binned Pantheon+ distance moduli
- `local_h0_measurements.json`: Collection of local H0 values

## Data Format

All CSV files use comma-separated values with header row.
JSON files follow standard format with parameter dictionaries.

## Usage

```python
from hrc_observations import ObservationalData

data = ObservationalData(data_dir='./data')
print(data.summary())
```

## License

Observational data is subject to the licensing terms of the original publications.
See individual sources for details.

## Notes

- Some data is hardcoded in `hrc_observations.py` using published values
- Full likelihood analysis requires downloading original covariance matrices
- Binned data is used for computational efficiency in MCMC
