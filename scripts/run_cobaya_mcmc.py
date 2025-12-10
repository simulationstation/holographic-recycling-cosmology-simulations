#!/usr/bin/env python3
"""Run Cobaya MCMC for LCDM or HMDE T06D cosmological models.

This script provides a unified interface for running MCMC chains
with either the standard LCDM model or the HMDE T06D horizon-memory
dark energy model using Cobaya + CAMB.

Features:
- Automatic download of Planck 2018 likelihood data
- Support for multiple parallel chains
- Resume from checkpoints
- Quick test mode for validation (single-point evaluation)
- Short sanity chain mode
- Wrapper for cobaya-run with proper paths

Usage:
    # Run LCDM baseline
    python scripts/run_cobaya_mcmc.py --model lcdm

    # Run HMDE T06D model
    python scripts/run_cobaya_mcmc.py --model hmde

    # Single-point evaluation test (no MCMC, just test likelihoods)
    python scripts/run_cobaya_mcmc.py --model lcdm --test
    python scripts/run_cobaya_mcmc.py --model hmde --test

    # Short sanity chains (2000 samples, relaxed convergence)
    python scripts/run_cobaya_mcmc.py --model lcdm --short
    python scripts/run_cobaya_mcmc.py --model hmde --short

    # Parallel chains
    python scripts/run_cobaya_mcmc.py --model hmde --parallel 4

    # With custom config
    python scripts/run_cobaya_mcmc.py --config path/to/config.yaml
"""

import argparse
import os
import sys
import subprocess
import copy
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def check_cobaya_installation():
    """Check if Cobaya and CAMB are properly installed."""
    try:
        import cobaya
        print(f"[OK] Cobaya version: {cobaya.__version__}")
    except ImportError:
        print("[ERROR] Cobaya not installed. Run: pip install cobaya")
        return False

    try:
        import camb
        print(f"[OK] CAMB version: {camb.__version__}")
    except ImportError:
        print("[ERROR] CAMB not installed. Run: pip install camb")
        return False

    return True


def download_planck_data(force: bool = False):
    """Download Planck 2018 likelihood data using cobaya-install.

    Args:
        force: Force re-download even if data exists
    """
    packages_path = Path.home() / "cobaya_packages"
    packages_path.mkdir(exist_ok=True)

    # Check if likelihoods already exist
    planck_path = packages_path / "data" / "planck_2018"
    if planck_path.exists() and not force:
        print(f"[OK] Planck data already exists at {planck_path}")
        return str(packages_path)

    print("[INFO] Downloading Planck 2018 likelihood data...")
    print("[INFO] This may take a while (~2-5 GB download)")

    # Use cobaya-install command (Cobaya 3.x syntax)
    likelihoods = [
        "planck_2018_highl_plik.TTTEEE_lite",
        "planck_2018_lensing.clik",
        "bao.sdss_dr12_consensus_bao",
        "sn.pantheonplus",
    ]

    cmd = ["cobaya-install"] + likelihoods + ["-p", str(packages_path)]

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Planck data installed at {packages_path}")
        return str(packages_path)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download Planck data: {e}")
        return None


def get_config_path(model: str) -> Path:
    """Get configuration file path for model.

    Args:
        model: Model name ('lcdm' or 'hmde')

    Returns:
        Path to YAML configuration file
    """
    configs = {
        'lcdm': PROJECT_ROOT / "cobaya_configs" / "lcdm_planck_bao_sne.yaml",
        'hmde': PROJECT_ROOT / "cobaya_configs" / "hmde_t06d_planck_bao_sne.yaml",
        't06d': PROJECT_ROOT / "cobaya_configs" / "hmde_t06d_planck_bao_sne.yaml",
    }

    model_key = model.lower()
    if model_key not in configs:
        raise ValueError(f"Unknown model: {model}. Available: {list(configs.keys())}")

    config_path = configs[model_key]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path


def run_single_point_test(model: str, packages_path: str = None):
    """Run a single-point likelihood evaluation to test configuration.

    This uses Cobaya's evaluate sampler to test that all likelihoods
    work correctly at the reference point.

    Args:
        model: Model name ('lcdm' or 'hmde')
        packages_path: Path to cobaya packages

    Returns:
        True if evaluation succeeded with finite likelihood
    """
    import yaml
    from cobaya.yaml import yaml_load_file
    from cobaya.run import run

    config_path = get_config_path(model)
    print(f"\n[INFO] Running single-point test for {model.upper()}")
    print(f"[INFO] Configuration: {config_path}")
    print("=" * 60)

    # Load configuration
    info = yaml_load_file(str(config_path))

    # Get reference values from params
    ref_values = {}
    for param, spec in info.get('params', {}).items():
        if isinstance(spec, dict) and 'ref' in spec:
            ref_values[param] = spec['ref']

    print(f"[INFO] Reference values: {ref_values}")

    # Replace sampler with evaluate
    info['sampler'] = {'evaluate': {'override': ref_values}}

    # Set packages path
    if packages_path:
        info['packages_path'] = packages_path

    # Remove output for test
    if 'output' in info:
        del info['output']

    # Don't resume or force for test
    info['resume'] = False
    info['force'] = True

    # Set PYTHONPATH for custom theory
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(PROJECT_ROOT)
    os.environ['PYTHONPATH'] = env['PYTHONPATH']

    try:
        updated_info, sampler = run(info)

        # Get results
        if hasattr(sampler, 'products') and sampler.products():
            products = sampler.products()
            print("\n[RESULTS]")

            # Check for logposterior
            if 'sample' in products:
                sample = products['sample']
                # Sample is a pandas DataFrame in newer Cobaya
                print(f"  Sample columns: {list(sample.columns) if hasattr(sample, 'columns') else 'N/A'}")

                # Look for log-likelihood columns
                loglike_cols = [c for c in sample.columns if 'chi2' in c.lower() or 'loglike' in c.lower() or 'log' in c.lower()]
                print(f"  Log-likelihood columns: {loglike_cols}")

                for col in loglike_cols:
                    val = sample[col].iloc[0] if len(sample) > 0 else float('nan')
                    print(f"    {col}: {val}")

                # Check if any are -inf
                total_loglike = sum(sample[c].iloc[0] for c in sample.columns if c.startswith('chi2__'))
                print(f"\n  Total chi2: {total_loglike}")

                if total_loglike == float('inf') or total_loglike == float('-inf'):
                    print("\n[ERROR] Likelihood is -inf at reference point!")
                    return False

        print("\n[OK] Single-point evaluation completed successfully")
        return True

    except Exception as e:
        print(f"\n[ERROR] Single-point evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_short_config(base_config: Path, output_dir: Path) -> Path:
    """Create a short-run configuration with reduced samples for sanity testing.

    Args:
        base_config: Path to base configuration
        output_dir: Directory for output

    Returns:
        Path to modified configuration file
    """
    import yaml

    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for short sanity run
    config['sampler']['mcmc']['max_samples'] = 2000
    config['sampler']['mcmc']['Rminus1_stop'] = 0.3  # Relaxed convergence
    config['sampler']['mcmc']['Rminus1_cl_stop'] = 0.5
    config['sampler']['mcmc']['max_tries'] = 50000

    # Update output path
    orig_output = config.get('output', 'results/mcmc/chains')
    model_name = base_config.stem  # e.g., 'lcdm_planck_bao_sne'
    config['output'] = str(output_dir / model_name / "chains")

    # Force and no resume for clean short run
    config['resume'] = False
    config['force'] = True

    short_config_path = output_dir / f"short_{base_config.name}"
    with open(short_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return short_config_path


def run_cobaya(config_path: Path, packages_path: str = None,
               parallel: int = 1, resume: bool = True,
               force: bool = False, debug: bool = False):
    """Run Cobaya MCMC with specified configuration.

    Args:
        config_path: Path to YAML configuration
        packages_path: Path to cobaya packages (data)
        parallel: Number of parallel chains
        resume: Resume from checkpoint if available
        force: Force overwrite existing chains
        debug: Enable debug output
    """
    cmd = ["cobaya-run", str(config_path)]

    if packages_path:
        cmd.extend(["-p", packages_path])

    if parallel > 1:
        cmd.extend(["--parallel", str(parallel)])

    if resume and not force:
        cmd.append("--resume")

    if force:
        cmd.append("--force")

    if debug:
        cmd.append("--debug")

    print(f"\n[INFO] Running Cobaya MCMC")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print(f"[INFO] Started at: {datetime.now().isoformat()}")
    print("=" * 60)

    try:
        # Set PYTHONPATH to include project root for custom theory class
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(PROJECT_ROOT)

        subprocess.run(cmd, check=True, env=env)
        print("=" * 60)
        print(f"[OK] MCMC completed at: {datetime.now().isoformat()}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Cobaya run failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Run interrupted by user")
        return False

    return True


def validate_config(config_path: Path):
    """Validate a Cobaya configuration without running MCMC.

    Args:
        config_path: Path to YAML configuration

    Returns:
        True if configuration is valid
    """
    import yaml

    print(f"\n[INFO] Validating configuration: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required = ['theory', 'likelihood', 'params', 'sampler']
        for section in required:
            if section not in config:
                print(f"[ERROR] Missing required section: {section}")
                return False

        # Check for HMDE theory
        if 'hrc2.cobaya_hmde_model.HMDE_T06D' in config.get('theory', {}):
            print("[INFO] Configuration includes HMDE T06D theory")
            # Verify T06D parameters
            params = config.get('params', {})
            if 'delta_w' not in params or 'a_w' not in params:
                print("[WARNING] T06D parameters (delta_w, a_w) not found in params")

        # Check likelihoods
        likelihoods = list(config.get('likelihood', {}).keys())
        print(f"[INFO] Likelihoods: {likelihoods}")

        print("[OK] Configuration appears valid")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to validate config: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Cobaya MCMC for cosmological models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model lcdm              Run LCDM baseline (full MCMC)
  %(prog)s --model hmde              Run HMDE T06D model (full MCMC)
  %(prog)s --model lcdm --test       Single-point likelihood test (no MCMC)
  %(prog)s --model hmde --test       Single-point likelihood test (no MCMC)
  %(prog)s --model lcdm --short      Short sanity chain (2000 samples)
  %(prog)s --model hmde --short      Short sanity chain (2000 samples)
  %(prog)s --config custom.yaml      Use custom configuration
  %(prog)s --download-data           Download Planck data only
  %(prog)s --validate --model hmde   Validate configuration
        """
    )

    parser.add_argument(
        '--model', '-m',
        choices=['lcdm', 'hmde', 't06d'],
        help='Model to run (lcdm or hmde/t06d)'
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to custom Cobaya configuration file'
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='Number of parallel chains (default: 1)'
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Single-point likelihood evaluation test (no MCMC sampling)'
    )

    parser.add_argument(
        '--short', '-s',
        action='store_true',
        help='Short sanity chain (2000 samples, relaxed convergence)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        default=False,
        help='Resume from checkpoint (default: False)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite existing chains'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate configuration without running'
    )

    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download Planck likelihood data only'
    )

    parser.add_argument(
        '--packages-path',
        type=Path,
        default=Path.home() / "cobaya_packages",
        help='Path to cobaya packages directory (default: ~/cobaya_packages)'
    )

    args = parser.parse_args()

    # Check installation
    if not check_cobaya_installation():
        sys.exit(1)

    # Handle download-only mode
    if args.download_data:
        packages_path = download_planck_data(force=True)
        if packages_path:
            print(f"\n[OK] Planck data available at: {packages_path}")
        sys.exit(0 if packages_path else 1)

    # Determine model
    if not args.model and not args.config:
        parser.error("Either --model or --config must be specified")

    # Determine packages path
    packages_path = str(args.packages_path)
    if not args.packages_path.exists():
        print(f"[WARNING] Packages path not found: {packages_path}")
        print("[INFO] Run with --download-data first to get Planck likelihoods")

    # Handle test mode (single-point evaluation)
    if args.test:
        if args.model:
            success = run_single_point_test(args.model, packages_path)
            sys.exit(0 if success else 1)
        else:
            parser.error("--test requires --model")

    # Determine configuration file
    if args.config:
        config_path = args.config.absolute()
    else:
        config_path = get_config_path(args.model)

    print(f"[INFO] Using configuration: {config_path}")

    # Validation mode
    if args.validate:
        success = validate_config(config_path)
        sys.exit(0 if success else 1)

    # Create short config if needed
    if args.short:
        short_dir = PROJECT_ROOT / "results" / "mcmc_short"
        short_dir.mkdir(parents=True, exist_ok=True)
        config_path = create_short_config(config_path, short_dir)
        print(f"[INFO] Created short-run configuration: {config_path}")
        # Ensure output directory exists
        import yaml
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        output_dir = Path(cfg['output']).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run MCMC
    success = run_cobaya(
        config_path=config_path,
        packages_path=packages_path,
        parallel=args.parallel,
        resume=args.resume,
        force=args.force or args.short,  # Force for short runs
        debug=args.debug
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
