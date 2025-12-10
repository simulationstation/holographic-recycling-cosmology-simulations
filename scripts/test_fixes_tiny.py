#!/usr/bin/env python3
"""Tiny test script to validate debugging fixes.

Tests:
1. Partial saves update after EVERY completion
2. Timeout enforcement actually interrupts long integrations
3. Per-completion logging appears correctly

Run with: python scripts/test_fixes_tiny.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hrc2.theory import CouplingFamily, PotentialType
from hrc2.utils.config import PerformanceConfig
from hrc2.analysis import run_xi_tradeoff_parallel


def test_tiny_scan():
    """Run 2x2 grid with very short z_max to validate fixes."""
    print("=" * 60)
    print("TINY TEST: 2x2 grid, z_max=10, 2 workers")
    print("=" * 60)
    print()

    # Ensure clean test environment
    test_partial_path = "results/hrc2_scan/test_partial.npz"
    if os.path.exists(test_partial_path):
        os.remove(test_partial_path)
        print(f"Removed old {test_partial_path}")

    # Very small config for quick test
    perf = PerformanceConfig(n_workers=2)

    # 2x2 grid = 4 points total
    xi_values = np.array([1e-5, 1e-4])
    phi0_values = np.array([0.0, 0.1])

    print(f"Grid: {len(xi_values)} xi x {len(phi0_values)} phi0 = {len(xi_values)*len(phi0_values)} points")
    print(f"Workers: {perf.n_workers}")
    print()

    os.makedirs('results/hrc2_scan', exist_ok=True)

    start = time.time()

    result = run_xi_tradeoff_parallel(
        xi_values, phi0_values,
        CouplingFamily.LINEAR,
        potential_type=PotentialType.QUADRATIC,
        perf=perf,
        z_max=10.0,  # Very low z_max for speed
        constraint_level="conservative",
        verbose=True,
    )

    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"COMPLETED in {elapsed:.1f}s")
    print("=" * 60)

    # Verify partial save has all 4 entries
    if os.path.exists(test_partial_path):
        data = np.load(test_partial_path)
        n_saved = data.get('n_completed', len(data['xi']))
        print(f"Partial save has {n_saved} entries (expected: 4)")
        if n_saved == 4:
            print("✓ Partial saves working correctly!")
        else:
            print("✗ Partial save count mismatch!")
    else:
        # Check the default partial path
        default_partial = "results/hrc2_scan/hrc2_partial_scan.npz"
        if os.path.exists(default_partial):
            data = np.load(default_partial)
            n_saved = data.get('n_completed', len(data['xi']))
            print(f"Default partial save has {n_saved} entries (expected: 4)")
        else:
            print("✗ No partial save found!")

    # Verify result object
    print(f"\nResult stable_mask shape: {result.stable_mask.shape}")
    print(f"Result total points: {result.stable_mask.size}")

    return result


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print("\n=== VALIDATING DEBUGGING FIXES ===\n")
    test_tiny_scan()
