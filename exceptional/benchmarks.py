"""
Benchmark suite for exceptional Lie algebra sparse engine.

Computes compression ratios and sparse vs dense bracket timing
for all five exceptional algebras: G2, F4, E6, E7, E8.

Run: py exceptional/benchmarks.py
"""

import sys
import os
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import numpy as np
from exceptional.engine import ExceptionalAlgebra


def benchmark_algebra(name: str, n_ops: int = 10000) -> dict:
    """Benchmark a single algebra.

    Returns dict with timing and compression data.
    """
    print(f"  Building {name}...", end="", flush=True)
    t0 = time.time()
    alg = ExceptionalAlgebra(name)
    build_time = time.time() - t0
    print(f" done ({build_time:.2f}s)")

    dim = alg.dim
    n_sc = alg.n_structure_constants
    full_n3 = dim ** 3
    compression = full_n3 / n_sc if n_sc > 0 else float('inf')

    rng = np.random.RandomState(42)

    # Sparse bracket timing
    xs = rng.randn(n_ops, dim)
    ys = rng.randn(n_ops, dim)

    t0 = time.time()
    for i in range(n_ops):
        alg.bracket(xs[i], ys[i])
    sparse_time = time.time() - t0

    # Dense bracket timing (using adjoint matrices)
    t0 = time.time()
    for i in range(n_ops):
        x, y = xs[i], ys[i]
        X = np.einsum('i,iab->ab', x, alg.gen_array)
        Y = np.einsum('i,iab->ab', y, alg.gen_array)
        comm = X @ Y - Y @ X
        # Project back
        np.einsum('ab,kba->k', comm, alg.gen_array)
    dense_time = time.time() - t0

    speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')

    return {
        "name": name,
        "dim": dim,
        "rank": alg.rank,
        "roots": alg.n_roots,
        "nonzero_f": n_sc,
        "full_n3": full_n3,
        "compression": compression,
        "sparse_time": sparse_time,
        "dense_time": dense_time,
        "speedup": speedup,
        "build_time": build_time,
    }


def main():
    print("=" * 90)
    print("  Exceptional Lie Algebra Benchmark Suite")
    print("  DHL-MM Sparse Structure Constant Engine")
    print("=" * 90)

    n_ops = 10000
    print(f"\n  Operations per algebra: {n_ops}\n")

    results = []
    for name in ["G2", "F4", "E6", "E7", "E8"]:
        try:
            r = benchmark_algebra(name, n_ops=n_ops)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print table
    print(f"\n{'='*90}")
    header = (
        f"{'Algebra':>8} {'Dim':>5} {'Rank':>5} {'Roots':>6} "
        f"{'Nonzero f':>10} {'Full n^3':>12} {'Compress':>10} {'Speedup':>10}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        line = (
            f"{r['name']:>8} {r['dim']:>5} {r['rank']:>5} {r['roots']:>6} "
            f"{r['nonzero_f']:>10,} {r['full_n3']:>12,} "
            f"{r['compression']:>9.1f}x {r['speedup']:>9.1f}x"
        )
        print(line)

    print("-" * 90)

    # Timing details
    print(f"\n{'Algebra':>8} {'Sparse (s)':>12} {'Dense (s)':>12} {'Build (s)':>12}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['name']:>8} {r['sparse_time']:>12.4f} {r['dense_time']:>12.4f} "
            f"{r['build_time']:>12.2f}"
        )
    print("-" * 50)

    print(f"\n{'='*90}")
    print("  Benchmark complete.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
