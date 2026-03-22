"""
Benchmark: sparse bracket kernel across all five exceptional Lie algebras.

Compares three methods:
  1. Dense matmul (baseline): full 248x248 matrix multiply
  2. NumPy sparse: vectorized gather-multiply-scatter via np.add.at
  3. C extension: compiled pybind11 kernel (if available)

Usage:
    py benchmarks/sparse_kernel_bench.py
"""
import sys
import os
import time
import numpy as np

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptional.engine import ExceptionalAlgebra
from dhl_mm.csparse import sparse_bracket, has_c_extension


# Algebra configs: (name, iterations)
ALGEBRAS = [
    ("G2",  10000),
    ("F4",  10000),
    ("E6",  1000),
    ("E7",  1000),
    ("E8",  1000),
]


def bench_dense_matmul(alg, x, y, n_iter):
    """Baseline: build adjoint matrix for x, then matmul with y."""
    # Build the dense adjoint action matrix ad_x[k,j] = sum_i x[i] * f_{ij}^k
    ad_x = np.einsum("i,ijk->kj", x, alg.gen_array)

    # Warm up
    _ = ad_x @ y

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = ad_x @ y
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def bench_numpy_sparse(alg, x, y, n_iter):
    """NumPy sparse: gather-multiply-scatter via np.add.at."""
    fI, fJ, fK, fC = alg.fI, alg.fJ, alg.fK, alg.fC
    dim = alg._dim

    # Warm up
    result = np.zeros(dim)
    contributions = x[fI] * y[fJ] * fC
    np.add.at(result, fK, contributions)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        result = np.zeros(dim)
        contributions = x[fI] * y[fJ] * fC
        np.add.at(result, fK, contributions)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def bench_c_extension(alg, x, y, n_iter):
    """C extension: compiled pybind11 kernel."""
    if not has_c_extension():
        return None

    fI = alg.fI.astype(np.int32)
    fJ = alg.fJ.astype(np.int32)
    fK = alg.fK.astype(np.int32)
    fC = alg.fC.astype(np.float64)
    dim = alg._dim

    # Warm up
    _ = sparse_bracket(x, y, fI, fJ, fK, fC, dim)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = sparse_bracket(x, y, fI, fJ, fK, fC, dim)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def main():
    print(f"C extension available: {has_c_extension()}")
    print()

    header = f"{'Algebra':<9}{'Dim':>5}  {'NNZ':>7}  {'Method':<18}{'Time(ms)':>10}  {'Speedup vs Dense':>16}"
    print(header)
    print("-" * len(header))

    for name, n_iter in ALGEBRAS:
        try:
            alg = ExceptionalAlgebra(name)
        except Exception as e:
            print(f"{name:<9}  (failed to load: {e})")
            continue

        dim = alg._dim
        nnz = len(alg.fI)
        x = np.random.randn(dim).astype(np.float64)
        y = np.random.randn(dim).astype(np.float64)

        # Dense baseline
        t_dense = bench_dense_matmul(alg, x, y, n_iter)

        # NumPy sparse
        t_numpy = bench_numpy_sparse(alg, x, y, n_iter)

        # C extension
        t_c = bench_c_extension(alg, x, y, n_iter)

        # Print results
        print(f"{name:<9}{dim:>5}  {nnz:>7}  {'Dense matmul':<18}{t_dense:>10.4f}  {'1.0x':>16}")
        speedup_np = t_dense / t_numpy if t_numpy > 0 else 0
        print(f"{'':<9}{'':>5}  {'':>7}  {'NumPy sparse':<18}{t_numpy:>10.4f}  {f'{speedup_np:.1f}x':>16}")

        if t_c is not None:
            speedup_c = t_dense / t_c if t_c > 0 else 0
            print(f"{'':<9}{'':>5}  {'':>7}  {'C extension':<18}{t_c:>10.4f}  {f'{speedup_c:.1f}x':>16}")
        else:
            print(f"{'':<9}{'':>5}  {'':>7}  {'C extension':<18}{'(not compiled)':>10}  {'N/A':>16}")

        # Correctness check: compare numpy sparse vs C extension
        if t_c is not None:
            fI = alg.fI.astype(np.int32)
            fJ = alg.fJ.astype(np.int32)
            fK = alg.fK.astype(np.int32)
            fC = alg.fC.astype(np.float64)
            ref = np.zeros(dim)
            np.add.at(ref, fK, x[fI] * y[fJ] * fC)
            c_result = sparse_bracket(x, y, fI, fJ, fK, fC, dim)
            maxdiff = np.max(np.abs(ref - c_result))
            print(f"{'':<9}{'':>5}  {'':>7}  {'  (max diff vs np)':<18}{maxdiff:>10.2e}")

        print()


if __name__ == "__main__":
    main()
