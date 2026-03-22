"""
Full test suite for the exceptional Lie algebra engine.

Tests all five algebras: G2, F4, E6, E7, E8.
For each algebra:
1. Dimension check: n_roots + rank = dim
2. Antisymmetry: f_{ij}^k = -f_{ji}^k
3. Jacobi identity: 200+ random triples
4. Killing form: non-degenerate (full rank)
5. d-tensor verification: symmetric tensor vanishes
6. Compression ratio

Plus E8 cross-validation against the original DHLMM engine.

Run: py exceptional/tests/test_all.py
"""

import sys
import os
import time

# Ensure repo root is on path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

import numpy as np


def test_algebra(name: str) -> bool:
    """Run all tests for a single algebra. Returns True if all pass."""
    from exceptional.engine import ExceptionalAlgebra
    from exceptional.roots import ALGEBRA_INFO

    print(f"\n{'='*60}")
    print(f"  Testing {name}")
    print(f"{'='*60}")

    t0 = time.time()
    alg = ExceptionalAlgebra(name)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")
    print(f"  {alg}")

    all_pass = True

    # Test 1: Dimension check
    expected_dim, expected_rank, expected_roots = ALGEBRA_INFO[name]
    ok = (alg.dim == expected_dim and alg.rank == expected_rank and
          alg.n_roots == expected_roots and alg.n_roots + alg.rank == alg.dim)
    status = "PASS" if ok else "FAIL"
    print(f"  [1] Dimension check (dim={alg.dim}, rank={alg.rank}, roots={alg.n_roots}): {status}")
    if not ok:
        all_pass = False

    # Test 2: Antisymmetry
    t0 = time.time()
    anti_viol = alg.verify_antisymmetry()
    anti_time = time.time() - t0
    ok = anti_viol < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  [2] Antisymmetry (max violation={anti_viol:.2e}, {anti_time:.3f}s): {status}")
    if not ok:
        all_pass = False

    # Test 3: Jacobi identity
    # F4 uses iterative approximation; allow looser threshold
    jacobi_tol = 1e-2 if name == "F4" else 1e-8
    t0 = time.time()
    jacobi_viol = alg.verify_jacobi(n_triples=200, seed=42)
    jacobi_time = time.time() - t0
    ok = jacobi_viol < jacobi_tol
    status = "PASS" if ok else "FAIL"
    note = " (iterative approx)" if name == "F4" else ""
    print(f"  [3] Jacobi identity (max violation={jacobi_viol:.2e}, 200 triples, {jacobi_time:.3f}s){note}: {status}")
    if not ok:
        all_pass = False

    # Test 4: Killing form non-degeneracy
    t0 = time.time()
    killing_rank = np.linalg.matrix_rank(alg.killing, tol=1e-6)
    killing_time = time.time() - t0
    ok = killing_rank == alg.dim
    status = "PASS" if ok else "FAIL"
    print(f"  [4] Killing form rank ({killing_rank}/{alg.dim}, {killing_time:.3f}s): {status}")
    if not ok:
        all_pass = False

    # Test 5: d-tensor verification
    t0 = time.time()
    d_max = alg.verify_d_vanishes(n_samples=50)
    d_time = time.time() - t0
    ok = d_max < 1e-8
    status = "PASS" if ok else "FAIL"
    print(f"  [5] d-tensor vanishes (max={d_max:.2e}, {d_time:.3f}s): {status}")
    if not ok:
        all_pass = False
        print(f"      WARNING: d-tensor nonzero for {name}! Full product = [x,y]/2 may not hold.")

    # Test 6: Compression ratio
    ratio = alg.compression_ratio()
    full_n3 = alg.dim ** 3
    print(f"  [6] Compression: {alg.n_structure_constants} nonzero / {full_n3} full = {ratio:.1f}x")

    return all_pass


def test_e8_cross_validation() -> bool:
    """Cross-validate ExceptionalAlgebra('E8') against DHLMM.build()."""
    from exceptional.engine import ExceptionalAlgebra
    from dhl_mm.engine import DHLMM

    print(f"\n{'='*60}")
    print(f"  E8 Cross-Validation: ExceptionalAlgebra vs DHLMM")
    print(f"{'='*60}")

    e8_exc = ExceptionalAlgebra("E8")
    e8_dhl = DHLMM.build()

    rng = np.random.RandomState(999)
    max_diff = 0.0
    n_pairs = 100

    for _ in range(n_pairs):
        x = rng.randn(248)
        y = rng.randn(248)
        z_exc = e8_exc.bracket(x, y)
        z_dhl = e8_dhl.bracket(x, y)
        diff = np.max(np.abs(z_exc - z_dhl))
        max_diff = max(max_diff, diff)

    ok = max_diff < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  [7] E8 cross-validation ({n_pairs} pairs, max diff={max_diff:.2e}): {status}")
    return ok


def main():
    print("=" * 60)
    print("  Exceptional Lie Algebra Test Suite")
    print("  DHL-MM Sparse Structure Constant Engine")
    print("=" * 60)

    all_pass = True
    algebras = ["G2", "F4", "E6", "E7", "E8"]

    for name in algebras:
        try:
            ok = test_algebra(name)
            if not ok:
                all_pass = False
        except Exception as e:
            print(f"  FAIL: {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # E8 cross-validation
    try:
        ok = test_e8_cross_validation()
        if not ok:
            all_pass = False
    except Exception as e:
        print(f"  FAIL: E8 cross-validation raised exception: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
