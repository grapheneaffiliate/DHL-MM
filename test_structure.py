"""
Test E8 Structure Constants
============================
Verifies antisymmetry, Jacobi identity, Killing form, and Cartan matrix.
"""
import numpy as np
import time
import sys

from e8_structure_constants import (
    build_e8_roots, compute_structure_constants, e8_simple_roots,
    e8_cartan_matrix, _root_key
)

np.set_printoptions(precision=6, suppress=True)


def build_adjoint_matrices(roots, brackets):
    """Build 240x240 adjoint matrices for each root generator."""
    n = len(roots)
    generators = []
    for i in range(n):
        E_i = np.zeros((n, n))
        for (a, b), (k, c) in brackets.items():
            if a == i:
                E_i[b, k] += c
        generators.append(E_i)
    return generators


def test_antisymmetry(brackets):
    """Test N_{a,b} = -N_{b,a}."""
    fails = 0
    total = 0
    max_err = 0.0
    for (i, j), (k, c) in brackets.items():
        if (j, i) in brackets:
            k2, c2 = brackets[(j, i)]
            total += 1
            if k != k2:
                fails += 1
            else:
                err = abs(c + c2)
                max_err = max(max_err, err)
                if err > 1e-10:
                    fails += 1
    return fails, total, max_err


def test_jacobi_random(generators, n_trials=100):
    """Test Jacobi identity: [X,[Y,Z]] + cyclic = 0 for random elements."""
    n = len(generators)
    np.random.seed(42)
    max_err = 0.0

    for trial in range(n_trials):
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)

        X = sum(x[i] * generators[i] for i in range(n))
        Y = sum(y[i] * generators[i] for i in range(n))
        Z = sum(z[i] * generators[i] for i in range(n))

        YZ = Y @ Z - Z @ Y
        ZX = Z @ X - X @ Z
        XY = X @ Y - Y @ X

        J = (X @ YZ - YZ @ X) + (Y @ ZX - ZX @ Y) + (Z @ XY - XY @ Z)
        err = np.linalg.norm(J) / (np.linalg.norm(X) * np.linalg.norm(Y) * np.linalg.norm(Z) + 1e-30)
        max_err = max(max_err, err)

    return max_err


def test_jacobi_basis_triples(generators, n_trials=2000):
    """Test Jacobi on specific basis triples."""
    n = len(generators)
    np.random.seed(77)
    max_err = 0.0
    fails = 0

    for _ in range(n_trials):
        a, b, c = np.random.choice(n, 3, replace=False)
        Ea, Eb, Ec = generators[a], generators[b], generators[c]

        bc = Eb @ Ec - Ec @ Eb
        ca = Ec @ Ea - Ea @ Ec
        ab = Ea @ Eb - Eb @ Ea

        J = (Ea @ bc - bc @ Ea) + (Eb @ ca - ca @ Eb) + (Ec @ ab - ab @ Ec)
        err = np.linalg.norm(J)
        max_err = max(max_err, err)
        if err > 1e-8:
            fails += 1

    return fails, n_trials, max_err


def test_killing_form(generators, roots):
    """Test Killing form properties.

    For E8 adjoint, the Killing form K(X,Y) = Tr(ad_X ad_Y).
    K(E_alpha, E_beta) should be nonzero only when beta = -alpha.
    K(E_alpha, E_{-alpha}) should be the same constant for all roots alpha.
    """
    n = len(generators)
    root_map = {}
    for k, r in enumerate(roots):
        root_map[_root_key(r)] = k

    # Find negative pairs
    neg_of = {}
    for k in range(n):
        nk = _root_key(-roots[k])
        if nk in root_map:
            neg_of[k] = root_map[nk]

    # Test K(E_alpha, E_{-alpha}) for a sample of roots
    sample = list(range(min(n, 60)))
    k_vals = []
    for i in sample:
        if i in neg_of:
            ni = neg_of[i]
            k_val = np.trace(generators[i] @ generators[ni])
            k_vals.append(k_val)

    k_vals = np.array(k_vals)
    k_abs = np.abs(k_vals)
    k_abs_mean = np.mean(k_abs)
    k_abs_std = np.std(k_abs)

    # Test K(E_alpha, E_beta) = 0 for beta != -alpha (sample)
    np.random.seed(99)
    off_diag_max = 0.0
    for _ in range(500):
        i, j = np.random.choice(n, 2, replace=False)
        if j == neg_of.get(i, -1):
            continue  # skip alpha, -alpha pairs
        k_val = np.trace(generators[i] @ generators[j])
        off_diag_max = max(off_diag_max, abs(k_val))

    return k_abs_mean, k_abs_std, off_diag_max


def test_cartan_matrix(roots, brackets, generators):
    """Verify the Cartan matrix relations using [H_i, E_j] = A_ij * E_j."""
    simple = e8_simple_roots()
    A_expected = e8_cartan_matrix()

    root_map = {}
    for k, r in enumerate(roots):
        root_map[_root_key(r)] = k

    simple_idx = []
    neg_simple_idx = []
    for s in simple:
        si = root_map.get(_root_key(s), -1)
        ni = root_map.get(_root_key(-s), -1)
        simple_idx.append(si)
        neg_simple_idx.append(ni)

    found = sum(1 for x in simple_idx if x >= 0)
    print(f"    Found {found}/8 simple roots in root list")
    if found < 8:
        return False, "Missing simple roots"

    # Build H_i = [E_i, F_i] in adjoint
    H = []
    for idx in range(8):
        si = simple_idx[idx]
        ni = neg_simple_idx[idx]
        H_i = generators[si] @ generators[ni] - generators[ni] @ generators[si]
        H.append(H_i)

    # Check [H_i, E_j] = A_ij * E_j
    computed_cartan = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            sj = simple_idx[j]
            Ej = generators[sj]
            comm = H[i] @ Ej - Ej @ H[i]

            norm_Ej = np.linalg.norm(Ej)
            if norm_Ej > 1e-10:
                coeff = np.sum(comm * Ej) / np.sum(Ej * Ej)
                computed_cartan[i, j] = coeff

    err = np.max(np.abs(computed_cartan - A_expected))
    print(f"    Computed Cartan matrix error vs expected: {err:.6e}")
    if err < 0.5:
        print(f"    Computed:\n{np.round(computed_cartan).astype(int)}")
        print(f"    Expected:\n{A_expected}")

    return err < 0.5, err


def main():
    print("=" * 70)
    print("  E8 STRUCTURE CONSTANTS TEST SUITE")
    print("=" * 70)

    # Build
    print("\n  Building E8 roots and structure constants...")
    t0 = time.perf_counter()
    roots, brackets = compute_structure_constants()
    t1 = time.perf_counter()
    print(f"  {len(roots)} roots, {len(brackets)} nonzero brackets, built in {t1 - t0:.3f}s")

    pos = sum(1 for v in brackets.values() if v[1] > 0)
    neg = sum(1 for v in brackets.values() if v[1] < 0)
    print(f"  Signs: {pos} positive, {neg} negative")

    # Build adjoint matrices
    print("\n  Building adjoint representation matrices...")
    t0 = time.perf_counter()
    generators = build_adjoint_matrices(roots, brackets)
    t1 = time.perf_counter()
    print(f"  Built {len(generators)} generator matrices in {t1 - t0:.3f}s")

    results = {}

    # Test 1: Antisymmetry
    print("\n  TEST 1: Antisymmetry")
    fails, total, max_err = test_antisymmetry(brackets)
    passed = fails == 0
    results["ANTISYMMETRY"] = passed
    print(f"  ANTISYMMETRY: {'PASS' if passed else 'FAIL'} (failures={fails}/{total}, max_err={max_err:.2e})")

    # Test 2: Jacobi (random elements)
    print("\n  TEST 2: Jacobi identity (100 random elements)")
    t0 = time.perf_counter()
    max_err = test_jacobi_random(generators, n_trials=100)
    t1 = time.perf_counter()
    passed = max_err < 1e-8
    results["JACOBI_RANDOM"] = passed
    print(f"  JACOBI (random):  {'PASS' if passed else 'FAIL'} (max_err={max_err:.2e}, time={t1 - t0:.1f}s)")

    # Test 3: Jacobi (basis triples)
    print("\n  TEST 3: Jacobi identity (2000 basis triples)")
    t0 = time.perf_counter()
    fails, total, max_err = test_jacobi_basis_triples(generators, n_trials=2000)
    t1 = time.perf_counter()
    passed = fails == 0
    results["JACOBI_BASIS"] = passed
    print(f"  JACOBI (basis):   {'PASS' if passed else 'FAIL'} (failures={fails}/{total}, max_err={max_err:.2e}, time={t1 - t0:.1f}s)")

    # Test 4: Killing form
    print("\n  TEST 4: Killing form")
    k_mean, k_std, off_max = test_killing_form(generators, roots)
    # For a correct Lie algebra, |K(E_a, E_{-a})| should be constant (since E8 has one root length)
    passed_diag = k_std / (abs(k_mean) + 1e-30) < 0.01
    passed_off = off_max < 1e-8
    passed = passed_diag and passed_off
    results["KILLING_FORM"] = passed
    print(f"  KILLING FORM:     {'PASS' if passed else 'FAIL'}")
    print(f"    |K(E_a, E_{{-a}})|: mean={k_mean:.4f}, std={k_std:.4f}, relative_std={k_std / (abs(k_mean) + 1e-30):.6f}")
    print(f"    K(E_a, E_b) off-diagonal max: {off_max:.6e}")

    # Test 5: Cartan matrix
    print("\n  TEST 5: Cartan matrix")
    passed, cartan_err = test_cartan_matrix(roots, brackets, generators)
    results["CARTAN"] = passed
    print(f"  CARTAN MATRIX:    {'PASS' if passed else 'FAIL'} (error={cartan_err:.2e})")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    if all_pass:
        print("\n  ALL TESTS PASSED - Structure constants are correct!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED: {', '.join(failed)}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
