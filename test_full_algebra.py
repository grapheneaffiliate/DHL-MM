"""
Test and Benchmark for Full 248-dim E8 Algebra
================================================

Tests:
1. Antisymmetry of structure constants
2. Jacobi identity for root + negative root triples
3. Leibniz rule: [H_i, [E_a, E_b]] = [H_i,E_a]E_b + E_a[H_i,E_b]
4. Jacobi for random 248-dim vectors (machine epsilon)
5. Jacobi for basis triples (exhaustive sampling)

Benchmark:
- Structure constant bracket vs full matrix bracket (248-dim)
- Count nonzero entries
- Speedup measurement
"""
import numpy as np
import time
import sys

from e8_full_algebra import (
    compute_full_structure_constants, build_248_adjoint_matrices,
    lie_bracket_248, lie_bracket_248_matrix
)
from e8_structure_constants import (
    build_e8_roots, e8_simple_roots, _root_key
)

np.set_printoptions(precision=8, suppress=True)


def main():
    print("=" * 70)
    print("  E8 FULL 248-DIM ALGEBRA: TEST SUITE + BENCHMARK")
    print("=" * 70)

    # ===== BUILD =====
    print("\n  Phase 1: Building full 248-dim structure constants...")
    t0 = time.perf_counter()
    roots, I, J, K, C, roots_simple = compute_full_structure_constants(verbose=True)
    t_build_sc = time.perf_counter() - t0
    print(f"  Built structure constants in {t_build_sc:.3f}s")
    print(f"  Total nonzero entries: {len(I)}")

    print("\n  Phase 2: Building 248x248 adjoint matrices...")
    t0 = time.perf_counter()
    generators = build_248_adjoint_matrices(roots, I, J, K, C)
    t_build_mat = time.perf_counter() - t0
    print(f"  Built {len(generators)} generator matrices in {t_build_mat:.3f}s")

    # Root and Cartan lookup
    simple = e8_simple_roots()
    root_map = {}
    for k, r in enumerate(roots):
        root_map[_root_key(r)] = k
    neg_of = {}
    for k in range(240):
        nk = _root_key(-roots[k])
        if nk in root_map:
            neg_of[k] = root_map[nk]

    simple_idx = [root_map[_root_key(s)] for s in simple]

    results = {}

    # ===== TEST 1: Antisymmetry =====
    print("\n" + "-" * 70)
    print("  TEST 1: Antisymmetry of structure constants")
    # Build a dict for fast lookup
    sc_dict = {}
    for idx in range(len(I)):
        key = (int(I[idx]), int(J[idx]))
        if key not in sc_dict:
            sc_dict[key] = []
        sc_dict[key].append((int(K[idx]), float(C[idx])))

    # Check: for each (i,j,k,c), we should have (j,i,k,-c)
    fails = 0
    checked = 0
    for idx in range(len(I)):
        i, j, k, c = int(I[idx]), int(J[idx]), int(K[idx]), float(C[idx])
        reverse_key = (j, i)
        if reverse_key in sc_dict:
            # Find matching k entry
            found = False
            for (k2, c2) in sc_dict[reverse_key]:
                if k2 == k:
                    if abs(c + c2) > 1e-10:
                        fails += 1
                    found = True
                    break
            if not found:
                fails += 1
        else:
            fails += 1
        checked += 1

    passed = fails == 0
    results["ANTISYMMETRY"] = passed
    print(f"  ANTISYMMETRY: {'PASS' if passed else 'FAIL'} (failures={fails}/{checked})")

    # ===== TEST 2: Jacobi for (alpha, -alpha, beta) triples =====
    print("\n" + "-" * 70)
    print("  TEST 2: Jacobi for (E_alpha, E_{-alpha}, E_beta) triples")
    print("  (This is the triple that FAILED with 240-dim algebra)")

    np.random.seed(42)
    n_test = 500
    max_err = 0.0
    fails_2 = 0
    tested = 0

    for trial in range(n_test):
        # Pick a random root alpha and a random root beta != +-alpha
        alpha_idx = np.random.randint(240)
        if alpha_idx not in neg_of:
            continue
        neg_alpha_idx = neg_of[alpha_idx]

        beta_idx = np.random.randint(240)
        while beta_idx == alpha_idx or beta_idx == neg_alpha_idx:
            beta_idx = np.random.randint(240)

        tested += 1
        Ea = generators[alpha_idx]
        Ena = generators[neg_alpha_idx]
        Eb = generators[beta_idx]

        # [Ea, [Ena, Eb]] + [Ena, [Eb, Ea]] + [Eb, [Ea, Ena]]
        comm_na_b = Ena @ Eb - Eb @ Ena
        comm_b_a = Eb @ Ea - Ea @ Eb
        comm_a_na = Ea @ Ena - Ena @ Ea

        jac = (Ea @ comm_na_b - comm_na_b @ Ea) + \
              (Ena @ comm_b_a - comm_b_a @ Ena) + \
              (Eb @ comm_a_na - comm_a_na @ Eb)

        err = np.linalg.norm(jac)
        max_err = max(max_err, err)
        if err > 1e-13:  # FP64: Jacobi error within machine precision
            fails_2 += 1
            if fails_2 <= 3:
                print(f"    FAIL triple: alpha={alpha_idx}, -alpha={neg_alpha_idx}, beta={beta_idx}, err={err:.2e}")

    passed = fails_2 == 0
    results["JACOBI_ALPHA_NEG_ALPHA"] = passed
    print(f"  JACOBI (alpha,-alpha,beta): {'PASS' if passed else 'FAIL'} "
          f"(failures={fails_2}/{tested}, max_err={max_err:.2e})")

    # ===== TEST 3: Leibniz rule [H_i, [E_a, E_b]] =====
    print("\n" + "-" * 70)
    print("  TEST 3: Leibniz rule: [H_i, [E_a, E_b]] = [H_i,E_a]E_b + E_a[H_i,E_b]")

    max_err_3 = 0.0
    fails_3 = 0
    tested_3 = 0

    for h_idx in range(8):
        H = generators[240 + h_idx]
        for trial in range(50):
            a, b = np.random.choice(240, 2, replace=False)
            Ea, Eb = generators[a], generators[b]

            # LHS: [H, [E_a, E_b]]
            comm_ab = Ea @ Eb - Eb @ Ea
            lhs = H @ comm_ab - comm_ab @ H

            # RHS: [H, E_a] @ E_b - E_b @ [H, E_a] + E_a @ [H, E_b] - [H, E_b] @ E_a
            # Actually Leibniz: [H, [A,B]] = [[H,A], B] + [A, [H,B]]
            H_Ea = H @ Ea - Ea @ H
            H_Eb = H @ Eb - Eb @ H
            rhs = (H_Ea @ Eb - Eb @ H_Ea) + (Ea @ H_Eb - H_Eb @ Ea)

            err = np.linalg.norm(lhs - rhs)
            max_err_3 = max(max_err_3, err)
            tested_3 += 1
            if err > 1e-13:  # FP64: Leibniz error within machine precision
                fails_3 += 1

    passed = fails_3 == 0
    results["LEIBNIZ"] = passed
    print(f"  LEIBNIZ: {'PASS' if passed else 'FAIL'} (failures={fails_3}/{tested_3}, max_err={max_err_3:.2e})")

    # ===== TEST 4: Jacobi for random 248-dim vectors =====
    print("\n" + "-" * 70)
    print("  TEST 4: Jacobi identity for random 248-dim vectors (100 trials)")

    np.random.seed(123)
    max_err_4 = 0.0

    for trial in range(100):
        x = np.random.randn(248)
        y = np.random.randn(248)
        z = np.random.randn(248)

        X = sum(x[i] * generators[i] for i in range(248) if abs(x[i]) > 1e-15)
        Y = sum(y[i] * generators[i] for i in range(248) if abs(y[i]) > 1e-15)
        Z = sum(z[i] * generators[i] for i in range(248) if abs(z[i]) > 1e-15)

        YZ = Y @ Z - Z @ Y
        ZX = Z @ X - X @ Z
        XY = X @ Y - Y @ X

        jac = (X @ YZ - YZ @ X) + (Y @ ZX - ZX @ Y) + (Z @ XY - XY @ Z)
        norms = np.linalg.norm(X) * np.linalg.norm(Y) * np.linalg.norm(Z)
        err = np.linalg.norm(jac) / (norms + 1e-30)
        max_err_4 = max(max_err_4, err)

    passed = max_err_4 < 1e-13  # FP64: random 248-dim Jacobi within machine precision
    results["JACOBI_RANDOM_248"] = passed
    print(f"  JACOBI (random 248-dim): {'PASS' if passed else 'FAIL'} (max_relative_err={max_err_4:.2e})")

    # ===== TEST 5: Jacobi for basis triples (sampling) =====
    print("\n" + "-" * 70)
    print("  TEST 5: Jacobi for basis triples (2000 random triples from 248 generators)")

    np.random.seed(77)
    max_err_5 = 0.0
    fails_5 = 0
    n_trials_5 = 2000

    for _ in range(n_trials_5):
        a, b, c = np.random.choice(248, 3, replace=False)
        Ea, Eb, Ec = generators[a], generators[b], generators[c]

        bc = Eb @ Ec - Ec @ Eb
        ca = Ec @ Ea - Ea @ Ec
        ab = Ea @ Eb - Eb @ Ea

        jac = (Ea @ bc - bc @ Ea) + (Eb @ ca - ca @ Eb) + (Ec @ ab - ab @ Ec)
        err = np.linalg.norm(jac)
        max_err_5 = max(max_err_5, err)
        if err > 1e-13:  # FP64: basis triple Jacobi within machine precision
            fails_5 += 1
            if fails_5 <= 5:
                print(f"    FAIL: triple ({a},{b},{c}), err={err:.2e}")

    passed = fails_5 == 0
    results["JACOBI_BASIS_248"] = passed
    print(f"  JACOBI (basis 248): {'PASS' if passed else 'FAIL'} "
          f"(failures={fails_5}/{n_trials_5}, max_err={max_err_5:.2e})")

    # ===== TEST 5b: Jacobi directly on structure constants (5000 triples) =====
    print("\n" + "-" * 70)
    print("  TEST 5b: Jacobi directly on structure constants (5000 random triples)")

    from collections import defaultdict
    sc_lookup = defaultdict(list)
    for idx in range(len(I)):
        sc_lookup[(int(I[idx]), int(J[idx]))].append((int(K[idx]), float(C[idx])))

    np.random.seed(55)
    max_err_5b = 0.0
    fails_5b = 0
    n_trials_5b = 5000

    for _ in range(n_trials_5b):
        a, b, c = np.random.choice(248, 3, replace=False)
        jac = np.zeros(248)

        for (d, c1) in sc_lookup.get((b, c), []):
            for (e, c2) in sc_lookup.get((a, d), []):
                jac[e] += c1 * c2
        for (d, c1) in sc_lookup.get((c, a), []):
            for (e, c2) in sc_lookup.get((b, d), []):
                jac[e] += c1 * c2
        for (d, c1) in sc_lookup.get((a, b), []):
            for (e, c2) in sc_lookup.get((c, d), []):
                jac[e] += c1 * c2

        err = np.max(np.abs(jac))
        max_err_5b = max(max_err_5b, err)
        if err > 1e-13:  # FP64: SC-direct Jacobi within machine precision
            fails_5b += 1

    passed = fails_5b == 0
    results["JACOBI_SC_DIRECT"] = passed
    print(f"  JACOBI (SC direct): {'PASS' if passed else 'FAIL'} "
          f"(failures={fails_5b}/{n_trials_5b}, max_err={max_err_5b:.2e})")

    # ===== TEST 6: Structure constant bracket matches matrix bracket =====
    print("\n" + "-" * 70)
    print("  TEST 6: Structure constant bracket == matrix bracket")
    print("    (Compare SC result as 248x248 matrix vs direct matrix commutator)")

    np.random.seed(99)
    max_err_6 = 0.0
    for trial in range(20):
        x = np.random.randn(248) * 0.1
        y = np.random.randn(248) * 0.1

        # Structure constant bracket -> coefficient vector
        z_sc = lie_bracket_248(x, y, I, J, K, C)

        # Convert both to matrices and compare
        X_mat = sum(x[i] * generators[i] for i in range(248) if abs(x[i]) > 1e-15)
        Y_mat = sum(y[i] * generators[i] for i in range(248) if abs(y[i]) > 1e-15)
        bracket_mat_direct = X_mat @ Y_mat - Y_mat @ X_mat

        # Convert SC result to matrix
        Z_mat_sc = sum(z_sc[i] * generators[i] for i in range(248) if abs(z_sc[i]) > 1e-15)

        err = np.linalg.norm(bracket_mat_direct - Z_mat_sc) / (np.linalg.norm(bracket_mat_direct) + 1e-30)
        max_err_6 = max(max_err_6, err)

    passed = max_err_6 < 1e-12  # FP64: SC vs matrix bracket agreement
    results["SC_VS_MATRIX"] = passed
    print(f"  SC vs MATRIX bracket: {'PASS' if passed else 'FAIL'} (max_relative_err={max_err_6:.2e})")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    if all_pass:
        print("\n  ALL TESTS PASSED!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED: {', '.join(failed)}")

    # ===== BENCHMARK =====
    print("\n" + "=" * 70)
    print("  BENCHMARK: 248-DIM STRUCTURE CONSTANTS vs FULL MATRIX MULTIPLY")
    print("=" * 70)

    # Count nonzero entries by type
    n_rr_to_root = 0
    n_rr_to_cartan = 0
    n_cartan_root = 0
    n_root_cartan = 0
    for idx in range(len(I)):
        i, j, k = int(I[idx]), int(J[idx]), int(K[idx])
        if i < 240 and j < 240 and k < 240:
            n_rr_to_root += 1
        elif i < 240 and j < 240 and k >= 240:
            n_rr_to_cartan += 1
        elif i >= 240 and j < 240:
            n_cartan_root += 1
        elif i < 240 and j >= 240:
            n_root_cartan += 1

    print(f"\n  Nonzero structure constant entries:")
    print(f"    [E,E] -> E (root-root):       {n_rr_to_root}")
    print(f"    [E,E] -> H (root-negroot):     {n_rr_to_cartan}")
    print(f"    [H,E] -> E (Cartan-root):      {n_cartan_root}")
    print(f"    [E,H] -> E (root-Cartan):      {n_root_cartan}")
    print(f"    Total:                          {len(I)}")
    print(f"    Expected approx:                ~17280")

    # Speed benchmark
    np.random.seed(42)
    x = np.random.randn(248) * 0.1
    y = np.random.randn(248) * 0.1

    # Warm up
    _ = lie_bracket_248(x, y, I, J, K, C)

    # Structure constant bracket timing
    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        z = lie_bracket_248(x, y, I, J, K, C)
    t_sc = time.perf_counter() - t0

    # Full matrix bracket timing
    # Pre-build X, Y matrices
    X_mat = sum(x[i] * generators[i] for i in range(248) if abs(x[i]) > 1e-15)
    Y_mat = sum(y[i] * generators[i] for i in range(248) if abs(y[i]) > 1e-15)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        bracket_mat = X_mat @ Y_mat - Y_mat @ X_mat
    t_mat = time.perf_counter() - t0

    print(f"\n  Speed ({n_iter} iterations):")
    print(f"    Structure constant bracket: {t_sc:.4f}s ({t_sc/n_iter*1e6:.0f} us/iter)")
    print(f"    Full 248x248 matrix bracket: {t_mat:.4f}s ({t_mat/n_iter*1e6:.0f} us/iter)")
    print(f"    Speedup (SC vs matrix MM):   {t_mat/t_sc:.2f}x")

    # Operation counts
    n_ops_sc = len(I)  # multiply-adds for SC
    n_ops_mm = 2 * 248**3  # two matrix multiplies for [X,Y] = XY - YX
    print(f"\n  Operation counts:")
    print(f"    Structure constants:  {n_ops_sc:,} multiply-adds")
    print(f"    Full matrix [X,Y]:    {n_ops_mm:,} multiply-adds (2 x 248^3)")
    print(f"    Theoretical ratio:    {n_ops_mm/n_ops_sc:.1f}x")

    # Jacobi verification final statement
    print(f"\n  Jacobi identity verification:")
    print(f"    248-dim basis triples (2000):  max_err = {max_err_5:.2e}")
    print(f"    248-dim random vectors (100):  max_err = {max_err_4:.2e}")
    print(f"    (alpha, -alpha, beta) (500):   max_err = {max_err:.2e}")

    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL TESTS PASSED - Full 248-dim E8 algebra is correct!")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
