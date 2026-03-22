"""
E8 Structure Constants with Correct Signs
==========================================

Uses the Frenkel-Kac cocycle on the simple root basis to determine
the signs of the structure constants N_{alpha,beta}.

The cocycle is: epsilon(a, b) = (-1)^{a^T B b} where B is an F_2
bilinear form satisfying B + B^T = Cartan matrix mod 2.
"""
import numpy as np
from itertools import product
import time


def build_e8_roots():
    """Build 240 E8 roots in R^8 (standard orthonormal basis).

    Uses ODD number of minus signs for the half-spin roots
    (compatible with the Bourbaki E8 simple root convention).
    """
    roots = []
    # D8 type: +-e_i +- e_j for i < j (112 roots)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0.0] * 8
                    v[i] = float(si)
                    v[j] = float(sj)
                    roots.append(v)
    # Half-spin type: (+-1/2)^8 with odd number of minus signs (128 roots)
    for signs in product([0.5, -0.5], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 1:
            roots.append(list(signs))
    return np.array(roots, dtype=np.float64)


def _root_key(v):
    """Hashable key for a root vector."""
    return tuple(int(round(2 * x)) for x in v)


def e8_simple_roots():
    """E8 simple roots."""
    return np.array([
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1, -1],
        [-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float64)


def e8_cartan_matrix():
    """E8 Cartan matrix."""
    simple = e8_simple_roots()
    A = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            A[i, j] = round(2 * np.dot(simple[i], simple[j]) / np.dot(simple[j], simple[j]))
    return A.astype(int)


def compute_structure_constants(verbose=True):
    """Compute E8 structure constants with correct Frenkel-Kac signs.

    Returns:
        roots: ndarray (240, 8) - root vectors
        brackets: dict (i, j) -> (k, coeff) for [E_i, E_j] = coeff * E_k
    """
    all_roots = build_e8_roots()
    simple = e8_simple_roots()
    A = e8_cartan_matrix()
    n_roots = len(all_roots)

    # Root lookup
    root_map = {}
    for k, r in enumerate(all_roots):
        root_map[_root_key(r)] = k

    # Convert roots to simple root basis (integer coords)
    S_inv = np.linalg.inv(simple)
    roots_simple = np.round(all_roots @ S_inv).astype(int)

    # Verify
    recon = roots_simple.astype(float) @ simple
    err = np.max(np.abs(recon - all_roots))
    assert err < 1e-10, f"Basis conversion failed: {err}"

    # Build F_2 bilinear form B such that B + B^T = A mod 2
    # Choose B upper-triangular: B[i][j] = |A[i][j]| % 2 for i < j
    B_form = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(i + 1, 8):
            B_form[i, j] = abs(A[i, j]) % 2
    # Diagonal: B[i][i] can be 0 or 1. Set to 0 for now.
    # Note: with diagonal=0, the cocycle squares to epsilon(a,a) = (-1)^0 = +1.

    # Verify: B + B^T = A mod 2 (ignoring diagonal since A_ii = 2 = 0 mod 2)
    for i in range(8):
        for j in range(8):
            if i != j:
                lhs = (B_form[i, j] + B_form[j, i]) % 2
                rhs = abs(A[i, j]) % 2
                assert lhs == rhs, f"B form check failed at ({i},{j})"

    if verbose:
        print(f"    F_2 bilinear form B verified")

    # Cocycle: epsilon(a, b) = (-1)^{a^T B b} (in simple root coords)
    # Precompute roots_simple for fast access
    rs = roots_simple  # (240, 8) integer array

    def cocycle(i, j):
        """Compute epsilon(root_i, root_j)."""
        a = rs[i]
        b = rs[j]
        # Compute a^T B b mod 2
        exp = 0
        for p in range(8):
            if a[p] == 0:
                continue
            for q in range(p + 1, 8):
                if B_form[p, q] != 0 and b[q] != 0:
                    exp += a[p] * b[q]
        return 1 if exp % 2 == 0 else -1

    # Verify cocycle property on simple roots:
    # epsilon(e_i, e_j) * epsilon(e_j, e_i) should equal (-1)^{A[i][j]}
    # since <alpha_i, alpha_j> = A[i][j] for simply-laced.
    if verbose:
        print(f"    Verifying cocycle on simple roots...")

    simple_idx = []
    for s in simple:
        simple_idx.append(root_map[_root_key(s)])

    cocycle_ok = True
    for i in range(8):
        for j in range(8):
            si, sj = simple_idx[i], simple_idx[j]
            eps_ij = cocycle(si, sj)
            eps_ji = cocycle(sj, si)
            expected = (-1) ** abs(A[i, j])
            if i == j:
                # A[i][i] = 2, (-1)^2 = 1
                expected = 1
            if eps_ij * eps_ji != expected:
                if verbose:
                    print(f"    FAIL: roots {i},{j}: eps*eps={eps_ij * eps_ji}, expected={expected}")
                cocycle_ok = False
    if verbose:
        print(f"    Cocycle check: {'PASS' if cocycle_ok else 'FAIL'}")

    # Compute structure constants
    if verbose:
        print(f"    Computing brackets and string lengths...")

    brackets = {}
    for i in range(n_roots):
        for j in range(n_roots):
            if i == j:
                continue
            s = all_roots[i] + all_roots[j]
            key = _root_key(s)
            if key in root_map:
                k = root_map[key]

                # String length: r = max int such that root[j] - r * root[i] is a root
                r = 0
                while True:
                    test = all_roots[j] - (r + 1) * all_roots[i]
                    if _root_key(test) in root_map:
                        r += 1
                    else:
                        break

                N_abs = r + 1
                sign = cocycle(i, j)
                brackets[(i, j)] = (k, float(sign * N_abs))

    if verbose:
        pos = sum(1 for v in brackets.values() if v[1] > 0)
        neg = sum(1 for v in brackets.values() if v[1] < 0)
        print(f"    {len(brackets)} brackets: {pos} positive, {neg} negative")

    return all_roots, brackets


def compute_structure_constants_sparse():
    """Sparse array version."""
    roots, brackets = compute_structure_constants()
    I, J, K, C = [], [], [], []
    for (i, j), (k, c) in brackets.items():
        I.append(i)
        J.append(j)
        K.append(k)
        C.append(c)
    return (roots,
            np.array(I, dtype=np.int32),
            np.array(J, dtype=np.int32),
            np.array(K, dtype=np.int32),
            np.array(C, dtype=np.float64))


if __name__ == "__main__":
    print("Building E8 structure constants with Frenkel-Kac cocycle...")
    t0 = time.time()
    roots, brackets = compute_structure_constants()
    t1 = time.time()
    print(f"\nDone in {t1 - t0:.2f}s: {len(roots)} roots, {len(brackets)} brackets")

    # Antisymmetry check
    fails = 0
    for (i, j), (k, c) in brackets.items():
        if (j, i) in brackets:
            k2, c2 = brackets[(j, i)]
            if k != k2 or abs(c + c2) > 1e-10:
                fails += 1
    print(f"Antisymmetry failures: {fails}/{len(brackets)}")
