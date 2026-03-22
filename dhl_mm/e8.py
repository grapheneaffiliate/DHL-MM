"""
E8 root system, structure constants, and full 248-dim algebra.

Uses the Frenkel-Kac cocycle for correct signs. Builds:
- 240 roots in R^8
- 16,694 nonzero structure constants f_{ij}^k
- Full 248-dim algebra (240 root + 8 Cartan generators)
- 248x248 adjoint representation matrices
"""

import numpy as np
from itertools import product


DIM = 248


def build_roots():
    """Build 240 E8 roots in R^8."""
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0.0] * 8
                    v[i] = float(si)
                    v[j] = float(sj)
                    roots.append(v)
    for signs in product([0.5, -0.5], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 1:
            roots.append(list(signs))
    return np.array(roots, dtype=np.float64)


def _root_key(v):
    return tuple(int(round(2 * x)) for x in v)


def simple_roots():
    """E8 simple roots (Bourbaki convention)."""
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


def cartan_matrix():
    """E8 Cartan matrix."""
    sr = simple_roots()
    A = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            A[i, j] = round(2 * np.dot(sr[i], sr[j]) / np.dot(sr[j], sr[j]))
    return A.astype(int)


def compute_structure_constants(verbose=False):
    """Compute 240-root structure constants with Frenkel-Kac cocycle signs.

    Returns (roots, brackets_dict) where brackets[(i,j)] = (k, coefficient).
    """
    all_roots = build_roots()
    sr = simple_roots()
    A = cartan_matrix()
    n_roots = len(all_roots)

    root_map = {}
    for k, r in enumerate(all_roots):
        root_map[_root_key(r)] = k

    S_inv = np.linalg.inv(sr)
    roots_simple = np.round(all_roots @ S_inv).astype(int)

    recon = roots_simple.astype(float) @ sr
    assert np.max(np.abs(recon - all_roots)) < 1e-10

    B_form = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(i + 1, 8):
            B_form[i, j] = abs(A[i, j]) % 2

    rs = roots_simple

    def cocycle(i, j):
        a, b = rs[i], rs[j]
        exp = 0
        for p in range(8):
            if a[p] == 0:
                continue
            for q in range(p + 1, 8):
                if B_form[p, q] != 0 and b[q] != 0:
                    exp += a[p] * b[q]
        return 1 if exp % 2 == 0 else -1

    brackets = {}
    for i in range(n_roots):
        for j in range(n_roots):
            if i == j:
                continue
            s = all_roots[i] + all_roots[j]
            key = _root_key(s)
            if key in root_map:
                k = root_map[key]
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

    return all_roots, brackets


def compute_full_structure_constants(verbose=False):
    """Full 248-dim structure constants (240 root + 8 Cartan generators).

    Returns (roots, I, J, K, C, roots_simple) as sparse arrays.
    """
    all_roots, brackets = compute_structure_constants(verbose=verbose)
    sr = simple_roots()
    A = cartan_matrix()
    n_roots = len(all_roots)

    S_inv = np.linalg.inv(sr)
    roots_simple = np.round(all_roots @ S_inv).astype(int)

    root_map = {}
    for k, r in enumerate(all_roots):
        root_map[_root_key(r)] = k

    neg_of = {}
    for k in range(n_roots):
        nk = _root_key(-all_roots[k])
        if nk in root_map:
            neg_of[k] = root_map[nk]

    B_form = np.zeros((8, 8), dtype=int)
    for p in range(8):
        for q in range(p + 1, 8):
            B_form[p, q] = abs(A[p, q]) % 2

    def cocycle(i, j):
        a, b = roots_simple[i], roots_simple[j]
        exp = 0
        for p in range(8):
            if a[p] == 0:
                continue
            for q in range(p + 1, 8):
                if B_form[p, q] != 0 and b[q] != 0:
                    exp += a[p] * b[q]
        return 1 if exp % 2 == 0 else -1

    I_list, J_list, K_list, C_list = [], [], [], []

    for (i, j), (k, c) in brackets.items():
        I_list.append(i)
        J_list.append(j)
        K_list.append(k)
        C_list.append(c)

    for i in range(n_roots):
        if i not in neg_of:
            continue
        j = neg_of[i]
        sign = cocycle(i, j)
        a = roots_simple[i]
        for k in range(8):
            if a[k] != 0:
                I_list.append(i)
                J_list.append(j)
                K_list.append(240 + k)
                C_list.append(float(sign * a[k]))

    for i in range(n_roots):
        a = roots_simple[i]
        for k in range(8):
            ip = sum(int(a[j]) * int(A[j, k]) for j in range(8))
            if ip != 0:
                I_list.append(240 + k)
                J_list.append(i)
                K_list.append(i)
                C_list.append(float(ip))
                I_list.append(i)
                J_list.append(240 + k)
                K_list.append(i)
                C_list.append(float(-ip))

    return (
        all_roots,
        np.array(I_list, dtype=np.int32),
        np.array(J_list, dtype=np.int32),
        np.array(K_list, dtype=np.int32),
        np.array(C_list, dtype=np.float64),
        roots_simple,
    )


def build_adjoint_matrices(roots, I, J, K, C):
    """Build 248x248 adjoint representation matrices for all 248 generators."""
    generators = [np.zeros((DIM, DIM)) for _ in range(DIM)]
    for idx in range(len(I)):
        i, j, k, c = I[idx], J[idx], K[idx], C[idx]
        generators[i][k, j] += c
    return generators


def lie_bracket(x, y, I, J, K, C):
    """Compute [x, y] using sparse structure constants. O(|f|)."""
    contributions = x[I] * y[J] * C
    result = np.zeros(DIM)
    np.add.at(result, K, contributions)
    return result
