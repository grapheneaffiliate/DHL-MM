"""
E8 Full 248-Dimensional Algebra
================================

Extends the 240 root generators with 8 Cartan generators to form
the complete E8 Lie algebra of dimension 248.

Generators:
  0-239:   E_alpha (root generators, one per root)
  240-247: H_1,...,H_8 (Cartan generators, one per simple root)

Structure constants:
  [E_alpha, E_beta]   = N_{alpha,beta} * E_{alpha+beta}  (when alpha+beta is a root)
  [E_alpha, E_{-alpha}] = sum_k a_k * H_k               (where alpha = sum_k a_k * alpha_k)
  [H_i, E_alpha]      = a_i * E_alpha                    (a_i = simple root coord)
  [H_i, H_j]          = 0
"""
import numpy as np
import time

from e8_structure_constants import (
    build_e8_roots, compute_structure_constants, e8_simple_roots,
    e8_cartan_matrix, _root_key
)


def compute_full_structure_constants(verbose=True):
    """Returns structure constants for the full 248-dim E8 algebra.

    Generators indexed 0-239: root generators E_alpha
    Generators indexed 240-247: Cartan generators H_1,...,H_8

    Returns:
        roots: ndarray (240, 8) - root vectors
        I, J, K, C: arrays where [gen_I[n], gen_J[n]] = C[n] * gen_K[n]
        roots_simple: ndarray (240, 8) - roots in simple root basis (integer coords)
    """
    # Get the 240 root-root brackets from existing code
    all_roots, brackets = compute_structure_constants(verbose=verbose)
    simple = e8_simple_roots()
    n_roots = len(all_roots)

    # Convert roots to simple root basis (integer coordinates)
    S_inv = np.linalg.inv(simple)
    roots_simple = np.round(all_roots @ S_inv).astype(int)

    # Verify conversion
    recon = roots_simple.astype(float) @ simple
    err = np.max(np.abs(recon - all_roots))
    assert err < 1e-10, f"Basis conversion error: {err}"

    # Build root lookup for negative root detection
    root_map = {}
    for k, r in enumerate(all_roots):
        root_map[_root_key(r)] = k

    # Find negative pairs
    neg_of = {}
    for k in range(n_roots):
        nk = _root_key(-all_roots[k])
        if nk in root_map:
            neg_of[k] = root_map[nk]

    # Collect all structure constant entries
    I_list, J_list, K_list, C_list = [], [], [], []

    # 1. Root-root brackets: [E_alpha, E_beta] = N * E_{alpha+beta} (already computed)
    for (i, j), (k, c) in brackets.items():
        I_list.append(i)
        J_list.append(j)
        K_list.append(k)
        C_list.append(c)

    # 2. [E_alpha, E_{-alpha}] = epsilon(alpha, -alpha) * sum_k a_k * H_k
    #    where a_k are simple root coordinates of alpha,
    #    and epsilon is the Frenkel-Kac cocycle sign.
    #    Rebuild the cocycle (same B_form as in e8_structure_constants.py)
    A = e8_cartan_matrix()
    B_form = np.zeros((8, 8), dtype=int)
    for p in range(8):
        for q in range(p + 1, 8):
            B_form[p, q] = abs(A[p, q]) % 2

    def cocycle(i, j):
        """Compute epsilon(root_i, root_j) using Frenkel-Kac cocycle."""
        a = roots_simple[i]
        b = roots_simple[j]
        exp = 0
        for p in range(8):
            if a[p] == 0:
                continue
            for q in range(p + 1, 8):
                if B_form[p, q] != 0 and b[q] != 0:
                    exp += a[p] * b[q]
        return 1 if exp % 2 == 0 else -1

    for i in range(n_roots):
        if i not in neg_of:
            continue
        j = neg_of[i]
        # Cocycle sign for this specific (alpha, -alpha) pair
        sign = cocycle(i, j)
        # Simple root coordinates of root i
        a = roots_simple[i]
        for k in range(8):
            if a[k] != 0:
                I_list.append(i)
                J_list.append(j)
                K_list.append(240 + k)
                C_list.append(float(sign * a[k]))

    # 3. [H_k, E_alpha] = <alpha_k, alpha> * E_alpha
    #    where <alpha_k, alpha> = sum_j a_j(alpha) * A_{jk} (Cartan matrix)
    #    This is the inner product, NOT the simple root coordinate.
    for i in range(n_roots):
        a = roots_simple[i]  # simple root coords of root i
        for k in range(8):
            # <alpha_k, root_i> = sum_j a[j] * A[j, k]
            ip = sum(int(a[j]) * int(A[j, k]) for j in range(8))
            if ip != 0:
                # [H_k, E_i] = ip * E_i
                I_list.append(240 + k)
                J_list.append(i)
                K_list.append(i)
                C_list.append(float(ip))
                # [E_i, H_k] = -ip * E_i
                I_list.append(i)
                J_list.append(240 + k)
                K_list.append(i)
                C_list.append(float(-ip))

    # 4. [H_i, H_j] = 0 (no entries needed)

    I_arr = np.array(I_list, dtype=np.int32)
    J_arr = np.array(J_list, dtype=np.int32)
    K_arr = np.array(K_list, dtype=np.int32)
    C_arr = np.array(C_list, dtype=np.float64)

    if verbose:
        # Count by type
        root_root = sum(1 for ii, jj in zip(I_list, J_list)
                        if ii < 240 and jj < 240 and K_list[I_list.index(ii)] < 240)
        n_total = len(I_list)
        n_rr = len(brackets)
        n_cartan_from_roots = sum(1 for ii, jj, kk in zip(I_list, J_list, K_list)
                                   if ii < 240 and jj < 240 and kk >= 240)
        n_hk_e = sum(1 for ii in I_list if ii >= 240)
        n_e_hk = sum(1 for ii, jj in zip(I_list, J_list) if ii < 240 and jj >= 240)
        print(f"    Full 248-dim structure constants:")
        print(f"      Root-root [E,E]->E:     {n_rr}")
        print(f"      Root-negroot [E,E]->H:  {n_cartan_from_roots}")
        print(f"      Cartan-root [H,E]->E:   {n_hk_e}")
        print(f"      Root-Cartan [E,H]->E:   {n_e_hk}")
        print(f"      Total nonzero entries:   {n_total}")

    return all_roots, I_arr, J_arr, K_arr, C_arr, roots_simple


def build_248_adjoint_matrices(roots, I, J, K, C):
    """Build 248x248 adjoint matrices for all 248 generators.

    Convention: (ad_i)_{k,j} = f_{ij}^k, so that ad_i(e_j) = sum_k f_{ij}^k e_k.
    This ensures [ad_X, ad_Y] = ad_{[X,Y]}.
    """
    n = 248
    generators = []
    for g in range(n):
        E_g = np.zeros((n, n))
        generators.append(E_g)

    for idx in range(len(I)):
        i, j, k, c = I[idx], J[idx], K[idx], C[idx]
        # [gen_i, gen_j] = c * gen_k
        # (ad_i)_{k,j} = c: the k-th component of ad_i(e_j)
        generators[i][k, j] += c

    return generators


def lie_bracket_248(x, y, I, J, K, C):
    """Compute [x, y] using structure constants.

    x, y: 248-dim coefficient vectors.
    Returns: 248-dim coefficient vector for [x, y].
    """
    # Vectorized: compute all contributions then scatter-add by K index
    contributions = x[I] * y[J] * C
    result = np.zeros(248)
    np.add.at(result, K, contributions)
    return result


def lie_bracket_248_matrix(x, y, generators, killing_inv=None):
    """Compute [x, y] using full 248x248 adjoint matrices.

    Returns: 248-dim coefficient vector from reading off the matrix bracket.
    """
    n = 248
    X = sum(x[i] * generators[i] for i in range(n) if abs(x[i]) > 1e-15)
    Y = sum(y[i] * generators[i] for i in range(n) if abs(y[i]) > 1e-15)
    bracket_mat = X @ Y - Y @ X

    # Read off coefficients using Killing form:
    # K_{ij} = Tr(ad_i @ ad_j), then coeffs = K^{-1} @ traces
    # where traces[i] = Tr(bracket_mat @ ad_i)
    if killing_inv is None:
        K_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = np.trace(generators[i] @ generators[j])
                K_mat[i, j] = val
                K_mat[j, i] = val
        killing_inv = np.linalg.pinv(K_mat)

    traces = np.array([np.trace(bracket_mat @ generators[i]) for i in range(n)])
    coeffs = killing_inv @ traces
    return coeffs


if __name__ == "__main__":
    print("=" * 70)
    print("  E8 FULL 248-DIMENSIONAL ALGEBRA")
    print("=" * 70)

    print("\n  Building full 248-dim structure constants...")
    t0 = time.perf_counter()
    roots, I, J, K, C, roots_simple = compute_full_structure_constants(verbose=True)
    t1 = time.perf_counter()
    print(f"  Built in {t1 - t0:.3f}s")
    print(f"  Total nonzero structure constants: {len(I)}")
