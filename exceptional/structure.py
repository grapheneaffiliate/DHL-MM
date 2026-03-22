"""
Structure constant computation for all five exceptional Lie algebras.

Strategy:
- Simply-laced (E6, E7, E8): Extract from E8's verified Frenkel-Kac cocycle
  structure constants. Filter to subalgebra generators and remap indices.
- G2: Frenkel-Kac cocycle with coroot normalization.
- F4: Chevalley basis via adjoint-matrix commutator propagation + Killing
  form projection.

All structure constants f_{ij}^k satisfy:
  [T_i, T_j] = sum_k f_{ij}^k T_k

Generators are ordered: root generators (indices 0..n_roots-1), then Cartan
generators (indices n_roots..dim-1).
"""

import numpy as np
from typing import Dict, Tuple, List
from .roots import build_root_system, _root_key, ALGEBRA_INFO


def _express_in_simple_basis(roots: np.ndarray, simple: np.ndarray) -> np.ndarray:
    """Express roots in the simple root basis."""
    if simple.shape[0] == simple.shape[1]:
        S_inv = np.linalg.inv(simple)
        coeffs = np.round(roots @ S_inv).astype(int)
    else:
        S_pinv = np.linalg.pinv(simple)
        coeffs = np.round(roots @ S_pinv).astype(int)
    recon = coeffs.astype(float) @ simple
    err = np.max(np.abs(recon - roots))
    assert err < 1e-10, f"Root reconstruction error {err} too large"
    return coeffs


def _build_cocycle_structure_constants(
    name: str,
    roots: np.ndarray,
    simple: np.ndarray,
    cartan: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build structure constants using the Frenkel-Kac cocycle.

    Works for G2 (and all simply-laced types). Mirrors dhl_mm/e8.py.
    """
    n_roots = len(roots)
    rank = len(simple)
    dim = n_roots + rank

    roots_sb = _express_in_simple_basis(roots, simple)

    scale = 2
    root_map: Dict[tuple, int] = {}
    for k, r in enumerate(roots):
        root_map[_root_key(r, scale)] = k

    neg_of: Dict[int, int] = {}
    for k in range(n_roots):
        nk = _root_key(-roots[k], scale)
        if nk in root_map:
            neg_of[k] = root_map[nk]

    B_form = np.zeros((rank, rank), dtype=int)
    for i in range(rank):
        for j in range(i + 1, rank):
            B_form[i, j] = abs(cartan[i, j]) % 2

    rs = roots_sb

    def cocycle(i: int, j: int) -> int:
        a, b = rs[i], rs[j]
        exp = 0
        for p in range(rank):
            if a[p] == 0:
                continue
            for q in range(p + 1, rank):
                if B_form[p, q] != 0 and b[q] != 0:
                    exp += a[p] * b[q]
        return 1 if exp % 2 == 0 else -1

    I_list: List[int] = []
    J_list: List[int] = []
    K_list: List[int] = []
    C_list: List[float] = []

    # Root-root brackets
    for i in range(n_roots):
        for j in range(n_roots):
            if i == j:
                continue
            s = roots[i] + roots[j]
            key = _root_key(s, scale)
            if key in root_map:
                k = root_map[key]
                r = 0
                while _root_key(roots[j] - (r + 1) * roots[i], scale) in root_map:
                    r += 1
                I_list.append(i)
                J_list.append(j)
                K_list.append(k)
                C_list.append(float(cocycle(i, j) * (r + 1)))

    # [E_alpha, E_{-alpha}] = cocycle * coroot
    A_inv = np.linalg.inv(cartan.astype(float))
    for i in range(n_roots):
        if i not in neg_of:
            continue
        j = neg_of[i]
        sign = cocycle(i, j)
        alpha = roots[i]
        alpha_sq = np.dot(alpha, alpha)
        v = np.array([2.0 * np.dot(simple[p], alpha) / alpha_sq for p in range(rank)])
        d = A_inv @ v
        for k in range(rank):
            if abs(d[k]) > 1e-12:
                I_list.append(i)
                J_list.append(j)
                K_list.append(n_roots + k)
                C_list.append(float(sign * d[k]))

    # [H_k, E_i] and [E_i, H_k]
    for i in range(n_roots):
        a = roots_sb[i]
        for k in range(rank):
            ip = sum(int(a[j]) * int(cartan[j, k]) for j in range(rank))
            if ip != 0:
                I_list.append(n_roots + k)
                J_list.append(i)
                K_list.append(i)
                C_list.append(float(ip))
                I_list.append(i)
                J_list.append(n_roots + k)
                K_list.append(i)
                C_list.append(float(-ip))

    return (
        np.array(I_list, dtype=np.int32),
        np.array(J_list, dtype=np.int32),
        np.array(K_list, dtype=np.int32),
        np.array(C_list, dtype=np.float64),
    )


def _build_f4_chevalley(
    roots: np.ndarray,
    simple: np.ndarray,
    cartan: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build F4 structure constants via Chevalley basis + Killing form projection.

    Algorithm:
    1. Build ad(H_k) and ad(E_{alpha_i}) for simple roots with all-positive N
       convention (N_{alpha_i, beta} = q+1 > 0 for all root string pairs).
    2. Propagate ad(E_gamma) for positive roots via spanning tree commutators.
    3. Fix coroot and Cartan entries for all positive root ad matrices.
    4. Build approximate ad(E_{-gamma}) for negative roots by solving the
       commutator equation [ad(E_gamma), ad(E_{-gamma})] = ad(H_gamma).
    5. Refine ALL ad matrices via Killing form projection until the Jacobi
       identity is satisfied to machine epsilon.
    6. Extract structure constants from the converged ad matrices.
    """
    n_roots = len(roots)
    rank = len(simple)
    dim = n_roots + rank

    roots_sb = _express_in_simple_basis(roots, simple)
    heights = np.sum(roots_sb, axis=1)
    roots_sb_int = roots_sb.astype(int)

    scale = 2
    root_map: Dict[tuple, int] = {}
    for k, r in enumerate(roots):
        root_map[_root_key(r, scale)] = k

    neg_of: Dict[int, int] = {}
    for k in range(n_roots):
        nk = _root_key(-roots[k], scale)
        if nk in root_map:
            neg_of[k] = root_map[nk]

    simple_idx: List[int] = []
    for i in range(rank):
        for k in range(n_roots):
            if np.allclose(roots[k], simple[i]):
                simple_idx.append(k)
                break

    pos_indices = sorted(
        [i for i in range(n_roots) if heights[i] > 0],
        key=lambda i: heights[i]
    )

    A_inv = np.linalg.inv(cartan.astype(float))

    def root_string_q(a_idx: int, b_idx: int) -> int:
        q = 0
        while _root_key(roots[b_idx] - (q + 1) * roots[a_idx], scale) in root_map:
            q += 1
        return q

    sum_root: Dict[Tuple[int, int], int] = {}
    for a in range(n_roots):
        for b in range(n_roots):
            if a == b:
                continue
            sk = _root_key(roots[a] + roots[b], scale)
            if sk in root_map:
                sum_root[(a, b)] = root_map[sk]

    # ---- Step 1: Build ad(H_k) and ad(E_{alpha_i}) ----
    ad = np.zeros((dim, dim, dim), dtype=np.float64)

    for i in range(rank):
        h_idx = n_roots + i
        for a in range(n_roots):
            eig = sum(roots_sb_int[a, j] * int(cartan[j, i]) for j in range(rank))
            if eig != 0:
                ad[h_idx][a, a] = float(eig)

    for si_idx in simple_idx:
        for j in range(rank):
            eig = sum(roots_sb_int[si_idx, p] * int(cartan[p, j]) for p in range(rank))
            if eig != 0:
                ad[si_idx][si_idx, n_roots + j] = -float(eig)
        alpha = roots[si_idx]
        alpha_sq = np.dot(alpha, alpha)
        v = np.array([2.0 * np.dot(simple[p], alpha) / alpha_sq for p in range(rank)])
        h_coeffs = A_inv @ v
        for k in range(rank):
            if abs(h_coeffs[k]) > 1e-12:
                ad[si_idx][n_roots + k, neg_of[si_idx]] = h_coeffs[k]
        for b in range(n_roots):
            if b == si_idx or b == neg_of.get(si_idx, -1):
                continue
            if (si_idx, b) not in sum_root:
                continue
            c = sum_root[(si_idx, b)]
            q = root_string_q(si_idx, b)
            ad[si_idx][c, b] = float(q + 1)

    # ---- Step 2: Spanning tree propagation ----
    built_pos = set(simple_idx)
    parent: Dict[int, Tuple[int, int, float]] = {}

    for gamma in pos_indices:
        if gamma in built_pos:
            continue
        for si_idx in simple_idx:
            delta_vec = roots[gamma] - roots[si_idx]
            dk = _root_key(delta_vec, scale)
            if dk not in root_map:
                continue
            delta = root_map[dk]
            if delta not in built_pos:
                continue
            q = root_string_q(si_idx, delta)
            n_tree = float(q + 1)
            parent[gamma] = (si_idx, delta, n_tree)
            comm = ad[si_idx] @ ad[delta] - ad[delta] @ ad[si_idx]
            ad[gamma] = comm / n_tree
            built_pos.add(gamma)
            break

    assert len(built_pos) == len(pos_indices), \
        f"Positive spanning tree incomplete: {len(built_pos)}/{len(pos_indices)}"

    # ---- Step 3: Fix coroot and Cartan entries ----
    for a in pos_indices:
        na = neg_of[a]
        alpha = roots[a]
        alpha_sq = np.dot(alpha, alpha)
        v = np.array([2.0 * np.dot(simple[p], alpha) / alpha_sq for p in range(rank)])
        h_coeffs = A_inv @ v
        for k in range(rank):
            ad[a][n_roots + k, na] = h_coeffs[k]
        for j in range(rank):
            eig = sum(roots_sb_int[a, p] * int(cartan[p, j]) for p in range(rank))
            ad[a][a, n_roots + j] = -float(eig) if eig != 0 else 0.0

    # ---- Step 4: Build negative root ad matrices ----
    neg_ordered = sorted(
        [i for i in range(n_roots) if heights[i] < 0],
        key=lambda i: -heights[i]
    )

    for ng in neg_ordered:
        g = neg_of[ng]
        alpha = roots[g]
        alpha_sq = np.dot(alpha, alpha)
        v_g = np.array([2.0 * np.dot(simple[p], alpha) / alpha_sq for p in range(rank)])
        h_coeffs_g = A_inv @ v_g

        ad_Hgamma = np.zeros((dim, dim))
        for k in range(rank):
            if abs(h_coeffs_g[k]) > 1e-12:
                ad_Hgamma += h_coeffs_g[k] * ad[n_roots + k]

        X = np.zeros((dim, dim))
        for j in range(rank):
            eig = sum(roots_sb_int[ng, p] * int(cartan[p, j]) for p in range(rank))
            if eig != 0:
                X[ng, n_roots + j] = -float(eig)
        for k in range(rank):
            if abs(h_coeffs_g[k]) > 1e-12:
                X[n_roots + k, g] = -h_coeffs_g[k]

        unknowns: List[Tuple[int, int]] = []
        for b in range(n_roots):
            if b == ng or b == g:
                continue
            if (ng, b) not in sum_root:
                continue
            c = sum_root[(ng, b)]
            unknowns.append((b, c))

        n_unk = len(unknowns)
        if n_unk > 0:
            RHS = ad_Hgamma - (ad[g] @ X - X @ ad[g])
            b_sys = RHS.flatten()
            A_sys = np.zeros((dim * dim, n_unk))
            for k_idx, (bk, ck) in enumerate(unknowns):
                for r in range(dim):
                    A_sys[r * dim + bk, k_idx] += ad[g][r, ck]
                for s in range(dim):
                    A_sys[ck * dim + s, k_idx] -= ad[g][bk, s]
            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            for k_idx, (bk, ck) in enumerate(unknowns):
                X[ck, bk] = sol[k_idx]

        ad[ng] = X

    # ---- Step 5: Killing form projection refinement ----
    for _iteration in range(20):
        K = np.einsum('iab,jba->ij', ad, ad)
        kr = np.linalg.matrix_rank(K, tol=1e-6)
        if kr < dim:
            break
        K_inv = np.linalg.inv(K)

        new_ad = np.zeros((dim, dim, dim), dtype=np.float64)
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                comm = ad[i] @ ad[j] - ad[j] @ ad[i]
                if np.max(np.abs(comm)) < 1e-15:
                    continue
                traces = np.einsum('ab,lba->l', comm, ad)
                f_ij = K_inv @ traces
                for k in range(dim):
                    if abs(f_ij[k]) > 1e-14:
                        new_ad[i][k, j] = f_ij[k]

        # Enforce antisymmetry
        for i in range(dim):
            for j in range(i + 1, dim):
                for k in range(dim):
                    avg = (new_ad[i][k, j] - new_ad[j][k, i]) / 2.0
                    new_ad[i][k, j] = avg
                    new_ad[j][k, i] = -avg

        ad = new_ad

    # ---- Step 6: Extract structure constants ----
    I_list: List[int] = []
    J_list: List[int] = []
    K_list: List[int] = []
    C_list: List[float] = []

    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            for k in range(dim):
                val = ad[i][k, j]
                if abs(val) > 1e-14:
                    I_list.append(i)
                    J_list.append(j)
                    K_list.append(k)
                    C_list.append(val)

    return (
        np.array(I_list, dtype=np.int32),
        np.array(J_list, dtype=np.int32),
        np.array(K_list, dtype=np.int32),
        np.array(C_list, dtype=np.float64),
    )


def _compute_structure_constants_from_e8(
    name: str,
    roots: np.ndarray,
    simple: np.ndarray,
    cartan: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract structure constants from E8 for simply-laced subalgebras (E6, E7)."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dhl_mm.e8 import (
        build_roots as e8_build_roots,
        compute_full_structure_constants,
    )

    e8_roots, e8_I, e8_J, e8_K, e8_C, e8_roots_sb = compute_full_structure_constants()
    all_e8_roots = e8_build_roots()

    n_e8_roots = len(all_e8_roots)
    n_sub_roots = len(roots)
    rank = len(simple)

    scale = 2
    sub_root_map = {}
    for k, r in enumerate(roots):
        sub_root_map[_root_key(r, scale)] = k

    e8_to_sub = {}
    for k in range(n_e8_roots):
        rk = _root_key(all_e8_roots[k], scale)
        if rk in sub_root_map:
            e8_to_sub[k] = sub_root_map[rk]

    if name == "E6":
        e8_simple_idx = [2, 3, 4, 5, 6, 7]
    elif name == "E7":
        e8_simple_idx = [1, 2, 3, 4, 5, 6, 7]
    else:
        e8_simple_idx = list(range(8))

    e8_cartan_to_sub = {}
    for sub_i, e8_i in enumerate(e8_simple_idx):
        e8_cartan_to_sub[240 + e8_i] = n_sub_roots + sub_i

    def map_idx(e8_idx: int) -> int:
        if e8_idx < 240:
            return e8_to_sub.get(e8_idx, -1)
        else:
            return e8_cartan_to_sub.get(e8_idx, -1)

    I_list, J_list, K_list, C_list = [], [], [], []
    for idx in range(len(e8_I)):
        i8, j8, k8 = int(e8_I[idx]), int(e8_J[idx]), int(e8_K[idx])
        si = map_idx(i8)
        sj = map_idx(j8)
        sk = map_idx(k8)
        if si >= 0 and sj >= 0 and sk >= 0:
            I_list.append(si)
            J_list.append(sj)
            K_list.append(sk)
            C_list.append(e8_C[idx])

    return (
        np.array(I_list, dtype=np.int32),
        np.array(J_list, dtype=np.int32),
        np.array(K_list, dtype=np.int32),
        np.array(C_list, dtype=np.float64),
    )


def compute_structure_constants(
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute full structure constants for a named exceptional algebra.

    Returns:
        (roots, simple_roots, cartan_matrix, I, J, K, C)
    """
    roots, simple, cartan = build_root_system(name)

    if name == "E8":
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dhl_mm.e8 import compute_full_structure_constants
        e8_roots, I, J, K, C, _ = compute_full_structure_constants()
        return e8_roots, simple, cartan, I, J, K, C
    elif name in ("E6", "E7"):
        I, J, K, C = _compute_structure_constants_from_e8(name, roots, simple, cartan)
    elif name == "G2":
        I, J, K, C = _build_cocycle_structure_constants(name, roots, simple, cartan)
    elif name == "F4":
        I, J, K, C = _build_f4_chevalley(roots, simple, cartan)
    else:
        raise ValueError(f"Unknown algebra: {name}")

    return roots, simple, cartan, I, J, K, C
