"""
Root system builders for all five exceptional Lie algebras.

Each builder returns:
- roots: ndarray of shape (n_roots, ambient_dim), all root vectors
- simple_roots: ndarray of shape (rank, ambient_dim)
- cartan_matrix: ndarray of shape (rank, rank), integer entries

Root system conventions:
- G2: Embedded in R^3, hyperplane x1+x2+x3=0. 12 roots, rank 2.
- F4: In R^4. 48 roots, rank 4. Mixed lengths (long^2=2, short^2=1).
- E6: Extracted from E8 root system. 72 roots, rank 6.
- E7: Extracted from E8 root system. 126 roots, rank 7.
- E8: Delegates to dhl_mm.e8. 240 roots, rank 8.
"""

import numpy as np
from typing import Tuple

# Algebra metadata: (dim, rank, n_roots)
ALGEBRA_INFO = {
    "G2": (14, 2, 12),
    "F4": (52, 4, 48),
    "E6": (78, 6, 72),
    "E7": (133, 7, 126),
    "E8": (248, 8, 240),
}


def _root_key(v: np.ndarray, scale: int = 2) -> tuple:
    """Hashable key for a root vector, rounding to nearest 1/scale."""
    return tuple(int(round(scale * x)) for x in v)


def build_g2() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build G2 root system in R^3 (hyperplane x1+x2+x3=0).

    Short roots (length^2=2): +/-(e_i - e_j) for i<j
    Long roots (length^2=6): +/-(2e_i - e_j - e_k) for {i,j,k}={1,2,3}
    Simple roots: alpha1 = e1-e2 (short), alpha2 = -2e1+e2+e3 (long)
    """
    roots = []
    e = np.eye(3)

    # Short roots: +/-(e_i - e_j)
    for i in range(3):
        for j in range(3):
            if i != j:
                roots.append(e[i] - e[j])

    # Long roots: +/-(2e_i - e_j - e_k) for permutations
    for i in range(3):
        others = [k for k in range(3) if k != i]
        roots.append(2 * e[i] - e[others[0]] - e[others[1]])
        roots.append(-2 * e[i] + e[others[0]] + e[others[1]])

    roots = np.array(roots, dtype=np.float64)
    assert len(roots) == 12, f"G2 should have 12 roots, got {len(roots)}"

    # Simple roots (Bourbaki)
    simple = np.array([
        [1, -1, 0],        # alpha1 = e1-e2 (short)
        [-2, 1, 1],        # alpha2 = -2e1+e2+e3 (long)
    ], dtype=np.float64)

    # Cartan matrix
    cartan = np.array([[2, -1], [-3, 2]], dtype=int)

    return roots, simple, cartan


def build_f4() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build F4 root system in R^4.

    Long roots (length^2=2): +/-e_i +/- e_j for i<j (24 roots)
    Short roots (length^2=1): +/-e_i (8 roots)
    Short roots (length^2=1): (1/2)(+/-1, +/-1, +/-1, +/-1) (16 roots)
    Simple roots (Bourbaki): a1=e2-e3, a2=e3-e4, a3=e4, a4=(e1-e2-e3-e4)/2
    """
    roots = []
    e = np.eye(4)

    # Long roots: +/-e_i +/- e_j for i<j
    for i in range(4):
        for j in range(i + 1, 4):
            for si in [1, -1]:
                for sj in [1, -1]:
                    roots.append(si * e[i] + sj * e[j])

    # Short roots: +/-e_i
    for i in range(4):
        roots.append(e[i])
        roots.append(-e[i])

    # Short roots: half-integer
    from itertools import product as iprod
    for signs in iprod([0.5, -0.5], repeat=4):
        roots.append(np.array(signs))

    roots = np.array(roots, dtype=np.float64)
    assert len(roots) == 48, f"F4 should have 48 roots, got {len(roots)}"

    simple = np.array([
        [0, 1, -1, 0],           # alpha1 = e2-e3
        [0, 0, 1, -1],           # alpha2 = e3-e4
        [0, 0, 0, 1],            # alpha3 = e4
        [0.5, -0.5, -0.5, -0.5], # alpha4 = (e1-e2-e3-e4)/2
    ], dtype=np.float64)

    cartan = np.array([
        [2, -1, 0, 0],
        [-1, 2, -2, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 2],
    ], dtype=int)

    return roots, simple, cartan


def _build_e8_roots() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full E8 root system."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dhl_mm.e8 import build_roots, simple_roots, cartan_matrix
    roots = build_roots()
    sr = simple_roots()
    cm = cartan_matrix()
    return roots, sr, cm


def build_e6() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build E6 root system by extracting from E8.

    E8 Dynkin diagram: a0-a1-a2-a3-a4-a5-a6 with a7 branching off a4.
    E6 sub-diagram: {a2,a3,a4,a5,a6,a7} (remove a0, a1).
    This gives chain a2-a3-a4-a5-a6 with a7 branching at a4 = E6 type.
    E6 roots = E8 roots with zero coefficients for a0 and a1 in simple root basis.
    """
    e8_roots, e8_simple, e8_cartan = _build_e8_roots()

    # Express E8 roots in simple root basis
    S_inv = np.linalg.inv(e8_simple)
    roots_sb = np.round(e8_roots @ S_inv).astype(int)

    # E6: remove a0, a1 -> filter roots with c0=0, c1=0
    mask = (roots_sb[:, 0] == 0) & (roots_sb[:, 1] == 0)
    e6_roots_e8 = e8_roots[mask]

    assert len(e6_roots_e8) == 72, f"E6 should have 72 roots, got {len(e6_roots_e8)}"

    # E6 simple roots in R^8 (ambient E8 space): {a2,a3,a4,a5,a6,a7}
    e6_simple_idx = [2, 3, 4, 5, 6, 7]
    e6_simple = e8_simple[e6_simple_idx]

    # E6 Cartan matrix
    cartan = np.zeros((6, 6), dtype=int)
    for i in range(6):
        for j in range(6):
            si, sj = e6_simple[i], e6_simple[j]
            cartan[i, j] = round(2 * np.dot(si, sj) / np.dot(sj, sj))

    return e6_roots_e8, e6_simple, cartan


def build_e7() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build E7 root system by extracting from E8.

    E8 Dynkin diagram: a0-a1-a2-a3-a4-a5-a6 with a7 branching off a4.
    E7 sub-diagram: {a1,a2,a3,a4,a5,a6,a7} (remove a0).
    This gives chain a1-a2-a3-a4-a5-a6 with a7 branching at a4 = E7 type.
    E7 roots = E8 roots with zero coefficient for a0 in simple root basis.
    """
    e8_roots, e8_simple, e8_cartan = _build_e8_roots()

    S_inv = np.linalg.inv(e8_simple)
    roots_sb = np.round(e8_roots @ S_inv).astype(int)

    # E7: remove a0 -> filter roots with c0=0
    mask = (roots_sb[:, 0] == 0)
    e7_roots_e8 = e8_roots[mask]

    assert len(e7_roots_e8) == 126, f"E7 should have 126 roots, got {len(e7_roots_e8)}"

    # E7 simple roots in R^8: {a1,a2,a3,a4,a5,a6,a7}
    e7_simple_idx = [1, 2, 3, 4, 5, 6, 7]
    e7_simple = e8_simple[e7_simple_idx]

    cartan = np.zeros((7, 7), dtype=int)
    for i in range(7):
        for j in range(7):
            si, sj = e7_simple[i], e7_simple[j]
            cartan[i, j] = round(2 * np.dot(si, sj) / np.dot(sj, sj))

    return e7_roots_e8, e7_simple, cartan


def build_e8() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build E8 root system (delegates to dhl_mm.e8)."""
    return _build_e8_roots()


def build_root_system(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build root system for named exceptional algebra.

    Args:
        name: One of "G2", "F4", "E6", "E7", "E8"

    Returns:
        (roots, simple_roots, cartan_matrix)
    """
    builders = {
        "G2": build_g2,
        "F4": build_f4,
        "E6": build_e6,
        "E7": build_e7,
        "E8": build_e8,
    }
    if name not in builders:
        raise ValueError(f"Unknown algebra: {name}. Must be one of {list(builders.keys())}")
    return builders[name]()
