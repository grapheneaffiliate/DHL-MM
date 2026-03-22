"""
Casimir degree analysis and d-tensor verification for exceptional Lie algebras.

Casimir invariant degrees (independent Casimir operators):
- G2:  2, 6
- F4:  2, 6, 8, 12
- E6:  2, 5, 6, 8, 9, 12
- E7:  2, 6, 8, 10, 12, 14, 18
- E8:  2, 8, 12, 14, 18, 20, 24, 30

NONE of the exceptional algebras have a degree-3 Casimir invariant.
Therefore the totally symmetric d-tensor d_{ij}^k should vanish for ALL of them,
meaning the DHL-MM full-product trick (x*y projected = [x,y]/2) works universally.

This module provides computational verification of d=0 for each algebra.
"""

import numpy as np
from typing import Dict, Tuple

# Casimir degrees for each exceptional algebra
CASIMIR_DEGREES: Dict[str, Tuple[int, ...]] = {
    "G2": (2, 6),
    "F4": (2, 6, 8, 12),
    "E6": (2, 5, 6, 8, 9, 12),
    "E7": (2, 6, 8, 10, 12, 14, 18),
    "E8": (2, 8, 12, 14, 18, 20, 24, 30),
}

# Dual Coxeter numbers (for Killing form normalization)
DUAL_COXETER: Dict[str, int] = {
    "G2": 4,
    "F4": 9,
    "E6": 12,
    "E7": 18,
    "E8": 30,
}


def has_cubic_casimir(name: str) -> bool:
    """Check if the algebra has a degree-3 Casimir invariant.

    Returns False for all exceptional algebras (none have degree 3).
    """
    if name not in CASIMIR_DEGREES:
        raise ValueError(f"Unknown algebra: {name}")
    return 3 in CASIMIR_DEGREES[name]


def verify_d_vanishes(gen_array: np.ndarray, n_samples: int = 50, seed: int = 12345) -> float:
    """Verify the symmetric d-tensor vanishes by sampling.

    The d-tensor is defined as:
        d_{ij}^k = Tr(ad_i @ ad_j @ ad_k) + Tr(ad_j @ ad_i @ ad_k)

    This should be zero for all i,j,k when there is no cubic Casimir.

    Args:
        gen_array: Array of adjoint representation matrices, shape (dim, dim, dim)
        n_samples: Number of random (i,j) pairs to test
        seed: Random seed for reproducibility

    Returns:
        Maximum absolute value of d_{ij}^k found (should be ~0).
    """
    dim = gen_array.shape[0]
    rng = np.random.RandomState(seed)
    max_sym = 0.0

    for _ in range(n_samples):
        i = rng.randint(dim)
        j = rng.randint(dim)
        if i == j:
            continue
        P_ij = gen_array[i] @ gen_array[j]
        P_ji = gen_array[j] @ gen_array[i]
        # d_{ij}^k = Tr(ad_i ad_j ad_k) + Tr(ad_j ad_i ad_k)
        # = Tr((ad_i ad_j + ad_j ad_i) ad_k) for all k
        # = sum over matrix entries: ((P_ij + P_ji) @ ad_k) trace
        # Efficiently: t_k = Tr((P_ij + P_ji) @ ad_k)
        sym = P_ij + P_ji
        t = np.einsum('ab,kba->k', sym, gen_array)
        max_sym = max(max_sym, np.max(np.abs(t)))

    return max_sym


def casimir_eigenvalue_adjoint(name: str) -> float:
    """Quadratic Casimir eigenvalue in the adjoint representation.

    For the normalization where long roots have length^2 = 2:
    C_2(adj) = 2 * h^v  (dual Coxeter number)
    But with Killing form normalization Tr(ad_i ad_j) = 2*h^v * delta_ij for root gens,
    the eigenvalue is just 2*h^v.

    Actually the standard result is C_2(adj) = 2*g where g is the dual Coxeter number.
    """
    return 2.0 * DUAL_COXETER[name]
