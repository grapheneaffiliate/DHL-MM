"""
ExceptionalAlgebra — Unified DHL-MM engine for all five exceptional Lie algebras.

Provides a single interface for G2, F4, E6, E7, E8 with:
- Sparse structure constants (precomputed)
- O(|f|) Lie bracket
- Full product projection (valid when d-tensor vanishes)
- Killing form
- d-tensor verification

Usage:
    from exceptional import ExceptionalAlgebra

    g2 = ExceptionalAlgebra("G2")
    x = np.random.randn(14)
    y = np.random.randn(14)
    z = g2.bracket(x, y)
    k = g2.killing_form(x, y)
    d_max = g2.verify_d_vanishes()
"""

import numpy as np
from typing import Optional

from .roots import ALGEBRA_INFO
from .structure import compute_structure_constants
from .casimir import (
    has_cubic_casimir,
    verify_d_vanishes as _verify_d_vanishes,
    CASIMIR_DEGREES,
    DUAL_COXETER,
)


class ExceptionalAlgebra:
    """DHL-MM sparse engine for an exceptional Lie algebra.

    Precomputes root system and structure constants on initialization.
    Provides sparse Lie bracket, full product, and Killing form.

    Attributes:
        name: Algebra name (G2, F4, E6, E7, E8)
        roots: Root vectors, shape (n_roots, ambient_dim)
        simple_roots: Simple root vectors, shape (rank, ambient_dim)
        cartan_matrix: Cartan matrix, shape (rank, rank)
        fI, fJ, fK, fC: Sparse structure constant arrays
        gen_array: Adjoint representation matrices, shape (dim, dim, dim)
        killing: Killing form matrix, shape (dim, dim)
    """

    def __init__(self, name: str) -> None:
        """Initialize and precompute all algebra data.

        Args:
            name: One of "G2", "F4", "E6", "E7", "E8"
        """
        if name not in ALGEBRA_INFO:
            raise ValueError(f"Unknown algebra: {name}. Must be one of {list(ALGEBRA_INFO.keys())}")

        self.name = name
        self._dim, self._rank, self._n_roots = ALGEBRA_INFO[name]

        # Compute structure constants
        result = compute_structure_constants(name)
        self.roots = result[0]
        self.simple_roots_arr = result[1]
        self.cartan_mat = result[2]
        self.fI = result[3]
        self.fJ = result[4]
        self.fK = result[5]
        self.fC = result[6]

        # Build adjoint representation matrices
        self.gen_array = self._build_adjoint()

        # Compute Killing form
        self.killing = np.einsum('iab,jba->ij', self.gen_array, self.gen_array)

        # d-tensor status
        self._d_verified = False
        self._d_max = None
        self._d_vanishes = not has_cubic_casimir(name)  # theoretical expectation

    def _build_adjoint(self) -> np.ndarray:
        """Build adjoint representation matrices from sparse structure constants."""
        d = self._dim
        gen = np.zeros((d, d, d), dtype=np.float64)
        for idx in range(len(self.fI)):
            i, j, k = int(self.fI[idx]), int(self.fJ[idx]), int(self.fK[idx])
            c = self.fC[idx]
            gen[i][k, j] += c
        return gen

    @property
    def dim(self) -> int:
        """Dimension of the Lie algebra."""
        return self._dim

    @property
    def rank(self) -> int:
        """Rank of the Lie algebra."""
        return self._rank

    @property
    def n_roots(self) -> int:
        """Number of roots in the root system."""
        return self._n_roots

    @property
    def n_structure_constants(self) -> int:
        """Number of nonzero structure constant entries."""
        return len(self.fI)

    def bracket(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Lie bracket [x, y] using sparse structure constants. O(|f|).

        Args:
            x: Algebra element, shape (dim,)
            y: Algebra element, shape (dim,)

        Returns:
            [x, y] as a dim-vector
        """
        result = np.zeros(self._dim)
        contributions = x[self.fI] * y[self.fJ] * self.fC
        np.add.at(result, self.fK, contributions)
        return result

    def full_product(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Full product x*y projected to the Lie algebra.

        When the d-tensor vanishes (no cubic Casimir), this equals [x,y]/2.
        This is true for ALL exceptional algebras.

        Args:
            x: Algebra element, shape (dim,)
            y: Algebra element, shape (dim,)

        Returns:
            Projected product as a dim-vector
        """
        return self.bracket(x, y) / 2.0

    def killing_form(self, x: np.ndarray, y: np.ndarray) -> float:
        """Killing form K(x, y) = Tr(ad(x) @ ad(y)).

        Args:
            x: Algebra element, shape (dim,)
            y: Algebra element, shape (dim,)

        Returns:
            Scalar Killing form value
        """
        return float(x @ self.killing @ y)

    def verify_d_vanishes(self, n_samples: int = 50) -> float:
        """Verify the symmetric d-tensor vanishes.

        Computes d_{ij}^k = Tr(ad_i ad_j ad_k) + Tr(ad_j ad_i ad_k)
        for random (i,j) pairs. Should be zero for all exceptional algebras
        since none have a degree-3 Casimir invariant.

        Args:
            n_samples: Number of random pairs to check

        Returns:
            Maximum absolute d-tensor component found (should be ~0)
        """
        self._d_max = _verify_d_vanishes(self.gen_array, n_samples=n_samples)
        self._d_verified = True
        return self._d_max

    def verify_jacobi(self, n_triples: int = 200, seed: int = 42) -> float:
        """Verify Jacobi identity on random triples.

        [x,[y,z]] + [y,[z,x]] + [z,[x,y]] = 0

        Returns maximum violation norm.
        """
        rng = np.random.RandomState(seed)
        max_viol = 0.0
        for _ in range(n_triples):
            x = rng.randn(self._dim)
            y = rng.randn(self._dim)
            z = rng.randn(self._dim)
            t1 = self.bracket(x, self.bracket(y, z))
            t2 = self.bracket(y, self.bracket(z, x))
            t3 = self.bracket(z, self.bracket(x, y))
            viol = np.max(np.abs(t1 + t2 + t3))
            max_viol = max(max_viol, viol)
        return max_viol

    def verify_antisymmetry(self) -> float:
        """Verify f_{ij}^k = -f_{ji}^k for all structure constant entries.

        Returns maximum violation.
        """
        # Build dict of (i,j,k) -> c
        sc = {}
        for idx in range(len(self.fI)):
            key = (int(self.fI[idx]), int(self.fJ[idx]), int(self.fK[idx]))
            sc[key] = self.fC[idx]

        max_viol = 0.0
        for (i, j, k), c in sc.items():
            c_rev = sc.get((j, i, k), 0.0)
            viol = abs(c + c_rev)
            max_viol = max(max_viol, viol)
        return max_viol

    def compression_ratio(self) -> float:
        """Compute compression ratio: dim^3 / n_structure_constants."""
        full = self._dim ** 3
        return full / self.n_structure_constants if self.n_structure_constants > 0 else float('inf')

    def __repr__(self) -> str:
        return (
            f"ExceptionalAlgebra({self.name}: dim={self.dim}, rank={self.rank}, "
            f"roots={self.n_roots}, nonzero_f={self.n_structure_constants})"
        )
