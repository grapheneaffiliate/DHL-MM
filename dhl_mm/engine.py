"""
DHL-MM Engine: the complete sparse E8 multiplication framework.
"""

import numpy as np

from .e8 import (
    DIM,
    compute_full_structure_constants,
    build_adjoint_matrices,
    lie_bracket,
)
from .zphi import ZPhi, PHI
from .defect import DefectMonitor


class DHLMM:
    """Dynamic Hodge-Lie Matrix Multiplication engine.

    Multiplies 248-dimensional E8 Lie algebra elements using 16,694
    precomputed structure constants instead of full 248x248 matrix
    multiplication. 913x fewer operations, verified to machine epsilon.

    Usage:
        engine = DHLMM.build()
        z = engine.bracket(x, y)        # [x, y] via sparse constants
        z = engine.full_product(x, y)   # full product projected to Lie algebra
        z, h2 = engine.product_with_defect(x, y)  # with error monitoring
    """

    def __init__(self, data):
        self.data = data
        self.fI = data['I']
        self.fJ = data['J']
        self.fK = data['K']
        self.fC = data['C']
        self.gen_array = data['gen_array']
        self.killing = data['killing']
        self.killing_inv = data['killing_inv']
        self.casimir = data['casimir_eigenvalue']
        self.monitor = DefectMonitor(self.casimir)

        self._fC_int = []
        for c in self.fC:
            ci = int(round(c))
            assert abs(c - ci) < 1e-10
            self._fC_int.append(ci)

    @classmethod
    def build(cls, verbose=False):
        """Build a DHLMM engine from scratch (precomputes all E8 data)."""
        roots, I, J, K, C, roots_simple = compute_full_structure_constants(verbose=verbose)
        generators = build_adjoint_matrices(roots, I, J, K, C)
        gen_array = np.array(generators)
        killing = np.einsum('iab,jba->ij', gen_array, gen_array)
        killing_inv = np.linalg.inv(killing)

        data = {
            'roots': roots,
            'roots_simple': roots_simple,
            'I': I, 'J': J, 'K': K, 'C': C,
            'generators': generators,
            'gen_array': gen_array,
            'killing': killing,
            'killing_inv': killing_inv,
            'killing_trace': np.trace(killing),
            'casimir_eigenvalue': 60.0,
        }
        return cls(data)

    @property
    def n_structure_constants(self):
        """Number of nonzero structure constant entries."""
        return len(self.fI)

    def bracket(self, x, y):
        """Lie bracket [x, y] using sparse structure constants. O(16,694)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return lie_bracket(x, y, self.fI, self.fJ, self.fK, self.fC)

    def full_product(self, x, y):
        """Full product x*y projected to Lie algebra.

        For E8: since d_{ij}^k = 0 (no cubic Casimir), this equals [x,y]/2.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        result = np.zeros(DIM)
        contributions = x[self.fI] * y[self.fJ] * self.fC
        np.add.at(result, self.fK, contributions)
        return result / 2.0

    def full_product_reference(self, x, y):
        """Full product via matrix multiplication (slow reference)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X = np.einsum('i,iab->ab', x, self.gen_array)
        Y = np.einsum('i,iab->ab', y, self.gen_array)
        XY = X @ Y
        traces = np.einsum('ab,kba->k', XY, self.gen_array)
        return self.killing_inv @ traces

    def killing_form(self, x, y):
        """Killing form K(x, y)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return x @ self.killing @ y

    def product_with_defect(self, x, y):
        """Full product with defect monitoring and optional lattice pruning."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        result = self.bracket(x, y)
        h2, should_prune = self.monitor.update(result)
        if should_prune:
            result = self.monitor.prune(result)
        return result, h2

    def zphi_bracket(self, x_zphi, y_zphi):
        """Exact Z[phi] Lie bracket using integer arithmetic only."""
        result = [ZPhi(0, 0) for _ in range(DIM)]
        fC_int = self._fC_int
        for idx in range(len(self.fI)):
            i, j, k = int(self.fI[idx]), int(self.fJ[idx]), int(self.fK[idx])
            xi = x_zphi[i]
            yj = y_zphi[j]
            if xi.a == 0 and xi.b == 0:
                continue
            if yj.a == 0 and yj.b == 0:
                continue
            c = fC_int[idx]
            contrib = xi * yj * c
            result[k] = result[k] + contrib
        return result

    def verify_d_vanishes(self, n_samples=50):
        """Verify that the symmetric d-tensor vanishes (E8 has no cubic Casimir).

        Returns the maximum symmetric component found (should be ~0).
        """
        rng = np.random.RandomState(12345)
        max_sym = 0.0
        for _ in range(n_samples):
            i = rng.randint(DIM)
            j = rng.randint(DIM)
            if i == j:
                continue
            P_ij = self.gen_array[i] @ self.gen_array[j]
            P_ji = self.gen_array[j] @ self.gen_array[i]
            t_ijk = np.einsum('ab,kba->k', P_ij, self.gen_array)
            t_jik = np.einsum('ab,kba->k', P_ji, self.gen_array)
            sym = t_ijk + t_jik
            max_sym = max(max_sym, np.max(np.abs(sym)))
        return max_sym
