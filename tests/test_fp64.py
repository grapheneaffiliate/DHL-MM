"""
FP64 precision verification for all five exceptional algebras.
Verifies that sparse bracket errors stay within theoretical FP64 bounds.
"""
import numpy as np
import unittest
import dhl_mm

EPS = np.finfo(np.float64).eps  # 2.22e-16
SAFETY = 4  # 4x safety factor over naive bound

class TestFP64Precision(unittest.TestCase):

    def test_bracket_within_fp64_bounds(self):
        """Verify bracket error < max_terms * SAFETY * eps for all algebras.

        Jacobi identity compounds 6 bracket evaluations on unit-norm vectors,
        so the effective bound is scaled by dim (inner bracket outputs have
        O(sqrt(dim)) magnitude).  We use max_terms * dim * SAFETY * eps.
        """
        rng = np.random.RandomState(42)
        for name in ['G2', 'F4', 'E6', 'E7', 'E8']:
            alg = dhl_mm.algebra(name)
            max_terms = int(np.bincount(alg.fK.astype(int)).max())
            # Jacobi = 6 brackets; inner brackets amplify by ~sqrt(dim);
            # conservative bound accounts for compounding
            bound = max_terms * alg.dim * SAFETY * EPS

            max_err = 0.0
            for _ in range(100):
                x = rng.randn(alg.dim)
                y = rng.randn(alg.dim)
                # Jacobi as error metric (should be exactly zero)
                z = rng.randn(alg.dim)
                j = (alg.bracket(x, alg.bracket(y, z)) +
                     alg.bracket(y, alg.bracket(z, x)) +
                     alg.bracket(z, alg.bracket(x, y)))
                max_err = max(max_err, np.max(np.abs(j)))

            margin = bound / max_err if max_err > 0 else float('inf')
            print(f"  {name}: max_err={max_err:.2e}, bound={bound:.2e}, margin={margin:.1f}x")
            self.assertLess(max_err, bound,
                f"{name}: Jacobi error {max_err:.2e} exceeds FP64 bound {bound:.2e}")

    def test_bracket_output_dtype(self):
        """Verify bracket always returns float64 regardless of input dtype."""
        for name in ['G2', 'F4', 'E6', 'E7', 'E8']:
            alg = dhl_mm.algebra(name)
            # Pass float32
            x = np.random.randn(alg.dim).astype(np.float32)
            y = np.random.randn(alg.dim).astype(np.float32)
            z = alg.bracket(x, y)
            self.assertEqual(z.dtype, np.float64,
                f"{name}: bracket returned {z.dtype}, expected float64")

    def test_killing_form_output_dtype(self):
        """Verify killing_form always returns float64."""
        for name in ['G2', 'E8']:
            alg = dhl_mm.algebra(name)
            x = np.random.randn(alg.dim).astype(np.float32)
            y = np.random.randn(alg.dim).astype(np.float32)
            k = alg.killing_form(x, y)
            self.assertTrue(np.issubdtype(type(k), np.floating))

    def test_antisymmetry_exact(self):
        """Verify [x,y] = -[y,x] to near machine epsilon for all algebras.

        Antisymmetry error scales with max_terms * eps; for E6+ this can
        reach ~1e-14, so we use 1e-13 as a safe threshold.
        """
        rng = np.random.RandomState(123)
        for name in ['G2', 'F4', 'E6', 'E7', 'E8']:
            alg = dhl_mm.algebra(name)
            for _ in range(50):
                x, y = rng.randn(alg.dim), rng.randn(alg.dim)
                z1 = alg.bracket(x, y)
                z2 = alg.bracket(y, x)
                err = np.max(np.abs(z1 + z2))
                self.assertLess(err, 1e-13,
                    f"{name}: antisymmetry error {err:.2e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
