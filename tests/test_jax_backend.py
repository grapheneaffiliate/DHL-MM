"""
Tests for the JAX backend of DHL-MM sparse Lie bracket.

All tests are skipped gracefully if JAX is not installed.
Run with: py tests/test_jax_backend.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import jax
    import jax.numpy as jnp
    # Use double precision by default for numerical accuracy
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

ALGEBRAS = ["G2", "F4", "E6", "E7", "E8"]
DIMS = {"G2": 14, "F4": 52, "E6": 78, "E7": 133, "E8": 248}


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestJaxBracketMatchesNumpy(unittest.TestCase):
    """JaxSparseBracket output matches numpy algebra.bracket() for all algebras."""

    def test_bracket_matches_numpy(self):
        from dhl_mm.jax_backend import JaxSparseBracket
        import dhl_mm

        rng = np.random.RandomState(42)

        for name in ALGEBRAS:
            with self.subTest(algebra=name):
                alg = dhl_mm.algebra(name)
                jbracket = JaxSparseBracket.from_algebra(name)
                dim = alg.dim

                for _ in range(10):
                    x_np = rng.randn(dim)
                    y_np = rng.randn(dim)

                    z_np = alg.bracket(x_np, y_np)
                    z_jax = jbracket(jnp.array(x_np), jnp.array(y_np))

                    np.testing.assert_allclose(
                        np.array(z_jax), z_np, atol=1e-12, rtol=1e-12,
                        err_msg=f"{name}: JAX bracket differs from numpy"
                    )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestAntisymmetry(unittest.TestCase):
    """bracket(x, y) = -bracket(y, x)."""

    def test_antisymmetry(self):
        from dhl_mm.jax_backend import JaxSparseBracket

        rng = np.random.RandomState(99)

        for name in ALGEBRAS:
            with self.subTest(algebra=name):
                jbracket = JaxSparseBracket.from_algebra(name)
                dim = DIMS[name]

                for _ in range(5):
                    x = jnp.array(rng.randn(dim))
                    y = jnp.array(rng.randn(dim))

                    z_xy = jbracket(x, y)
                    z_yx = jbracket(y, x)

                    np.testing.assert_allclose(
                        np.array(z_xy), -np.array(z_yx), atol=1e-12,
                        err_msg=f"{name}: antisymmetry violated"
                    )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestJacobi(unittest.TestCase):
    """Jacobi identity: [x,[y,z]] + [y,[z,x]] + [z,[x,y]] = 0."""

    def test_jacobi(self):
        from dhl_mm.jax_backend import JaxSparseBracket

        rng = np.random.RandomState(7)

        for name in ALGEBRAS:
            with self.subTest(algebra=name):
                jbracket = JaxSparseBracket.from_algebra(name)
                dim = DIMS[name]

                for _ in range(5):
                    x = jnp.array(rng.randn(dim))
                    y = jnp.array(rng.randn(dim))
                    z = jnp.array(rng.randn(dim))

                    t1 = jbracket(x, jbracket(y, z))
                    t2 = jbracket(y, jbracket(z, x))
                    t3 = jbracket(z, jbracket(x, y))

                    violation = np.max(np.abs(np.array(t1 + t2 + t3)))
                    self.assertLess(
                        violation, 1e-10,
                        f"{name}: Jacobi violation {violation:.2e}"
                    )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestGradient(unittest.TestCase):
    """jax.grad works through the bracket."""

    def test_gradient(self):
        from dhl_mm.jax_backend import JaxSparseBracket

        for name in ["G2", "F4"]:
            with self.subTest(algebra=name):
                jbracket = JaxSparseBracket.from_algebra(name)
                dim = DIMS[name]

                x = jnp.ones(dim, dtype=jnp.float64)
                y = jnp.ones(dim, dtype=jnp.float64) * 2.0

                def loss_fn(x_):
                    return jnp.sum(jbracket(x_, y))

                grad_fn = jax.grad(loss_fn)
                g = grad_fn(x)

                # Gradient should be a finite array of the right shape
                self.assertEqual(g.shape, (dim,))
                self.assertTrue(jnp.all(jnp.isfinite(g)),
                                f"{name}: gradient contains non-finite values")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestJitCompilation(unittest.TestCase):
    """JIT version produces same result as non-JIT."""

    def test_jit_compilation(self):
        from dhl_mm.jax_backend import JaxSparseBracket

        rng = np.random.RandomState(123)

        for name in ["G2", "E6"]:
            with self.subTest(algebra=name):
                jbracket = JaxSparseBracket.from_algebra(name)
                dim = DIMS[name]

                x = jnp.array(rng.randn(dim))
                y = jnp.array(rng.randn(dim))

                z_nojit = jbracket(x, y)
                z_jit = jbracket.bracket_jit(x, y)

                np.testing.assert_allclose(
                    np.array(z_jit), np.array(z_nojit), atol=1e-14,
                    err_msg=f"{name}: JIT result differs from non-JIT"
                )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestVmapBatch(unittest.TestCase):
    """batch_bracket matches loop over individual pairs."""

    def test_vmap_batch(self):
        from dhl_mm.jax_backend import JaxLieAlgebra

        rng = np.random.RandomState(55)
        batch_size = 8

        for name in ["G2", "F4"]:
            with self.subTest(algebra=name):
                alg = JaxLieAlgebra(name)
                dim = DIMS[name]

                xs = jnp.array(rng.randn(batch_size, dim))
                ys = jnp.array(rng.randn(batch_size, dim))

                # Batched via vmap
                z_batch = alg.batch_bracket(xs, ys)

                # Loop over individual pairs
                for i in range(batch_size):
                    z_i = alg.bracket(xs[i], ys[i])
                    np.testing.assert_allclose(
                        np.array(z_batch[i]), np.array(z_i), atol=1e-12,
                        err_msg=f"{name}: batch[{i}] differs from individual"
                    )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestKillingFormMatchesNumpy(unittest.TestCase):
    """JaxSparseKillingForm matches numpy Killing form."""

    def test_killing_form_matches_numpy(self):
        from dhl_mm.jax_backend import JaxSparseKillingForm
        import dhl_mm

        rng = np.random.RandomState(77)

        for name in ALGEBRAS:
            with self.subTest(algebra=name):
                alg = dhl_mm.algebra(name)
                jkill = JaxSparseKillingForm.from_algebra(name)
                dim = alg.dim

                for _ in range(10):
                    x_np = rng.randn(dim)
                    y_np = rng.randn(dim)

                    k_np = alg.killing_form(x_np, y_np)
                    k_jax = jkill(jnp.array(x_np), jnp.array(y_np))

                    np.testing.assert_allclose(
                        float(k_jax), k_np, atol=1e-10, rtol=1e-10,
                        err_msg=f"{name}: JAX killing form differs from numpy"
                    )


class TestImportWithoutJax(unittest.TestCase):
    """Importing the module doesn't crash even without JAX."""

    def test_import_without_jax(self):
        # The module should always be importable; if JAX is missing,
        # it sets _HAS_JAX = False and classes raise on instantiation.
        # We verify the module-level import doesn't raise.
        import importlib
        mod = importlib.import_module("dhl_mm.jax_backend")
        self.assertTrue(hasattr(mod, "_HAS_JAX"))
        self.assertTrue(hasattr(mod, "JaxSparseBracket"))
        self.assertTrue(hasattr(mod, "JaxSparseKillingForm"))
        self.assertTrue(hasattr(mod, "JaxLieAlgebra"))


if __name__ == "__main__":
    unittest.main()
