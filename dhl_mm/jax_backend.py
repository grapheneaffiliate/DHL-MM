"""
JAX backend for DHL-MM sparse Lie bracket.

Provides JIT-compiled, differentiable sparse bracket operations using JAX.
JAX is an optional dependency -- the module imports cleanly without it but
raises an informative error when classes are instantiated.

Usage:
    from dhl_mm.jax_backend import JaxSparseBracket

    bracket = JaxSparseBracket.from_algebra("E8")
    z = bracket(x, y)           # JIT-compiled sparse bracket
    grad_fn = jax.grad(lambda x: bracket(x, y).sum())
    g = grad_fn(x)              # automatic differentiation
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_jvp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _require_jax():
    """Raise an informative error if JAX is not installed."""
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for the JAX backend but is not installed. "
            "Install it with: pip install jax jaxlib"
        )


# ---------------------------------------------------------------------------
# Module-level pure function with custom JVP for forward-mode AD
# ---------------------------------------------------------------------------

if _HAS_JAX:
    @custom_jvp
    def _bracket_fn(x, y, I, J, K, C, dim):
        """Pure-function sparse Lie bracket for JAX tracing."""
        xi = x[I]
        yj = y[J]
        contributions = xi * yj * C
        result = jnp.zeros(dim, dtype=x.dtype).at[K].add(contributions)
        return result

    @_bracket_fn.defjvp
    def _bracket_fn_jvp(primals, tangents):
        x, y, I, J, K, C, dim = primals
        dx, dy, *_ = tangents
        primal_out = _bracket_fn(x, y, I, J, K, C, dim)
        # d/dx [x,y] . dx = [dx, y] and d/dy [x,y] . dy = [x, dy]
        tangent_out = _bracket_fn(dx, y, I, J, K, C, dim) + _bracket_fn(x, dy, I, J, K, C, dim)
        return primal_out, tangent_out


# ---------------------------------------------------------------------------
# JaxSparseBracket
# ---------------------------------------------------------------------------

class JaxSparseBracket:
    """Sparse Lie bracket [x, y] using JAX arrays and scatter-add.

    Supports JIT compilation and automatic differentiation.
    """

    def __init__(self, I, J, K, C, dim):
        """Initialize with index arrays and structure constants.

        Args:
            I: int32 array of first indices, shape (n_entries,)
            J: int32 array of second indices, shape (n_entries,)
            K: int32 array of output indices, shape (n_entries,)
            C: float64 array of structure constants, shape (n_entries,)
            dim: dimension of the Lie algebra
        """
        _require_jax()
        self.I = jnp.asarray(I, dtype=jnp.int32)
        self.J = jnp.asarray(J, dtype=jnp.int32)
        self.K = jnp.asarray(K, dtype=jnp.int32)
        self.C = jnp.asarray(C, dtype=jnp.float64)
        self.dim = int(dim)

    @classmethod
    def from_algebra(cls, name):
        """Build a JaxSparseBracket from a named exceptional Lie algebra.

        Args:
            name: One of "G2", "F4", "E6", "E7", "E8"

        Returns:
            JaxSparseBracket instance
        """
        _require_jax()
        from . import algebra
        alg = algebra(name)
        return cls(
            I=np.asarray(alg.fI, dtype=np.int32),
            J=np.asarray(alg.fJ, dtype=np.int32),
            K=np.asarray(alg.fK, dtype=np.int32),
            C=np.asarray(alg.fC, dtype=np.float64),
            dim=alg.dim,
        )

    def __call__(self, x, y):
        """Compute the sparse Lie bracket [x, y].

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            jax array of shape (dim,), the bracket [x, y]
        """
        xi = x[self.I]
        yj = y[self.J]
        contributions = xi * yj * self.C
        result = jnp.zeros(self.dim, dtype=x.dtype).at[self.K].add(contributions)
        return result

    def bracket_jit(self, x, y):
        """JIT-compiled version of the sparse Lie bracket.

        On the first call, JAX traces and compiles the function.
        Subsequent calls with same-shaped inputs reuse the compiled version.

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            jax array of shape (dim,), the bracket [x, y]
        """
        if not hasattr(self, '_jit_fn'):
            self._jit_fn = jax.jit(self.__call__)
        return self._jit_fn(x, y)


# ---------------------------------------------------------------------------
# JaxSparseKillingForm
# ---------------------------------------------------------------------------

class JaxSparseKillingForm:
    """Killing form K(x, y) = x @ killing_matrix @ y using JAX."""

    def __init__(self, killing_matrix):
        """Initialize with the Killing form matrix.

        Args:
            killing_matrix: (dim, dim) array, the Killing form matrix
        """
        _require_jax()
        self.killing = jnp.asarray(killing_matrix, dtype=jnp.float64)

    @classmethod
    def from_algebra(cls, name):
        """Build a JaxSparseKillingForm from a named exceptional Lie algebra.

        Args:
            name: One of "G2", "F4", "E6", "E7", "E8"

        Returns:
            JaxSparseKillingForm instance
        """
        _require_jax()
        from . import algebra
        alg = algebra(name)
        return cls(killing_matrix=np.asarray(alg.killing, dtype=np.float64))

    def __call__(self, x, y):
        """Compute K(x, y) = x @ killing @ y.

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            scalar jax array
        """
        return x @ self.killing @ y


# ---------------------------------------------------------------------------
# JaxLieAlgebra -- convenience wrapper
# ---------------------------------------------------------------------------

class JaxLieAlgebra:
    """Convenience wrapper combining bracket, Killing form, and utilities.

    Provides a high-level API for JAX-accelerated Lie algebra computations
    including JIT compilation, automatic differentiation, and batched
    operations via vmap.
    """

    def __init__(self, algebra_name):
        """Initialize a JAX Lie algebra engine.

        Args:
            algebra_name: One of "G2", "F4", "E6", "E7", "E8"
        """
        _require_jax()
        self.algebra_name = algebra_name.upper()
        self._bracket_op = JaxSparseBracket.from_algebra(self.algebra_name)
        self._killing_op = JaxSparseKillingForm.from_algebra(self.algebra_name)
        self.dim = self._bracket_op.dim

    def bracket(self, x, y):
        """Compute the Lie bracket [x, y].

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            jax array of shape (dim,)
        """
        return self._bracket_op(x, y)

    def killing_form(self, x, y):
        """Compute the Killing form K(x, y).

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            scalar jax array
        """
        return self._killing_op(x, y)

    def full_product(self, x, y):
        """Full product x*y projected to the Lie algebra.

        For all exceptional algebras, the d-tensor vanishes so this
        equals bracket(x, y) / 2.

        Args:
            x: jax array of shape (dim,)
            y: jax array of shape (dim,)

        Returns:
            jax array of shape (dim,)
        """
        return self.bracket(x, y) / 2.0

    def batch_bracket(self, xs, ys):
        """Batched Lie bracket using jax.vmap.

        Args:
            xs: jax array of shape (batch, dim)
            ys: jax array of shape (batch, dim)

        Returns:
            jax array of shape (batch, dim)
        """
        return jax.vmap(self.bracket)(xs, ys)

    def __repr__(self):
        return f"JaxLieAlgebra({self.algebra_name}, dim={self.dim})"
