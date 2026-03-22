"""
Optional C extension for sparse bracket computation.
Falls back to numpy if C extension not compiled.

Usage:
    from dhl_mm.csparse import sparse_bracket, has_c_extension

    result = sparse_bracket(x, y, I, J, K, C, dim)
"""
import numpy as np

_HAS_C_EXTENSION = False
try:
    from dhl_mm._csparse import (
        sparse_bracket as _c_bracket,
        sparse_bracket_batched as _c_bracket_batched,
        sparse_bracket_backward as _c_backward,
        sparse_bracket_backward_batched as _c_backward_batched,
        sparse_bracket_f32 as _c_bracket_f32,
        sparse_bracket_batched_f32 as _c_bracket_batched_f32,
        sparse_bracket_backward_f32 as _c_backward_f32,
        sparse_bracket_backward_batched_f32 as _c_backward_batched_f32,
    )
    _HAS_C_EXTENSION = True
except ImportError:
    pass


def has_c_extension():
    """Return True if the compiled C extension is available."""
    return _HAS_C_EXTENSION


def sparse_bracket(x, y, I, J, K, C, dim):
    """Sparse Lie bracket using C extension if available, numpy fallback.

    Computes result[k] = sum_{n where K[n]==k} C[n] * x[I[n]] * y[J[n]]

    Args:
        x: (dim,) or (batch, dim) array
        y: (dim,) or (batch, dim) array
        I, J, K: (n_entries,) int32 index arrays
        C: (n_entries,) float array of structure constants
        dim: algebra dimension

    Returns:
        result: same shape as x
    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    if _HAS_C_EXTENSION:
        is_f32 = (x.dtype == np.float32)
        if x.ndim == 1:
            if is_f32:
                return _c_bracket_f32(x, y, I, J, K, C.astype(np.float32), dim)
            return _c_bracket(
                x.astype(np.float64), y.astype(np.float64),
                I.astype(np.int32), J.astype(np.int32), K.astype(np.int32),
                C.astype(np.float64), dim
            )
        elif x.ndim == 2:
            if is_f32:
                return _c_bracket_batched_f32(x, y, I, J, K, C.astype(np.float32), dim)
            return _c_bracket_batched(
                x.astype(np.float64), y.astype(np.float64),
                I.astype(np.int32), J.astype(np.int32), K.astype(np.int32),
                C.astype(np.float64), dim
            )
        else:
            raise ValueError(f"x must be 1-D or 2-D, got ndim={x.ndim}")

    # Numpy fallback
    if x.ndim == 1:
        result = np.zeros(dim, dtype=x.dtype)
        contributions = x[I] * y[J] * C
        np.add.at(result, K, contributions)
        return result
    elif x.ndim == 2:
        batch = x.shape[0]
        result = np.zeros((batch, dim), dtype=x.dtype)
        # Vectorized over batch: x[:, I] is (batch, n_entries)
        contributions = x[:, I] * y[:, J] * C[np.newaxis, :]
        np.add.at(result, (np.arange(batch)[:, np.newaxis], K[np.newaxis, :]), contributions)
        return result
    else:
        raise ValueError(f"x must be 1-D or 2-D, got ndim={x.ndim}")


def sparse_bracket_backward(x, y, grad_output, I, J, K, C, dim):
    """Backward pass for sparse Lie bracket.

    Args:
        x: (dim,) or (batch, dim) array
        y: (dim,) or (batch, dim) array
        grad_output: same shape as x — gradient of loss w.r.t. bracket output
        I, J, K: (n_entries,) int32 index arrays
        C: (n_entries,) float array of structure constants
        dim: algebra dimension

    Returns:
        (grad_x, grad_y): gradients w.r.t. x and y
    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    grad_output = np.ascontiguousarray(grad_output)

    if _HAS_C_EXTENSION:
        is_f32 = (x.dtype == np.float32)
        if x.ndim == 1:
            if is_f32:
                return _c_backward_f32(x, y, grad_output, I, J, K, C.astype(np.float32), dim)
            return _c_backward(
                x.astype(np.float64), y.astype(np.float64),
                grad_output.astype(np.float64),
                I.astype(np.int32), J.astype(np.int32), K.astype(np.int32),
                C.astype(np.float64), dim
            )
        elif x.ndim == 2:
            if is_f32:
                return _c_backward_batched_f32(x, y, grad_output, I, J, K, C.astype(np.float32), dim)
            return _c_backward_batched(
                x.astype(np.float64), y.astype(np.float64),
                grad_output.astype(np.float64),
                I.astype(np.int32), J.astype(np.int32), K.astype(np.int32),
                C.astype(np.float64), dim
            )
        else:
            raise ValueError(f"x must be 1-D or 2-D, got ndim={x.ndim}")

    # Numpy fallback
    if x.ndim == 1:
        grad_x = np.zeros(dim, dtype=x.dtype)
        grad_y = np.zeros(dim, dtype=y.dtype)
        cg = C * grad_output[K]
        np.add.at(grad_x, I, cg * y[J])
        np.add.at(grad_y, J, cg * x[I])
        return grad_x, grad_y
    elif x.ndim == 2:
        batch = x.shape[0]
        grad_x = np.zeros((batch, dim), dtype=x.dtype)
        grad_y = np.zeros((batch, dim), dtype=y.dtype)
        cg = C[np.newaxis, :] * grad_output[:, K]  # (batch, n_entries)
        batch_idx = np.arange(batch)[:, np.newaxis]
        np.add.at(grad_x, (batch_idx, I[np.newaxis, :]), cg * y[:, J])
        np.add.at(grad_y, (batch_idx, J[np.newaxis, :]), cg * x[:, I])
        return grad_x, grad_y
    else:
        raise ValueError(f"x must be 1-D or 2-D, got ndim={x.ndim}")
