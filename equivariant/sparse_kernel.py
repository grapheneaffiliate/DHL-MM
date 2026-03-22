"""
PyTorch autograd wrapper around DHL-MM sparse structure constant engine.

Provides differentiable Lie bracket and Killing form operations that
register structure constants as non-trainable buffers and support
both CPU and GPU computation.

Supports all five exceptional Lie algebras via from_algebra() factory.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class SparseLieBracketFn(torch.autograd.Function):
    """Differentiable sparse Lie bracket via scatter_add / gather."""

    @staticmethod
    def forward(ctx, x, y, I, J, K, C, dim):
        """
        Compute z[..., k] = sum_{entries where K==k} x[..., I] * y[..., J] * C

        Args:
            x: (..., dim) tensor
            y: (..., dim) tensor
            I, J, K: (n_entries,) int64 index tensors
            C: (n_entries,) float tensor of structure constants
            dim: algebra dimension (int)
        """
        ctx.save_for_backward(x, y, I, J, K, C)
        ctx.dim = dim

        # Gather values at sparse indices
        x_I = x[..., I]  # (..., n_entries)
        y_J = y[..., J]  # (..., n_entries)
        contributions = x_I * y_J * C  # (..., n_entries)

        # Scatter-add into result
        batch_shape = x.shape[:-1]
        result = x.new_zeros(*batch_shape, dim)
        K_expanded = K.expand_as(contributions)
        result.scatter_add_(-1, K_expanded, contributions)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for sparse Lie bracket.

        grad_x[..., i] = sum over entries where I==i of: y[..., J] * C * grad_output[..., K]
        grad_y[..., j] = sum over entries where J==j of: x[..., I] * C * grad_output[..., K]
        """
        x, y, I, J, K, C = ctx.saved_tensors
        dim = ctx.dim

        # Gather grad_output at K positions
        grad_K = grad_output[..., K]  # (..., n_entries)
        x_I = x[..., I]
        y_J = y[..., J]

        # grad_x contributions: y[..., J] * C * grad_output[..., K], scattered by I
        grad_x_contribs = y_J * C * grad_K  # (..., n_entries)
        batch_shape = x.shape[:-1]
        grad_x = x.new_zeros(*batch_shape, dim)
        I_expanded = I.expand_as(grad_x_contribs)
        grad_x.scatter_add_(-1, I_expanded, grad_x_contribs)

        # grad_y contributions: x[..., I] * C * grad_output[..., K], scattered by J
        grad_y_contribs = x_I * C * grad_K  # (..., n_entries)
        grad_y = y.new_zeros(*batch_shape, dim)
        J_expanded = J.expand_as(grad_y_contribs)
        grad_y.scatter_add_(-1, J_expanded, grad_y_contribs)

        # No grads for I, J, K, C, dim
        return grad_x, grad_y, None, None, None, None, None


class SparseLieBracket(nn.Module):
    """
    Differentiable Lie bracket [x, y] using sparse structure constants.

    If I, J, K, C are not provided, builds them from DHLMM (E8 default).
    Structure constants are registered as buffers (not trainable, but
    move to GPU with the model).

    Use from_algebra(name) to build for any exceptional algebra.
    """

    def __init__(self, algebra_dim=248, I=None, J=None, K=None, C=None):
        super().__init__()
        self.algebra_dim = algebra_dim

        if I is None or J is None or K is None or C is None:
            from dhl_mm import DHLMM
            engine = DHLMM.build()
            I_np = engine.fI.astype(np.int64)
            J_np = engine.fJ.astype(np.int64)
            K_np = engine.fK.astype(np.int64)
            C_np = engine.fC.astype(np.float64)
            self.register_buffer("I", torch.from_numpy(I_np))
            self.register_buffer("J", torch.from_numpy(J_np))
            self.register_buffer("K", torch.from_numpy(K_np))
            self.register_buffer("C", torch.from_numpy(C_np))
        else:
            self.register_buffer("I", I.long() if isinstance(I, torch.Tensor) else torch.tensor(I, dtype=torch.long))
            self.register_buffer("J", J.long() if isinstance(J, torch.Tensor) else torch.tensor(J, dtype=torch.long))
            self.register_buffer("K", K.long() if isinstance(K, torch.Tensor) else torch.tensor(K, dtype=torch.long))
            self.register_buffer("C", C.double() if isinstance(C, torch.Tensor) else torch.tensor(C, dtype=torch.float64))

    @classmethod
    def from_algebra(cls, name: str):
        """
        Build a SparseLieBracket for any exceptional Lie algebra.

        Args:
            name: One of "G2", "F4", "E6", "E7", "E8"

        Returns:
            SparseLieBracket instance with structure constants from the algebra.
        """
        from dhl_mm.exceptional_engine import ExceptionalAlgebra
        alg = ExceptionalAlgebra(name)
        I = torch.from_numpy(alg.fI.astype(np.int64))
        J = torch.from_numpy(alg.fJ.astype(np.int64))
        K = torch.from_numpy(alg.fK.astype(np.int64))
        C = torch.from_numpy(alg.fC.astype(np.float64))
        return cls(algebra_dim=alg.dim, I=I, J=J, K=K, C=C)

    def forward(self, x, y):
        """
        Compute [x, y] using sparse structure constants.

        Args:
            x: (..., algebra_dim) tensor
            y: (..., algebra_dim) tensor

        Returns:
            z: (..., algebra_dim) tensor = [x, y]
        """
        # Ensure C matches input dtype for the multiplication
        C = self.C.to(dtype=x.dtype)
        return SparseLieBracketFn.apply(x, y, self.I, self.J, self.K, C, self.algebra_dim)


class SparseKillingForm(nn.Module):
    """
    Killing form K(x, y) = x @ killing_matrix @ y.

    The Killing matrix is registered as a buffer.

    Use from_algebra(name) to build for any exceptional algebra.
    """

    def __init__(self, killing_matrix=None):
        super().__init__()
        if killing_matrix is None:
            from dhl_mm import DHLMM
            engine = DHLMM.build()
            killing_matrix = engine.killing
        if isinstance(killing_matrix, np.ndarray):
            killing_matrix = torch.from_numpy(killing_matrix.astype(np.float64))
        self.register_buffer("killing", killing_matrix)

    @classmethod
    def from_algebra(cls, name: str):
        """
        Build a SparseKillingForm for any exceptional Lie algebra.

        Args:
            name: One of "G2", "F4", "E6", "E7", "E8"

        Returns:
            SparseKillingForm instance with the Killing matrix from the algebra.
        """
        from dhl_mm.exceptional_engine import ExceptionalAlgebra
        alg = ExceptionalAlgebra(name)
        killing = torch.from_numpy(alg.killing.astype(np.float64))
        return cls(killing_matrix=killing)

    def forward(self, x, y):
        """
        Compute Killing form K(x, y) = x @ killing @ y, batched.

        Args:
            x: (..., dim) tensor
            y: (..., dim) tensor

        Returns:
            scalar or (...,) tensor of Killing form values
        """
        killing = self.killing.to(dtype=x.dtype)
        # (..., dim) @ (dim, dim) -> (..., dim), then dot with y
        Ky = torch.einsum("...i,ij->...j", x, killing)
        return torch.einsum("...i,...i->...", Ky, y)
