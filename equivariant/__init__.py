"""
Equivariant neural network layers using DHL-MM sparse structure constants.

Provides differentiable PyTorch wrappers around the E8 Lie algebra engine
for building equivariant graph neural networks.
"""

from .sparse_kernel import SparseLieBracket, SparseLieBracketFn, SparseKillingForm
from .layers import LieConvLayer, ClebschGordanDecomposer
from .model import ExceptionalEGNN

__all__ = [
    "SparseLieBracket",
    "SparseLieBracketFn",
    "SparseKillingForm",
    "LieConvLayer",
    "ClebschGordanDecomposer",
    "ExceptionalEGNN",
]
