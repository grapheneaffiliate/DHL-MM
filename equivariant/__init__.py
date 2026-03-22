"""
Equivariant neural network layers using DHL-MM sparse structure constants.

Provides differentiable PyTorch wrappers around exceptional Lie algebra engines
for building equivariant graph neural networks. Supports all five exceptional
algebras: G2, F4, E6, E7, E8.
"""

from .sparse_kernel import SparseLieBracket, SparseLieBracketFn, SparseKillingForm
from .layers import (
    LieConvLayer,
    EquivariantLieConvLayer,
    AdjointLinearLayer,
    AdjointBilinearLayer,
    ClebschGordanDecomposer,
)
from .model import ExceptionalEGNN

__all__ = [
    "SparseLieBracket",
    "SparseLieBracketFn",
    "SparseKillingForm",
    "LieConvLayer",
    "EquivariantLieConvLayer",
    "AdjointLinearLayer",
    "AdjointBilinearLayer",
    "ClebschGordanDecomposer",
    "ExceptionalEGNN",
]
