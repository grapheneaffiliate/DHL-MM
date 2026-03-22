"""
Equivariant neural network layers built on sparse Lie bracket kernel.

LieConvLayer: Message-passing layer using Lie bracket for equivariant updates.
ClebschGordanDecomposer: Tensor product decomposition via structure constants.
"""

import torch
import torch.nn as nn

from .sparse_kernel import SparseLieBracket


class LieConvLayer(nn.Module):
    """
    Equivariant message-passing layer using sparse Lie bracket.

    For each edge (i, j):
        message_ij = bracket(W_msg @ features[i], features[j])
    Aggregate:
        agg_i = sum_j message_ij
    Update:
        features_i = features_i + MLP(agg_i)
    """

    def __init__(self, algebra_dim=248, hidden_dim=64, structure_constants=None):
        """
        Args:
            algebra_dim: dimension of the Lie algebra (default 248 for E8)
            hidden_dim: hidden dimension of the update MLP
            structure_constants: dict with keys 'I', 'J', 'K', 'C' or None to build from DHLMM
        """
        super().__init__()
        self.algebra_dim = algebra_dim
        self.hidden_dim = hidden_dim

        # Build the sparse bracket kernel
        if structure_constants is not None:
            self.bracket = SparseLieBracket(
                algebra_dim=algebra_dim,
                I=structure_constants['I'],
                J=structure_constants['J'],
                K=structure_constants['K'],
                C=structure_constants['C'],
            )
        else:
            self.bracket = SparseLieBracket(algebra_dim=algebra_dim)

        # Linear map for message: projects source features before bracket
        self.W_msg = nn.Linear(algebra_dim, algebra_dim, bias=False)

        # Update MLP: transforms aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(algebra_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, algebra_dim),
        )

        # Layer norm for stabilizing training
        self.layer_norm = nn.LayerNorm(algebra_dim)

    def forward(self, features, edge_index):
        """
        Forward pass.

        Args:
            features: (n_nodes, algebra_dim) node feature tensor
            edge_index: (2, n_edges) int64 tensor of [source, target] indices

        Returns:
            updated_features: (n_nodes, algebra_dim)
        """
        src_idx = edge_index[0]  # (n_edges,)
        tgt_idx = edge_index[1]  # (n_edges,)

        n_nodes = features.shape[0]

        # Get source and target features for each edge
        src_feats = features[src_idx]  # (n_edges, algebra_dim)
        tgt_feats = features[tgt_idx]  # (n_edges, algebra_dim)

        # Project source features
        src_projected = self.W_msg(src_feats)  # (n_edges, algebra_dim)

        # Compute bracket messages
        messages = self.bracket(src_projected, tgt_feats)  # (n_edges, algebra_dim)

        # Aggregate messages by target node (scatter_add)
        agg = features.new_zeros(n_nodes, self.algebra_dim)
        tgt_expanded = tgt_idx.unsqueeze(-1).expand_as(messages)
        agg.scatter_add_(0, tgt_expanded, messages)

        # Update with residual connection
        updated = features + self.update_mlp(agg)
        updated = self.layer_norm(updated)

        return updated


class ClebschGordanDecomposer(nn.Module):
    """
    Clebsch-Gordan-like decomposition using Lie algebra structure constants.

    For a Lie algebra, the tensor product of two adjoint representations
    decomposes as: adj x adj = 1 + adj + sym^2 + ...

    The structure constants provide the projection onto the adjoint (antisymmetric)
    component, which is the Lie bracket itself. The symmetric component is
    related to the Killing form.

    This module provides both projections as a differentiable decomposition.
    """

    def __init__(self, algebra_dim=248, structure_constants=None):
        super().__init__()
        self.algebra_dim = algebra_dim

        if structure_constants is not None:
            self.bracket = SparseLieBracket(
                algebra_dim=algebra_dim,
                I=structure_constants['I'],
                J=structure_constants['J'],
                K=structure_constants['K'],
                C=structure_constants['C'],
            )
        else:
            self.bracket = SparseLieBracket(algebra_dim=algebra_dim)

    def decompose(self, v1, v2):
        """
        Decompose the tensor product of two algebra elements into components.

        Returns:
            antisymmetric: [v1, v2] (adjoint component, the Lie bracket)
            symmetric: element-wise product v1 * v2 (approximation to
                       symmetric tensor component, used as a learnable feature)
            scalar: dot product v1 . v2 (trivial/singlet component)
        """
        # Antisymmetric (adjoint) component: the Lie bracket
        antisym = self.bracket(v1, v2)

        # Scalar (singlet) component: inner product
        scalar = torch.sum(v1 * v2, dim=-1, keepdim=True)

        # Symmetric component approximation: element-wise product
        # (In the full theory this would use the d-tensor, which vanishes for E8,
        #  so we use element-wise product as a learnable proxy)
        sym = v1 * v2

        return antisym, sym, scalar
