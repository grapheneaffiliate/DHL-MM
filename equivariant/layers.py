"""
Equivariant neural network layers built on sparse Lie bracket kernel.

LieConvLayer: Unconstrained message-passing layer (baseline).
AdjointLinearLayer: Schur's lemma-constrained linear map (scalar * x).
AdjointBilinearLayer: Adjoint-invariant bilinear (bracket + killing).
EquivariantLieConvLayer: Properly equivariant message-passing layer.
ClebschGordanDecomposer: Tensor product decomposition via structure constants.
"""

import torch
import torch.nn as nn

from .sparse_kernel import SparseLieBracket, SparseKillingForm


class AdjointLinearLayer(nn.Module):
    """
    Adjoint-equivariant linear layer: alpha * x.

    By Schur's lemma, the only linear map V -> V commuting with
    the adjoint action on a simple Lie algebra irrep is a scalar multiple
    of the identity.

    Args:
        algebra_name: Name of the exceptional algebra (default "E8").
    """

    def __init__(self, algebra_name: str = "E8"):
        super().__init__()
        self.algebra_name = algebra_name
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x: (..., algebra_dim) tensor

        Returns:
            alpha * x: (..., algebra_dim) tensor
        """
        return self.alpha * x


class AdjointBilinearLayer(nn.Module):
    """
    Adjoint-invariant bilinear layer: adj x adj -> adj.

    For simple Lie algebras with vanishing d-tensor (all exceptional algebras),
    the ONLY adjoint-equivariant bilinear map adj x adj -> adj is the Lie bracket.
    (The Killing form maps to the trivial rep, not the adjoint.)

    Computes: alpha * [x, y]

    Also exposes the Killing form K(x, y) as a scalar invariant for downstream use.

    Args:
        algebra_name: Name of the exceptional algebra (default "E8").
    """

    def __init__(self, algebra_name: str = "E8"):
        super().__init__()
        self.algebra_name = algebra_name
        self.bracket = SparseLieBracket.from_algebra(algebra_name)
        self.killing_form = SparseKillingForm.from_algebra(algebra_name)
        self.algebra_dim = self.bracket.algebra_dim
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, y):
        """
        Args:
            x: (..., algebra_dim) tensor
            y: (..., algebra_dim) tensor

        Returns:
            alpha * [x, y]: (..., algebra_dim) tensor (adjoint-equivariant)
        """
        bracket_xy = self.bracket(x, y)  # (..., algebra_dim)
        return self.alpha * bracket_xy

    def killing_scalar(self, x, y):
        """Compute Killing form K(x, y) as an invariant scalar."""
        return self.killing_form(x, y)


class EquivariantLieConvLayer(nn.Module):
    """
    Properly adjoint-equivariant message-passing layer.

    Uses Schur's lemma constraints throughout:
      - Message projection: AdjointLinearLayer (scalar * source features)
      - Message computation: AdjointBilinearLayer (bracket + killing)
      - Update: bracket-based nonlinearity x -> x + alpha * [x, W(x)]

    This follows the Lie Neurons pattern (Lin et al. 2024), ensuring
    exact equivariance under the adjoint action of the Lie group.

    Args:
        algebra_name: Name of the exceptional algebra (default "E8").
    """

    def __init__(self, algebra_name: str = "E8"):
        super().__init__()
        self.algebra_name = algebra_name

        # Build bracket for this algebra
        self.bracket = SparseLieBracket.from_algebra(algebra_name)
        self.algebra_dim = self.bracket.algebra_dim

        # Equivariant message projection (Schur: scalar * identity)
        self.msg_projection = AdjointLinearLayer(algebra_name)

        # Equivariant message computation (bracket + killing)
        self.msg_bilinear = AdjointBilinearLayer(algebra_name)

        # Update nonlinearity: x -> x + update_scale * [x, W(x)]
        self.update_W = AdjointLinearLayer(algebra_name)
        self.update_scale = nn.Parameter(torch.tensor(0.1))

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

        # Project source features (equivariant: scalar multiplication)
        src_projected = self.msg_projection(src_feats)  # (n_edges, algebra_dim)

        # Compute equivariant messages via bracket + killing
        messages = self.msg_bilinear(src_projected, tgt_feats)  # (n_edges, algebra_dim)

        # Aggregate messages by target node (scatter_add)
        agg = features.new_zeros(n_nodes, self.algebra_dim)
        tgt_expanded = tgt_idx.unsqueeze(-1).expand_as(messages)
        agg.scatter_add_(0, tgt_expanded, messages)

        # Equivariant update: x + scale * [x, W(x)]
        Wx = self.update_W(agg)  # (n_nodes, algebra_dim)
        bracket_update = self.bracket(agg, Wx)  # (n_nodes, algebra_dim)
        updated = features + agg + self.update_scale * bracket_update

        return updated


class LieConvLayer(nn.Module):
    """
    Unconstrained message-passing layer using sparse Lie bracket (baseline).

    For each edge (i, j):
        message_ij = bracket(W_msg @ features[i], features[j])
    Aggregate:
        agg_i = sum_j message_ij
    Update:
        features_i = features_i + MLP(agg_i)

    This is NOT strictly equivariant (W_msg breaks equivariance) but
    serves as an expressive baseline.
    """

    def __init__(self, algebra_dim=248, hidden_dim=64, structure_constants=None,
                 algebra_name=None):
        """
        Args:
            algebra_dim: dimension of the Lie algebra (default 248 for E8)
            hidden_dim: hidden dimension of the update MLP
            structure_constants: dict with keys 'I', 'J', 'K', 'C' or None
            algebra_name: if provided, build bracket from this algebra
        """
        super().__init__()
        self.algebra_dim = algebra_dim
        self.hidden_dim = hidden_dim

        # Build the sparse bracket kernel
        if algebra_name is not None:
            self.bracket = SparseLieBracket.from_algebra(algebra_name)
            self.algebra_dim = self.bracket.algebra_dim
            algebra_dim = self.algebra_dim
        elif structure_constants is not None:
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
