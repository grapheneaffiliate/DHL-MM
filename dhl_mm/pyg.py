"""
PyTorch Geometric integration for DHL-MM sparse Lie bracket convolution.

Requires torch_geometric (optional dependency).
Install: pip install torch-geometric
"""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    MessagePassing = nn.Module  # fallback base class


class LieBracketConv(MessagePassing if _HAS_PYG else nn.Module):
    """Equivariant message-passing layer using sparse Lie bracket.

    Node features are vectors in an exceptional Lie algebra.
    Messages are computed via the sparse bracket [src, tgt].
    Updates use bracket-based nonlinearity following Lie Neurons (Lin et al. 2024).

    Args:
        algebra_name: One of "G2", "F4", "E6", "E7", "E8"
        equivariant: If True, use Schur's-lemma-constrained linear maps

    Example:
        conv = LieBracketConv("E8")
        x = torch.randn(100, 248)  # 100 nodes with E8-valued features
        edge_index = ...  # (2, num_edges)
        out = conv(x, edge_index)  # (100, 248)
    """
    def __init__(self, algebra_name="E8", equivariant=True):
        if not _HAS_PYG:
            raise ImportError(
                "LieBracketConv requires torch_geometric. "
                "Install it with: pip install torch-geometric"
            )
        super().__init__(aggr='add')

        # Build sparse bracket from the exceptional algebra
        from equivariant.sparse_kernel import SparseLieBracket, SparseKillingForm

        self.algebra_name = algebra_name
        self.bracket = SparseLieBracket.from_algebra(algebra_name)
        self.algebra_dim = self.bracket.algebra_dim

        # Learnable parameters (Schur's lemma: scalar multiples)
        if equivariant:
            self.msg_scale = nn.Parameter(torch.tensor(1.0))
            self.update_scale = nn.Parameter(torch.tensor(0.1))
            self.skip_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.msg_proj = nn.Linear(self.algebra_dim, self.algebra_dim)
            self.update_mlp = nn.Sequential(
                nn.Linear(self.algebra_dim, self.algebra_dim),
                nn.GELU(),
                nn.Linear(self.algebra_dim, self.algebra_dim),
            )
        self.equivariant = equivariant

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: target node features, x_j: source node features
        if self.equivariant:
            return self.msg_scale * self.bracket(x_j, x_i)
        else:
            return self.bracket(self.msg_proj(x_j), x_i)

    def update(self, aggr_out, x):
        if self.equivariant:
            # Bracket-based nonlinearity: x + alpha * [agg, scale * agg]
            nonlin = self.bracket(aggr_out, self.update_scale * aggr_out)
            return self.skip_scale * x + aggr_out + nonlin
        else:
            return x + self.update_mlp(aggr_out)
