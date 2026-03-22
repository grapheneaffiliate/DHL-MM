"""
ExceptionalEGNN: Equivariant Graph Neural Network using exceptional Lie algebra structure.

Architecture:
    Input MLP (in_dim -> algebra_dim) ->
    N x [LieConvLayer or EquivariantLieConvLayer] ->
    Killing form pooling (+ optional bracket pooling) ->
    Output MLP (pooled_dim -> out_dim)

Supports all five exceptional algebras: G2, F4, E6, E7, E8.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .sparse_kernel import SparseLieBracket, SparseKillingForm
from .layers import LieConvLayer, EquivariantLieConvLayer


class ExceptionalEGNN(nn.Module):
    """
    Equivariant Graph Neural Network with exceptional Lie algebra structure.

    Uses sparse structure constants for efficient equivariant message passing
    and the Killing form for invariant pooling.

    Args:
        in_dim: input node feature dimension
        hidden_dim: MLP hidden dimension
        out_dim: output prediction dimension
        n_layers: number of LieConv blocks
        algebra_dim: dimension of the Lie algebra (auto-set if algebra_name given)
        algebra_name: name of exceptional algebra ("G2","F4","E6","E7","E8"), default "E8"
        equivariant: if True, use EquivariantLieConvLayer; else use LieConvLayer
        bracket_pooling: if True, add ||[h_i, h_j]||^2 summed over neighbors as feature
        structure_constants: dict with 'I','J','K','C' or None
        killing_matrix: numpy/torch tensor or None
        task: 'graph' for graph-level, 'node' for node-level prediction
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4, algebra_dim=None,
                 algebra_name="E8", equivariant=False, bracket_pooling=False,
                 structure_constants=None, killing_matrix=None, task='graph'):
        super().__init__()
        self.algebra_name = algebra_name
        self.equivariant_mode = equivariant
        self.bracket_pooling = bracket_pooling
        self.task = task

        # Determine algebra_dim from algebra_name if not explicitly given
        if algebra_dim is not None:
            self.algebra_dim = algebra_dim
        else:
            _dim_map = {"G2": 14, "F4": 52, "E6": 78, "E7": 133, "E8": 248}
            self.algebra_dim = _dim_map.get(algebra_name, 248)

        algebra_dim = self.algebra_dim

        # Input projection: map arbitrary features to algebra space
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, algebra_dim),
        )

        # Stack of layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if equivariant:
                self.layers.append(
                    EquivariantLieConvLayer(algebra_name=algebra_name)
                )
            else:
                self.layers.append(
                    LieConvLayer(
                        algebra_dim=algebra_dim,
                        hidden_dim=hidden_dim,
                        structure_constants=structure_constants,
                        algebra_name=algebra_name if structure_constants is None else None,
                    )
                )

        # Killing form for invariant features
        if killing_matrix is not None:
            self.killing_form = SparseKillingForm(killing_matrix)
        else:
            self.killing_form = SparseKillingForm.from_algebra(algebra_name)

        # Bracket for bracket_pooling
        if bracket_pooling:
            self.pool_bracket = SparseLieBracket.from_algebra(algebra_name)

        # Output MLP dimension depends on pooling options
        pool_dim = 1  # Killing form self-inner-product
        if bracket_pooling:
            pool_dim += 1  # bracket norm squared
        # For graph task: pool invariant scalars
        # For node task: predict directly from algebra features
        if task == 'graph':
            self.output_mlp = nn.Sequential(
                nn.Linear(algebra_dim + pool_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.output_mlp = nn.Sequential(
                nn.Linear(algebra_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x: (n_nodes, in_dim) input node features
            edge_index: (2, n_edges) edge indices
            batch: (n_nodes,) graph membership indices for batched graphs.
                   If None and task=='graph', assumes single graph.

        Returns:
            If task=='graph': (n_graphs, out_dim)
            If task=='node': (n_nodes, out_dim)
        """
        # Project to algebra space
        h = self.input_mlp(x)  # (n_nodes, algebra_dim)

        # Apply equivariant layers
        for layer in self.layers:
            h = layer(h, edge_index)

        if self.task == 'graph':
            # Compute invariant scalar per node via Killing form: K(h, h)
            killing_scalar = self.killing_form(h, h)  # (n_nodes,)
            killing_scalar = killing_scalar.unsqueeze(-1)  # (n_nodes, 1)

            invariant_feats = [h, killing_scalar]

            # Optional bracket pooling: ||[h_i, h_j]||^2 summed over neighbors
            if self.bracket_pooling:
                src_idx = edge_index[0]
                tgt_idx = edge_index[1]
                n_nodes = h.shape[0]

                src_feats = h[src_idx]
                tgt_feats = h[tgt_idx]
                bracket_ij = self.pool_bracket(src_feats, tgt_feats)  # (n_edges, algebra_dim)
                bracket_norm_sq = (bracket_ij ** 2).sum(dim=-1)  # (n_edges,)

                # Scatter-add bracket norms to target nodes
                bracket_pool = h.new_zeros(n_nodes)
                bracket_pool.scatter_add_(0, tgt_idx, bracket_norm_sq)
                invariant_feats.append(bracket_pool.unsqueeze(-1))  # (n_nodes, 1)

            # Concatenate all invariant features
            node_feats = torch.cat(invariant_feats, dim=-1)

            # Pool over nodes in each graph
            if batch is None:
                # Single graph: mean pool
                pooled = node_feats.mean(dim=0, keepdim=True)
            else:
                # Batched graphs: scatter mean
                n_graphs = batch.max().item() + 1
                pooled = x.new_zeros(n_graphs, node_feats.shape[-1])
                counts = x.new_zeros(n_graphs, 1)
                pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(node_feats), node_feats)
                counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch.unsqueeze(-1), dtype=x.dtype))
                pooled = pooled / counts.clamp(min=1)

            return self.output_mlp(pooled)
        else:
            # Node-level prediction
            return self.output_mlp(h)
