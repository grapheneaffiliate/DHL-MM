"""
ExceptionalEGNN: Equivariant Graph Neural Network using E8 Lie algebra structure.

Architecture:
    Input MLP (in_dim -> algebra_dim) ->
    N x [LieConvLayer + LayerNorm + Residual] ->
    Killing form pooling ->
    Output MLP (pooled_dim -> out_dim)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .sparse_kernel import SparseLieBracket, SparseKillingForm
from .layers import LieConvLayer


class ExceptionalEGNN(nn.Module):
    """
    Equivariant Graph Neural Network with E8 (or other exceptional) Lie algebra structure.

    Uses sparse structure constants for efficient equivariant message passing
    and the Killing form for invariant pooling.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4, algebra_dim=248,
                 structure_constants=None, killing_matrix=None, task='graph'):
        """
        Args:
            in_dim: input node feature dimension
            hidden_dim: MLP hidden dimension
            out_dim: output prediction dimension
            n_layers: number of LieConvLayer blocks
            algebra_dim: dimension of the Lie algebra (248 for E8)
            structure_constants: dict with 'I','J','K','C' or None to build from DHLMM
            killing_matrix: numpy array or torch tensor, or None to build from DHLMM
            task: 'graph' for graph-level prediction, 'node' for node-level
        """
        super().__init__()
        self.algebra_dim = algebra_dim
        self.task = task

        # Input projection: map arbitrary features to algebra space
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, algebra_dim),
        )

        # Stack of equivariant layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                LieConvLayer(
                    algebra_dim=algebra_dim,
                    hidden_dim=hidden_dim,
                    structure_constants=structure_constants,
                )
            )

        # Killing form for invariant features
        self.killing_form = SparseKillingForm(killing_matrix)

        # Output MLP
        if task == 'graph':
            # Graph-level: pool nodes then predict
            # Killing form self-inner-product gives a scalar per node,
            # then sum-pool over nodes, plus algebra_dim mean features
            self.output_mlp = nn.Sequential(
                nn.Linear(algebra_dim + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            # Node-level: predict per node
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
            # Compute invariant scalar per node via Killing form
            killing_scalar = self.killing_form(h, h)  # (n_nodes,)
            killing_scalar = killing_scalar.unsqueeze(-1)  # (n_nodes, 1)

            # Concatenate algebra features with Killing invariant
            node_feats = torch.cat([h, killing_scalar], dim=-1)  # (n_nodes, algebra_dim+1)

            # Pool over nodes in each graph
            if batch is None:
                # Single graph: mean pool
                pooled = node_feats.mean(dim=0, keepdim=True)  # (1, algebra_dim+1)
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
