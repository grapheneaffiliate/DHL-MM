"""
Synthetic benchmark for the equivariant DHL-MM PyTorch library.

Compares sparse DHL-MM kernel vs dense matrix multiply for Lie bracket,
and trains ExceptionalEGNN on a synthetic graph regression task.

No external dependencies beyond numpy and torch.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from dhl_mm import DHLMM
from dhl_mm.e8 import DIM

from equivariant.sparse_kernel import SparseLieBracket, SparseKillingForm
from equivariant.model import ExceptionalEGNN


class SyntheticGraphDataset:
    """
    Random graphs with algebra-valued node features.

    Target: Killing form invariant of sum of pairwise brackets.
    This ensures the target respects the algebra symmetry.
    """

    def __init__(self, n_graphs=500, n_nodes_range=(5, 20), algebra_dim=DIM, seed=42):
        self.n_graphs = n_graphs
        self.n_nodes_range = n_nodes_range
        self.algebra_dim = algebra_dim

        rng = np.random.RandomState(seed)

        # Build engine for computing targets
        engine = DHLMM.build()

        self.graphs = []
        for _ in range(n_graphs):
            n_nodes = rng.randint(n_nodes_range[0], n_nodes_range[1] + 1)

            # Random algebra-valued node features (sparse for realism)
            node_features = rng.randn(n_nodes, algebra_dim) * 0.1

            # Random edges (connected graph with some extra edges)
            edges_src = []
            edges_tgt = []
            # Chain to ensure connectivity
            for i in range(n_nodes - 1):
                edges_src.extend([i, i + 1])
                edges_tgt.extend([i + 1, i])
            # Extra random edges
            n_extra = rng.randint(0, n_nodes)
            for _ in range(n_extra):
                s = rng.randint(n_nodes)
                t = rng.randint(n_nodes)
                if s != t:
                    edges_src.extend([s, t])
                    edges_tgt.extend([t, s])

            edge_index = np.array([edges_src, edges_tgt], dtype=np.int64)

            # Compute target: Killing form of sum of pairwise brackets along edges
            bracket_sum = np.zeros(algebra_dim)
            for e in range(edge_index.shape[1]):
                s, t = edge_index[0, e], edge_index[1, e]
                bracket_sum += engine.bracket(node_features[s], node_features[t])

            target = engine.killing_form(bracket_sum, bracket_sum)
            # Normalize target to reasonable range
            target = np.float64(target) / (n_nodes ** 2)

            self.graphs.append({
                'nodes': torch.tensor(node_features, dtype=torch.float32),
                'edge_index': torch.tensor(edge_index, dtype=torch.long),
                'target': torch.tensor([target], dtype=torch.float32),
                'n_nodes': n_nodes,
            })

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g['nodes'], g['edge_index'], g['target']


def benchmark_forward_pass(n_iters=100, batch_size=32, algebra_dim=DIM):
    """Compare forward pass time: sparse DHL-MM kernel vs dense matmul."""
    print("Building DHL-MM engine...")
    engine = DHLMM.build()

    # Build dense structure constant tensor for comparison
    dense_f = np.zeros((algebra_dim, algebra_dim, algebra_dim), dtype=np.float64)
    for idx in range(len(engine.fI)):
        i, j, k = int(engine.fI[idx]), int(engine.fJ[idx]), int(engine.fK[idx])
        c = engine.fC[idx]
        dense_f[i, j, k] += c

    dense_f_torch = torch.tensor(dense_f, dtype=torch.float32)

    # Sparse kernel
    bracket = SparseLieBracket(algebra_dim=algebra_dim)
    bracket.eval()

    # Random inputs
    x = torch.randn(batch_size, algebra_dim, dtype=torch.float32)
    y = torch.randn(batch_size, algebra_dim, dtype=torch.float32)

    # Warm up
    for _ in range(5):
        _ = bracket(x, y)
        _ = torch.einsum('bi,bj,ijk->bk', x, y, dense_f_torch)

    # Time sparse
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_iters):
        z_sparse = bracket(x, y)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_sparse = (time.perf_counter() - t0) / n_iters * 1000  # ms

    # Time dense
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_iters):
        z_dense = torch.einsum('bi,bj,ijk->bk', x, y, dense_f_torch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_dense = (time.perf_counter() - t0) / n_iters * 1000  # ms

    # Memory comparison
    import tracemalloc
    tracemalloc.start()
    _ = bracket(x, y)
    _, mem_sparse = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = torch.einsum('bi,bj,ijk->bk', x, y, dense_f_torch)
    _, mem_dense = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_sparse_mb = mem_sparse / 1024 / 1024
    mem_dense_mb = mem_dense / 1024 / 1024

    # Also report the size of stored constants
    sparse_const_bytes = (bracket.I.nelement() * 8 + bracket.J.nelement() * 8 +
                          bracket.K.nelement() * 8 + bracket.C.nelement() * 8)
    dense_const_bytes = dense_f_torch.nelement() * 4
    sparse_const_mb = sparse_const_bytes / 1024 / 1024
    dense_const_mb = dense_const_bytes / 1024 / 1024

    return {
        't_sparse': t_sparse,
        't_dense': t_dense,
        'speedup': t_dense / t_sparse if t_sparse > 0 else float('inf'),
        'mem_sparse_peak': mem_sparse_mb,
        'mem_dense_peak': mem_dense_mb,
        'sparse_const_mb': sparse_const_mb,
        'dense_const_mb': dense_const_mb,
    }


def benchmark_training(n_steps=100, n_graphs=200, hidden_dim=32, n_layers=2):
    """Train ExceptionalEGNN on synthetic data and report loss curve."""
    print("\nGenerating synthetic dataset...")
    dataset = SyntheticGraphDataset(n_graphs=n_graphs, n_nodes_range=(5, 12),
                                    algebra_dim=DIM)

    print("Building model...")
    model = ExceptionalEGNN(
        in_dim=DIM,
        hidden_dim=hidden_dim,
        out_dim=1,
        n_layers=n_layers,
        algebra_dim=DIM,
        task='graph',
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []
    print(f"Training for {n_steps} steps...")
    for step in range(n_steps):
        idx = step % len(dataset)
        nodes, edge_index, target = dataset[idx]

        optimizer.zero_grad()
        pred = model(nodes, edge_index)  # (1, 1)
        loss = criterion(pred.squeeze(), target.squeeze())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: loss = {loss.item():.6f}")

    return losses


def main():
    print("=" * 70)
    print("DHL-MM Equivariant Neural Network Benchmark")
    print("=" * 70)
    print(f"Algebra: E8 (dim={DIM})")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # Forward pass benchmark
    print("-" * 70)
    print("1. Forward Pass Benchmark (sparse vs dense)")
    print("-" * 70)
    results = benchmark_forward_pass(n_iters=50, batch_size=32)

    print()
    print(f"{'Metric':<30s} {'Sparse (DHL-MM)':>18s} {'Dense (matmul)':>18s}")
    print("-" * 70)
    print(f"{'Forward time (ms/batch)':<30s} {results['t_sparse']:>17.3f}  {results['t_dense']:>17.3f}")
    print(f"{'Constant storage (MB)':<30s} {results['sparse_const_mb']:>17.3f}  {results['dense_const_mb']:>17.3f}")
    print(f"{'Peak memory (MB)':<30s} {results['mem_sparse_peak']:>17.3f}  {results['mem_dense_peak']:>17.3f}")
    print(f"{'Speedup':<30s} {results['speedup']:>17.1f}x")

    # Training benchmark
    print()
    print("-" * 70)
    print("2. Training Benchmark (ExceptionalEGNN on synthetic data)")
    print("-" * 70)
    losses = benchmark_training(n_steps=100, n_graphs=200, hidden_dim=32, n_layers=2)

    print()
    print(f"{'Metric':<30s} {'Value':>18s}")
    print("-" * 50)
    print(f"{'Training loss (step 0)':<30s} {losses[0]:>18.6f}")
    print(f"{'Training loss (step 100)':<30s} {losses[-1]:>18.6f}")
    print(f"{'Loss reduction':<30s} {(1 - losses[-1]/losses[0])*100:>17.1f}%")
    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
