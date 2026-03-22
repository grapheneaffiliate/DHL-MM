"""
Benchmark for the equivariant DHL-MM PyTorch library.

Benchmarks all 5 exceptional algebras:
1. Sparse vs dense bracket forward+backward wall time
2. EquivariantLieConvLayer forward+backward on synthetic graph

No external dependencies beyond numpy, torch, and exceptional/.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from equivariant.sparse_kernel import SparseLieBracket, SparseKillingForm
from equivariant.layers import EquivariantLieConvLayer
from equivariant.model import ExceptionalEGNN

ALL_ALGEBRAS = ["G2", "F4", "E6", "E7", "E8"]
DIM_MAP = {"G2": 14, "F4": 52, "E6": 78, "E7": 133, "E8": 248}


def benchmark_bracket_sparse_vs_dense(n_iters=50, batch_size=32):
    """Compare sparse bracket forward+backward vs dense for all algebras."""
    from exceptional.engine import ExceptionalAlgebra

    results = {}

    for name in ALL_ALGEBRAS:
        dim = DIM_MAP[name]
        alg = ExceptionalAlgebra(name)

        # Build sparse bracket
        bracket = SparseLieBracket.from_algebra(name)

        # Build dense structure constant tensor
        dense_f = np.zeros((dim, dim, dim), dtype=np.float64)
        for idx in range(len(alg.fI)):
            i, j, k = int(alg.fI[idx]), int(alg.fJ[idx]), int(alg.fK[idx])
            c = alg.fC[idx]
            dense_f[i, j, k] += c
        dense_f_torch = torch.tensor(dense_f, dtype=torch.float32)

        # Random inputs with grad
        x = torch.randn(batch_size, dim, dtype=torch.float32, requires_grad=True)
        y = torch.randn(batch_size, dim, dtype=torch.float32, requires_grad=True)

        # Warm up
        for _ in range(3):
            z = bracket(x, y)
            z.sum().backward()
            x.grad = None
            y.grad = None

        # Time sparse forward+backward
        t0 = time.perf_counter()
        for _ in range(n_iters):
            z = bracket(x, y)
            z.sum().backward()
            x.grad = None
            y.grad = None
        t_sparse = (time.perf_counter() - t0) / n_iters * 1000  # ms

        # Time dense forward+backward
        x2 = x.detach().requires_grad_(True)
        y2 = y.detach().requires_grad_(True)
        for _ in range(3):
            z = torch.einsum('bi,bj,ijk->bk', x2, y2, dense_f_torch)
            z.sum().backward()
            x2.grad = None
            y2.grad = None

        t0 = time.perf_counter()
        for _ in range(n_iters):
            z = torch.einsum('bi,bj,ijk->bk', x2, y2, dense_f_torch)
            z.sum().backward()
            x2.grad = None
            y2.grad = None
        t_dense = (time.perf_counter() - t0) / n_iters * 1000  # ms

        n_entries = len(alg.fI)
        sparse_mb = n_entries * 4 * 8 / 1024 / 1024  # I,J,K (int64) + C (float64)
        dense_mb = dim ** 3 * 4 / 1024 / 1024  # float32

        results[name] = {
            'dim': dim,
            'n_entries': n_entries,
            't_sparse_ms': t_sparse,
            't_dense_ms': t_dense,
            'speedup': t_dense / t_sparse if t_sparse > 0 else float('inf'),
            'sparse_mb': sparse_mb,
            'dense_mb': dense_mb,
        }

    return results


def benchmark_equivariant_layer(n_iters=20):
    """Benchmark EquivariantLieConvLayer forward+backward on synthetic graph."""
    n_nodes = 20
    n_edges_per_dir = 30  # 60 directed edges total

    # Build graph
    rng = np.random.RandomState(42)
    src_list = []
    tgt_list = []
    # Chain
    for i in range(n_nodes - 1):
        src_list.extend([i, i + 1])
        tgt_list.extend([i + 1, i])
    # Extra random edges to reach ~60
    while len(src_list) < n_edges_per_dir * 2:
        s = rng.randint(n_nodes)
        t = rng.randint(n_nodes)
        if s != t:
            src_list.extend([s, t])
            tgt_list.extend([t, s])
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

    results = {}
    for name in ALL_ALGEBRAS:
        dim = DIM_MAP[name]
        layer = EquivariantLieConvLayer(algebra_name=name)
        layer.train()

        features = torch.randn(n_nodes, dim, dtype=torch.float32, requires_grad=True)

        # Warm up
        for _ in range(2):
            out = layer(features, edge_index)
            out.sum().backward()
            features.grad = None

        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = layer(features, edge_index)
            out.sum().backward()
            features.grad = None
        elapsed = (time.perf_counter() - t0) / n_iters * 1000  # ms

        results[name] = {
            'dim': dim,
            'ms_per_forward': elapsed,
            'n_nodes': n_nodes,
            'n_edges': edge_index.shape[1],
        }

    return results


def main():
    print("=" * 80)
    print("DHL-MM Equivariant Neural Network Benchmark — All 5 Exceptional Algebras")
    print("=" * 80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # 1. Sparse vs Dense bracket
    print("-" * 80)
    print("1. Sparse vs Dense Bracket (forward + backward, batch=32)")
    print("-" * 80)

    bracket_results = benchmark_bracket_sparse_vs_dense(n_iters=50, batch_size=32)

    header = f"{'Algebra':<8s} {'Dim':>5s} {'Entries':>8s} {'Sparse ms':>10s} {'Dense ms':>10s} {'Speedup':>8s} {'Sparse MB':>10s} {'Dense MB':>10s}"
    print(header)
    print("-" * len(header))
    for name in ALL_ALGEBRAS:
        r = bracket_results[name]
        print(f"{name:<8s} {r['dim']:>5d} {r['n_entries']:>8d} {r['t_sparse_ms']:>10.2f} {r['t_dense_ms']:>10.2f} {r['speedup']:>7.1f}x {r['sparse_mb']:>10.3f} {r['dense_mb']:>10.3f}")

    print()

    # 2. EquivariantLieConvLayer benchmark
    print("-" * 80)
    print("2. EquivariantLieConvLayer (forward + backward, 20 nodes, ~60 edges)")
    print("-" * 80)

    layer_results = benchmark_equivariant_layer(n_iters=20)

    header2 = f"{'Algebra':<8s} {'Dim':>5s} {'Nodes':>6s} {'Edges':>6s} {'ms/forward':>12s}"
    print(header2)
    print("-" * len(header2))
    for name in ALL_ALGEBRAS:
        r = layer_results[name]
        print(f"{name:<8s} {r['dim']:>5d} {r['n_nodes']:>6d} {r['n_edges']:>6d} {r['ms_per_forward']:>12.2f}")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
