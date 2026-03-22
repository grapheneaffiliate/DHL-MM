"""
Tests for the equivariant DHL-MM PyTorch library.

Tests cover:
- Gradient correctness (autograd.gradcheck)
- Antisymmetry of the Lie bracket
- Consistency between PyTorch sparse kernel and NumPy reference
- Batched vs single-sample consistency
- Killing form symmetry
- Training convergence on synthetic data

Runnable both as pytest and standalone script.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from dhl_mm import DHLMM
from dhl_mm.e8 import DIM, lie_bracket

from equivariant.sparse_kernel import SparseLieBracket, SparseLieBracketFn, SparseKillingForm
from equivariant.layers import LieConvLayer, ClebschGordanDecomposer
from equivariant.model import ExceptionalEGNN


# Shared engine and bracket (built once)
_engine = None
_bracket = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = DHLMM.build()
    return _engine


def _get_bracket():
    global _bracket
    if _bracket is None:
        engine = _get_engine()
        I = torch.from_numpy(engine.fI.astype(np.int64))
        J = torch.from_numpy(engine.fJ.astype(np.int64))
        K = torch.from_numpy(engine.fK.astype(np.int64))
        C = torch.from_numpy(engine.fC.astype(np.float64))
        _bracket = SparseLieBracket(algebra_dim=DIM, I=I, J=J, K=K, C=C)
    return _bracket


def test_gradient_check():
    """torch.autograd.gradcheck on SparseLieBracket with float64."""
    bracket = _get_bracket()

    # Small random inputs in float64 for gradcheck
    x = torch.randn(DIM, dtype=torch.float64, requires_grad=True)
    y = torch.randn(DIM, dtype=torch.float64, requires_grad=True)

    I = bracket.I
    J = bracket.J
    K = bracket.K
    C = bracket.C.double()

    def fn(x, y):
        return SparseLieBracketFn.apply(x, y, I, J, K, C, DIM)

    passed = torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "Gradient check failed!"
    print("  [PASS] test_gradient_check")


def test_antisymmetry():
    """bracket(x, y) = -bracket(y, x)"""
    bracket = _get_bracket()
    rng = np.random.RandomState(123)

    for trial in range(10):
        x_np = rng.randn(DIM)
        y_np = rng.randn(DIM)
        x = torch.tensor(x_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)

        z_xy = bracket(x, y)
        z_yx = bracket(y, x)

        diff = torch.max(torch.abs(z_xy + z_yx)).item()
        assert diff < 1e-10, f"Antisymmetry violated: max diff = {diff}"

    print("  [PASS] test_antisymmetry (10 random pairs)")


def test_consistency_with_numpy():
    """Sparse PyTorch kernel matches numpy dhl_mm.bracket on 100 random pairs."""
    bracket = _get_bracket()
    engine = _get_engine()
    rng = np.random.RandomState(456)

    max_diff = 0.0
    for trial in range(100):
        x_np = rng.randn(DIM)
        y_np = rng.randn(DIM)

        # NumPy reference
        z_np = lie_bracket(x_np, y_np, engine.fI, engine.fJ, engine.fK, engine.fC)

        # PyTorch sparse
        x_t = torch.tensor(x_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)
        z_t = bracket(x_t, y_t).detach().numpy()

        diff = np.max(np.abs(z_np - z_t))
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-10, f"NumPy consistency failed: max diff = {max_diff}"
    print(f"  [PASS] test_consistency_with_numpy (100 pairs, max diff = {max_diff:.2e})")


def test_batch_consistency():
    """Batched output matches loop over individual pairs."""
    bracket = _get_bracket()
    rng = np.random.RandomState(789)
    batch_size = 8

    x_np = rng.randn(batch_size, DIM)
    y_np = rng.randn(batch_size, DIM)

    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    # Batched
    z_batch = bracket(x, y).detach().numpy()

    # Individual
    max_diff = 0.0
    for i in range(batch_size):
        xi = torch.tensor(x_np[i], dtype=torch.float64)
        yi = torch.tensor(y_np[i], dtype=torch.float64)
        zi = bracket(xi, yi).detach().numpy()
        diff = np.max(np.abs(z_batch[i] - zi))
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-10, f"Batch consistency failed: max diff = {max_diff}"
    print(f"  [PASS] test_batch_consistency (batch_size={batch_size}, max diff = {max_diff:.2e})")


def test_killing_form_symmetry():
    """K(x,y) = K(y,x)"""
    engine = _get_engine()
    killing_form = SparseKillingForm(engine.killing)
    rng = np.random.RandomState(101)

    for trial in range(10):
        x_np = rng.randn(DIM)
        y_np = rng.randn(DIM)
        x = torch.tensor(x_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)

        kxy = killing_form(x, y).item()
        kyx = killing_form(y, x).item()

        diff = abs(kxy - kyx)
        assert diff < 1e-8, f"Killing form symmetry violated: diff = {diff}"

    print("  [PASS] test_killing_form_symmetry (10 random pairs)")


def test_killing_form_consistency():
    """PyTorch Killing form matches NumPy engine.killing_form."""
    engine = _get_engine()
    killing_form = SparseKillingForm(engine.killing)
    rng = np.random.RandomState(202)

    max_diff = 0.0
    for trial in range(20):
        x_np = rng.randn(DIM)
        y_np = rng.randn(DIM)

        k_np = engine.killing_form(x_np, y_np)

        x = torch.tensor(x_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)
        k_t = killing_form(x, y).item()

        diff = abs(k_np - k_t)
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-6, f"Killing form consistency failed: max diff = {max_diff}"
    print(f"  [PASS] test_killing_form_consistency (20 pairs, max diff = {max_diff:.2e})")


def test_cg_decomposer():
    """ClebschGordanDecomposer returns correct shapes and antisymmetric component."""
    bracket = _get_bracket()
    decomposer = ClebschGordanDecomposer(algebra_dim=DIM)
    # Share the bracket kernel to avoid rebuilding
    decomposer.bracket = bracket

    rng = np.random.RandomState(303)
    x_np = rng.randn(DIM)
    y_np = rng.randn(DIM)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    antisym, sym, scalar = decomposer.decompose(x, y)

    assert antisym.shape == (DIM,), f"Wrong antisym shape: {antisym.shape}"
    assert sym.shape == (DIM,), f"Wrong sym shape: {sym.shape}"
    assert scalar.shape == (1,), f"Wrong scalar shape: {scalar.shape}"

    # antisym should equal bracket(x, y)
    z_bracket = bracket(x, y)
    diff = torch.max(torch.abs(antisym - z_bracket)).item()
    assert diff < 1e-10, f"CG antisym != bracket: diff = {diff}"

    print("  [PASS] test_cg_decomposer")


def test_lie_conv_layer_shapes():
    """LieConvLayer produces correct output shapes."""
    engine = _get_engine()
    I = torch.from_numpy(engine.fI.astype(np.int64))
    J = torch.from_numpy(engine.fJ.astype(np.int64))
    K = torch.from_numpy(engine.fK.astype(np.int64))
    C = torch.from_numpy(engine.fC.astype(np.float64))

    sc = {'I': I, 'J': J, 'K': K, 'C': C}
    layer = LieConvLayer(algebra_dim=DIM, hidden_dim=32, structure_constants=sc)
    layer = layer.float()

    n_nodes = 10
    features = torch.randn(n_nodes, DIM, dtype=torch.float32)
    # Simple chain graph
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    tgt = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    out = layer(features, edge_index)
    assert out.shape == (n_nodes, DIM), f"Wrong output shape: {out.shape}"

    print("  [PASS] test_lie_conv_layer_shapes")


def test_model_forward():
    """ExceptionalEGNN forward pass produces correct output shapes."""
    engine = _get_engine()

    model = ExceptionalEGNN(
        in_dim=DIM,
        hidden_dim=32,
        out_dim=1,
        n_layers=2,
        algebra_dim=DIM,
        task='graph',
    )

    n_nodes = 8
    x = torch.randn(n_nodes, DIM, dtype=torch.float32)
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    tgt = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    out = model(x, edge_index)
    assert out.shape == (1, 1), f"Wrong output shape: {out.shape}"

    # Check that gradients flow
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed through the model!"

    print("  [PASS] test_model_forward")


def test_training_convergence():
    """ExceptionalEGNN loss decreases on synthetic data over 50 steps."""
    engine = _get_engine()

    model = ExceptionalEGNN(
        in_dim=DIM,
        hidden_dim=32,
        out_dim=1,
        n_layers=2,
        algebra_dim=DIM,
        task='graph',
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Generate a small fixed dataset
    rng = np.random.RandomState(999)
    data = []
    for _ in range(10):
        n = rng.randint(5, 12)
        nodes = torch.randn(n, DIM, dtype=torch.float32) * 0.1
        src = list(range(n - 1)) + list(range(1, n))
        tgt = list(range(1, n)) + list(range(n - 1))
        ei = torch.tensor([src, tgt], dtype=torch.long)
        target = torch.tensor([rng.randn()], dtype=torch.float32)
        data.append((nodes, ei, target))

    losses = []
    for step in range(50):
        nodes, ei, target = data[step % len(data)]
        optimizer.zero_grad()
        pred = model(nodes, ei)
        loss = criterion(pred.squeeze(), target.squeeze())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check that loss decreased: compare average of first 5 vs last 5
    early_avg = np.mean(losses[:5])
    late_avg = np.mean(losses[-5:])
    assert late_avg < early_avg, (
        f"Training did not converge: early avg = {early_avg:.6f}, late avg = {late_avg:.6f}"
    )

    print(f"  [PASS] test_training_convergence (loss: {early_avg:.4f} -> {late_avg:.4f})")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Gradient check (autograd.gradcheck)", test_gradient_check),
        ("Antisymmetry [x,y] = -[y,x]", test_antisymmetry),
        ("Consistency with NumPy", test_consistency_with_numpy),
        ("Batch consistency", test_batch_consistency),
        ("Killing form symmetry", test_killing_form_symmetry),
        ("Killing form consistency", test_killing_form_consistency),
        ("CG decomposer", test_cg_decomposer),
        ("LieConvLayer shapes", test_lie_conv_layer_shapes),
        ("Model forward pass", test_model_forward),
        ("Training convergence", test_training_convergence),
    ]

    print("=" * 60)
    print("Equivariant DHL-MM Test Suite")
    print("=" * 60)
    print(f"Building E8 engine (dim={DIM})...")
    _get_engine()
    print("Engine built.\n")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"Running: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
