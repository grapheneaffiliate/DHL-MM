# DHL-MM: Dynamic Hodge-Lie Matrix Multiplication

Fast Lie algebra multiplication for all five exceptional algebras (G₂, F₄, E₆, E₇, E₈) using sparse structure constants with algebraic error control. Includes a differentiable PyTorch kernel for equivariant neural networks.

## What It Does

Replaces dense matrix multiplication with a **sparse gather-multiply-scatter** operation over precomputed structure constants. For E₈ this means 16,694 operations instead of 15.3 million — **913× fewer**. The same principle applies to all exceptional Lie algebras, verified to machine epsilon across all five.

## Compression Table

| Algebra | Dim | Roots | Nonzero f | Full n³ | Compression | Jacobi Error |
|---------|-----|-------|-----------|---------|-------------|-------------|
| G₂ | 14 | 12 | 120 | 2,744 | 22.9× | ~1e-14 |
| F₄ | 52 | 48 | 1,196 | 140,608 | 117.6× | ~1e-16 |
| E₆ | 78 | 72 | 2,208 | 474,552 | 215× | ~1e-13 |
| E₇ | 133 | 126 | 5,544 | 2,352,637 | 424× | ~1e-13 |
| E₈ | 248 | 240 | 16,694 | 15,252,992 | 913× | ~1e-16 |

All algebras verified to machine epsilon. No approximations.

## Why It Works

None of the exceptional Lie algebras have a cubic Casimir invariant:

| Algebra | Casimir Degrees |
|---------|----------------|
| G₂ | 2, 6 |
| F₄ | 2, 6, 8, 12 |
| E₆ | 2, 5, 6, 8, 9, 12 |
| E₇ | 2, 6, 8, 10, 12, 14, 18 |
| E₈ | 2, 8, 12, 14, 18, 20, 24, 30 |

No degree 3 means the symmetric structure constants d_{ij}^k vanish identically for **all five**. The full product T_i·T_j projected onto the Lie algebra equals [T_i, T_j]/2 exactly. The entire product is determined by the antisymmetric structure constants alone.

## Project Structure

```
dhl_mm/            Core E₈ library (pip-installable)
  engine.py          DHLMM class: bracket, full_product, Z[phi] arithmetic, defect monitor
  e8.py              E₈ root system + Frenkel-Kac structure constants
  zphi.py            Exact Z[phi] = {a + b*phi : a,b in Z} arithmetic
  defect.py          Friedmann-style h²(t) error tracker

exceptional/       All 5 exceptional algebras (G₂, F₄, E₆, E₇, E₈)
  engine.py          ExceptionalAlgebra(name) unified class
  roots.py           Root system builders
  structure.py       Structure constant computation (Chevalley basis)
  casimir.py         Casimir degree analysis + d-tensor verification
  benchmarks.py      Compression ratio table

equivariant/       PyTorch equivariant neural network layers
  sparse_kernel.py   Differentiable autograd wrapper (SparseLieBracket)
  layers.py          LieConvLayer, ClebschGordanDecomposer
  model.py           ExceptionalEGNN architecture
  benchmark.py       Synthetic benchmark
```

## Quick Start

### E₈ (core library)

```python
from dhl_mm import DHLMM
import numpy as np

engine = DHLMM.build()
x, y = np.random.randn(248), np.random.randn(248)

z = engine.bracket(x, y)          # [x, y] — 913x fewer ops than matmul
p = engine.full_product(x, y)     # x*y projected to Lie algebra (= [x,y]/2)
```

### Any exceptional algebra

```python
from exceptional import ExceptionalAlgebra

alg = ExceptionalAlgebra("F4")    # or "G2", "E6", "E7", "E8"
x, y = np.random.randn(alg.dim), np.random.randn(alg.dim)

z = alg.bracket(x, y)
k = alg.killing_form(x, y)
d = alg.verify_d_vanishes()       # confirms d-tensor = 0
```

### PyTorch (differentiable)

```python
from equivariant import SparseLieBracket, ExceptionalEGNN
import torch

bracket = SparseLieBracket()      # E₈ structure constants as buffers
x = torch.randn(248, requires_grad=True)
y = torch.randn(248, requires_grad=True)
z = bracket(x, y)                 # full autograd support
z.sum().backward()                # gradients flow through sparse scatter-add

model = ExceptionalEGNN(in_dim=248, hidden_dim=64, out_dim=1, n_layers=4)
```

### Benchmarks and tests

```bash
pip install numpy

py dhl_mm_v2.py                    # E₈ framework with benchmarks
py test_structure.py               # E₈ algebraic identity verification
py test_full_algebra.py            # E₈ full 248-dim test suite
py exceptional/benchmarks.py       # All 5 algebras compression table
py exceptional/tests/test_all.py   # All 5 algebras test suite
py equivariant/tests/test_equivariance.py  # PyTorch gradient + consistency tests
```

## How It Works

### 1. Sparse Structure Constant Engine
Instead of multiplying n×n matrices, the product of two Lie algebra elements X = Σ xᵢTᵢ, Y = Σ yⱼTⱼ is:

```
(XY)_k = (1/2) Σ_{(i,j,k) ∈ f} xᵢ · yⱼ · f_{ij}^k
```

where f_{ij}^k are precomputed sparse entries. This is a gather-multiply-scatter operation.

### 2. Z[φ] Exact Arithmetic
Coefficients stored as integer pairs (a, b) representing a + bφ where φ is the golden ratio. Multiplication uses φ² = φ + 1. No floating-point accumulation errors for algebraic inputs.

### 3. Defect Equation Monitor
```
h²(t) = h²_Λ - (κ/3) · ρ_defect(t)
```
Tracks accumulated deviation from the Z[φ] lattice. When h²(t) drops below threshold, coefficients are projected back to the nearest lattice point. Self-correcting computation.

### 4. Differentiable PyTorch Kernel
The sparse bracket is wrapped in a custom `torch.autograd.Function` with correct backward pass (the adjoint of scatter-add is gather). Structure constants are registered as module buffers — they move to GPU with the model but are not trainable.

## Applications

- Quantum simulation (Hamiltonian commutators, Trotter-Suzuki decomposition)
- Lattice gauge theory (exceptional gauge group computations)
- Lie group integration (Runge-Kutta-Munthe-Kaas methods)
- Equivariant neural networks (molecular property prediction, physics-informed ML)
- Post-quantum cryptography (E₈ lattice operations)
- Geometric algebra / Clifford algebra acceleration
- E₈ lattice computations (GSM physics solver)

See [APPLICATIONS_ROADMAP.md](APPLICATIONS_ROADMAP.md) for detailed directions.

## Requirements

- Python 3.8+
- NumPy
- PyTorch (for equivariant/ only)

## License

MIT
