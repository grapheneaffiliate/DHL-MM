# Equivariant DHL-MM

PyTorch library that uses DHL-MM's sparse structure constant engine as the fast inner kernel for equivariant neural network layers.

## Architecture

```
sparse_kernel.py    - Differentiable autograd wrapper (SparseLieBracket, SparseKillingForm)
layers.py           - LieConvLayer, ClebschGordanDecomposer
model.py            - ExceptionalEGNN architecture
benchmark.py        - Synthetic benchmark (no external dependencies beyond torch)
tests/              - Comprehensive test suite
```

## Quick Start

```python
from equivariant import SparseLieBracket, ExceptionalEGNN
import torch

# Low-level: differentiable Lie bracket
bracket = SparseLieBracket()  # builds E8 structure constants
x = torch.randn(248, requires_grad=True)
y = torch.randn(248, requires_grad=True)
z = bracket(x, y)  # [x, y] with full autograd support
z.sum().backward()  # gradients flow through sparse scatter-add

# High-level: equivariant GNN
model = ExceptionalEGNN(in_dim=248, hidden_dim=64, out_dim=1, n_layers=4)
nodes = torch.randn(10, 248)
edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
prediction = model(nodes, edge_index)
```

## Running Tests

```bash
py equivariant/tests/test_equivariance.py
```

## Running Benchmark

```bash
py equivariant/benchmark.py
```

## Key Design Decisions

- Structure constants (I, J, K, C) are registered as `nn.Module` buffers: they move to GPU with the model but are not trainable parameters
- The backward pass of `SparseLieBracketFn` implements the adjoint of scatter-add (which is gather), ensuring correct gradients
- Default algebra is E8 (dim=248, ~16,694 nonzero structure constants) but any algebra can be used by passing custom I, J, K, C tensors
- No dependency on torch_geometric; edge_index is a plain (2, n_edges) tensor
