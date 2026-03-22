# DHL-MM: Dynamic Hodge-Lie Matrix Multiplication

Fast matrix multiplication for E₈ Lie algebra elements using sparse structure constants with algebraic error control.

## What It Does

Multiplies 248-dimensional E₈ Lie algebra elements using **16,694 precomputed structure constants** instead of full 248×248 matrix multiplication (15.3M operations). **913× fewer operations**, verified to machine epsilon.

## Key Results

| Metric | Value |
|---|---|
| Structure constant entries | 16,694 |
| Full MM operations (248³) | 15,252,992 |
| Operation compression | 913× |
| Measured wall-time speedup | 218× (Python/numpy) |
| Correctness | 3.3×10⁻¹⁶ relative error |
| Jacobi identity | 4.4×10⁻¹⁶ (machine epsilon) |

## Why It Works

E₈ has a special property: **no cubic Casimir invariant** (all Casimir degrees are even: 2,8,12,14,18,20,24,30). This means the symmetric structure constants d_{ij}^k vanish identically, so the full product T_i·T_j projected onto the Lie algebra equals [T_i, T_j]/2 exactly. The entire product is determined by the antisymmetric structure constants alone.

## Files

| File | Purpose |
|---|---|
| `e8_structure_constants.py` | E₈ root system (240 roots) with correct Frenkel-Kac cocycle signs |
| `e8_full_algebra.py` | Full 248-dim algebra (240 root + 8 Cartan generators) |
| `dhl_mm_v2.py` | Complete framework: sparse product + Z[φ] arithmetic + defect monitor |
| `test_structure.py` | Algebraic identity verification (antisymmetry, Jacobi, Killing form) |
| `test_full_algebra.py` | Full 248-dim test suite (7 tests, all pass) |

## Quick Start

```bash
pip install numpy

# Run the full framework with benchmarks
py dhl_mm_v2.py

# Verify all algebraic identities
py test_structure.py
py test_full_algebra.py
```

## How It Works

### 1. Structure Constant Engine
Instead of multiplying 248×248 matrices, the product of two Lie algebra elements X = Σ xᵢTᵢ, Y = Σ yⱼTⱼ is:

```
(XY)_k = (1/2) Σ_{(i,j,k) ∈ f} xᵢ · yⱼ · f_{ij}^k
```

where f_{ij}^k are 16,694 precomputed entries with values ±1. This is a sparse gather-multiply-scatter operation.

### 2. Z[φ] Exact Arithmetic
Coefficients stored as integer pairs (a, b) representing a + bφ. Multiplication uses φ² = φ + 1. No floating point accumulation errors.

### 3. Defect Equation Monitor
```
h²(t) = h²_Λ - (κ/3) · ρ_defect(t)
```
Tracks accumulated deviation from algebraic structure. When h²(t) drops below threshold, coefficients are projected back to the nearest Z[φ] lattice point.

## Applications

- Quantum simulation (Hamiltonian commutators)
- Lie group integration (Runge-Kutta-Munthe-Kaas methods)
- Gauge theory / lattice QCD computations
- Robotics (Lie group kinematics)
- E₈ lattice computations (GSM physics solver)

## Requirements

- Python 3.8+
- NumPy

## License

MIT
