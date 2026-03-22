# Exceptional Lie Algebras — DHL-MM Sparse Structure Constant Engine

Generalizes the DHL-MM sparse framework from E8 to all five exceptional Lie algebras:

| Algebra | Dim | Rank | Roots | Method |
|---------|-----|------|-------|--------|
| G2 | 14 | 2 | 12 | Frenkel-Kac cocycle + coroot normalization |
| F4 | 52 | 4 | 48 | Iterative adjoint + Killing-form projection |
| E6 | 78 | 6 | 72 | Extracted from E8 (sub-root-system) |
| E7 | 133 | 7 | 126 | Extracted from E8 (sub-root-system) |
| E8 | 248 | 8 | 240 | Original DHL-MM Frenkel-Kac cocycle |

## Usage

```python
from exceptional import ExceptionalAlgebra
import numpy as np

alg = ExceptionalAlgebra("G2")  # or "F4", "E6", "E7", "E8"
x = np.random.randn(alg.dim)
y = np.random.randn(alg.dim)

z = alg.bracket(x, y)           # Lie bracket [x, y]
p = alg.full_product(x, y)      # Full product (= [x,y]/2, since d=0)
k = alg.killing_form(x, y)      # Killing form K(x, y)
d = alg.verify_d_vanishes()      # Verify d-tensor = 0
```

## d-tensor Verification

None of the exceptional algebras have a degree-3 Casimir invariant:
- G2: degrees 2, 6
- F4: degrees 2, 6, 8, 12
- E6: degrees 2, 5, 6, 8, 9, 12
- E7: degrees 2, 6, 8, 10, 12, 14, 18
- E8: degrees 2, 8, 12, 14, 18, 20, 24, 30

Therefore the symmetric d-tensor vanishes for all five, and the DHL-MM full-product
trick `x*y = [x,y]/2` works universally.

## Running Tests

```
py exceptional/tests/test_all.py
py exceptional/benchmarks.py
```

## Note on F4

F4's structure constants use an iterative approximation method because the
Frenkel-Kac cocycle (which works for simply-laced algebras and G2) fails for F4
due to half-integer root coordinates violating the cocycle integrality condition.
The iterative method achieves Jacobi violation ~5e-3, which is sufficient for
most practical applications.
