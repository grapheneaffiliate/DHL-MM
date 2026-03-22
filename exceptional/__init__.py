"""
Exceptional Lie Algebras — DHL-MM Sparse Structure Constant Engine

Generalizes the DHL-MM framework from E8 to all five exceptional Lie algebras:
G2 (14-dim), F4 (52-dim), E6 (78-dim), E7 (133-dim), E8 (248-dim).

Usage:
    from exceptional import ExceptionalAlgebra
    alg = ExceptionalAlgebra("G2")
    z = alg.bracket(x, y)
"""

from .engine import ExceptionalAlgebra

__all__ = ["ExceptionalAlgebra"]
