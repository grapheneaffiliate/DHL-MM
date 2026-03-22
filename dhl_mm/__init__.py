"""
DHL-MM: Dynamic Hodge-Lie Matrix Multiplication
================================================

Fast matrix multiplication for E8 Lie algebra elements using sparse
structure constants. 913x fewer operations than full 248x248 matrix
multiplication, verified to machine epsilon.

Quick start:
    from dhl_mm import DHLMM

    engine = DHLMM.build()
    z = engine.bracket(x, y)
"""

__version__ = "1.0.0"

from .engine import DHLMM
from .zphi import ZPhi, quantize, spectral_decompose, PHI
from .defect import DefectMonitor
from .e8 import DIM, build_roots, simple_roots, cartan_matrix

__all__ = [
    "DHLMM",
    "ZPhi",
    "quantize",
    "spectral_decompose",
    "DefectMonitor",
    "PHI",
    "DIM",
    "build_roots",
    "simple_roots",
    "cartan_matrix",
]
