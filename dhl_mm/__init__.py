"""
DHL-MM: Dynamic Hodge-Lie Matrix Multiplication
================================================

Fast matrix multiplication for exceptional Lie algebra elements using sparse
structure constants. Up to 913x fewer operations than full matrix
multiplication, verified to machine epsilon.

Supports all five exceptional algebras: G2, F4, E6, E7, E8.

Quick start:
    import dhl_mm

    # Load any exceptional algebra (cached, <100ms)
    e8 = dhl_mm.algebra('E8')
    z = e8.bracket(x, y)

    # Or use the E8-specific engine directly
    from dhl_mm import DHLMM
    engine = DHLMM.build()
    z = engine.bracket(x, y)
"""

__version__ = "0.1.3"

import os as _os
import numpy as _np

from .engine import DHLMM
from .zphi import ZPhi, quantize, spectral_decompose, PHI
from .defect import DefectMonitor
from .e8 import DIM, build_roots, simple_roots, cartan_matrix
from .quantum import LieHamiltonian, EquivariantTrotterSuzuki, E8SpinLattice
from .lattice import GaugeLattice

_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "data")


def _load_from_cache(name: str):
    """Load an ExceptionalAlgebra from precomputed .npz cache.

    Args:
        name: Algebra name (G2, F4, E6, E7, E8)

    Returns:
        ExceptionalAlgebra instance if cache exists, None otherwise.
    """
    from .exceptional_engine import ExceptionalAlgebra

    cache_path = _os.path.join(_DATA_DIR, f"{name}_constants.npz")
    if not _os.path.exists(cache_path):
        return None

    data = _np.load(cache_path)
    data_dict = {
        'fI': data['fI'],
        'fJ': data['fJ'],
        'fK': data['fK'],
        'fC': data['fC'],
        'killing': data['killing'],
        'dim': data['dim'],
        'rank': data['rank'],
        'n_roots': data['n_roots'],
    }
    return ExceptionalAlgebra.from_cache(name, data_dict)


def algebra(name: str):
    """Load an exceptional Lie algebra engine with precomputed constants.

    Returns an object with .bracket(x, y), .killing_form(x, y), .full_product(x, y),
    .dim, .rank, .n_roots, .n_structure_constants, .algebra_name

    Tries loading from precomputed .npz cache first for fast startup (<100ms).
    Falls back to live computation if the .npz file is missing.

    Args:
        name: One of "G2", "F4", "E6", "E7", "E8"

    Returns:
        ExceptionalAlgebra instance
    """
    # Normalize name
    name = name.upper()

    # Try cache first
    cached = _load_from_cache(name)
    if cached is not None:
        return cached

    # Fall back to live computation
    from .exceptional_engine import ExceptionalAlgebra
    return ExceptionalAlgebra(name)


__all__ = [
    "algebra",
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
    "LieHamiltonian",
    "EquivariantTrotterSuzuki",
    "E8SpinLattice",
    "GaugeLattice",
]
