#!/usr/bin/env python
"""Precompute structure constants for all five exceptional Lie algebras.

Saves .npz files to dhl_mm/data/ containing:
- fI, fJ, fK, fC: sparse structure constant arrays
- killing: Killing form matrix
- dim, rank, n_roots: algebra metadata
"""

import os
import sys
import time
import numpy as np

# Ensure the repo root is on the path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from dhl_mm.exceptional_engine import ExceptionalAlgebra

ALGEBRAS = ["G2", "F4", "E6", "E7", "E8"]

DATA_DIR = os.path.join(repo_root, "dhl_mm", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def precompute_one(name: str) -> None:
    """Precompute and save structure constants for one algebra."""
    print(f"  Computing {name}...", end=" ", flush=True)
    t0 = time.time()

    alg = ExceptionalAlgebra(name)

    out_path = os.path.join(DATA_DIR, f"{name}_constants.npz")
    np.savez_compressed(
        out_path,
        fI=alg.fI,
        fJ=alg.fJ,
        fK=alg.fK,
        fC=alg.fC,
        killing=alg.killing,
        dim=np.array(alg.dim),
        rank=np.array(alg.rank),
        n_roots=np.array(alg.n_roots),
    )

    elapsed = time.time() - t0
    file_size = os.path.getsize(out_path)
    print(
        f"dim={alg.dim}, rank={alg.rank}, roots={alg.n_roots}, "
        f"f={alg.n_structure_constants}, "
        f"file={file_size / 1024:.1f}KB, time={elapsed:.1f}s"
    )


def main():
    print("Precomputing structure constants for exceptional Lie algebras")
    print(f"Output directory: {DATA_DIR}")
    print()

    t_total = time.time()
    for name in ALGEBRAS:
        precompute_one(name)

    print(f"\nAll done in {time.time() - t_total:.1f}s")

    # Verify cache loading
    print("\nVerifying cache loading...")
    for name in ALGEBRAS:
        path = os.path.join(DATA_DIR, f"{name}_constants.npz")
        data = np.load(path)
        cached = ExceptionalAlgebra.from_cache(name, dict(data))
        x = np.random.randn(cached.dim)
        y = np.random.randn(cached.dim)
        z = cached.bracket(x, y)
        print(f"  {name}: dim={cached.dim}, ||[x,y]||={np.linalg.norm(z):.4f} OK")


if __name__ == "__main__":
    main()
