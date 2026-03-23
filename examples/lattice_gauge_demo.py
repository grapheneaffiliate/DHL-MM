"""
Lattice gauge theory demo for exceptional Lie groups.

Demonstrates:
1. E8 lattice gauge theory on an 8x8 lattice
2. Metropolis thermalization with observable monitoring
3. Wilson loop measurements
4. E8 vs G2 comparison on a 4x4 lattice
5. Timing and performance summary

Run:
    py examples/lattice_gauge_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dhl_mm.lattice import GaugeLattice


def main():
    wall_start = time.time()

    # ------------------------------------------------------------------
    # 1. E8 lattice: 8x8, hot start, 100 Metropolis sweeps
    # ------------------------------------------------------------------
    print("=" * 60)
    print("E8 Lattice Gauge Theory  --  8x8 lattice")
    print("=" * 60)

    t0 = time.time()
    lat_e8 = GaugeLattice((8, 8), "E8")
    t_load = time.time() - t0
    print(f"Algebra loaded in {t_load:.2f}s")
    print(f"  dim = {lat_e8.alg.dim}, "
          f"n_structure_constants = {lat_e8.alg.n_structure_constants}")
    print(f"  Sites: {lat_e8.n_sites}, Links: {lat_e8.n_links}, "
          f"Plaquettes: {lat_e8.n_plaquettes}")

    lat_e8.hot_start(scale=0.5, seed=42)

    print("\nThermalization (100 sweeps, beta=1.0, step_size=0.05):")
    t0 = time.time()
    result_e8 = lat_e8.thermalize(n_sweeps=100, beta=1.0, step_size=0.05, seed=99)
    t_therm = time.time() - t0
    print(f"  Completed in {t_therm:.1f}s")

    for i in range(0, 100, 10):
        print(f"  Sweep {i+1:3d}: action={result_e8['actions'][i]:12.2f}  "
              f"acc_rate={result_e8['acceptance_rates'][i]:.3f}  "
              f"avg_plaq={result_e8['average_plaquettes'][i]:.4f}")

    # ------------------------------------------------------------------
    # 2. Wilson loop measurements after thermalization
    # ------------------------------------------------------------------
    print("\nWilson loop measurements (Killing norm, averaged):")
    t0 = time.time()
    loops = lat_e8.measure_wilson_loops(max_size=3)
    t_loops = time.time() - t0
    for (m, n), val in sorted(loops.items()):
        print(f"  {m}x{n}: {val:.4f}")
    print(f"  Measured in {t_loops:.2f}s")

    # ------------------------------------------------------------------
    # 3. E8 vs G2 comparison on 4x4 lattice
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("E8 vs G2 Comparison  --  4x4 lattice, 50 sweeps")
    print("=" * 60)

    # E8
    t0 = time.time()
    lat_e8_small = GaugeLattice((4, 4), "E8")
    lat_e8_small.hot_start(scale=0.5, seed=10)
    res_e8 = lat_e8_small.thermalize(n_sweeps=50, beta=1.0, step_size=0.05, seed=20)
    t_e8 = time.time() - t0

    # G2
    t0 = time.time()
    lat_g2 = GaugeLattice((4, 4), "G2")
    lat_g2.hot_start(scale=0.5, seed=10)
    res_g2 = lat_g2.thermalize(n_sweeps=50, beta=1.0, step_size=0.05, seed=20)
    t_g2 = time.time() - t0

    print(f"E8 (dim={lat_e8_small.alg.dim}): final action = {res_e8['actions'][-1]:.2f}, "
          f"time = {t_e8:.2f}s")
    print(f"G2 (dim={lat_g2.alg.dim}): final action = {res_g2['actions'][-1]:.2f}, "
          f"time = {t_g2:.2f}s")
    print(f"E8/G2 action ratio: {res_e8['actions'][-1] / res_g2['actions'][-1]:.2f}")

    # ------------------------------------------------------------------
    # 4. Plot thermalization curves
    # ------------------------------------------------------------------
    print("\nSaving thermalization plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        sweeps_e8 = list(range(1, len(res_e8['actions']) + 1))
        sweeps_g2 = list(range(1, len(res_g2['actions']) + 1))

        ax.plot(sweeps_e8, res_e8['actions'], 'b-', linewidth=1.5,
                label=f"E8 (dim={lat_e8_small.alg.dim})")
        ax.plot(sweeps_g2, res_g2['actions'], 'r-', linewidth=1.5,
                label=f"G2 (dim={lat_g2.alg.dim})")
        ax.set_xlabel("Sweep", fontsize=12)
        ax.set_ylabel("Wilson action", fontsize=12)
        ax.set_title("Lattice gauge theory thermalization: E8 vs G2 (4x4)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(os.path.dirname(__file__), "lattice_thermalization.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Saved to {plot_path}")
    except ImportError:
        print("  matplotlib not available -- skipping plot")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    wall_total = time.time() - wall_start
    print("\n" + "=" * 60)
    print(f"Total wall-clock time: {wall_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
