"""
Quantum simulation demo for DHL-MM.

Demonstrates Lie-algebra-valued lattice dynamics (NOT Hilbert-space QM)
using sparse structure constants from DHL-MM.

Produces three plots:
  1. killing_norm_conservation.png — Killing norm vs time step
  2. correlation_spreading.png    — correlations between sites vs time
  3. algebra_comparison.png       — Killing norm drift: G2 vs E8
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from dhl_mm.quantum import E8SpinLattice

# ---------------------------------------------------------------------------
# 1. E8 lattice: 8 sites, 500 steps
# ---------------------------------------------------------------------------
print("=" * 60)
print("DHL-MM Quantum Simulation Demo")
print("Lie-algebra-valued lattice dynamics (adjoint flow)")
print("=" * 60)

n_sites = 8
dt = 0.001
steps = 500

print(f"\n[1] Creating E8 spin lattice with {n_sites} sites ...")
t0 = time.perf_counter()
lattice = E8SpinLattice(n_sites, algebra_name="E8", coupling=1.0)
t_load = time.perf_counter() - t0
print(f"    Algebra dim = {lattice.dim}, loaded in {t_load:.2f}s")

state0 = lattice.random_initial_state(scale=0.1, seed=42)

print(f"\n[2] Evolving {steps} steps with dt={dt}, order=2 ...")
t0 = time.perf_counter()
trajectory = lattice.evolve(state0, dt=dt, steps=steps, order=2)
t_evolve = time.perf_counter() - t0
print(f"    Evolution completed in {t_evolve:.2f}s")

# Compute Killing norms and correlations along trajectory
killing_norms = []
corr_01, corr_03, corr_07 = [], [], []
sample_every = max(1, steps // 200)  # sample ~200 points for plots

sample_indices = list(range(0, steps + 1, sample_every))
print(f"\n[3] Computing observables at {len(sample_indices)} sample points ...")
t0 = time.perf_counter()
for idx in sample_indices:
    s = trajectory[idx]
    killing_norms.append(lattice.measure_killing_norm(s))
    corr_01.append(lattice.measure_correlations(s, 0, 1))
    corr_03.append(lattice.measure_correlations(s, 0, 3))
    corr_07.append(lattice.measure_correlations(s, 0, 7))
t_obs = time.perf_counter() - t0
print(f"    Observables computed in {t_obs:.2f}s")

# ---------------------------------------------------------------------------
# 2. Algebra comparison: G2 vs E8 (4 sites, 200 steps)
# ---------------------------------------------------------------------------
print(f"\n[4] Algebra comparison: G2 vs E8 (4 sites, 200 steps) ...")
comp_sites = 4
comp_steps = 200
comp_dt = 0.001

results_comp = {}
for alg_name in ["G2", "E8"]:
    t0 = time.perf_counter()
    lat = E8SpinLattice(comp_sites, algebra_name=alg_name, coupling=1.0)
    s0 = lat.random_initial_state(scale=0.1, seed=99)
    traj = lat.evolve(s0, dt=comp_dt, steps=comp_steps, order=2)
    norms = [lat.measure_killing_norm(traj[i]) for i in range(comp_steps + 1)]
    elapsed = time.perf_counter() - t0
    results_comp[alg_name] = norms
    print(f"    {alg_name}: dim={lat.dim}, time={elapsed:.2f}s, "
          f"norm drift={abs(norms[-1] - norms[0]):.2e}")

# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = [i * dt for i in sample_indices]

    # Plot 1: Killing norm conservation
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_axis, killing_norms, "b-", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Killing Norm")
    ax.set_title(f"Killing Norm Conservation — E8 lattice ({n_sites} sites)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/killing_norm_conservation.png", dpi=150)
    plt.close(fig)
    print("\n    Saved examples/killing_norm_conservation.png")

    # Plot 2: Correlation spreading
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_axis, corr_01, label="K(x_0, x_1)", linewidth=1.2)
    ax.plot(time_axis, corr_03, label="K(x_0, x_3)", linewidth=1.2)
    ax.plot(time_axis, corr_07, label="K(x_0, x_7)", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Killing Form Correlation")
    ax.set_title(f"Correlation Spreading — E8 lattice ({n_sites} sites)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/correlation_spreading.png", dpi=150)
    plt.close(fig)
    print("    Saved examples/correlation_spreading.png")

    # Plot 3: Algebra comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    comp_time = [i * comp_dt for i in range(comp_steps + 1)]
    for alg_name, norms in results_comp.items():
        norm0 = norms[0]
        drift = [abs(n - norm0) for n in norms]
        ax.plot(comp_time, drift, label=alg_name, linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("|Killing Norm - Initial|")
    ax.set_title(f"Killing Norm Drift — G2 vs E8 ({comp_sites} sites)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/algebra_comparison.png", dpi=150)
    plt.close(fig)
    print("    Saved examples/algebra_comparison.png")

except ImportError:
    print("\n    matplotlib not installed — skipping plots.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print(f"  Algebra load:      {t_load:.2f}s")
print(f"  Evolution ({steps} steps): {t_evolve:.2f}s")
print(f"  Observable calc:   {t_obs:.2f}s")
print(f"  Total wall clock:  {t_load + t_evolve + t_obs:.2f}s")
print("=" * 60)
