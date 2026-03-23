"""
RKMK Lie group integrator demo.

1. Rigid body on E8: RK4 integration, Killing norm conservation
2. Convergence test on G2: verify 4th-order convergence
3. Compare Euler vs RK2 vs RK4 accuracy
4. Save convergence plot
5. Print timing
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dhl_mm.integrators import RKMKIntegrator, LieGroupFlow, ConvergenceTest


def demo_rigid_body():
    """Demo 1: Rigid body on E8 with Killing norm conservation."""
    print("=" * 60)
    print("Demo 1: Rigid body dynamics on E8")
    print("=" * 60)

    flow = LieGroupFlow("E8", integrator_method='rk4')
    dim = flow.alg.dim
    rng = np.random.RandomState(42)

    y0 = rng.randn(dim) * 0.1
    inertia = np.abs(rng.randn(dim)) + 1.0

    f = flow.rigid_body(y0, inertia)

    t0 = time.perf_counter()
    result = flow.integrator.solve(f, y0, (0.0, 1.0), dt=0.001, method='rk4')
    elapsed = time.perf_counter() - t0

    norms = result['killing_norms']
    norm_0 = norms[0]
    max_drift = max(abs(kn - norm_0) for kn in norms)
    rel_drift = max_drift / abs(norm_0) if abs(norm_0) > 0 else max_drift

    print(f"  Algebra dim:       {dim}")
    print(f"  Steps:             {len(norms) - 1}")
    print(f"  Initial K(y,y):    {norm_0:.6e}")
    print(f"  Final K(y,y):      {norms[-1]:.6e}")
    print(f"  Max absolute drift: {max_drift:.6e}")
    print(f"  Max relative drift: {rel_drift:.6e}")
    print(f"  Time:              {elapsed:.3f}s")
    print()


def demo_convergence():
    """Demo 2: Convergence test on G2 showing 4th-order convergence."""
    print("=" * 60)
    print("Demo 2: Convergence test on G2")
    print("=" * 60)

    ct = ConvergenceTest("G2")
    result = ct.test_order(method='rk4', n_steps_list=[50, 100, 200, 400, 800])

    print(f"  Method: RK4")
    print(f"  {'Steps':>8s}  {'dt':>12s}  {'Error':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    for dt, err in zip(result['step_sizes'], result['errors']):
        n = int(round(1.0 / dt))
        print(f"  {n:8d}  {dt:12.6e}  {err:12.6e}")
    print(f"  Measured order: {result['measured_order']:.2f} (expected ~4.0)")
    print()
    return result


def demo_comparison():
    """Demo 3: Compare Euler vs RK2 vs RK4 accuracy."""
    print("=" * 60)
    print("Demo 3: Euler vs RK2 vs RK4 comparison")
    print("=" * 60)

    ct = ConvergenceTest("G2")
    n_steps_list = [50, 100, 200, 400, 800]

    results = {}
    for method in ['euler', 'rk2', 'rk4']:
        t0 = time.perf_counter()
        res = ct.test_order(method=method, n_steps_list=n_steps_list)
        elapsed = time.perf_counter() - t0
        results[method] = res
        print(f"  {method:>6s}: order={res['measured_order']:.2f}, "
              f"finest error={res['errors'][-1]:.2e}, time={elapsed:.3f}s")

    print()
    return results


def save_plot(results):
    """Demo 4: Save convergence plot."""
    print("=" * 60)
    print("Demo 4: Saving convergence plot")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        markers = {'euler': 'o', 'rk2': 's', 'rk4': '^'}
        colors = {'euler': '#e74c3c', 'rk2': '#3498db', 'rk4': '#2ecc71'}
        expected_orders = {'euler': 1, 'rk2': 2, 'rk4': 4}

        for method, res in results.items():
            dts = np.array(res['step_sizes'])
            errs = np.array(res['errors'])
            order = res['measured_order']
            label = f"{method.upper()} (measured order={order:.2f})"
            ax.loglog(dts, errs, marker=markers[method], color=colors[method],
                      linewidth=2, markersize=8, label=label)

        # Reference slopes
        dt_ref = np.array([results['euler']['step_sizes'][0],
                           results['euler']['step_sizes'][-1]])
        for order, ls in [(1, ':'), (2, '--'), (4, '-.')]:
            scale = results['euler']['errors'][0] / (dt_ref[0] ** order)
            ax.loglog(dt_ref, scale * dt_ref ** order, color='gray',
                      linestyle=ls, alpha=0.5, label=f'slope {order}')

        ax.set_xlabel('Step size (dt)', fontsize=12)
        ax.set_ylabel('Error (L2 norm)', fontsize=12)
        ax.set_title('RKMK Integrator Convergence (G2 adjoint flow)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(os.path.dirname(__file__), "integrator_convergence.png")
        fig.savefig(out_path, dpi=150)
        print(f"  Plot saved to: {out_path}")
        plt.close(fig)
    except ImportError:
        print("  matplotlib not available -- skipping plot")

    print()


def demo_timing():
    """Demo 5: Timing breakdown."""
    print("=" * 60)
    print("Demo 5: Timing")
    print("=" * 60)

    for alg_name in ['G2', 'E8']:
        rk = RKMKIntegrator(alg_name)
        dim = rk.alg.dim
        rng = np.random.RandomState(42)
        H = rng.randn(dim) * 0.1
        y0 = rng.randn(dim) * 0.1

        def f(t, y, _H=H, _rk=rk):
            return _rk.alg.bracket(_H, y)

        for method in ['euler', 'rk2', 'rk4']:
            t0 = time.perf_counter()
            rk.solve(f, y0, (0.0, 1.0), dt=0.01, method=method)
            elapsed = time.perf_counter() - t0
            print(f"  {alg_name} {method:>6s}: 100 steps in {elapsed:.4f}s "
                  f"({elapsed/100*1e6:.1f} us/step)")

    print()


if __name__ == "__main__":
    demo_rigid_body()
    rk4_result = demo_convergence()
    all_results = demo_comparison()
    save_plot(all_results)
    demo_timing()
    print("All demos completed successfully.")
