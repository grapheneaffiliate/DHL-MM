"""
Tests for dhl_mm.integrators module.

RKMK Lie group integrators: Euler, RK2, RK4 with BCH corrections.

Runnable as:
    py tests/test_integrators.py
    py -m pytest tests/test_integrators.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dhl_mm.integrators import RKMKIntegrator, LieGroupFlow, ConvergenceTest


class TestEuler(unittest.TestCase):
    """Tests for Lie-Euler integrator."""

    @classmethod
    def setUpClass(cls):
        cls.rk = RKMKIntegrator("G2")
        cls.dim = cls.rk.alg.dim
        cls.rng = np.random.RandomState(42)

    def test_euler_basic(self):
        """Euler step produces correct shape, nonzero output."""
        H = self.rng.randn(self.dim) * 0.3
        y0 = self.rng.randn(self.dim) * 0.3

        def f(t, y):
            return self.rk.alg.bracket(H, y)

        y1 = self.rk.euler(f, y0, 0.0, 0.01)
        self.assertEqual(y1.shape, (self.dim,))
        self.assertGreater(np.linalg.norm(y1 - y0), 0.0,
                           "Euler step should change the state")


class TestRK2(unittest.TestCase):
    """Tests for RKMK second-order integrator."""

    @classmethod
    def setUpClass(cls):
        cls.rk = RKMKIntegrator("G2")
        cls.dim = cls.rk.alg.dim

    def test_rk2_better_than_euler(self):
        """RK2 has less error than Euler for the same dt."""
        rng = np.random.RandomState(77)
        H = rng.randn(self.dim) * 0.3
        y0 = rng.randn(self.dim) * 0.3

        def f(t, y):
            return self.rk.alg.bracket(H, y)

        # Compute reference with very small steps (Euler, 10000 steps)
        dt_ref = 0.0001
        n_ref = 10000
        t_final = dt_ref * n_ref
        y_ref = y0.copy()
        for i in range(n_ref):
            y_ref = self.rk.euler(f, y_ref, i * dt_ref, dt_ref)

        # Euler with large dt
        dt = t_final / 100
        y_euler = y0.copy()
        for i in range(100):
            y_euler = self.rk.euler(f, y_euler, i * dt, dt)

        # RK2 with same large dt
        y_rk2 = y0.copy()
        for i in range(100):
            y_rk2 = self.rk.rk2(f, y_rk2, i * dt, dt)

        err_euler = np.linalg.norm(y_euler - y_ref)
        err_rk2 = np.linalg.norm(y_rk2 - y_ref)
        self.assertLess(err_rk2, err_euler,
                        f"RK2 error {err_rk2:.2e} should be < Euler error {err_euler:.2e}")


class TestRK4(unittest.TestCase):
    """Tests for RKMK fourth-order integrator."""

    @classmethod
    def setUpClass(cls):
        cls.rk = RKMKIntegrator("G2")
        cls.dim = cls.rk.alg.dim

    def test_rk4_better_than_rk2(self):
        """RK4 has less error than RK2 for the same dt on a nonlinear problem."""
        rng = np.random.RandomState(88)
        y0 = rng.randn(self.dim) * 0.1
        inertia = np.abs(rng.randn(self.dim)) + 1.0
        inv_inertia = 1.0 / inertia
        bracket = self.rk.alg.bracket

        def f(t, y):
            return bracket(y, inv_inertia * y)

        # Reference: integrate with very fine dt using plain RK4
        dt_fine = 0.0001
        n_fine = 10000
        y_ref = y0.copy()
        for i in range(n_fine):
            t = i * dt_fine
            k1 = dt_fine * f(t, y_ref)
            k2 = dt_fine * f(t + dt_fine/2, y_ref + k1/2)
            k3 = dt_fine * f(t + dt_fine/2, y_ref + k2/2)
            k4 = dt_fine * f(t + dt_fine, y_ref + k3)
            y_ref = y_ref + (k1 + 2*k2 + 2*k3 + k4) / 6

        # RK2 and RK4 with larger dt
        dt = 1.0 / 100
        y_rk2 = y0.copy()
        y_rk4 = y0.copy()
        for i in range(100):
            t = i * dt
            y_rk2 = self.rk.rk2(f, y_rk2, t, dt)
            y_rk4 = self.rk.rk4(f, y_rk4, t, dt)

        err_rk2 = np.linalg.norm(y_rk2 - y_ref)
        err_rk4 = np.linalg.norm(y_rk4 - y_ref)
        self.assertLess(err_rk4, err_rk2,
                        f"RK4 error {err_rk4:.2e} should be < RK2 error {err_rk2:.2e}")

    def test_rk4_order(self):
        """Verify 4th-order convergence: halving dt reduces error by ~16x."""
        ct = ConvergenceTest("G2")
        result = ct.test_order(method='rk4', n_steps_list=[100, 200, 400, 800])
        self.assertGreater(result['measured_order'], 3.5,
                           f"Measured order {result['measured_order']:.2f} should be > 3.5")


class TestAdjointFlow(unittest.TestCase):
    """Tests for adjoint flow conservation."""

    @classmethod
    def setUpClass(cls):
        cls.flow = LieGroupFlow("G2", integrator_method='rk4')
        cls.dim = cls.flow.alg.dim

    def test_adjoint_flow_conserves_killing(self):
        """Adjoint flow dy/dt = [H, y] preserves K(y, y)."""
        rng = np.random.RandomState(99)
        H = rng.randn(self.dim) * 0.1
        y0 = rng.randn(self.dim) * 0.1

        f = self.flow.adjoint_flow(y0, H)
        result = self.flow.integrator.solve(f, y0, (0.0, 0.1), dt=0.001, method='rk4')

        norm_0 = result['killing_norms'][0]
        norms = result['killing_norms']
        max_drift = max(abs(kn - norm_0) for kn in norms)
        self.assertLess(max_drift, 1e-6,
                        f"Killing norm drift {max_drift:.2e} exceeds 1e-6")


class TestRigidBody(unittest.TestCase):
    """Tests for rigid body dynamics."""

    @classmethod
    def setUpClass(cls):
        cls.flow = LieGroupFlow("G2", integrator_method='rk4')
        cls.dim = cls.flow.alg.dim

    def test_rigid_body_conserves_energy(self):
        """Rigid body preserves K(y, y) (Casimir invariant)."""
        rng = np.random.RandomState(111)
        y0 = rng.randn(self.dim) * 0.1
        inertia = np.abs(rng.randn(self.dim)) + 1.0  # positive inertia

        f = self.flow.rigid_body(y0, inertia)
        result = self.flow.integrator.solve(f, y0, (0.0, 0.1), dt=0.001, method='rk4')

        norm_0 = result['killing_norms'][0]
        norms = result['killing_norms']
        max_drift = max(abs(kn - norm_0) for kn in norms)
        self.assertLess(max_drift, 1e-6,
                        f"Casimir drift {max_drift:.2e} exceeds 1e-6")


class TestSolveInterface(unittest.TestCase):
    """Tests for the solve() interface."""

    def test_solve_interface(self):
        """solve() returns correct dict structure."""
        rk = RKMKIntegrator("G2")
        dim = rk.alg.dim
        rng = np.random.RandomState(55)
        H = rng.randn(dim) * 0.1
        y0 = rng.randn(dim) * 0.1

        def f(t, y):
            return rk.alg.bracket(H, y)

        result = rk.solve(f, y0, (0.0, 0.1), dt=0.01, method='rk4')
        self.assertIn('times', result)
        self.assertIn('states', result)
        self.assertIn('killing_norms', result)
        self.assertEqual(len(result['times']), 11)  # 10 steps + initial
        self.assertEqual(len(result['states']), 11)
        self.assertEqual(len(result['killing_norms']), 11)
        self.assertEqual(result['states'][0].shape, (dim,))
        self.assertAlmostEqual(result['times'][0], 0.0)
        self.assertAlmostEqual(result['times'][-1], 0.1, places=10)


if __name__ == "__main__":
    unittest.main()
