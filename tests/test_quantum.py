"""
Tests for dhl_mm.quantum module.

All tests use Lie-algebra-valued dynamics (adjoint flow on the algebra),
NOT Hilbert-space quantum mechanics.

Runnable as:
    py tests/test_quantum.py
    py -m pytest tests/test_quantum.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dhl_mm.quantum import LieHamiltonian, EquivariantTrotterSuzuki, E8SpinLattice


class TestCommutator(unittest.TestCase):
    """Tests for the sparse commutator."""

    @classmethod
    def setUpClass(cls):
        cls.ham = LieHamiltonian("G2")
        cls.dim = cls.ham.alg.dim
        cls.rng = np.random.RandomState(123)

    def test_commutator_antisymmetry(self):
        """[H, rho] = -[rho, H] for random H, rho."""
        for _ in range(10):
            H = self.rng.randn(self.dim)
            rho = self.rng.randn(self.dim)
            c1 = self.ham.commutator(H, rho)
            c2 = self.ham.commutator(rho, H)
            np.testing.assert_allclose(c1, -c2, atol=1e-12,
                                       err_msg="Antisymmetry violated")

    def test_jacobi_identity(self):
        """Jacobi identity: [x,[y,z]] + [y,[z,x]] + [z,[x,y]] = 0."""
        bracket = self.ham.alg.bracket
        for _ in range(10):
            x = self.rng.randn(self.dim)
            y = self.rng.randn(self.dim)
            z = self.rng.randn(self.dim)
            t1 = bracket(x, bracket(y, z))
            t2 = bracket(y, bracket(z, x))
            t3 = bracket(z, bracket(x, y))
            np.testing.assert_allclose(t1 + t2 + t3, 0.0, atol=1e-10,
                                       err_msg="Jacobi identity violated")

    def test_conserved_check(self):
        """[H, H] = 0 for any H."""
        H = self.rng.randn(self.dim)
        val = self.ham.conserved_check(H, H)
        self.assertLess(val, 1e-12, "[H, H] should be zero")


class TestEvolution(unittest.TestCase):
    """Tests for time evolution and norm conservation."""

    @classmethod
    def setUpClass(cls):
        cls.ham = LieHamiltonian("G2")
        cls.dim = cls.ham.alg.dim

    def test_killing_norm_conservation(self):
        """Killing norm should drift < 1e-4 over 100 steps with small dt."""
        rng = np.random.RandomState(42)
        H = rng.randn(self.dim) * 0.1
        rho_0 = rng.randn(self.dim) * 0.1
        traj = self.ham.evolve(H, rho_0, dt=0.001, steps=100, order=2)
        norm_0 = self.ham.killing_norm(traj[0])
        norm_f = self.ham.killing_norm(traj[-1])
        drift = abs(norm_f - norm_0)
        self.assertLess(drift, 1e-4,
                        f"Killing norm drift {drift:.2e} exceeds 1e-4")

    def test_second_order_better(self):
        """Second-order Trotter should have less drift than first-order."""
        rng = np.random.RandomState(77)
        H = rng.randn(self.dim) * 0.1
        rho_0 = rng.randn(self.dim) * 0.1

        traj1 = self.ham.evolve(H, rho_0, dt=0.01, steps=100, order=1)
        traj2 = self.ham.evolve(H, rho_0, dt=0.01, steps=100, order=2)

        norm_0 = self.ham.killing_norm(rho_0)
        drift1 = abs(self.ham.killing_norm(traj1[-1]) - norm_0)
        drift2 = abs(self.ham.killing_norm(traj2[-1]) - norm_0)

        self.assertLess(drift2, drift1,
                        f"Order-2 drift {drift2:.2e} should be < order-1 drift {drift1:.2e}")


class TestTrotterSuzuki(unittest.TestCase):
    """Tests for the EquivariantTrotterSuzuki wrapper."""

    def test_evolve_returns_dict(self):
        ham = LieHamiltonian("G2")
        ts = EquivariantTrotterSuzuki(ham)
        rng = np.random.RandomState(55)
        H = rng.randn(ham.alg.dim) * 0.1
        rho_0 = rng.randn(ham.alg.dim) * 0.1
        result = ts.evolve(H, rho_0, t_final=0.01, dt=0.001, order=2)
        self.assertIn('states', result)
        self.assertIn('times', result)
        self.assertIn('killing_norms', result)
        self.assertEqual(len(result['states']), 11)  # 10 steps + initial
        self.assertEqual(len(result['times']), 11)


class TestLattice(unittest.TestCase):
    """Tests for the E8SpinLattice."""

    def test_lattice_basic(self):
        """Create lattice, evolve 10 steps, verify state shapes."""
        lat = E8SpinLattice(4, algebra_name="G2", coupling=1.0)
        state = lat.random_initial_state(scale=0.1, seed=10)
        self.assertEqual(state.shape, (4, lat.dim))

        traj = lat.evolve(state, dt=0.001, steps=10, order=2)
        self.assertEqual(len(traj), 11)  # 10 steps + initial
        for s in traj:
            self.assertEqual(s.shape, (4, lat.dim))

    def test_lattice_killing_norm(self):
        """Killing norm measurement returns a scalar."""
        lat = E8SpinLattice(3, algebra_name="G2", coupling=1.0)
        state = lat.random_initial_state(scale=0.1, seed=20)
        kn = lat.measure_killing_norm(state)
        self.assertIsInstance(kn, float)

    def test_lattice_correlations(self):
        """Correlation measurement returns a scalar."""
        lat = E8SpinLattice(3, algebra_name="G2", coupling=1.0)
        state = lat.random_initial_state(scale=0.1, seed=30)
        c = lat.measure_correlations(state, 0, 1)
        self.assertIsInstance(c, float)


if __name__ == "__main__":
    unittest.main()
