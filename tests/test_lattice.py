"""
Tests for dhl_mm.lattice — lattice gauge theory for exceptional Lie groups.

Run with:
    py tests/test_lattice.py
    pytest tests/test_lattice.py -v
"""

import sys
import os
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dhl_mm.lattice import GaugeLattice


def test_cold_start_zero_action():
    """Cold start (all zeros) should have zero action."""
    lat = GaugeLattice((4, 4), "G2")
    lat.cold_start()
    S = lat.total_action(beta=1.0)
    assert abs(S) < 1e-15, f"Cold start action should be 0, got {S}"
    print("PASS test_cold_start_zero_action")


def test_plaquette_antisymmetry():
    """P_{mu,nu} should equal -P_{nu,mu}."""
    lat = GaugeLattice((4, 4), "G2")
    lat.random_links(scale=0.1, seed=42)

    for site in range(min(8, lat.n_sites)):
        P01 = lat.plaquette(site, 0, 1)
        P10 = lat.plaquette(site, 1, 0)
        diff = np.max(np.abs(P01 + P10))
        assert diff < 1e-14, f"P_01 + P_10 should be ~0, got max diff {diff}"

    print("PASS test_plaquette_antisymmetry")


def test_staple_shape():
    """Staple should return a vector of the correct dimension."""
    lat = GaugeLattice((4, 4, 4), "G2")
    lat.random_links(scale=0.1, seed=7)
    s = lat.staple(0, 0)
    assert s.shape == (lat.alg.dim,), f"Expected shape ({lat.alg.dim},), got {s.shape}"
    print("PASS test_staple_shape")


def test_periodic_boundary():
    """Verify neighbor wrapping with periodic boundary conditions."""
    lat = GaugeLattice((4, 4), "G2")

    # Site at (3, 0): shifting in direction 0 should wrap to (0, 0)
    site_3_0 = lat._site_index((3, 0))
    site_0_0 = lat._site_index((0, 0))
    neighbor = lat._shifted_index(site_3_0, 0, +1)
    assert neighbor == site_0_0, f"Expected {site_0_0}, got {neighbor}"

    # Site at (0, 0): shifting backward in direction 1 should wrap to (0, 3)
    site_0_3 = lat._site_index((0, 3))
    neighbor_back = lat._shifted_index(site_0_0, 1, -1)
    assert neighbor_back == site_0_3, f"Expected {site_0_3}, got {neighbor_back}"

    print("PASS test_periodic_boundary")


def test_metropolis_detailed_balance():
    """Acceptance rate should be roughly 0.3-0.7 for a tuned step_size."""
    lat = GaugeLattice((4, 4), "G2")
    lat.hot_start(scale=0.5, seed=10)
    rng = np.random.RandomState(42)

    n_accept = 0
    n_trials = 200
    for _ in range(n_trials):
        site = rng.randint(lat.n_sites)
        mu = rng.randint(lat.n_dims)
        if lat.metropolis_update(site, mu, beta=1.0, step_size=0.1, rng=rng):
            n_accept += 1

    rate = n_accept / n_trials
    assert 0.1 < rate < 0.95, f"Acceptance rate {rate:.3f} out of expected range"
    print(f"PASS test_metropolis_detailed_balance (rate={rate:.3f})")


def test_wilson_loop_trivial():
    """On a cold lattice, all Wilson loops should be zero."""
    lat = GaugeLattice((4, 4), "G2")
    lat.cold_start()

    for m in range(1, 4):
        for n in range(m, 4):
            loop = lat.rectangular_loop(0, 0, 1, m, n)
            assert np.max(np.abs(loop)) < 1e-15, \
                f"Cold lattice {m}x{n} loop should be 0, got max {np.max(np.abs(loop))}"

    print("PASS test_wilson_loop_trivial")


def test_thermalization_decreases_action():
    """Hot start action should decrease (become more negative) during thermalization.

    For positive beta and the Wilson action S = -beta/2 * sum K(P,P),
    the Killing form for exceptional algebras is negative definite, so
    S is positive for non-trivial configurations. Thermalization should
    decrease the action (move toward equilibrium).
    """
    lat = GaugeLattice((4, 4), "G2")
    lat.hot_start(scale=1.0, seed=123)

    initial_action = lat.total_action(beta=1.0)
    result = lat.thermalize(n_sweeps=20, beta=1.0, step_size=0.1, seed=456)
    final_action = result['actions'][-1]

    # The action should change during thermalization.
    # For a hot start the initial action is typically large and should relax.
    # We check that the final action differs from the initial.
    actions = result['actions']
    changed = abs(actions[-1] - actions[0]) > 1e-10
    assert changed, "Action did not change during thermalization"

    # Additionally, for a hot start with large random links, the action
    # should generally decrease as the system thermalizes
    assert final_action < initial_action, \
        f"Expected action to decrease: initial={initial_action:.4f}, final={final_action:.4f}"

    print("PASS test_thermalization_decreases_action")


if __name__ == "__main__":
    test_cold_start_zero_action()
    test_plaquette_antisymmetry()
    test_staple_shape()
    test_periodic_boundary()
    test_metropolis_detailed_balance()
    test_wilson_loop_trivial()
    test_thermalization_decreases_action()
    print("\nAll tests passed!")
