"""
Quantum simulation module for DHL-MM.

Lie-algebra-valued time evolution using sparse structure constants.
This simulates adjoint-action dynamics (evolving elements of the algebra
under commutator flow), NOT full Hilbert-space quantum mechanics.

The core operation -- the commutator [H, rho] -- uses the sparse bracket
from DHL-MM, giving up to 913x fewer operations than dense matrix methods.
"""

import numpy as np
from typing import List, Dict, Optional
import dhl_mm


class LieHamiltonian:
    """Hamiltonian defined as a Lie algebra element.

    Provides commutator-based time evolution on the algebra, NOT on a
    Hilbert space. The state rho is a vector in the Lie algebra, and
    evolution is driven by the adjoint action [H, rho].
    """

    def __init__(self, algebra_name: str = "E8") -> None:
        """Load a DHL-MM algebra engine.

        Args:
            algebra_name: Name of the exceptional Lie algebra (G2, F4, E6, E7, E8).
        """
        self.alg = dhl_mm.algebra(algebra_name)
        self.algebra_name = algebra_name

    def commutator(self, H: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Compute the sparse Lie bracket [H, rho].

        Args:
            H: Hamiltonian vector in the Lie algebra, shape (dim,).
            rho: State vector in the Lie algebra, shape (dim,).

        Returns:
            [H, rho] as a dim-vector.
        """
        return self.alg.bracket(H, rho)

    def evolve(self, H: np.ndarray, rho_0: np.ndarray, dt: float,
               steps: int, order: int = 2) -> List[np.ndarray]:
        """Trotter-Suzuki time evolution of an algebra element under [H, .].

        This is Lie-algebra-valued evolution (adjoint dynamics), not
        Hilbert-space quantum mechanics.

        For order=1: rho_{n+1} = rho_n + dt * [H, rho_n]
        For order=2: rho_{n+1} = rho_n + dt*[H, rho_n] + (dt^2/2)*[H, [H, rho_n]]

        Args:
            H: Hamiltonian algebra element, shape (dim,).
            rho_0: Initial state, shape (dim,).
            dt: Time step.
            steps: Number of time steps.
            order: Trotter order (1 or 2).

        Returns:
            List of states at each step (length steps+1, including initial).
        """
        trajectory = [rho_0.copy()]
        rho = rho_0.copy()
        bracket = self.alg.bracket
        for _ in range(steps):
            c1 = bracket(H, rho)
            if order >= 2:
                c2 = bracket(H, c1)
                rho = rho + dt * c1 + (dt * dt / 2.0) * c2
            else:
                rho = rho + dt * c1
            trajectory.append(rho.copy())
        return trajectory

    def conserved_check(self, H: np.ndarray, observable: np.ndarray) -> float:
        """Check whether an observable commutes with the Hamiltonian.

        Returns the norm of [H, observable]. If near zero, the observable
        is a conserved quantity under H-evolution.

        Args:
            H: Hamiltonian algebra element, shape (dim,).
            observable: Observable algebra element, shape (dim,).

        Returns:
            Norm of [H, observable] (scalar).
        """
        return float(np.linalg.norm(self.alg.bracket(H, observable)))

    def killing_norm(self, x: np.ndarray) -> float:
        """Killing norm K(x, x).

        Args:
            x: Algebra element, shape (dim,).

        Returns:
            Scalar Killing form value K(x, x).
        """
        return self.alg.killing_form(x, x)


class EquivariantTrotterSuzuki:
    """Trotter-Suzuki integrator for Lie-algebra-valued dynamics.

    Wraps a LieHamiltonian and provides structured time-stepping with
    diagnostics (Killing norm tracking for drift detection).
    """

    def __init__(self, hamiltonian: LieHamiltonian) -> None:
        """Store the LieHamiltonian for evolution.

        Args:
            hamiltonian: A LieHamiltonian instance.
        """
        self.ham = hamiltonian

    def first_order(self, H: np.ndarray, rho: np.ndarray, dt: float) -> np.ndarray:
        """Single first-order Trotter step.

        rho_{n+1} = rho_n + dt * [H, rho_n]

        Args:
            H: Hamiltonian algebra element.
            rho: Current state.
            dt: Time step.

        Returns:
            Updated state.
        """
        return rho + dt * self.ham.commutator(H, rho)

    def second_order(self, H: np.ndarray, rho: np.ndarray, dt: float) -> np.ndarray:
        """Single second-order symmetric Trotter step.

        rho_{n+1} = rho_n + dt*[H, rho_n] + (dt^2/2)*[H, [H, rho_n]]

        Args:
            H: Hamiltonian algebra element.
            rho: Current state.
            dt: Time step.

        Returns:
            Updated state.
        """
        c1 = self.ham.commutator(H, rho)
        c2 = self.ham.commutator(H, c1)
        return rho + dt * c1 + (dt * dt / 2.0) * c2

    def evolve(self, H: np.ndarray, rho_0: np.ndarray, t_final: float,
               dt: float, order: int = 2) -> Dict:
        """Full evolution from t=0 to t_final.

        Args:
            H: Hamiltonian algebra element.
            rho_0: Initial state.
            t_final: Final time.
            dt: Time step size.
            order: 1 or 2.

        Returns:
            Dict with keys:
                'states': list of state arrays at each step
                'times': list of float times
                'killing_norms': list of K(rho, rho) at each step (for drift tracking)
        """
        steps = int(round(t_final / dt))
        step_fn = self.second_order if order == 2 else self.first_order

        states = [rho_0.copy()]
        times = [0.0]
        killing_norms = [self.ham.killing_norm(rho_0)]

        rho = rho_0.copy()
        for i in range(steps):
            rho = step_fn(H, rho, dt)
            states.append(rho.copy())
            times.append((i + 1) * dt)
            killing_norms.append(self.ham.killing_norm(rho))

        return {
            'states': states,
            'times': times,
            'killing_norms': killing_norms,
        }


class E8SpinLattice:
    """Lattice of Lie-algebra-valued spins with nearest-neighbor coupling.

    Each site i carries a vector x_i in the Lie algebra. The evolution
    equation for site i is:

        d(x_i)/dt = J * ([x_{i-1}, x_i] + [x_{i+1}, x_i])

    with open boundary conditions (sites 0 and n_sites-1 have one neighbor).

    This is a classical spin chain on the Lie algebra, NOT a quantum
    Hilbert-space lattice model.
    """

    def __init__(self, n_sites: int, algebra_name: str = "E8",
                 coupling: float = 1.0) -> None:
        """Create a lattice of algebra-valued sites.

        Args:
            n_sites: Number of sites.
            algebra_name: Lie algebra name (G2, F4, E6, E7, E8).
            coupling: Coupling constant J.
        """
        self.n_sites = n_sites
        self.coupling = coupling
        self.ham = LieHamiltonian(algebra_name)
        self.alg = self.ham.alg
        self.dim = self.alg.dim

    def build_hamiltonian(self) -> None:
        """Placeholder — the Hamiltonian for this lattice is implicit.

        The nearest-neighbor interaction is computed on-the-fly during
        evolution as H_eff(i) = J * (x_{i-1} + x_{i+1}), and the update
        for site i is [H_eff(i), x_i]. No single algebra-valued vector
        represents the full Hamiltonian for coupled sites.
        """
        pass  # interaction is computed during evolve()

    def random_initial_state(self, scale: float = 0.1,
                             seed: Optional[int] = None) -> np.ndarray:
        """Generate a random initial lattice state.

        Args:
            scale: Standard deviation of the random components.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (n_sites, dim).
        """
        rng = np.random.RandomState(seed)
        return rng.randn(self.n_sites, self.dim) * scale

    def evolve(self, state: np.ndarray, dt: float, steps: int,
               order: int = 2) -> List[np.ndarray]:
        """Evolve the full lattice under nearest-neighbor commutator flow.

        At each step, for each site i:
            d(x_i)/dt = J * ([x_{i-1}, x_i] + [x_{i+1}, x_i])

        Uses first or second order integration.

        Args:
            state: Initial state, shape (n_sites, dim).
            dt: Time step.
            steps: Number of steps.
            order: Integration order (1 or 2).

        Returns:
            List of state arrays at each step (length steps+1).
        """
        bracket = self.alg.bracket
        J = self.coupling
        n = self.n_sites
        trajectory = [state.copy()]
        current = state.copy()

        for _ in range(steps):
            # Compute first-order derivatives for all sites
            deriv1 = np.zeros_like(current)
            for i in range(n):
                if i > 0:
                    deriv1[i] += bracket(current[i - 1], current[i])
                if i < n - 1:
                    deriv1[i] += bracket(current[i + 1], current[i])
            deriv1 *= J

            if order >= 2:
                # Second-order correction: d2(x_i)/dt^2
                # We need [H_eff, deriv1_i] for each site, where H_eff
                # is the effective field from neighbors.
                # More precisely, for symmetric Trotter:
                # x_new = x + dt*f(x) + (dt^2/2)*df/dx * f(x)
                # For this coupled system, we approximate the second
                # derivative by computing the derivative of deriv1.
                # We evaluate the derivative operator at (current, deriv1).
                deriv2 = np.zeros_like(current)
                for i in range(n):
                    # d/dt of deriv1[i] involves how neighbors change
                    # plus how x_i changes
                    if i > 0:
                        # contribution from d/dt [x_{i-1}, x_i]
                        # = [dx_{i-1}/dt, x_i] + [x_{i-1}, dx_i/dt]
                        deriv2[i] += bracket(deriv1[i - 1], current[i])
                        deriv2[i] += bracket(current[i - 1], deriv1[i])
                    if i < n - 1:
                        deriv2[i] += bracket(deriv1[i + 1], current[i])
                        deriv2[i] += bracket(current[i + 1], deriv1[i])
                deriv2 *= J

                current = current + dt * deriv1 + (dt * dt / 2.0) * deriv2
            else:
                current = current + dt * deriv1

            trajectory.append(current.copy())

        return trajectory

    def measure_killing_norm(self, state: np.ndarray) -> float:
        """Sum of Killing norms K(x_i, x_i) over all sites.

        Args:
            state: Lattice state, shape (n_sites, dim).

        Returns:
            Total Killing norm (scalar).
        """
        total = 0.0
        for i in range(self.n_sites):
            total += self.alg.killing_form(state[i], state[i])
        return total

    def measure_correlations(self, state: np.ndarray, i: int, j: int) -> float:
        """Killing form correlation between two sites.

        Args:
            state: Lattice state, shape (n_sites, dim).
            i: First site index.
            j: Second site index.

        Returns:
            K(x_i, x_j) (scalar).
        """
        return self.alg.killing_form(state[i], state[j])
