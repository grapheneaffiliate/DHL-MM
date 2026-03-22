"""
Defect equation error monitor.

Friedmann-style h^2(t) tracker for structural integrity of computations.
When accumulated deviation from the Z[phi] lattice crosses a threshold,
coefficients are projected back to the nearest lattice point.
"""

import numpy as np

from .zphi import PHI, quantize


class DefectMonitor:
    """h^2(t) = h^2_lambda - (kappa/3) * rho_defect(t)

    h^2_lambda: baseline = Casimir eigenvalue (60 for E8 adjoint)
    kappa: coupling = 1/phi
    rho_defect(t): accumulated deviation from Z[phi] lattice
    """

    def __init__(self, casimir_eigenvalue=60.0):
        self.h2_lambda = casimir_eigenvalue
        self.kappa = 1.0 / PHI
        self.rho_defect = 0.0
        self.h2_history = [self.h2_lambda]
        self.defect_history = [0.0]
        self.pruning_events = 0
        self.threshold = 0.1 * self.h2_lambda

    def measure_defect(self, coefficients):
        """Measure mean squared distance from Z[phi] lattice."""
        total_defect = 0.0
        count = 0
        for c in coefficients:
            if abs(c) < 1e-15:
                continue
            _, err = quantize(c)
            total_defect += err**2
            count += 1
        return total_defect / max(1, count)

    def update(self, coefficients):
        """Update defect tracker. Returns (h2, should_prune)."""
        defect = self.measure_defect(coefficients)
        self.rho_defect += defect
        h2 = self.h2_lambda - (self.kappa / 3.0) * self.rho_defect
        self.h2_history.append(h2)
        self.defect_history.append(defect)
        should_prune = h2 < self.threshold
        if should_prune:
            self.pruning_events += 1
        return h2, should_prune

    def prune(self, coefficients):
        """Project coefficients to nearest Z[phi] lattice points."""
        result = np.zeros_like(coefficients)
        reset_defect = 0.0
        for idx, c in enumerate(coefficients):
            if abs(c) < 1e-15:
                continue
            zp, err = quantize(c)
            result[idx] = zp.to_float()
            reset_defect += err**2
        self.rho_defect = max(0, self.rho_defect - reset_defect)
        return result
