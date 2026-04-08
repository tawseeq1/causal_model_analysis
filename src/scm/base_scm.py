"""Abstract base SCM for symbolic structural assignments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Set

import numpy as np

from scm.structural_equations import StructuralEquation, gather_parents
from scm.solver import FixedPointResult, solve_fixed_point


class BaseSCM(ABC):
    """Structural causal model with optional latent variables and cyclic feedback.

    Subclasses define how structural equations are constructed and whether
    internal standardization (iSCM) is applied.
    """

    def __init__(
        self,
        n_vars: int,
        equations: Sequence[StructuralEquation],
        latent_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.n_vars = int(n_vars)
        self.equations: List[StructuralEquation] = list(equations)
        if len(self.equations) != self.n_vars:
            raise ValueError("equations list must have length n_vars")
        self.latent_indices: Set[int] = set(latent_indices or [])

    def structural_rhs(self, x_current: np.ndarray, history: np.ndarray) -> np.ndarray:
        """Evaluate deterministic structural RHS ``g(x_t, x_{t-1},...)`` without noise."""
        out = np.zeros(self.n_vars, dtype=float)
        for i, eq in enumerate(self.equations):
            parents = gather_parents(x_current, history, eq.parent_specs)
            out[i] = eq.evaluate(parents, 0.0)
        return out

    def simulate_timestep(
        self,
        history: np.ndarray,
        noise: np.ndarray,
        x_init: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        max_iter: int = 2000,
        damping: float = 1.0,
    ) -> FixedPointResult:
        """Solve ``x = g(x) + noise`` via fixed-point iteration (Bongers-style)."""
        if noise.shape[0] != self.n_vars:
            raise ValueError("noise must match n_vars")
        x0 = np.zeros(self.n_vars) if x_init is None else np.asarray(x_init, dtype=float).copy()

        def update(x: np.ndarray) -> np.ndarray:
            return self.structural_rhs(x, history) + noise

        return solve_fixed_point(x0, update, tol=tol, max_iter=max_iter, damping=damping)

    @abstractmethod
    def copy(self) -> "BaseSCM":
        """Independent copy for parallel seeds."""
