"""Bongers-style fixed-point solvers for cyclic structural assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class FixedPointResult:
    """Outcome of a fixed-point solve."""

    x: np.ndarray
    iterations: int
    converged: bool
    final_delta: float


def solve_fixed_point(
    x_init: np.ndarray,
    update: Callable[[np.ndarray], np.ndarray],
    tol: float = 1e-6,
    max_iter: int = 2000,
    damping: float = 1.0,
) -> FixedPointResult:
    """Iterate ``x <- (1-a)*x + a*G(x)`` until ``||G(x)-x|| < tol`` or ``max_iter``.

    This implements synchronous updates suitable for Bongers-style solvability
    of structural equations ``x = G(x)`` at a single time index.

    Parameters
    ----------
    x_init
        Initial iterate (e.g., previous timestep values).
    update
        Function ``G`` returning the structural RHS before noise (noise added outside).
    tol
        Convergence threshold on the update residual ``||G(x)-x||``.
    max_iter
        Maximum iterations.
    damping
        Damping factor ``a`` in ``(0,1]`` for stability on stiff feedback.
    """
    x = np.asarray(x_init, dtype=float).copy()
    a = float(np.clip(damping, 1e-6, 1.0))
    last_delta = float("inf")
    for k in range(max_iter):
        gx = np.asarray(update(x), dtype=float)
        step = gx - x
        last_delta = float(np.linalg.norm(step))
        if last_delta < tol:
            x_new = x + a * step
            return FixedPointResult(x=x_new, iterations=k + 1, converged=True, final_delta=last_delta)
        x = x + a * step
    return FixedPointResult(x=x, iterations=max_iter, converged=False, final_delta=last_delta)


def spectral_radius_upper_bound(W0: np.ndarray) -> float:
    """Upper bound on rho(W0) using power iteration (few steps)."""
    w = np.asarray(W0, dtype=float)
    n = w.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(30):
        v = w @ v
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return 0.0
        v = v / nv
    lam = float(v @ (w @ v))
    return abs(lam)
