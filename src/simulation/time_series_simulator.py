"""Roll forward SCMs over time with configurable noise and lags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph
from scm.base_scm import BaseSCM


NoiseKind = Literal["gaussian", "laplace", "uniform", "skew"]


@dataclass
class NoiseSpec:
    """Exogenous noise distribution per variable."""

    kind: NoiseKind = "gaussian"
    scale: float = 1.0
    scales: Optional[np.ndarray] = None  # per-var override

    def sample(self, n_vars: int, rng: np.random.Generator, t: int) -> np.ndarray:
        scales = np.full(n_vars, self.scale, dtype=float)
        if self.scales is not None:
            scales = np.asarray(self.scales, dtype=float).ravel()
            if scales.size != n_vars:
                raise ValueError("scales must match n_vars")
        if self.kind == "gaussian":
            return rng.normal(size=n_vars) * scales
        if self.kind == "laplace":
            return rng.laplace(size=n_vars) * scales
        if self.kind == "uniform":
            return (rng.random(n_vars) - 0.5) * 2 * scales * np.sqrt(3)
        if self.kind == "skew":
            # mild non-Gaussian mixture
            z = rng.normal(size=n_vars)
            skew = rng.binomial(1, 0.3, size=n_vars) * 2.0 - 1.0
            return z * (1.0 + 0.5 * skew) * scales
        raise ValueError(f"unknown noise kind {self.kind}")


@dataclass
class SimulationResult:
    """Observed trajectory and metadata."""

    data: np.ndarray  # shape (T, n_obs)
    full_state: np.ndarray  # shape (T, n_vars) including latent
    ground_truth: LaggedAdjacencyGraph
    observed_indices: np.ndarray
    convergence_failures: int = 0
    extras: dict = field(default_factory=dict)


class TimeSeriesSimulator:
    """Unrolled simulator with explicit lag history."""

    def __init__(
        self,
        scm: BaseSCM,
        max_lag: int,
        noise: NoiseSpec,
        observed_indices: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.scm = scm
        self.max_lag = int(max_lag)
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        if observed_indices is None:
            obs = [i for i in range(scm.n_vars) if i not in scm.latent_indices]
            self.observed_indices = np.asarray(obs, dtype=int)
        else:
            self.observed_indices = np.asarray(list(observed_indices), dtype=int)

    def run(
        self,
        length: int,
        ground_truth: LaggedAdjacencyGraph,
        fp_tol: float = 1e-6,
        fp_max_iter: int = 3000,
        damping: float = 1.0,
        burn_in: int = 0,
    ) -> SimulationResult:
        """Simulate ``length`` steps after optional burn-in."""
        T = int(length) + int(burn_in)
        n = self.scm.n_vars
        hist = np.zeros((self.max_lag, n), dtype=float)
        traj = np.zeros((T, n), dtype=float)
        fails = 0
        x_prev = np.zeros(n, dtype=float)
        for t in range(T):
            noise = self.noise.sample(n, self.rng, t)
            res = self.scm.simulate_timestep(
                hist,
                noise,
                x_init=x_prev,
                tol=fp_tol,
                max_iter=fp_max_iter,
                damping=damping,
            )
            if not res.converged:
                fails += 1
            x_t = res.x
            traj[t] = x_t
            if self.max_lag > 0:
                hist[1:] = hist[:-1]
                hist[0] = x_t
            x_prev = x_t

        if burn_in > 0:
            traj = traj[burn_in:]

        obs = traj[:, self.observed_indices]
        sub_gt = ground_truth.subgraph_observed(self.observed_indices)
        return SimulationResult(
            data=obs,
            full_state=traj,
            ground_truth=sub_gt,
            observed_indices=self.observed_indices.copy(),
            convergence_failures=fails,
        )
