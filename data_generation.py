"""Consolidated Synthetic Data Generation Techniques from Current Framework.

This file is a standalone script containing all the techniques, classes, and logic 
used to generate synthetic data, specifically designed for ChatGPT analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence, List, Optional, Literal, Tuple, Union, Set, Dict

import numpy as np


# ---------------------------------------------------------------------------
# 1. GRAPH REPRESENTATION (src/graph/graph_representation.py)
# ---------------------------------------------------------------------------

@dataclass
class LaggedAdjacencyGraph:
    """Directed graph over variables with explicit lag structure.

    Adjacency ``A[i, j, ell]`` is 1 (or a weight) if an edge exists from
    variable ``j`` at time ``t - ell`` to variable ``i`` at time ``t``.
    ``ell = 0`` is contemporaneous; ``ell > 0`` is lagged.
    """
    n_vars: int
    max_lag: int
    adjacency: np.ndarray
    var_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.adjacency = np.asarray(self.adjacency, dtype=float)
        expected = (self.n_vars, self.n_vars, self.max_lag + 1)
        if self.adjacency.shape != expected:
            raise ValueError(f"adjacency must have shape {expected}, got {self.adjacency.shape}")
        if self.var_names is not None and len(self.var_names) != self.n_vars:
            raise ValueError("var_names length must match n_vars")

    @classmethod
    def zeros(cls, n_vars: int, max_lag: int, var_names: Optional[Sequence[str]] = None) -> "LaggedAdjacencyGraph":
        """Empty graph (all zeros)."""
        adj = np.zeros((n_vars, n_vars, max_lag + 1), dtype=float)
        names = list(var_names) if var_names is not None else None
        return cls(n_vars=n_vars, max_lag=max_lag, adjacency=adj, var_names=names)

    @classmethod
    def from_binary(
        cls,
        adjacency: np.ndarray,
        var_names: Optional[Sequence[str]] = None,
    ) -> "LaggedAdjacencyGraph":
        adjacency = np.asarray(adjacency, dtype=float)
        n_vars, n2, n_lags = adjacency.shape
        max_lag = n_lags - 1
        return cls(n_vars=n_vars, max_lag=max_lag, adjacency=adjacency, var_names=list(var_names) if var_names else None)

    def copy(self) -> "LaggedAdjacencyGraph":
        names = list(self.var_names) if self.var_names else None
        return LaggedAdjacencyGraph(
            n_vars=self.n_vars,
            max_lag=self.max_lag,
            adjacency=self.adjacency.copy(),
            var_names=names,
        )

    def to_binary(self, threshold: float = 1e-8) -> "LaggedAdjacencyGraph":
        bin_adj = (np.abs(self.adjacency) > threshold).astype(float)
        return LaggedAdjacencyGraph(
            n_vars=self.n_vars,
            max_lag=self.max_lag,
            adjacency=bin_adj,
            var_names=list(self.var_names) if self.var_names else None,
        )

    def edge_list(self, weighted: bool = False) -> List[Tuple[int, int, int, float]]:
        out: List[Tuple[int, int, int, float]] = []
        for ell in range(self.max_lag + 1):
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    w = float(self.adjacency[i, j, ell])
                    if abs(w) > 0:
                        out.append((i, j, ell, w))
        return out

    def subgraph_observed(self, observed_indices: Sequence[int]) -> "LaggedAdjacencyGraph":
        idx = list(observed_indices)
        sub = self.adjacency[np.ix_(idx, idx, range(self.max_lag + 1))]
        names = [self.var_names[i] for i in idx] if self.var_names else None
        return LaggedAdjacencyGraph(n_vars=len(idx), max_lag=self.max_lag, adjacency=sub, var_names=names)


# ---------------------------------------------------------------------------
# 2. SOLVER (src/scm/solver.py)
# ---------------------------------------------------------------------------

@dataclass
class FixedPointResult:
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


# ---------------------------------------------------------------------------
# 3. STRUCTURAL EQUATIONS (src/scm/structural_equations.py)
# ---------------------------------------------------------------------------

@dataclass
class ParentIndex:
    var: int
    lag: int

class StructuralEquation(ABC):
    parent_specs: Tuple[ParentIndex, ...]
    @abstractmethod
    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        pass
    @abstractmethod
    def copy(self) -> "StructuralEquation":
        pass

class LinearEquation(StructuralEquation):
    def __init__(self, parent_specs: Sequence[ParentIndex], weights: np.ndarray, bias: float = 0.0) -> None:
        self.parent_specs = tuple(parent_specs)
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.bias = float(bias)

    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        pv = np.asarray(parent_values, dtype=float).ravel()
        return float(self.bias + float(np.dot(self.weights, pv)) + noise)

    def copy(self) -> "LinearEquation":
        return LinearEquation(self.parent_specs, self.weights.copy(), self.bias)

class PolynomialEquation(StructuralEquation):
    def __init__(self, parent_specs: Sequence[ParentIndex], linear: np.ndarray, quad: Optional[np.ndarray] = None, bias: float = 0.0) -> None:
        self.parent_specs = tuple(parent_specs)
        self.linear = np.asarray(linear, dtype=float).ravel()
        self.quad = np.asarray(quad, dtype=float) if quad is not None else None
        self.bias = float(bias)

    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        p = np.asarray(parent_values, dtype=float).ravel()
        y = float(self.bias + float(np.dot(self.linear, p)))
        if self.quad is not None:
            y += float(p @ self.quad @ p)
        return y + float(noise)

    def copy(self) -> "PolynomialEquation":
        q = self.quad.copy() if self.quad is not None else None
        return PolynomialEquation(self.parent_specs, self.linear.copy(), q, self.bias)

class MLPEquation(StructuralEquation):
    def __init__(self, parent_specs: Sequence[ParentIndex], hidden_dims: Sequence[int] = (16,), rng: Optional[np.random.Generator] = None) -> None:
        self.parent_specs = tuple(parent_specs)
        self.hidden_dims = tuple(hidden_dims)
        self.rng = rng or np.random.default_rng()
        d_in = len(self.parent_specs)
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        prev = d_in
        for h in self.hidden_dims:
            w = self.rng.normal(scale=0.5 / np.sqrt(prev), size=(prev, h))
            b = np.zeros(h)
            self._weights.append(w)
            self._biases.append(b)
            prev = h
        w_out = self.rng.normal(scale=0.5 / np.sqrt(prev), size=(prev, 1))
        self._weights.append(w_out)
        self._biases.append(np.zeros(1))

    def _forward(self, x: np.ndarray) -> float:
        h = x.astype(float)
        for i, (w, b) in enumerate(zip(self._weights[:-1], self._biases[:-1])):
            h = np.tanh(h @ w + b)
        out = h @ self._weights[-1] + self._biases[-1]
        return float(out.ravel()[0])

    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        p = np.asarray(parent_values, dtype=float).ravel()
        return self._forward(p) + float(noise)

    def copy(self) -> "MLPEquation":
        other = MLPEquation(self.parent_specs, self.hidden_dims, self.rng)
        other._weights = [w.copy() for w in self._weights]
        other._biases = [b.copy() for b in self._biases]
        return other

def gather_parents(x_current: np.ndarray, history: np.ndarray, specs: Sequence[ParentIndex]) -> np.ndarray:
    vals: List[float] = []
    for spec in specs:
        if spec.lag == 0:
            vals.append(float(x_current[spec.var]))
        else:
            idx = spec.lag - 1
            if idx >= history.shape[0]:
                vals.append(0.0)
            else:
                vals.append(float(history[idx, spec.var]))
    return np.asarray(vals, dtype=float)

def build_linear_from_adjacency(adjacency: np.ndarray, rng: np.random.Generator, weight_scale: float = 0.8, self_feedback_scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = np.asarray(adjacency, dtype=float)
    n, n2, nlag = adjacency.shape
    w = np.zeros_like(adjacency)
    sfb = self_feedback_scale if self_feedback_scale is not None else weight_scale * 0.5
    for ell in range(nlag):
        mask = adjacency[:, :, ell] != 0
        for i in range(n):
            for j in range(n):
                if not mask[i, j]:
                    continue
                if ell == 0 and i == j:
                    w[i, j, ell] = rng.uniform(-sfb, sfb)
                else:
                    sgn = -1.0 if rng.random() < 0.5 else 1.0
                    w[i, j, ell] = rng.uniform(0.1, weight_scale) * sgn
    return w, adjacency

def equations_from_weights(weights: np.ndarray, bias: Optional[np.ndarray] = None, nonlinear: str = "linear", rng: Optional[np.random.Generator] = None) -> List[StructuralEquation]:
    weights = np.asarray(weights, dtype=float)
    n, n2, nlag = weights.shape
    rng = rng or np.random.default_rng()
    bias_vec = np.zeros(n) if bias is None else np.asarray(bias, dtype=float).ravel()
    equations: List[StructuralEquation] = []
    for i in range(n):
        parents: List[ParentIndex] = []
        coefs: List[float] = []
        for ell in range(nlag):
            for j in range(n):
                val = weights[i, j, ell]
                if abs(val) < 1e-12:
                    continue
                parents.append(ParentIndex(var=j, lag=ell))
                coefs.append(float(val))
        specs = tuple(parents)
        if nonlinear == "linear":
            equations.append(LinearEquation(specs, np.asarray(coefs), bias=float(bias_vec[i])))
        elif nonlinear == "poly":
            d = len(specs)
            quad = rng.normal(scale=0.05, size=(d, d))
            quad = (quad + quad.T) / 2
            equations.append(PolynomialEquation(specs, np.asarray(coefs), quad=quad, bias=float(bias_vec[i])))
        elif nonlinear == "mlp":
            equations.append(MLPEquation(specs, hidden_dims=(12,), rng=rng))
    return equations


# ---------------------------------------------------------------------------
# 4. BASE SCM (src/scm/base_scm.py)
# ---------------------------------------------------------------------------

class BaseSCM(ABC):
    def __init__(self, n_vars: int, equations: Sequence[StructuralEquation], latent_indices: Optional[Sequence[int]] = None) -> None:
        self.n_vars = int(n_vars)
        self.equations: List[StructuralEquation] = list(equations)
        self.latent_indices: Set[int] = set(latent_indices or [])

    def structural_rhs(self, x_current: np.ndarray, history: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_vars, dtype=float)
        for i, eq in enumerate(self.equations):
            parents = gather_parents(x_current, history, eq.parent_specs)
            out[i] = eq.evaluate(parents, 0.0)
        return out

    def simulate_timestep(self, history: np.ndarray, noise: np.ndarray, x_init: Optional[np.ndarray] = None, tol: float = 1e-6, max_iter: int = 2000, damping: float = 1.0) -> FixedPointResult:
        x0 = np.zeros(self.n_vars) if x_init is None else np.asarray(x_init, dtype=float).copy()
        def update(x: np.ndarray) -> np.ndarray:
            return self.structural_rhs(x, history) + noise
        return solve_fixed_point(x0, update, tol=tol, max_iter=max_iter, damping=damping)

    @abstractmethod
    def copy(self) -> "BaseSCM":
        pass


# ---------------------------------------------------------------------------
# 5. CLASSICAL SCM (src/scm/classical_scm.py)
# ---------------------------------------------------------------------------

@dataclass
class ClassicalSCMConfig:
    adjacency: LaggedAdjacencyGraph
    nonlinear: str = "linear"  # linear | poly | mlp
    weight_scale: float = 0.8
    feedback_strength: float = 0.45
    bias: Optional[np.ndarray] = None
    rng: Optional[np.random.Generator] = None

class ClassicalSCM(BaseSCM):
    def __init__(self, weights: np.ndarray, equations: Sequence[StructuralEquation], latent_indices: Optional[Sequence[int]] = None) -> None:
        super().__init__(n_vars=len(equations), equations=equations, latent_indices=latent_indices)
        self.weights = np.asarray(weights, dtype=float)

    @classmethod
    def from_config(cls, cfg: ClassicalSCMConfig, latent_indices: Optional[Sequence[int]] = None) -> "ClassicalSCM":
        rng = cfg.rng or np.random.default_rng()
        adj = cfg.adjacency.adjacency
        w, _ = build_linear_from_adjacency(adj, rng, weight_scale=cfg.weight_scale, self_feedback_scale=cfg.feedback_strength)
        eqs = equations_from_weights(w, bias=cfg.bias, nonlinear=cfg.nonlinear, rng=rng)
        return cls(weights=w, equations=eqs, latent_indices=latent_indices)

    def copy(self) -> "ClassicalSCM":
        return ClassicalSCM(self.weights.copy(), [e.copy() for e in self.equations], latent_indices=list(self.latent_indices))


# ---------------------------------------------------------------------------
# 6. iSCM (src/scm/iscm.py)
# ---------------------------------------------------------------------------

@dataclass
class ISCMConfig:
    eps: float = 1e-8
    use_running_scale: bool = True

class ISCM(BaseSCM):
    def __init__(self, base_equations: Sequence[StructuralEquation], latent_indices: Optional[Sequence[int]] = None, cfg: Optional[ISCMConfig] = None) -> None:
        super().__init__(n_vars=len(base_equations), equations=list(base_equations), latent_indices=latent_indices)
        self.cfg = cfg or ISCMConfig()
        self._means: Dict[int, np.ndarray] = {}
        self._vars: Dict[int, np.ndarray] = {}
        self._momentum = 0.05

    @classmethod
    def from_classical(cls, scm: ClassicalSCM, cfg: Optional[ISCMConfig] = None) -> "ISCM":
        return cls(scm.equations, latent_indices=list(scm.latent_indices), cfg=cfg)

    def _standardize(self, i: int, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if self.cfg.use_running_scale:
            if i not in self._means:
                self._means[i] = v.copy()
                self._vars[i] = np.ones_like(v)
            else:
                self._means[i] = (1 - self._momentum) * self._means[i] + self._momentum * v
                self._vars[i] = (1 - self._momentum) * self._vars[i] + self._momentum * (v - self._means[i]) ** 2
            std = np.sqrt(self._vars[i] + self.cfg.eps)
            return (v - self._means[i]) / std
        mu = float(np.mean(v))
        std = float(np.std(v) + self.cfg.eps)
        return (v - mu) / std

    def structural_rhs(self, x_current: np.ndarray, history: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_vars, dtype=float)
        for i, eq in enumerate(self.equations):
            raw = gather_parents(x_current, history, eq.parent_specs)
            z = self._standardize(i, raw)
            out[i] = eq.evaluate(z, 0.0)
        return out

    def reset_running_stats(self) -> None:
        self._means.clear()
        self._vars.clear()

    def copy(self) -> "ISCM":
        other = ISCM([e.copy() for e in self.equations], latent_indices=list(self.latent_indices), cfg=self.cfg)
        other._means = {k: v.copy() for k, v in self._means.items()}
        other._vars = {k: v.copy() for k, v in self._vars.items()}
        return other


# ---------------------------------------------------------------------------
# 7. TIME SERIES SIMULATOR (src/simulation/time_series_simulator.py)
# ---------------------------------------------------------------------------

NoiseKind = Literal["gaussian", "laplace", "uniform", "skew"]

@dataclass
class NoiseSpec:
    kind: NoiseKind = "gaussian"
    scale: float = 1.0
    scales: Optional[np.ndarray] = None

    def sample(self, n_vars: int, rng: np.random.Generator, t: int) -> np.ndarray:
        scales = np.full(n_vars, self.scale, dtype=float)
        if self.scales is not None:
            scales = np.asarray(self.scales, dtype=float).ravel()
        if self.kind == "gaussian":
            return rng.normal(size=n_vars) * scales
        if self.kind == "laplace":
            return rng.laplace(size=n_vars) * scales
        if self.kind == "uniform":
            return (rng.random(n_vars) - 0.5) * 2 * scales * np.sqrt(3)
        if self.kind == "skew":
            z = rng.normal(size=n_vars)
            skew = rng.binomial(1, 0.3, size=n_vars) * 2.0 - 1.0
            return z * (1.0 + 0.5 * skew) * scales
        raise ValueError(f"unknown noise kind {self.kind}")

@dataclass
class SimulationResult:
    data: np.ndarray
    full_state: np.ndarray
    ground_truth: LaggedAdjacencyGraph
    observed_indices: np.ndarray
    convergence_failures: int = 0
    extras: dict = field(default_factory=dict)

class TimeSeriesSimulator:
    def __init__(self, scm: BaseSCM, max_lag: int, noise: NoiseSpec, observed_indices: Optional[Sequence[int]] = None, rng: Optional[np.random.Generator] = None) -> None:
        self.scm = scm
        self.max_lag = int(max_lag)
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        if observed_indices is None:
            obs = [i for i in range(scm.n_vars) if i not in scm.latent_indices]
            self.observed_indices = np.asarray(obs, dtype=int)
        else:
            self.observed_indices = np.asarray(list(observed_indices), dtype=int)

    def run(self, length: int, ground_truth: LaggedAdjacencyGraph, fp_tol: float = 1e-6, fp_max_iter: int = 3000, damping: float = 1.0, burn_in: int = 0) -> SimulationResult:
        T = int(length) + int(burn_in)
        n = self.scm.n_vars
        hist = np.zeros((self.max_lag, n), dtype=float)
        traj = np.zeros((T, n), dtype=float)
        fails = 0
        x_prev = np.zeros(n, dtype=float)
        for t in range(T):
            noise = self.noise.sample(n, self.rng, t)
            res = self.scm.simulate_timestep(hist, noise, x_init=x_prev, tol=fp_tol, max_iter=fp_max_iter, damping=damping)
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


# ---------------------------------------------------------------------------
# 8. SYNTHETIC DATASET BUILDER (src/data/synthetic_dataset.py)
# ---------------------------------------------------------------------------

GraphType = Literal["random", "chain", "cycle", "hub"]

@dataclass
class SyntheticDatasetConfig:
    n_observed: int = 4
    n_latent: int = 0
    max_lag: int = 2
    length: int = 500
    graph_type: GraphType = "random"
    edge_prob: float = 0.35
    scm_kind: Literal["classical", "iscm"] = "classical"
    nonlinear: str = "linear"
    noise_kind: str = "gaussian"
    noise_scale: float = 1.0
    feedback_strength: float = 0.45
    weight_scale: float = 0.8
    seed: int = 0
    fp_tol: float = 1e-6
    fp_max_iter: int = 3000
    damping: float = 1.0
    burn_in: int = 50

def _random_binary_matrix(n: int, max_lag: int, rng: np.random.Generator, p: float, allow_self_loops: bool = True) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for ell in range(max_lag + 1):
        for i in range(n):
            for j in range(n):
                if ell == 0 and i == j and not allow_self_loops:
                    continue
                if rng.random() < p:
                    A[i, j, ell] = 1.0
    return A

def _chain_graph(n: int, max_lag: int) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for i in range(1, n):
        A[i, i - 1, 1] = 1.0
    return A

def _cycle_graph(n: int, max_lag: int) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for i in range(n):
        A[i, (i - 1) % n, 0] = 1.0
    return A

def _hub_graph(n: int, max_lag: int, rng: np.random.Generator) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    hub = int(rng.integers(0, n))
    for j in range(n):
        if j == hub:
            continue
        A[j, hub, rng.integers(0, max_lag + 1)] = 1.0
    return A

def build_adjacency(cfg: SyntheticDatasetConfig, rng: np.random.Generator) -> LaggedAdjacencyGraph:
    n_tot = cfg.n_observed + cfg.n_latent
    if cfg.graph_type == "random":
        base = _random_binary_matrix(n_tot, cfg.max_lag, rng, cfg.edge_prob)
    elif cfg.graph_type == "chain":
        base = _chain_graph(n_tot, cfg.max_lag)
    elif cfg.graph_type == "cycle":
        base = _cycle_graph(n_tot, cfg.max_lag)
    elif cfg.graph_type == "hub":
        base = _hub_graph(n_tot, cfg.max_lag, rng)
    else:
        raise ValueError(cfg.graph_type)

    if cfg.n_latent > 0:
        obs_idx = np.arange(cfg.n_observed)
        lat_idx = np.arange(cfg.n_observed, n_tot)
        for z in lat_idx:
            targets = rng.choice(obs_idx, size=min(3, cfg.n_observed), replace=False)
            for tgt in targets:
                ell = int(rng.integers(0, cfg.max_lag + 1))
                base[tgt, z, ell] = 1.0

    names = [f"X{i}" for i in range(cfg.n_observed)] + [f"Z{i}" for i in range(cfg.n_latent)]
    return LaggedAdjacencyGraph(n_vars=n_tot, max_lag=cfg.max_lag, adjacency=base, var_names=names)

def build_synthetic_dataset(cfg: SyntheticDatasetConfig) -> Tuple[SimulationResult, Union[ClassicalSCM, ISCM]]:
    rng = np.random.default_rng(cfg.seed)
    adj = build_adjacency(cfg, rng)
    ccfg = ClassicalSCMConfig(
        adjacency=adj,
        nonlinear=cfg.nonlinear,
        weight_scale=cfg.weight_scale,
        feedback_strength=cfg.feedback_strength,
        rng=rng,
    )
    latent_indices = list(range(cfg.n_observed, cfg.n_observed + cfg.n_latent))
    classical = ClassicalSCM.from_config(ccfg, latent_indices=latent_indices)

    if cfg.scm_kind == "iscm":
        scm: Union[ClassicalSCM, ISCM] = ISCM.from_classical(classical)
        scm.reset_running_stats()
    else:
        scm = classical

    noise = NoiseSpec(kind=cfg.noise_kind, scale=cfg.noise_scale)
    sim = TimeSeriesSimulator(
        scm,
        max_lag=cfg.max_lag,
        noise=noise,
        observed_indices=np.arange(cfg.n_observed),
        rng=rng,
    )
    res = sim.run(
        length=cfg.length,
        ground_truth=adj,
        fp_tol=cfg.fp_tol,
        fp_max_iter=cfg.fp_max_iter,
        damping=cfg.damping,
        burn_in=cfg.burn_in,
    )
    return res, scm
