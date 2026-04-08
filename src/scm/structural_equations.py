"""Symbolic structural equations: linear, polynomial, optional MLP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ParentIndex:
    """Single parent reference: variable ``var`` at lag ``lag`` (0 = same time)."""

    var: int
    lag: int


class StructuralEquation(ABC):
    """Maps parent values (fixed order) plus exogenous noise to a scalar response."""

    parent_specs: Tuple[ParentIndex, ...]

    @abstractmethod
    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        """Evaluate structural assignment."""

    @abstractmethod
    def copy(self) -> "StructuralEquation":
        """Deep-enough copy for independent runs."""


class LinearEquation(StructuralEquation):
    """y = bias + sum_k w_k * parent_k + noise."""

    def __init__(self, parent_specs: Sequence[ParentIndex], weights: np.ndarray, bias: float = 0.0) -> None:
        self.parent_specs = tuple(parent_specs)
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.bias = float(bias)
        if self.weights.size != len(self.parent_specs):
            raise ValueError("weights must match number of parents")

    def evaluate(self, parent_values: np.ndarray, noise: float) -> float:
        pv = np.asarray(parent_values, dtype=float).ravel()
        return float(self.bias + float(np.dot(self.weights, pv)) + noise)

    def copy(self) -> "LinearEquation":
        return LinearEquation(self.parent_specs, self.weights.copy(), self.bias)


class PolynomialEquation(StructuralEquation):
    """y = bias + sum_k a_k * p_k + sum_{k<=m} b_{k,m} p_k p_m + noise (degree 2)."""

    def __init__(
        self,
        parent_specs: Sequence[ParentIndex],
        linear: np.ndarray,
        quad: Optional[np.ndarray] = None,
        bias: float = 0.0,
    ) -> None:
        self.parent_specs = tuple(parent_specs)
        self.linear = np.asarray(linear, dtype=float).ravel()
        self.quad = np.asarray(quad, dtype=float) if quad is not None else None
        self.bias = float(bias)
        d = len(self.parent_specs)
        if self.linear.size != d:
            raise ValueError("linear coef size must match parents")
        if self.quad is not None and self.quad.shape != (d, d):
            raise ValueError("quad must be (d,d)")

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
    """Small MLP with tanh activations (optional ``torch`` backend)."""

    def __init__(
        self,
        parent_specs: Sequence[ParentIndex],
        hidden_dims: Sequence[int] = (16,),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
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


def gather_parents(
    x_current: np.ndarray,
    history: np.ndarray,
    specs: Sequence[ParentIndex],
) -> np.ndarray:
    """Collect parent values given current state ``x_current`` and lag history.

    ``history`` has shape ``(max_lag, n_vars)`` where row 0 is ``t-1``, row 1 is ``t-2``, etc.
    """
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


def build_linear_from_adjacency(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    weight_scale: float = 0.8,
    self_feedback_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random stable-ish weights proportional to binary adjacency.

    Parameters
    ----------
    adjacency
        Shape ``(n, n, L+1)`` binary or weighted mask.
    weight_scale
        Global scale for off-diagonal (lagged) edges.
    self_feedback_scale
        If set, scales contemporaneous diagonal/feedback entries separately.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n, n2, nlag = adjacency.shape
    if n != n2:
        raise ValueError("adjacency must be square in first two dims")
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


def equations_from_weights(
    weights: np.ndarray,
    bias: Optional[np.ndarray] = None,
    nonlinear: str = "linear",
    rng: Optional[np.random.Generator] = None,
) -> List[StructuralEquation]:
    """Create per-variable equations from weight tensor ``(n,n,L+1)``."""
    weights = np.asarray(weights, dtype=float)
    n, n2, nlag = weights.shape
    if n != n2:
        raise ValueError("weights must be square on first two axes")
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
            equations.append(
                PolynomialEquation(
                    specs,
                    np.asarray(coefs),
                    quad=quad,
                    bias=float(bias_vec[i]),
                )
            )
        elif nonlinear == "mlp":
            equations.append(MLPEquation(specs, hidden_dims=(12,), rng=rng))
        else:
            raise ValueError(f"unknown nonlinear mode: {nonlinear}")
    return equations
