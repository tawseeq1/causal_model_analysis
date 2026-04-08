"""Lagged adjacency representation for time-series causal graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LaggedAdjacencyGraph:
    """Directed graph over variables with explicit lag structure.

    Adjacency ``A[i, j, ell]`` is 1 (or a weight) if an edge exists from
    variable ``j`` at time ``t - ell`` to variable ``i`` at time ``t``.
    ``ell = 0`` is contemporaneous; ``ell > 0`` is lagged.

    Attributes
    ----------
    n_vars
        Number of observed variables.
    max_lag
        Maximum lag index ``L`` (lags ``0..L`` inclusive).
    adjacency
        Array of shape ``(n_vars, n_vars, max_lag + 1)``.
    var_names
        Optional human-readable names aligned with the first axis.
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
        """Build from array; infer ``n_vars`` and ``max_lag`` from trailing dims."""
        adjacency = np.asarray(adjacency, dtype=float)
        if adjacency.ndim != 3:
            raise ValueError("adjacency must be 3-D (n_vars, n_vars, n_lags)")
        n_vars, n2, n_lags = adjacency.shape
        if n_vars != n2:
            raise ValueError("first two dimensions must match (square per lag slice)")
        max_lag = n_lags - 1
        return cls(n_vars=n_vars, max_lag=max_lag, adjacency=adjacency, var_names=list(var_names) if var_names else None)

    def copy(self) -> "LaggedAdjacencyGraph":
        """Deep copy of adjacency."""
        names = list(self.var_names) if self.var_names else None
        return LaggedAdjacencyGraph(
            n_vars=self.n_vars,
            max_lag=self.max_lag,
            adjacency=self.adjacency.copy(),
            var_names=names,
        )

    def to_binary(self, threshold: float = 1e-8) -> "LaggedAdjacencyGraph":
        """Binarize weights for structural comparison."""
        bin_adj = (np.abs(self.adjacency) > threshold).astype(float)
        return LaggedAdjacencyGraph(
            n_vars=self.n_vars,
            max_lag=self.max_lag,
            adjacency=bin_adj,
            var_names=list(self.var_names) if self.var_names else None,
        )

    def edge_list(self, weighted: bool = False) -> List[Tuple[int, int, int, float]]:
        """Return list of (target, source, lag, weight)."""
        out: List[Tuple[int, int, int, float]] = []
        for ell in range(self.max_lag + 1):
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    w = float(self.adjacency[i, j, ell])
                    if weighted:
                        if abs(w) > 0:
                            out.append((i, j, ell, w))
                    else:
                        if abs(w) > 0:
                            out.append((i, j, ell, w))
        return out

    def subgraph_observed(self, observed_indices: Sequence[int]) -> "LaggedAdjacencyGraph":
        """Restrict to a subset of observed variable indices."""
        idx = list(observed_indices)
        sub = self.adjacency[np.ix_(idx, idx, range(self.max_lag + 1))]
        names = [self.var_names[i] for i in idx] if self.var_names else None
        return LaggedAdjacencyGraph(n_vars=len(idx), max_lag=self.max_lag, adjacency=sub, var_names=names)
