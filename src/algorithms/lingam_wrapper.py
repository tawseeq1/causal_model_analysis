"""VARLiNGAM for linear non-Gaussian lagged causal discovery."""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms.base import CausalDiscoveryModel
from graph.graph_representation import LaggedAdjacencyGraph


class VARLiNGAMWrapper(CausalDiscoveryModel):
    """Wrapper around ``lingam.VARLiNGAM``."""

    def __init__(self, max_lag: int, criterion: str = "bic") -> None:
        self.max_lag = max_lag
        self.criterion = criterion
        self._graph: Optional[LaggedAdjacencyGraph] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        from lingam import VARLiNGAM

        data = np.asarray(data, dtype=float)
        _, n = data.shape
        model = VARLiNGAM(lags=self.max_lag, criterion=self.criterion)
        model.fit(data)
        mats = np.asarray(model.adjacency_matrices_, dtype=float)
        adj = np.zeros((n, n, self.max_lag + 1), dtype=float)
        w = np.zeros_like(adj)
        nlags_avail = min(mats.shape[0], self.max_lag + 1)
        for ell in range(nlags_avail):
            B = mats[ell]
            adj[:, :, ell] = (np.abs(B) > 1e-8).astype(float)
            w[:, :, ell] = B
        names = [f"X{i}" for i in range(n)]
        self._graph = LaggedAdjacencyGraph(n_vars=n, max_lag=self.max_lag, adjacency=adj, var_names=names)
        self._weights = w

    def get_graph(self) -> LaggedAdjacencyGraph:
        if self._graph is None:
            raise RuntimeError("call fit() first")
        return self._graph

    def get_edge_weights(self) -> Optional[np.ndarray]:
        return self._weights
