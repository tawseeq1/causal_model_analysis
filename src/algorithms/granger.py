"""Granger causality via statsmodels VAR and pairwise tests."""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms.base import CausalDiscoveryModel
from graph.graph_representation import LaggedAdjacencyGraph


class GrangerVARWrapper(CausalDiscoveryModel):
    """Pairwise Granger tests with Holm-adjusted p-values (optional)."""

    def __init__(self, max_lag: int, alpha: float = 0.05) -> None:
        self.max_lag = max_lag
        self.alpha = alpha
        self._graph: Optional[LaggedAdjacencyGraph] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = np.asarray(data, dtype=float)
        T, n = data.shape
        adj = np.zeros((n, n, self.max_lag + 1), dtype=float)
        pvals: list[tuple[int, int, int, float]] = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x = np.column_stack([data[:, i], data[:, j]])
                try:
                    gc = grangercausalitytests(x, maxlag=self.max_lag, verbose=False)
                except Exception:
                    continue
                for lag in range(1, self.max_lag + 1):
                    if lag not in gc:
                        continue
                    p = float(gc[lag][0]["ssr_ftest"][1])
                    pvals.append((i, j, lag, p))

        if pvals:
            from statsmodels.stats.multitest import multipletests

            p_arr = np.array([p for _, _, _, p in pvals])
            reject, _, _, _ = multipletests(p_arr, alpha=self.alpha, method="holm")
            for k, ok in enumerate(reject):
                if ok:
                    i, j, ell, _ = pvals[k]
                    adj[i, j, ell] = 1.0

        names = [f"X{i}" for i in range(n)]
        self._graph = LaggedAdjacencyGraph(n_vars=n, max_lag=self.max_lag, adjacency=adj, var_names=names)
        self._weights = adj

    def get_graph(self) -> LaggedAdjacencyGraph:
        if self._graph is None:
            raise RuntimeError("call fit() first")
        return self._graph

    def get_edge_weights(self) -> Optional[np.ndarray]:
        return self._weights
