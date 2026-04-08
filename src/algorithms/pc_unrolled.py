"""PC algorithm on time-unfolded (lag-augmented) observations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from algorithms.base import CausalDiscoveryModel
from graph.graph_representation import LaggedAdjacencyGraph


def unroll_time_series(data: np.ndarray, max_lag: int) -> Tuple[np.ndarray, int, int]:
    """Stack ``[X(t), X(t-1), ..., X(t-max_lag)]`` row-wise.

    Returns
    -------
    unfolded
        Shape ``(T - max_lag, N * (max_lag + 1))``.
    n_vars, max_lag
        Echoed for convenience.
    """
    data = np.asarray(data, dtype=float)
    T, n = data.shape
    rows = []
    for t in range(max_lag, T):
        blocks = [data[t - ell] for ell in range(max_lag + 1)]
        rows.append(np.concatenate(blocks, axis=0))
    return np.asarray(rows), n, max_lag


def map_unfolded_pc_to_lagged(
    adj_unfolded: np.ndarray,
    n_vars: int,
    max_lag: int,
) -> np.ndarray:
    """Map PC adjacency on unfolded nodes to ``(n_vars, n_vars, max_lag+1)``.

    Column layout per row: block ``ell`` occupies indices ``[ell*n_vars, (ell+1)*n_vars)``,
    representing ``X(t-ell)``.

    For each directed edge ``p -> q`` with ``p != q`` on unfolded graph, interpret as
    potential causal influence from source column ``p`` to target column ``q``. We
    project influences **into** current-time targets (block 0): ``q < n_vars``.

    - If ``q`` is in block 0: ``i = q``, ``j = p % n_vars``, ``ell = p // n_vars``.
    - If ``q`` is in a lagged block, we remap relative to the row's temporal anchor.
    """
    n_feat = n_vars * (max_lag + 1)
    if adj_unfolded.shape != (n_feat, n_feat):
        raise ValueError(f"expected adj shape {(n_feat, n_feat)}, got {adj_unfolded.shape}")
    A = np.zeros((n_vars, n_vars, max_lag + 1), dtype=float)
    for p in range(n_feat):
        for q in range(n_feat):
            if abs(adj_unfolded[p, q]) < 1e-12:
                continue
            ell_p = p // n_vars
            j = p % n_vars
            ell_q = q // n_vars
            i = q % n_vars
            lag = ell_p - ell_q
            if 0 <= lag <= max_lag:
                A[i, j, lag] = max(A[i, j, lag], float(abs(adj_unfolded[p, q])))
            elif lag < 0 and ell_q == 0:
                # future block pointing back; ignore or map to positive lag
                lag2 = ell_q - ell_p
                if 0 <= lag2 <= max_lag:
                    A[i, j, lag2] = max(A[i, j, lag2], float(abs(adj_unfolded[p, q])))
    return A


class PCUnrolledWrapper(CausalDiscoveryModel):
    """Constraint-based PC on lag-unfolded data matrix."""

    def __init__(self, max_lag: int, alpha: float = 0.05, stable: bool = True) -> None:
        self.max_lag = max_lag
        self.alpha = alpha
        self.stable = stable
        self._graph: Optional[LaggedAdjacencyGraph] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        from causallearn.search.ConstraintBased.PC import pc

        data = np.asarray(data, dtype=float)
        unfolded, n_vars, ml = unroll_time_series(data, self.max_lag)
        if ml != self.max_lag:
            raise RuntimeError("max_lag mismatch")
        cg = pc(unfolded, alpha=self.alpha, stable=self.stable)
        adj = np.asarray(cg.G.graph, dtype=float)
        lagged = map_unfolded_pc_to_lagged(adj, n_vars, self.max_lag)
        names = [f"X{i}" for i in range(n_vars)]
        self._graph = LaggedAdjacencyGraph(n_vars=n_vars, max_lag=self.max_lag, adjacency=lagged, var_names=names)
        self._weights = lagged

    def get_graph(self) -> LaggedAdjacencyGraph:
        if self._graph is None:
            raise RuntimeError("call fit() first")
        return self._graph

    def get_edge_weights(self) -> Optional[np.ndarray]:
        return self._weights
