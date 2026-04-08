"""Cycle-aware post-processing for lagged causal graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph


@dataclass
class CyclePostProcessConfig:
    """Hyperparameters for SCC merging and directional R² asymmetry."""

    min_strength: float = 0.25
    r2_gap_threshold: float = 0.05


class CycleAwarePostProcessor:
    """Heuristic refinement for feedback structure.

    Steps:
    1. Build a summary directed graph across all lags (union of edges).
    2. Detect strongly connected components (SCCs) with size > 1 (candidate cycles).
    3. Flag bidirectional reachability across lags as cycle candidates.
    4. For each unordered pair with edges in both directions (possibly different lags),
       compare residual asymmetry from simple linear regressions to break symmetry.
    """

    def __init__(self, cfg: Optional[CyclePostProcessConfig] = None) -> None:
        self.cfg = cfg or CyclePostProcessConfig()

    def _union_graph(self, g: LaggedAdjacencyGraph, thr: float) -> nx.DiGraph:
        G = nx.DiGraph()
        for i in range(g.n_vars):
            G.add_node(i)
        A = g.adjacency
        for ell in range(g.max_lag + 1):
            for i in range(g.n_vars):
                for j in range(g.n_vars):
                    if abs(A[i, j, ell]) > thr:
                        if G.has_edge(j, i):
                            G[j][i]["weight"] = max(G[j][i]["weight"], abs(A[i, j, ell]))
                        else:
                            G.add_edge(j, i, weight=float(abs(A[i, j, ell])))
        return G

    def _r2(self, y: np.ndarray, x: np.ndarray) -> float:
        """Univariate R² of ``y`` regressed on ``x`` (intercept)."""
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float).ravel()
        n = y.size
        if n < 8:
            return 0.0
        X = np.column_stack([np.ones(n), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    def refine(self, graph: LaggedAdjacencyGraph, data: Optional[np.ndarray] = None) -> LaggedAdjacencyGraph:
        """Return a copy of ``graph`` with refined orientations for cycle candidates."""
        out = graph.copy()
        A = out.adjacency.copy()
        thr = self.cfg.min_strength
        G = self._union_graph(out, thr=1e-8)

        sccs: List[Set[int]] = [c for c in nx.strongly_connected_components(G) if len(c) > 1]

        bidirectional_pairs: List[Tuple[int, int]] = []
        for i in range(out.n_vars):
            for j in range(i + 1, out.n_vars):
                if G.has_edge(i, j) and G.has_edge(j, i):
                    bidirectional_pairs.append((i, j))

        # For SCC members, strengthen edges across lags (mark as cycle candidates)
        for comp in sccs:
            comp_list = list(comp)
            for u in comp_list:
                for v in comp_list:
                    if u == v:
                        continue
                    for ell in range(out.max_lag + 1):
                        if abs(A[u, v, ell]) > 0:
                            A[u, v, ell] = max(abs(A[u, v, ell]), self.cfg.min_strength)

        if data is not None:
            X = np.asarray(data, dtype=float)
            for i, j in bidirectional_pairs:
                r_ij = self._r2(X[:, i], X[:, j])
                r_ji = self._r2(X[:, j], X[:, i])
                diff = r_ij - r_ji
                if diff > self.cfg.r2_gap_threshold:
                    for ell in range(out.max_lag + 1):
                        A[j, i, ell] *= 0.35
                elif diff < -self.cfg.r2_gap_threshold:
                    for ell in range(out.max_lag + 1):
                        A[i, j, ell] *= 0.35

        out.adjacency = A
        return out
