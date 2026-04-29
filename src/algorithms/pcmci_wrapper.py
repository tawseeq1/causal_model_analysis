"""PCMCI (Tigramite) wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.base import CausalDiscoveryModel
from graph.graph_representation import LaggedAdjacencyGraph


class PCMCIWrapper(CausalDiscoveryModel):
    """PCMCI with partial correlation conditional independence testing."""

    def __init__(
        self,
        max_lag: int,
        pc_alpha: float = 0.05,
        alpha_level: float = 0.05,
        cond_ind_test: str = "parcorr",
        verbosity: int = 0,
    ) -> None:
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity
        self._graph: Optional[LaggedAdjacencyGraph] = None
        self._val_matrix: Optional[np.ndarray] = None
        self._results: Dict[str, Any] = {}

    def fit(self, data: np.ndarray) -> None:
        import tigramite.data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        data = np.asarray(data, dtype=float)
        T, n = data.shape
        var_names = [f"X{i}" for i in range(n)]
        dataframe = pp.DataFrame(
            data=data,
            datatime=np.arange(T, dtype=int),
            var_names=var_names,
        )
        if self.cond_ind_test == "parcorr":
            cit = ParCorr(significance="analytic")
        elif self.cond_ind_test == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC

            cit = GPDC(significance="analytic")
        else:
            raise ValueError(f"unsupported cond_ind_test {self.cond_ind_test}")

        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=self.verbosity)
        results = pcmci.run_pcmci(
            tau_max=self.max_lag,
            pc_alpha=self.pc_alpha,
            alpha_level=self.alpha_level,
        )
        self._results = results
        graph = results["graph"]
        val_matrix = results["val_matrix"]
        adj = np.zeros((n, n, self.max_lag + 1), dtype=float)
        val_adj = np.zeros((n, n, self.max_lag + 1), dtype=float)
        for i in range(n):
            for j in range(n):
                for ell in range(self.max_lag + 1):
                    val = str(graph[i, j, ell]).strip()
                    if val == "-->":
                        adj[j, i, ell] = 1.0
                        val_adj[j, i, ell] = val_matrix[i, j, ell]
                    elif val in ("o-o", "x-x"):
                        adj[j, i, ell] = 0.5
                        val_adj[j, i, ell] = val_matrix[i, j, ell]
        self._val_matrix = val_adj
        self._graph = LaggedAdjacencyGraph(n_vars=n, max_lag=self.max_lag, adjacency=adj, var_names=var_names)

    def get_graph(self) -> LaggedAdjacencyGraph:
        if self._graph is None:
            raise RuntimeError("call fit() first")
        return self._graph

    def get_edge_weights(self) -> Optional[np.ndarray]:
        return self._val_matrix
