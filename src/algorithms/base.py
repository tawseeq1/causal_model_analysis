"""Abstract interface for causal discovery on multivariate time series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph


class CausalDiscoveryModel(ABC):
    """Unified API for lagged causal discovery."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the model from ``data`` with shape ``(T, n_vars)``."""

    @abstractmethod
    def get_graph(self) -> LaggedAdjacencyGraph:
        """Return discovered graph (binary or weighted)."""

    def get_edge_weights(self) -> Optional[np.ndarray]:
        """Optional edge weights aligned with ``get_graph()`` adjacency tensor."""
        return None
