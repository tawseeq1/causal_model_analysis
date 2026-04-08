"""Classical SCM built from lagged adjacency and optional nonlinearities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph
from scm.base_scm import BaseSCM
from scm.structural_equations import StructuralEquation, build_linear_from_adjacency, equations_from_weights


@dataclass
class ClassicalSCMConfig:
    """Configuration for sampling a classical SCM."""

    adjacency: LaggedAdjacencyGraph
    nonlinear: str = "linear"  # linear | poly | mlp
    weight_scale: float = 0.8
    feedback_strength: float = 0.45
    bias: Optional[np.ndarray] = None
    rng: Optional[np.random.Generator] = None


class ClassicalSCM(BaseSCM):
    """SCM with explicit lagged adjacency and sampled weights."""

    def __init__(
        self,
        weights: np.ndarray,
        equations: Sequence[StructuralEquation],
        latent_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(n_vars=len(equations), equations=equations, latent_indices=latent_indices)
        self.weights = np.asarray(weights, dtype=float)

    @classmethod
    def from_config(
        cls,
        cfg: ClassicalSCMConfig,
        latent_indices: Optional[Sequence[int]] = None,
    ) -> "ClassicalSCM":
        """Construct from adjacency mask and sampled weights."""
        rng = cfg.rng or np.random.default_rng()
        adj = cfg.adjacency.adjacency
        w, _ = build_linear_from_adjacency(
            adj,
            rng,
            weight_scale=cfg.weight_scale,
            self_feedback_scale=cfg.feedback_strength,
        )
        eqs = equations_from_weights(w, bias=cfg.bias, nonlinear=cfg.nonlinear, rng=rng)
        return cls(weights=w, equations=eqs, latent_indices=latent_indices)

    def copy(self) -> "ClassicalSCM":
        return ClassicalSCM(
            self.weights.copy(),
            [e.copy() for e in self.equations],
            latent_indices=list(self.latent_indices),
        )
