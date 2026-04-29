"""Internally standardized SCM (iSCM) to reduce variance-ordering artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from scm.base_scm import BaseSCM
from scm.classical_scm import ClassicalSCM
from scm.structural_equations import StructuralEquation, gather_parents


@dataclass
class ISCMConfig:
    """Controls for internal standardization."""

    eps: float = 1e-8
    use_running_scale: bool = True


class ISCM(BaseSCM):
    """Wraps a classical SCM by internally standardizing parents before evaluation.

    At each structural evaluation, parent values are transformed to z-scores
    using either per-run statistics (``use_running_scale=False``) or smoothed
    running moments (``use_running_scale=True``) to mimic internal normalization
    without changing the underlying graph.

    Running statistics are stored **per target variable index** so that parent
    vectors of different lengths do not share state.
    """

    def __init__(
        self,
        base_equations: Sequence[StructuralEquation],
        latent_indices: Optional[Sequence[int]] = None,
        cfg: Optional[ISCMConfig] = None,
    ) -> None:
        super().__init__(n_vars=len(base_equations), equations=list(base_equations), latent_indices=latent_indices)
        self.cfg = cfg or ISCMConfig()
        self._means: Dict[int, np.ndarray] = {}
        self._vars: Dict[int, np.ndarray] = {}
        self._momentum = 0.05

    @classmethod
    def from_classical(cls, scm: ClassicalSCM, cfg: Optional[ISCMConfig] = None) -> "ISCM":
        """Share equation topology with a classical SCM."""
        obj = cls(scm.equations, latent_indices=list(scm.latent_indices), cfg=cfg)
        obj.weights = scm.weights
        return obj

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
        """Reset running moments (call before a new independent trajectory)."""
        self._means.clear()
        self._vars.clear()

    def copy(self) -> "ISCM":
        other = ISCM([e.copy() for e in self.equations], latent_indices=list(self.latent_indices), cfg=self.cfg)
        other._means = {k: v.copy() for k, v in self._means.items()}
        other._vars = {k: v.copy() for k, v in self._vars.items()}
        return other
