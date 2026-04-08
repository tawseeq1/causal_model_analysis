"""Graph comparison metrics: P/R/F1, SHD, orientation, effect error."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph


@dataclass
class GraphMetricsResult:
    """Container for graph-level evaluation outputs."""

    precision: float
    recall: float
    f1: float
    shd: float
    orientation_accuracy: float
    n_predicted_edges: int
    n_true_edges: int
    n_correct_adj: int
    n_correct_oriented: int
    effect_mae: Optional[float] = None
    effect_rmse: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _binarize(A: np.ndarray, thr: float) -> np.ndarray:
    return (np.abs(A) > thr).astype(np.int8)


def _undirected_skeleton(A: np.ndarray) -> np.ndarray:
    """Skeleton: union of directed edges ignoring orientation."""
    return ((A + A.T) > 0).astype(np.int8)


def compare_graphs(
    truth: LaggedAdjacencyGraph,
    pred: LaggedAdjacencyGraph,
    threshold: float = 1e-8,
    weight_truth: Optional[LaggedAdjacencyGraph] = None,
    weight_pred: Optional[LaggedAdjacencyGraph] = None,
) -> GraphMetricsResult:
    """Compare predicted lagged graph to ground truth.

    Precision/recall/F1 are computed on **adjacency** (edge existence, any lag).

    Orientation accuracy counts an edge as correctly oriented only if the
    undirected skeleton matches **and** the directed edge set matches for that
    unordered pair at the same lag (allowing at most one direction in truth).

    SHD is Hamming distance between binary adjacency tensors (symmetrized per
    lag slice for structural error counting on directed edges: we use directed
    SHD on the stacked tensor).

    Parameters
    ----------
    truth, pred
        Graphs with identical ``n_vars`` and ``max_lag``.
    threshold
        Threshold for binarizing weighted adjacencies.
    weight_truth, weight_pred
        Optional weighted graphs for MAE/RMSE on overlapping directed edges.
    """
    if truth.n_vars != pred.n_vars or truth.max_lag != pred.max_lag:
        raise ValueError("truth and pred must share n_vars and max_lag")
    T = _binarize(truth.adjacency, threshold)
    P = _binarize(pred.adjacency, threshold)

    # Adjacency sets: directed edges (i <- j at lag ell)
    t_edges = np.stack([T[:, :, ell] for ell in range(truth.max_lag + 1)], axis=-1)
    p_edges = np.stack([P[:, :, ell] for ell in range(pred.max_lag + 1)], axis=-1)

    t_flat = t_edges.reshape(-1)
    p_flat = p_edges.reshape(-1)

    tp = int(np.sum((t_flat == 1) & (p_flat == 1)))
    fp = int(np.sum((t_flat == 0) & (p_flat == 1)))
    fn = int(np.sum((t_flat == 1) & (p_flat == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if tp == 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if tp == 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Directed SHD on full tensor
    shd = float(np.sum(t_flat != p_flat))

    n_true = int(np.sum(t_flat))
    n_pred = int(np.sum(p_flat))

    # Orientation: among positions where skeleton matches (union), check direction match
    # For each (i,j,ell), truth may have at most one of (i,j) or (j,i) for acyclic-ish comparison;
    # we count oriented correct when both agree exactly on directed tensor.
    n_correct_adj = tp
    n_correct_oriented = int(np.sum((T == P) & (T == 1)))

    # Alternative orientation metric: among recovered skeleton edges, fraction with correct direction
    # Build undirected overlap: positions where undirected skeleton matches
    orient_correct = 0
    orient_total = 0
    for ell in range(truth.max_lag + 1):
        Ut = _undirected_skeleton(T[:, :, ell])
        Up = _undirected_skeleton(P[:, :, ell])
        for i in range(truth.n_vars):
            for j in range(i + 1, truth.n_vars):
                if Ut[i, j] == 0:
                    continue
                # If truth has a single direction among (i,j)
                tij = T[i, j, ell] or T[j, i, ell]
                pij = P[i, j, ell] or P[j, i, ell]
                if tij and pij:
                    orient_total += 1
                    if T[i, j, ell] == P[i, j, ell] and T[j, i, ell] == P[j, i, ell]:
                        orient_correct += 1

    orient_acc = orient_correct / orient_total if orient_total > 0 else 1.0

    effect_mae: Optional[float] = None
    effect_rmse: Optional[float] = None
    if weight_truth is not None and weight_pred is not None:
        wt = weight_truth.adjacency
        wp = weight_pred.adjacency
        mask = (T == 1) & (P == 1)
        if np.any(mask):
            diff = wt[mask] - wp[mask]
            effect_mae = float(np.mean(np.abs(diff)))
            effect_rmse = float(np.sqrt(np.mean(diff**2)))

    return GraphMetricsResult(
        precision=prec,
        recall=rec,
        f1=f1,
        shd=shd,
        orientation_accuracy=orient_acc,
        n_predicted_edges=n_pred,
        n_true_edges=n_true,
        n_correct_adj=n_correct_adj,
        n_correct_oriented=n_correct_oriented,
        effect_mae=effect_mae,
        effect_rmse=effect_rmse,
    )
