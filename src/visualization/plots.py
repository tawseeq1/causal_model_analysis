"""Adjacency heatmaps and NetworkX visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from graph.graph_representation import LaggedAdjacencyGraph


def plot_adjacency_heatmap(adjacency: np.ndarray, path: Path, title: str = "") -> None:
    """Save a heatmap of flattened lag slices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(adjacency, dtype=float)
    n, n2, L = a.shape
    mat = np.concatenate([a[:, :, ell] for ell in range(L)], axis=1)
    fig, ax = plt.subplots(figsize=(4 + L, 4))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("lag blocks (j)")
    ax.set_ylabel("i")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lagged_graph_nx(
    graph: LaggedAdjacencyGraph,
    path: Path,
    layout_seed: int = 0,
) -> None:
    """Draw a union graph over all lags with edge labels ``(lag)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    G = nx.DiGraph()
    for i in range(graph.n_vars):
        name = graph.var_names[i] if graph.var_names else f"X{i}"
        G.add_node(i, label=name)
    for ell in range(graph.max_lag + 1):
        for i in range(graph.n_vars):
            for j in range(graph.n_vars):
                w = float(graph.adjacency[i, j, ell])
                if abs(w) < 1e-8:
                    continue
                if G.has_edge(j, i):
                    G[j][i]["label"] += f",{ell}"
                else:
                    G.add_edge(j, i, label=str(ell))
    pos = nx.spring_layout(G, seed=layout_seed)
    labels = nx.get_edge_attributes(G, "label")
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_color="#88c", node_size=600)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={i: (graph.var_names[i] if graph.var_names else f"X{i}") for i in G.nodes},
        font_size=9,
    )
    nx.draw_networkx_edges(G, pos, edge_color="#444", arrows=True, arrowsize=16)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics_summary(df: pd.DataFrame, path: Path, title: str = "") -> None:
    """Save a grouped bar chart of precision, recall, and F1 by algorithm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    work = df.copy()
    if "postprocessed" in work.columns:
        work = work[work["postprocessed"].astype(bool) == False]
    if work.empty:
        work = df.copy()
    algos = work["algorithm"].astype(str).tolist()
    x = np.arange(len(algos))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(algos) * 0.9), 5))
    ax.bar(x - w, work["precision"], width=w, label="Precision", color="#4477aa")
    ax.bar(x, work["recall"], width=w, label="Recall", color="#66ccee")
    ax.bar(x + w, work["f1"], width=w, label="F1", color="#228833")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.set_title(title or "Discovery metrics (non–post-processed rows)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_runtime_bar(df: pd.DataFrame, path: Path, title: str = "") -> None:
    """Save runtime bar chart in seconds (non–post-processed rows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    work = df.copy()
    if "postprocessed" in work.columns:
        work = work[work["postprocessed"].astype(bool) == False]
    if work.empty:
        work = df.copy()
    labs = work["algorithm"].astype(str).tolist()
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(max(8, len(labs) * 0.8), 4.5))
    ax.bar(x, work["runtime_sec"].to_numpy(), color="#cc6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title(title or "Algorithm runtime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
