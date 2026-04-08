"""Synthetic lagged graphs, latent confounders, and SCM instantiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from graph.graph_representation import LaggedAdjacencyGraph
from scm.classical_scm import ClassicalSCM, ClassicalSCMConfig
from scm.iscm import ISCM
from simulation.time_series_simulator import NoiseSpec, SimulationResult, TimeSeriesSimulator


GraphType = Literal["random", "chain", "cycle", "hub"]


@dataclass
class SyntheticDatasetConfig:
    """Specification for a synthetic experiment."""

    n_observed: int = 4
    n_latent: int = 0
    max_lag: int = 2
    length: int = 500
    graph_type: GraphType = "random"
    edge_prob: float = 0.35
    scm_kind: Literal["classical", "iscm"] = "classical"
    nonlinear: str = "linear"
    noise_kind: str = "gaussian"
    noise_scale: float = 1.0
    feedback_strength: float = 0.45
    weight_scale: float = 0.8
    seed: int = 0
    fp_tol: float = 1e-6
    fp_max_iter: int = 3000
    damping: float = 1.0
    burn_in: int = 50


def _random_binary_matrix(
    n: int,
    max_lag: int,
    rng: np.random.Generator,
    p: float,
    allow_self_loops: bool = True,
) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for ell in range(max_lag + 1):
        for i in range(n):
            for j in range(n):
                if ell == 0 and i == j and not allow_self_loops:
                    continue
                if rng.random() < p:
                    A[i, j, ell] = 1.0
    return A


def _chain_graph(n: int, max_lag: int) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for i in range(1, n):
        A[i, i - 1, 1] = 1.0
    return A


def _cycle_graph(n: int, max_lag: int) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    for i in range(n):
        A[i, (i - 1) % n, 0] = 1.0
    return A


def _hub_graph(n: int, max_lag: int, rng: np.random.Generator) -> np.ndarray:
    A = np.zeros((n, n, max_lag + 1))
    hub = int(rng.integers(0, n))
    for j in range(n):
        if j == hub:
            continue
        A[j, hub, rng.integers(0, max_lag + 1)] = 1.0
    return A


def build_adjacency(cfg: SyntheticDatasetConfig, rng: np.random.Generator) -> LaggedAdjacencyGraph:
    """Create a binary lagged adjacency for the full system (observed + latent)."""
    n_tot = cfg.n_observed + cfg.n_latent
    if cfg.graph_type == "random":
        base = _random_binary_matrix(n_tot, cfg.max_lag, rng, cfg.edge_prob)
    elif cfg.graph_type == "chain":
        base = _chain_graph(n_tot, cfg.max_lag)
    elif cfg.graph_type == "cycle":
        base = _cycle_graph(n_tot, cfg.max_lag)
    elif cfg.graph_type == "hub":
        base = _hub_graph(n_tot, cfg.max_lag, rng)
    else:
        raise ValueError(cfg.graph_type)

    # Latent confounders: connect latents to multiple observed children
    if cfg.n_latent > 0:
        obs_idx = np.arange(cfg.n_observed)
        lat_idx = np.arange(cfg.n_observed, n_tot)
        for z in lat_idx:
            targets = rng.choice(obs_idx, size=min(3, cfg.n_observed), replace=False)
            for tgt in targets:
                ell = int(rng.integers(0, cfg.max_lag + 1))
                base[tgt, z, ell] = 1.0

    names = [f"X{i}" for i in range(cfg.n_observed)] + [f"Z{i}" for i in range(cfg.n_latent)]
    return LaggedAdjacencyGraph(n_vars=n_tot, max_lag=cfg.max_lag, adjacency=base, var_names=names)


def build_synthetic_dataset(cfg: SyntheticDatasetConfig) -> Tuple[SimulationResult, Union[ClassicalSCM, ISCM]]:
    """Generate SCM, simulate observed series, and return ground-truth graph on observed nodes."""
    rng = np.random.default_rng(cfg.seed)
    adj = build_adjacency(cfg, rng)
    ccfg = ClassicalSCMConfig(
        adjacency=adj,
        nonlinear=cfg.nonlinear,
        weight_scale=cfg.weight_scale,
        feedback_strength=cfg.feedback_strength,
        rng=rng,
    )
    latent_indices = list(range(cfg.n_observed, cfg.n_observed + cfg.n_latent))
    classical = ClassicalSCM.from_config(ccfg, latent_indices=latent_indices)

    if cfg.scm_kind == "iscm":
        scm: Union[ClassicalSCM, ISCM] = ISCM.from_classical(classical)
        scm.reset_running_stats()
    else:
        scm = classical

    noise = NoiseSpec(kind=cfg.noise_kind, scale=cfg.noise_scale)  # type: ignore[arg-type]
    sim = TimeSeriesSimulator(
        scm,
        max_lag=cfg.max_lag,
        noise=noise,
        observed_indices=np.arange(cfg.n_observed),
        rng=rng,
    )
    res = sim.run(
        length=cfg.length,
        ground_truth=adj,
        fp_tol=cfg.fp_tol,
        fp_max_iter=cfg.fp_max_iter,
        damping=cfg.damping,
        burn_in=cfg.burn_in,
    )
    return res, scm
