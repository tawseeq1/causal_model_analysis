"""Run full benchmark pipelines from :class:`ExperimentConfig`."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from algorithms.base import CausalDiscoveryModel
from algorithms.granger import GrangerVARWrapper
from algorithms.lingam_wrapper import VARLiNGAMWrapper
from algorithms.pc_unrolled import PCUnrolledWrapper
from algorithms.pcmci_plus_wrapper import PCMCIPlusWrapper
from algorithms.pcmci_wrapper import PCMCIWrapper
from data.synthetic_dataset import SyntheticDatasetConfig, build_synthetic_dataset
from experiments.config_loader import ExperimentConfig
from graph.graph_representation import LaggedAdjacencyGraph
from graph.metrics import GraphMetricsResult, compare_graphs
from postprocessing.cycle_detection import CycleAwarePostProcessor
from utils.seed import set_global_seed
from visualization.plots import (
    plot_adjacency_heatmap,
    plot_lagged_graph_nx,
    plot_metrics_summary,
    plot_runtime_bar,
    plot_metric_bar,
)


def _make_model(name: str, cfg: ExperimentConfig) -> CausalDiscoveryModel:
    ml = cfg.max_lag
    n = name.lower()
    if n == "pcmci":
        return PCMCIWrapper(max_lag=ml, pc_alpha=cfg.pcmci_alpha, alpha_level=cfg.pcmci_alpha)
    if n in ("pcmciplus", "pcmci+"):
        return PCMCIPlusWrapper(max_lag=ml, pc_alpha=cfg.pcmci_alpha, alpha_level=cfg.pcmci_alpha)
    if n in ("pc", "pc_unrolled"):
        return PCUnrolledWrapper(max_lag=ml, alpha=cfg.pc_alpha)
    if n in ("granger",):
        return GrangerVARWrapper(max_lag=ml, alpha=cfg.pcmci_alpha)
    if n in ("varlingam", "lingam"):
        return VARLiNGAMWrapper(max_lag=ml)
    raise ValueError(f"unknown algorithm: {name}")


class ExperimentRunner:
    """Orchestrates simulation, discovery, optional post-processing, and CSV export."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        set_global_seed(self.cfg.seed)
        out_dir = Path(self.cfg.output_dir) / self.cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)

        syn = SyntheticDatasetConfig(
            n_observed=self.cfg.n_observed,
            n_latent=self.cfg.n_latent,
            max_lag=self.cfg.max_lag,
            length=self.cfg.length,
            graph_type=self.cfg.graph_type,  # type: ignore[arg-type]
            edge_prob=self.cfg.edge_prob,
            scm_kind=self.cfg.scm_kind,  # type: ignore[arg-type]
            nonlinear=self.cfg.nonlinear,
            noise_kind=self.cfg.noise_kind,
            noise_scale=self.cfg.noise_scale,
            feedback_strength=self.cfg.feedback_strength,
            weight_scale=self.cfg.weight_scale,
            seed=self.cfg.seed,
            fp_tol=self.cfg.fp_tol,
            fp_max_iter=self.cfg.fp_max_iter,
            damping=self.cfg.damping,
            burn_in=self.cfg.burn_in,
        )
        sim_res, _scm = build_synthetic_dataset(syn)
        truth = sim_res.ground_truth
        data = sim_res.data

        if self.cfg.save_plots:
            plot_adjacency_heatmap(truth.adjacency, out_dir / "truth_adj.png", title="Ground truth")
            plot_lagged_graph_nx(truth, out_dir / "truth_graph.png")

        # Always save ground-truth adjacency as CSV (one file per lag slice)
        n_vars = truth.n_vars
        vnames = truth.var_names if truth.var_names else [f"X{i}" for i in range(n_vars)]
        for ell in range(truth.max_lag + 1):
            df_lag = pd.DataFrame(
                truth.adjacency[:, :, ell].astype(int),
                index=vnames,
                columns=vnames,
            )
            df_lag.to_csv(out_dir / f"truth_adj_lag{ell}.csv")


        rows: List[Dict[str, Any]] = []
        for algo in self.cfg.algorithms:
            model = _make_model(algo, self.cfg)
            t0 = time.perf_counter()
            model.fit(data)
            runtime = time.perf_counter() - t0
            pred = model.get_graph()
            w_pred = model.get_edge_weights()
            w_pred_graph = None
            if w_pred is not None:
                w_pred_graph = LaggedAdjacencyGraph(
                    n_vars=truth.n_vars,
                    max_lag=truth.max_lag,
                    adjacency=w_pred,
                    var_names=truth.var_names
                )

            weight_truth = None
            if hasattr(_scm, "weights"):
                weight_truth = LaggedAdjacencyGraph(
                    n_vars=truth.n_vars,
                    max_lag=truth.max_lag,
                    adjacency=_scm.weights,
                    var_names=truth.var_names
                )
            metrics = compare_graphs(truth, pred, weight_truth=weight_truth, weight_pred=w_pred_graph)
            row = self._metrics_to_row(
                algo,
                metrics,
                runtime,
                postprocessed=False,
                scm_kind=self.cfg.scm_kind,
            )
            rows.append(row)

            if self.cfg.use_cycle_postprocess:
                cpp = CycleAwarePostProcessor()
                refined = cpp.refine(pred, data=data)
                m2 = compare_graphs(truth, refined, weight_truth=None, weight_pred=None)
                row2 = self._metrics_to_row(
                    algo + "_cpp",
                    m2,
                    runtime,
                    postprocessed=True,
                    scm_kind=self.cfg.scm_kind,
                )
                rows.append(row2)
                pred_for_plot = refined
            else:
                pred_for_plot = pred

            if self.cfg.save_plots:
                plot_adjacency_heatmap(
                    pred_for_plot.adjacency,
                    out_dir / f"pred_adj_{algo}.png",
                    title=f"Predicted ({algo})",
                )
                plot_lagged_graph_nx(
                    pred_for_plot,
                    out_dir / f"pred_graph_{algo}.png",
                )

        df = pd.DataFrame(rows)
        csv_path = out_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        meta_path = out_dir / "config.json"
        meta_path.write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        if self.cfg.save_plots:
            plot_metrics_summary(df, out_dir / "metrics_prf.png", title=f"{self.cfg.name} ({self.cfg.scm_kind})")
            plot_metric_bar(df, "shd", out_dir / "metrics_shd.png", title=f"{self.cfg.name} ({self.cfg.scm_kind}) - SHD", ylabel="SHD (Lower=Better)")
            plot_metric_bar(df, "orientation_accuracy", out_dir / "metrics_orientation.png", title=f"{self.cfg.name} ({self.cfg.scm_kind}) - Orientation", ylabel="Orientation Acc (Higher=Better)")
            plot_metric_bar(df, "effect_rmse", out_dir / "metrics_rmse.png", title=f"{self.cfg.name} ({self.cfg.scm_kind}) - RMSE", ylabel="RMSE (Lower=Better)")
            plot_runtime_bar(df, out_dir / "metrics_runtime.png", title=f"{self.cfg.name} ({self.cfg.scm_kind})")
        return df

    def _metrics_to_row(
        self,
        algo: str,
        m: GraphMetricsResult,
        runtime: float,
        postprocessed: bool,
        scm_kind: str,
    ) -> Dict[str, Any]:
        return {
            "algorithm": algo,
            "scm_kind": scm_kind,
            "postprocessed": postprocessed,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "shd": m.shd,
            "orientation_accuracy": m.orientation_accuracy,
            "n_predicted_edges": m.n_predicted_edges,
            "n_true_edges": m.n_true_edges,
            "effect_mae": m.effect_mae,
            "effect_rmse": m.effect_rmse,
            "runtime_sec": runtime,
        }
