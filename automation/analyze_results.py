"""Generates rigorous statistical analysis and visualizations from aggregated results.

Produces bar plots, line plots, heatmaps, and boxplots analyzing the 
sensitivity, SCM behavior, and failure modes of causal discovery algorithms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_algorithm_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """A. Algorithm Comparison: Average F1, SHD, Runtime across all conditions."""
    df_algo = df[df["postprocessed"] == False]  # Base comparison without postprocessing
    
    metrics = {
        "f1": "F1 Score (Higher=Better)",
        "precision": "Precision (Higher=Better)",
        "recall": "Recall (Higher=Better)",
        "orientation_accuracy": "Orientation Accuracy (Higher=Better)",
        "shd": "SHD (Lower=Better)",
        "effect_mae": "Effect MAE (Lower=Better)",
        "effect_rmse": "Effect RMSE (Lower=Better)",
        "n_predicted_edges": "Predicted Edges",
        "runtime_sec": "Runtime (Seconds)"
    }
    for metric, ylabel in metrics.items():
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_algo, x="algorithm", y=metric, capsize=0.1, errorbar="sd")
        plt.title(f"Algorithm Comparison: {ylabel}")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / f"algo_comparison_{metric}.png", dpi=300)
        plt.close()


def plot_sensitivity_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    """B. Sensitivity Analysis: Graph Size, Sample Size, Noise x Nonlinearity."""
    df_algo = df[df["postprocessed"] == False]

    key_metrics = ["f1", "shd", "orientation_accuracy"]
    
    for m in key_metrics:
        if m not in df.columns: continue
        
        # Lineplot: Performance vs Graph Size
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_algo, x="n_observed", y=m, hue="algorithm", marker="o", errorbar="se")
        plt.title(f"Performance vs Graph Size ({m.upper()})")
        plt.ylabel(m.upper())
        plt.xlabel("Number of Observed Variables")
        plt.legend(title="Algorithm")
        plt.tight_layout()
        plt.savefig(out_dir / f"sensitivity_graph_size_{m}.png", dpi=300)
        plt.close()
    
        # Lineplot: Performance vs Sample length
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_algo, x="length", y=m, hue="algorithm", marker="s", errorbar="se")
        plt.title(f"Performance vs Time Series Length ({m.upper()})")
        plt.ylabel(m.upper())
        plt.xlabel("Sample Size (Length)")
        plt.tight_layout()
        plt.savefig(out_dir / f"sensitivity_sample_length_{m}.png", dpi=300)
        plt.close()
    
        # Heatmap: Noise x Nonlinearity (Average across top algorithms)
        pivot_df = df_algo.pivot_table(index="noise_kind", columns="nonlinear", values=m, aggfunc="mean")
        if not pivot_df.empty:
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title(f"Average {m.upper()}: Noise Type vs Nonlinearity")
            plt.tight_layout()
            plt.savefig(out_dir / f"sensitivity_heatmap_{m}_noise_nonlinear.png", dpi=300)
            plt.close()


def plot_scm_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """C. SCM Comparison: Classical vs ISCM over identical algorithms."""
    if "scm_kind" not in df.columns:
        return
        
    df_algo = df[df["postprocessed"] == False]
    
    for m in ["f1", "shd", "orientation_accuracy"]:
        if m not in df.columns: continue
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_algo, x="algorithm", y=m, hue="scm_kind", capsize=0.1)
        plt.title(f"SCM Comparison: Classical vs ISCM ({m.upper()})")
        plt.ylabel(f"{m.upper()}")
        plt.legend(title="SCM Kind")
        plt.tight_layout()
        plt.savefig(out_dir / f"scm_comparison_{m}.png", dpi=300)
        plt.close()


def plot_failure_modes(df: pd.DataFrame, out_dir: Path) -> None:
    """D. Failure Modes: Cycles, Post-processing drops, Feedback Strength."""
    # Boxplot: Performance across Graph Types (Highlighting Cycles vs Hubs vs Random)
    if "graph_type" not in df.columns:
        return
        
    df_algo = df[df["postprocessed"] == False]
    
    for m in ["f1", "shd"]:
        if m not in df.columns: continue
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_algo, x="graph_type", y=m, hue="algorithm")
        plt.title(f"Failure Mode Detection: {m.upper()} by Underlying Graph Topology")
        plt.ylabel(f"{m.upper()}")
        plt.tight_layout()
        plt.savefig(out_dir / f"failure_graph_types_{m}.png", dpi=300)
        plt.close()
        
        # Special pointplot for cycles: Impact of Feedback Strength
        cycles_only = df_algo[df_algo["graph_type"] == "cycle"]
        if not cycles_only.empty and "feedback_strength" in cycles_only.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=cycles_only, x="feedback_strength", y=m, hue="algorithm", marker="X")
            plt.title(f"Cycle Graph Failure Mode: {m.upper()} vs Feedback Strength")
            plt.ylabel(f"{m.upper()}")
            plt.xlabel("Feedback Strength")
            plt.tight_layout()
            plt.savefig(out_dir / f"failure_cycles_feedback_{m}.png", dpi=300)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze results CSV and produce comprehensive plots.")
    parser.add_argument("--results-csv", type=str, default="outputs/grid_search/master_results.csv", help="Master aggregated CSV.")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis_plots", help="Directory to save figures.")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    out_dir = Path(args.out_dir)

    if not results_path.exists():
        print(f"Error: Could not find aggregated results at {results_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {results_path}...")
    df = pd.read_csv(results_path)
    
    if df.empty:
        print("Dataframe is empty.")
        return

    sns.set_theme(style="whitegrid", palette="muted")

    print("Generating Algorithm Comparisons...")
    plot_algorithm_comparison(df, out_dir)

    print("Generating Sensitivity Analysis...")
    plot_sensitivity_analysis(df, out_dir)

    print("Generating SCM Comparisons...")
    plot_scm_comparison(df, out_dir)

    print("Generating Failure Mode Analysis...")
    plot_failure_modes(df, out_dir)

    print(f"Analysis complete. All figures saved in {out_dir}")


if __name__ == "__main__":
    main()
