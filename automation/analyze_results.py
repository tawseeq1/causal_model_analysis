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
    
    metrics = {"f1": "F1 Score (Higher=Better)", "shd": "SHD (Lower=Better)", "runtime_sec": "Runtime (Seconds)"}
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

    # Lineplot: Performance vs Graph Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_algo, x="n_observed", y="f1", hue="algorithm", marker="o", errorbar="se")
    plt.title("Performance vs Graph Size (Number of Variables)")
    plt.ylabel("F1 Score")
    plt.xlabel("Number of Observed Variables")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_graph_size.png", dpi=300)
    plt.close()

    # Lineplot: Performance vs Sample length
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_algo, x="length", y="f1", hue="algorithm", marker="s", errorbar="se")
    plt.title("Performance vs Time Series Length")
    plt.ylabel("F1 Score")
    plt.xlabel("Sample Size (Length)")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_sample_length.png", dpi=300)
    plt.close()

    # Heatmap: Noise x Nonlinearity (Average F1 across top algorithms)
    pivot_df = df_algo.pivot_table(index="noise_kind", columns="nonlinear", values="f1", aggfunc="mean")
    if not pivot_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
        plt.title("Average F1 Score: Noise Type vs Nonlinearity")
        plt.tight_layout()
        plt.savefig(out_dir / "sensitivity_heatmap_noise_nonlinear.png", dpi=300)
        plt.close()


def plot_scm_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """C. SCM Comparison: Classical vs ISCM over identical algorithms."""
    if "scm_kind" not in df.columns:
        return
        
    df_algo = df[df["postprocessed"] == False]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_algo, x="algorithm", y="f1", hue="scm_kind", capsize=0.1)
    plt.title("SCM Comparison: Classical vs ISCM (F1 Score)")
    plt.ylabel("F1 Score")
    plt.legend(title="SCM Kind")
    plt.tight_layout()
    plt.savefig(out_dir / "scm_comparison_f1.png", dpi=300)
    plt.close()


def plot_failure_modes(df: pd.DataFrame, out_dir: Path) -> None:
    """D. Failure Modes: Cycles, Post-processing drops, Feedback Strength."""
    # Boxplot: Performance across Graph Types (Highlighting Cycles vs Hubs vs Random)
    if "graph_type" not in df.columns:
        return
        
    df_algo = df[df["postprocessed"] == False]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_algo, x="graph_type", y="f1", hue="algorithm")
    plt.title("Failure Mode Detection: F1 by Underlying Graph Topology")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_graph_types.png", dpi=300)
    plt.close()
    
    # Special pointplot for cycles: Impact of Feedback Strength
    cycles_only = df_algo[df_algo["graph_type"] == "cycle"]
    if not cycles_only.empty and "feedback_strength" in cycles_only.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=cycles_only, x="feedback_strength", y="f1", hue="algorithm", marker="X")
        plt.title("Cycle Graph Failure Mode: F1 vs Feedback Strength")
        plt.ylabel("F1 Score")
        plt.xlabel("Feedback Strength")
        plt.tight_layout()
        plt.savefig(out_dir / "failure_cycles_feedback.png", dpi=300)
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
