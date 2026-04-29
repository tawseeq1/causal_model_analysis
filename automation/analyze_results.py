"""Generates statistical analysis and visualizations from aggregated results.

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
    """A. Algorithm Comparison: Average metrics across the Base Configuration."""
    # We use the 'base' config runs to compare the overall algorithm performance
    df_algo = df[(df["postprocessed"] == False) & (df["varied_var"] == "base")]
    if df_algo.empty:
        # Fallback if no specific 'base' found
        df_algo = df[df["postprocessed"] == False]
    
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
        plt.title(f"General Case: {ylabel}")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / f"algo_comparison_{metric}.png", dpi=300)
        plt.close()


def plot_ablation_studies(df: pd.DataFrame, out_dir: Path) -> None:
    """B. Ablation Studies: Plot the effect of each of the 10 variables independently."""
    df_algo = df[df["postprocessed"] == False]

    # The 10 variables we varied
    variables = [
        "scm_kind", "nonlinear", "noise_kind", "graph_type",
        "n_observed", "edge_prob", "max_lag", "feedback_strength",
        "weight_scale", "length"
    ]

    key_metrics = ["f1", "shd"]

    for var in variables:
        # To see the true effect of `var`, we only look at rows where `var` was the ONE thing varied,
        # OR the row is the 'base' row (where nothing was varied, so it serves as the control point).
        ablation_df = df_algo[(df_algo["varied_var"] == var) | (df_algo["varied_var"] == "base")]
        
        if ablation_df.empty or ablation_df[var].nunique() <= 1:
            continue

        for m in key_metrics:
            if m not in df.columns: 
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Check if the variable is categorical or continuous to choose the plot type
            is_numeric = pd.api.types.is_numeric_dtype(ablation_df[var])
            
            if is_numeric:
                sns.lineplot(data=ablation_df, x=var, y=m, hue="algorithm", marker="o", errorbar=None)
            else:
                sns.barplot(data=ablation_df, x=var, y=m, hue="algorithm", errorbar=None)
                
            plt.title(f"Ablation Effect of {var} on {m.upper()}")
            plt.ylabel(m.upper())
            plt.xlabel(var.replace("_", " ").title())
            plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save into the analysis plots directory
            plt.savefig(out_dir / f"ablation_{var}_{m}.png", dpi=300)
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

    print("Generating Algorithm Comparisons (General Case)...")
    plot_algorithm_comparison(df, out_dir)

    print("Generating 10 Ablation Variable Plots...")
    plot_ablation_studies(df, out_dir)

    print(f"Analysis complete. All figures saved in {out_dir}")


if __name__ == "__main__":
    main()
