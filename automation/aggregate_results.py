"""Scans the outputs directory to load all modular `results.csv` files,
merges them into a single master pandas DataFrame mapping config parameters
to the metric rows, and computes descriptive summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate_results(runs_root: Path) -> pd.DataFrame:
    """Finds all configs and results.csv in the subdirectories of runs_root."""
    all_rows = []
    runs_root = runs_root.resolve()

    print(f"Scanning for result sets in {runs_root}...")

    # Look for any subdirectory containing both `config.json` and `results.csv`
    for config_file in runs_root.rglob("config.json"):
        exp_dir = config_file.parent
        results_file = exp_dir / "results.csv"

        if not results_file.exists():
            continue

        # Safely load config
        try:
            with config_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON {config_file}: {e}")
            continue

        # Safely load the metrics CSV
        try:
            df = pd.read_csv(results_file)
        except Exception as e:
            print(f"Failed to load CSV {results_file}: {e}")
            continue

        # Embed configuration fields into the rows
        # Defining the metadata keys we care about capturing
        meta_keys = [
            "name", "scm_kind", "nonlinear", "noise_kind", "graph_type",
            "n_observed", "edge_prob", "max_lag", "feedback_strength",
            "weight_scale", "length", "seed", "varied_var"
        ]

        for k in meta_keys:
            val = cfg.get(k, None)
            df[k] = val

        # We keep the experiment directory string for tracking back to specific logs
        df["exp_dir_name"] = exp_dir.name
        
        all_rows.append(df)

    if not all_rows:
        print("No valid results found. Returning empty DataFrame.")
        return pd.DataFrame()

    master_df = pd.concat(all_rows, ignore_index=True)
    return master_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate distributed experiment results.")
    parser.add_argument("--outputs-dir", type=str, default="outputs/grid_search", help="Root outputs directory containing run folders.")
    parser.add_argument("--save-path", type=str, default="outputs/grid_search/master_results.csv", help="Path to save the master aggregated CSV.")
    args = parser.parse_args()

    root_dir = Path(args.outputs_dir)
    save_path = Path(args.save_path)

    if not root_dir.exists():
        print(f"Outputs directory '{root_dir}' does not exist.")
        return

    master_df = aggregate_results(root_dir)

    if master_df.empty:
        print("Aggregation failed or yielded no results.")
        return

    # Create destination dir if missing
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save unaggregated master raw data
    master_df.to_csv(save_path, index=False)
    print(f"Master CSV saved to: {save_path} with {len(master_df)} total evaluated algorithm rows.")

    # Compute a quick grouped summary (mean & std for standard metrics across identical configs + algorithm)
    # The grouping keys are essentially everything except the specific seed and metrics.
    group_keys = [
        "algorithm", "scm_kind", "nonlinear", "noise_kind", "graph_type",
        "n_observed", "edge_prob", "max_lag", "feedback_strength", "weight_scale", "length", "postprocessed", "varied_var"
    ]
    
    # Existing metrics
    metrics = ["f1", "shd", "orientation_accuracy", "precision", "recall", "runtime_sec", "effect_mae", "effect_rmse"]
    metrics_present = [m for m in metrics if m in master_df.columns]

    summary_df = master_df.groupby(group_keys)[metrics_present].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-level columns from custom agg (e.g., f1_mean, f1_std)
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    summary_path = save_path.parent / "master_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV (Means & Stds) saved to: {summary_path}")


if __name__ == "__main__":
    main()
