"""Orchestrator script that runs the entire end-to-end experimental automation pipeline.

Workflow:
1. Generate configurations (grid search)
2. Run experiments (parallel execution)
3. Aggregate results (CSVs)
4. Analyze and plot (seaborn/pandas)
5. Generate Markdown report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], step_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"=== STEP: {step_name} ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed at step: {step_name}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full causal discovery experimental pipeline.")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of random grid configs to test. Defaults to 10 for quick testing.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for the runner.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    automation_dir = root_dir / "automation"
    
    gen_configs_py = automation_dir / "generate_configs.py"
    run_exp_py = automation_dir / "run_experiments.py"
    agg_res_py = automation_dir / "aggregate_results.py"
    analyze_py = automation_dir / "analyze_results.py"
    gen_rep_py = automation_dir / "generate_report.py"

    # 1. Generate Configs
    cmd_gen = [sys.executable, str(gen_configs_py)]
    if args.n_samples > 0:
        cmd_gen.extend(["--n-samples", str(args.n_samples)])
    run_cmd(cmd_gen, "Configuration Grid Generation")

    # 2. Run Experiments
    cmd_run = [sys.executable, str(run_exp_py), "--workers", str(args.workers), "--main-script", str(root_dir / "main.py")]
    run_cmd(cmd_run, "Experiment Execution")

    # 3. Aggregate
    cmd_agg = [sys.executable, str(agg_res_py)]
    run_cmd(cmd_agg, "Result Aggregation")

    # 4. Analyze
    cmd_anl = [sys.executable, str(analyze_py)]
    run_cmd(cmd_anl, "Statistical Analysis & Plotting")

    # 5. Report
    cmd_rep = [sys.executable, str(gen_rep_py)]
    run_cmd(cmd_rep, "Final Report Generation")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("Checked 'outputs/final_report.md' and 'outputs/analysis_plots/' for your complete benchmark overview.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
