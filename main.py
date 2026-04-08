"""CLI entry point for JSON-driven experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _setup_path()
    from experiments.config_loader import load_experiment_config
    from experiments.runner import ExperimentRunner

    parser = argparse.ArgumentParser(description="Causal discovery experiments")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to JSON config")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    runner = ExperimentRunner(cfg)
    df = runner.run()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
