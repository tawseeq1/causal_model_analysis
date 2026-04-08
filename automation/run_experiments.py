"""Executes the `main.py` pipeline for a batch of configuration JSONs.

Handles parallel execution using ThreadPoolExecutor, captures logs, 
and gracefully skips/logs failures to ensure the pipeline finishes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm is not installed in the environment."""
        return iterable


def run_experiment(config_path: Path, main_script: Path) -> dict:
    """Run a single experiment configuration using the CLI."""
    cmd = [sys.executable, str(main_script), "--config", str(config_path)]
    try:
        # Run process, capturing stdout and stderr
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 min timeout per config
        )
        return {"config": str(config_path), "success": True, "error": None}
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error in {config_path.name}: {e.stderr.strip()}", file=sys.stderr)
        return {"config": str(config_path), "success": False, "error": f"Process exited with {e.returncode}: {e.stderr.strip()}"}
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] {config_path.name} timed out after {e.timeout}s", file=sys.stderr)
        return {"config": str(config_path), "success": False, "error": f"Timeout {e.timeout}s"}
    except Exception as e:
        print(f"[ERROR] Exception on {config_path.name}: {str(e)}", file=sys.stderr)
        return {"config": str(config_path), "success": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple experiment configs.")
    parser.add_argument("--configs-dir", type=str, default="configs/grid_search", help="Directory with config JSONs.")
    parser.add_argument("--main-script", type=str, default="main.py", help="Path to main.py script.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    main_script = Path(args.main_script).resolve()

    if not configs_dir.exists() or not configs_dir.is_dir():
        print(f"Error: Configs directory '{configs_dir}' does not exist.")
        sys.exit(1)
        
    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        print(f"No JSON configs found in '{configs_dir}'. Exiting.")
        sys.exit(0)

    print(f"Found {len(config_files)} configurations to run.")
    print(f"Using {args.workers} parallel workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_experiment, conf, main_script): conf for conf in config_files}
        
        for future in tqdm(as_completed(futures), total=len(config_files), desc="Running Experiments"):
            results.append(future.result())

    # Summary
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    print("-" * 50)
    print(f"Execution complete! Total: {len(results)}, Success: {success_count}, Failed: {fail_count}")
    
    if fail_count > 0:
        print("Failures:")
        for r in results:
            if not r["success"]:
                print(f"  - {Path(r['config']).name}: {r['error']}")


if __name__ == "__main__":
    main()
