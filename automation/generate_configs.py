"""Generates JSON configuration files from a combinatorial grid of hyperparameters.

Can optionally sample `n_samples` to avoid massive full-grid runtimes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import uuid
from pathlib import Path


GRID = {
    # Core modeling components (Baseline vs Complex)
    "scm_kind": ["classical", "iscm"],
    "nonlinear": ["linear", "mlp"],
    
    # Noise distribution (Gaussian vs Non-Gaussian for algorithms like LiNGAM)
    "noise_kind": ["gaussian", "laplace"],
    
    # Structure (Standard vs Feedback)
    "graph_type": ["random", "cycle"],
    
    # Fixed parameters (reduced to 1 option to kill combinatorial explosion)
    "n_observed": [5],
    "edge_prob": [0.2],
    "max_lag": [2],
    "feedback_strength": [0.2],
    "weight_scale": [0.5],
    
    # Sample complexity (Small vs Large data regime)
    "length": [500, 2000],
}
# GRID = {
#     "scm_kind": ["classical", "iscm"],
#     "nonlinear": ["linear", "poly", "mlp"],
#     "noise_kind": ["gaussian", "laplace", "uniform", "skew"],
#     "graph_type": ["random", "chain", "cycle"],
#     "n_observed": [4, 6, 8],
#     "edge_prob": [0.2],
#     "max_lag": [1, 2, 3, 4],
#     "feedback_strength": [0.2],
#     "weight_scale": [0.5],
#     "length": [500, 1000, 1500, 2000],
# }


def generate_grid_dicts() -> list[dict]:
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = list(itertools.product(*values))
    
    dicts = []
    for combo in combinations:
        dicts.append(dict(zip(keys, combo)))
    return dicts


def build_config(params: dict) -> dict:
    """Wrap dynamic parameters with static base configuration."""
    # Hardcoded base parameters ensuring consistency across the pipeline
    cfg = {
        "name": f"exp_{uuid.uuid4().hex[:8]}",
        "output_dir": "outputs/grid_search",
        "algorithms": ["pcmci", "pcmciplus", "pc_unrolled", "granger", "varlingam"],
        "n_latent": 0,
        "seed": 42,  # Fixed seed for reproducibility initially, could be dynamic
        "save_plots": True,  # NOW ENABLED! Generates truth and predicted graph images per run
        "use_cycle_postprocess": True,
        "pcmci_alpha": 0.05,
        "pc_alpha": 0.05,
        "noise_scale": 1.0,
        "fp_tol": 1e-6,
        "fp_max_iter": 3000,
        "damping": 1.0,
        "burn_in": 50,
    }
    cfg.update(params)
    # Semantic name based on key parameters
    cfg["name"] = f"exp_{params['graph_type']}_{params['n_observed']}obs_{params['scm_kind']}_{params['nonlinear']}_lag{params['max_lag']}"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment configs.")
    parser.add_argument("--out-dir", type=str, default="configs/grid_search", help="Directory to save config JSONs.")
    parser.add_argument("--n-samples", type=int, default=None, help="Sample a random subset to avoid massive grids.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up old configurations to prevent duplicating grid setups
    for f in out_path.glob("*.json"):
        f.unlink()

    print(f"Generating full combinatorial grid...")
    all_params = generate_grid_dicts()
    print(f"Total theoretical combinations: {len(all_params)}")

    if args.n_samples is not None and args.n_samples < len(all_params):
        print(f"Sampling {args.n_samples} configurations (seed={args.seed})...")
        random.seed(args.seed)
        all_params = random.sample(all_params, args.n_samples)

    generated_files = []
    for i, params in enumerate(all_params):
        # Allow multiple seeds per structural config if desired by attaching seed to the combo
        cfg_dict = build_config(params)
        # Unique identifier to avoid overwriting identical param names
        filename = out_path / f"{cfg_dict['name']}_{uuid.uuid4().hex[:4]}.json"
        
        with filename.open("w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
            
        generated_files.append(filename)

    print(f"Successfully generated {len(generated_files)} JSON config files in '{out_path}'.")


if __name__ == "__main__":
    main()
