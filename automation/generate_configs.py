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


BASE_PARAMS = {
    "scm_kind": "classical",
    "nonlinear": "linear",
    "noise_kind": "gaussian",
    "graph_type": "random",
    "n_observed": 6,
    "edge_prob": 0.2,
    "max_lag": 2,
    "feedback_strength": 0.2,
    "weight_scale": 0.5,
    "length": 1000,
}

GRID = {
    "scm_kind": ["classical", "iscm"],
    "nonlinear": ["linear", "poly", "mlp"],
    "noise_kind": ["gaussian", "laplace", "uniform", "skew"],
    "graph_type": ["random", "chain", "cycle"],
    "n_observed": [4, 6, 8],
    "edge_prob": [0.2, 0.4],
    "max_lag": [1, 2, 3, 4],
    "feedback_strength": [0.2],
    "weight_scale": [0.5],
    "length": [500, 1000, 1500, 2000],
}


def generate_ablation_dicts() -> list[tuple[dict, str]]:
    dicts = []
    seen = set()
    for key, values in GRID.items():
        for val in values:
            combo = BASE_PARAMS.copy()
            combo[key] = val
            
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                dicts.append((combo, key))
    return dicts


def build_config(params: dict, is_general: bool = False, varied_var: str = "base") -> dict:
    """Wrap dynamic parameters with static base configuration."""
    cfg = {
        "name": f"exp_{uuid.uuid4().hex[:8]}",
        "output_dir": "outputs/grid_search",
        "algorithms": ["pcmci", "pcmciplus", "pc_unrolled", "granger", "varlingam"],
        "n_latent": 0,
        "seed": 42,
        "save_plots": is_general,
        "use_cycle_postprocess": True,
        "pcmci_alpha": 0.05,
        "pc_alpha": 0.05,
        "noise_scale": 1.0,
        "fp_tol": 1e-6,
        "fp_max_iter": 3000,
        "damping": 1.0,
        "burn_in": 50,
        "varied_var": varied_var,
    }
    cfg.update(params)
    
    if is_general:
        cfg["name"] = f"exp_GENERAL_BASE_{params['graph_type']}_{params['n_observed']}obs_{uuid.uuid4().hex[:4]}"
    else:
        cfg["name"] = f"exp_ablation_{varied_var}_{params[varied_var]}_{uuid.uuid4().hex[:4]}"
        
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment configs.")
    parser.add_argument("--out-dir", type=str, default="configs/grid_search", help="Directory to save config JSONs.")
    parser.add_argument("--n-samples", type=int, default=None, help="Ignored. Added for compatibility.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up old configurations to prevent duplicating grid setups
    for f in out_path.glob("*.json"):
        f.unlink()

    print("Generating Ablation Grid and General Case...")
    
    generated_files = []
    
    # 1. Generate the General Base Config (with plots enabled)
    base_cfg_dict = build_config(BASE_PARAMS, is_general=True, varied_var="base")
    base_filename = out_path / f"{base_cfg_dict['name']}.json"
    with base_filename.open("w", encoding="utf-8") as f:
        json.dump(base_cfg_dict, f, indent=2)
    generated_files.append(base_filename)
    print("Generated General Base Configuration.")

    # 2. Generate Ablation Configurations (one-at-a-time, plots disabled)
    ablation_params = generate_ablation_dicts()
    print(f"Total ablation combinations to test: {len(ablation_params)}")

    for params, varied_var in ablation_params:
        if params == BASE_PARAMS:
            # We already have the exact base config via the 'general' pass
            # But the 'general' pass has save_plots=True. We also want a data point for the ablation
            # with save_plots=False, so let's just generate it anyway to keep the CSV clean.
            pass
            
        cfg_dict = build_config(params, is_general=False, varied_var=varied_var)
        filename = out_path / f"{cfg_dict['name']}.json"
        
        with filename.open("w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
            
        generated_files.append(filename)

    print(f"Successfully generated {len(generated_files)} JSON config files in '{out_path}'.")


if __name__ == "__main__":
    main()
