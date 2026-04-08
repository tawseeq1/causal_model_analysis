"""Load experiment definitions from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    """Top-level experiment specification."""

    name: str = "default"
    seed: int = 0
    output_dir: str = "outputs"
    scm_kind: str = "classical"  # classical | iscm
    graph_type: str = "random"
    n_observed: int = 4
    n_latent: int = 0
    max_lag: int = 2
    length: int = 800
    edge_prob: float = 0.35
    nonlinear: str = "linear"
    noise_kind: str = "gaussian"
    noise_scale: float = 1.0
    feedback_strength: float = 0.45
    weight_scale: float = 0.8
    fp_tol: float = 1e-6
    fp_max_iter: int = 3000
    damping: float = 1.0
    burn_in: int = 50
    algorithms: List[str] = field(
        default_factory=lambda: ["pcmci", "pcmciplus", "pc_unrolled", "granger", "varlingam"]
    )
    pcmci_alpha: float = 0.05
    pc_alpha: float = 0.05
    use_cycle_postprocess: bool = True
    save_plots: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse JSON file into :class:`ExperimentConfig`."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    known = {f.name for f in fields(ExperimentConfig)}
    base: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in known:
            base[k] = v
        else:
            extra[k] = v
    cfg = ExperimentConfig(**base)
    cfg.extra = extra
    return cfg
