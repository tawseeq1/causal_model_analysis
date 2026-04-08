"""Deterministic seeding for NumPy, SciPy, and optional PyTorch."""

from __future__ import annotations

import os
import random
from typing import Any, Optional


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across common backends.

    Parameters
    ----------
    seed
        Integer seed applied to Python, NumPy, and (if importable) PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch may fail to import if compiled against a different NumPy ABI;
        # this is non-fatal — we simply skip torch seeding.
        pass


def rng(seed: Optional[int] = None) -> Any:
    """Return a NumPy Generator for local stochastic draws.

    Parameters
    ----------
    seed
        If None, uses non-deterministic entropy (still isolated per call site).
    """
    import numpy as np

    return np.random.default_rng(seed)
