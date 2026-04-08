"""Loaders for CauseMe benchmarks and public climate-like series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class CauseMeLoader:
    """Download and parse CauseMe challenge CSV files."""

    cache_dir: Path = Path("data_cache") / "causeme"

    def download_csv(self, url: str, filename: str) -> Path:
        """Fetch ``url`` into ``cache_dir/filename``."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / filename
        if dest.exists():
            return dest
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    def load(self, url: str, filename: str, max_rows: Optional[int] = None) -> np.ndarray:
        """Return numeric array ``(T, n_vars)`` from CSV."""
        path = self.download_csv(url, filename)
        df = pd.read_csv(path, nrows=max_rows)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("no numeric columns in CauseMe CSV")
        return numeric.to_numpy(dtype=float)


@dataclass
class ClimateLoader:
    """Fetch simple monthly climate indices (CSV) as a surrogate for reanalysis."""

    def load_noaa_nino34(self, cache_dir: Path = Path("data_cache") / "climate") -> Tuple[np.ndarray, list[str]]:
        """Download NOAA Niño 3.4 index (monthly) as a univariate series."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
        dest = cache_dir / "sstoi.indices"
        if not dest.exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
        text = dest.read_text(errors="ignore").splitlines()
        rows = []
        for line in text:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            rows.append(vals)
        arr = np.asarray(rows, dtype=float)
        cols = ["col_" + str(i) for i in range(arr.shape[1])]
        return arr, cols

    def load_era5_sample_optional(self, path: Optional[Path] = None) -> Optional[np.ndarray]:
        """If ``path`` to NetCDF exists, load surface temperature slice via ``xarray``."""
        if path is None or not path.exists():
            return None
        try:
            import xarray as xr
        except ImportError:
            return None
        ds = xr.open_dataset(path)
        # Heuristic: first data variable
        name = list(ds.data_vars)[0]
        da = ds[name]
        arr = np.asarray(da.values, dtype=float)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        return arr
