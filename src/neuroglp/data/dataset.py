"""Dataset loading utilities for NeuroGLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from neuroglp.data.openfda_client import fetch_openfda_reports, load_curated_dataset


def load_cached_or_benchmark_dataset(max_records: int = 10000) -> List[Dict[str, Any]]:
    """Load cached adverse event records or fall back to benchmark dataset."""
    try:
        records = fetch_openfda_reports(max_records=max_records, use_cache=True, offline=True)
        if records:
            return records
    except Exception:
        pass
    return load_curated_dataset()
