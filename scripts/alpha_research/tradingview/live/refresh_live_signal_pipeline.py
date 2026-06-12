"""Compatibility wrapper for the historical full live-refresh pipeline."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TARGET = REPO_ROOT / "scripts/alpha_research/tradingview/live/history/refresh_live_signal_pipeline.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
