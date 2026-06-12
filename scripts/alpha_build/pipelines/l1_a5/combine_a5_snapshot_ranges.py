"""Combine two or more complete A5 snapshot date ranges."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
TV_ALPHA = REPO_ROOT / "examples" / "alpha_research" / "tradingview" / "vnpy_alpha"
sys.path.insert(0, str(TV_ALPHA))
sys.path.insert(0, str(REPO_ROOT))

from build_a5_signal_snapshot import build_daily_filter, normalize_date, write_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Combine complete A5 snapshot ranges")
    parser.add_argument("--snapshot-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Combine snapshots."""
    args = parse_args()
    paths = [Path(path) for path in args.snapshot_path]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing snapshots: {missing}")

    snapshot = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    snapshot["signal_date"] = normalize_date(snapshot["signal_date"])
    if "parent_start_date" in snapshot.columns:
        snapshot["parent_start_date"] = normalize_date(snapshot["parent_start_date"])
    dedupe_key = ["event_id"] if "event_id" in snapshot.columns else ["signal_date", "vt_symbol", "buy_point_type"]
    snapshot = snapshot.drop_duplicates(dedupe_key)
    snapshot = snapshot.sort_values(["signal_date", "vt_symbol", "buy_point_type"]).reset_index(drop=True)
    daily_filter = build_daily_filter(snapshot)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "a5_signal_snapshot.parquet"
    daily_path = output_dir / "daily_filter_snapshot.csv"
    schema_path = output_dir / "snapshot_schema.md"
    snapshot.to_parquet(snapshot_path, index=False)
    daily_filter.to_csv(daily_path, index=False)
    if args.write_csv:
        snapshot.to_csv(output_dir / "a5_signal_snapshot.csv", index=False)
    write_schema(schema_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_snapshots": [str(path) for path in paths],
        "dedupe_key": dedupe_key,
        "rows": int(len(snapshot)),
        "date_min": str(snapshot["signal_date"].min().date()) if not snapshot.empty else None,
        "date_max": str(snapshot["signal_date"].max().date()) if not snapshot.empty else None,
    }
    (output_dir / "combine_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"inputs={len(paths)} rows={len(snapshot)}")
    print(f"wrote {snapshot_path}")


if __name__ == "__main__":
    main()
