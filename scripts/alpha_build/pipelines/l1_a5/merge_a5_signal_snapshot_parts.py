"""Merge sharded A5 signal snapshot outputs."""

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


DEFAULT_SNAPSHOT_NAME = "a5_signal_snapshot.parquet"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge A5 signal snapshot part outputs")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot-name", default=DEFAULT_SNAPSHOT_NAME)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def merge_snapshots(paths: list[Path]) -> pd.DataFrame:
    """Read, concatenate, deduplicate, and sort snapshot parts."""
    snapshot = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if "signal_date" in snapshot.columns:
        snapshot["signal_date"] = normalize_date(snapshot["signal_date"])
    if "parent_start_date" in snapshot.columns:
        snapshot["parent_start_date"] = normalize_date(snapshot["parent_start_date"])
    if "event_id" in snapshot.columns:
        snapshot = snapshot.drop_duplicates(["event_id"])
    else:
        key = [
            column
            for column in ["signal_date", "vt_symbol", "buy_point_type", "parent_start_date"]
            if column in snapshot.columns
        ]
        if key:
            snapshot = snapshot.drop_duplicates(key)
    sort_key = [column for column in ["signal_date", "vt_symbol", "buy_point_type"] if column in snapshot.columns]
    if sort_key:
        snapshot = snapshot.sort_values(sort_key).reset_index(drop=True)
    return snapshot


def main() -> None:
    """Merge snapshot parts."""
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    output_dir = Path(args.output_dir)
    paths = sorted(parts_dir.rglob(args.snapshot_name))
    if not paths:
        raise RuntimeError(f"no {args.snapshot_name} files found under {parts_dir}")

    snapshot = merge_snapshots(paths)
    daily_filter = build_daily_filter(snapshot)

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / args.snapshot_name
    daily_path = output_dir / "daily_filter_snapshot.csv"
    schema_path = output_dir / "snapshot_schema.md"
    snapshot.to_parquet(snapshot_path, index=False)
    daily_filter.to_csv(daily_path, index=False)
    if args.write_csv:
        snapshot.to_csv(output_dir / "a5_signal_snapshot.csv", index=False)
    write_schema(schema_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parts_dir": str(parts_dir),
        "snapshot_path": str(snapshot_path),
        "daily_filter_path": str(daily_path),
        "part_files": [str(path) for path in paths],
        "rows": int(len(snapshot)),
        "symbols": int(snapshot["vt_symbol"].nunique()) if "vt_symbol" in snapshot.columns else None,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"parts={len(paths)} rows={len(snapshot)}")
    print(f"wrote {snapshot_path}")
    print(f"wrote {daily_path}")


if __name__ == "__main__":
    main()
