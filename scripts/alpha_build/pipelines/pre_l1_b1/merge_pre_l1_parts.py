"""Merge sharded PRE_L1/B1 event outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


EVENT_FILES = (
    "pre_l1_near_start_events.parquet",
    "pre_l1_b1_strong_open_break_high_events.parquet",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge PRE_L1/B1 event part outputs")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Merge event parquet files."""
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    output_dir = Path(args.output_dir)
    paths: list[Path] = []
    event_name = ""
    for candidate in EVENT_FILES:
        paths = sorted(parts_dir.rglob(candidate))
        if paths:
            event_name = candidate
            break
    if not paths:
        raise RuntimeError(f"no PRE_L1/B1 event parquet files found under {parts_dir}")

    events = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if "signal_date" in events.columns:
        events["signal_date"] = pd.to_datetime(events["signal_date"]).dt.normalize()
    key = [column for column in ["signal_date", "vt_symbol", "pre_l1_type", "signal_idx"] if column in events.columns]
    if key:
        events = events.drop_duplicates(key)
    sort_key = [column for column in ["signal_date", "vt_symbol", "pre_l1_type"] if column in events.columns]
    if sort_key:
        events = events.sort_values(sort_key).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / event_name
    events.to_parquet(output_path, index=False)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parts_dir": str(parts_dir),
        "event_name": event_name,
        "part_files": [str(path) for path in paths],
        "rows": int(len(events)),
        "symbols": int(events["vt_symbol"].nunique()) if "vt_symbol" in events.columns else None,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"parts={len(paths)} rows={len(events)}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
